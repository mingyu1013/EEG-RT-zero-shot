import os
# ==== torch.compile / dynamo 완전 비활성화 (networkx 버그 회피) ====
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from models_eegnet import (
    EEGNetV4Encoder,
    EEGNetRegressorMLP,
    mae,
    rmse,
    nrmse,
    corr,
)

from preprocessing_ccd_resp import (
    RELEASE_ROOTS,
    get_common_eeg_channels,
    load_ccd_trials_all_releases,
    robust_zscore_per_subject,
    subjectwise_zscore_rt,
    extract_ccd_epochs_for_release,
)

# ============================================================
# Config
# ============================================================

SEED = 2025

# strict split 설정
# → Train / Val / Test(제로샷)
TRAIN_RELEASES = ["R2", "R3", "R5"]
VAL_RELEASES   = ["R1"]
TEST_RELEASES  = ["R4"]

# 학습 설정
BATCH_SIZE          = 64
NUM_EPOCHS_CCD      = 80
LR_CCD              = 1e-3
WEIGHT_DECAY        = 1e-4
EARLY_STOP_PATIENCE = 12

OUT_DIR = Path("ch5_ccd_only_outputs_mlp_strict_respLocked_subjRTz")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Utils
# ============================================================

def set_seed(seed: int = 2025):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            x = torch.randn(1).cuda()
            _ = x * 2
            print("[INFO] Using CUDA")
            return torch.device("cuda")
        except RuntimeError as e:
            print(f"[WARN] CUDA available이지만 실행 실패, CPU로 fallback: {e}")
    print("[INFO] Using CPU")
    return torch.device("cpu")


# ============================================================
# Dataset & train / eval
# ============================================================

class EEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def train_regressor(
    model: EEGNetRegressorMLP,
    X_tr: np.ndarray,
    y_tr_z: np.ndarray,
    X_val: np.ndarray,
    y_val_z: np.ndarray,
    device: torch.device,
) -> dict:
    train_ds = EEGDataset(X_tr, y_tr_z)
    val_ds   = EEGDataset(X_val, y_val_z)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    crit = torch.nn.MSELoss()
    opt  = torch.optim.Adam(model.parameters(), lr=LR_CCD, weight_decay=WEIGHT_DECAY)

    best_state = None
    best_val = np.inf
    best_ep = -1
    no_imp = 0
    hist = {"train_loss": [], "val_loss": []}

    for ep in range(1, NUM_EPOCHS_CCD + 1):
        # train
        model.train()
        tot, n = 0.0, 0
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            out = model(Xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            bs = Xb.size(0)
            tot += float(loss.item()) * bs
            n += bs
        tr_loss = tot / n

        # val
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                yb = yb.to(device)
                out = model(Xb)
                loss = crit(out, yb)
                bs = Xb.size(0)
                tot += float(loss.item()) * bs
                n += bs
        va_loss = tot / n

        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        print(f"[CCD-ONLY][{ep:03d}] train={tr_loss:.4f} | val={va_loss:.4f}")

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = model.state_dict()
            best_ep = ep
            no_imp = 0
        else:
            no_imp += 1

        if no_imp >= EARLY_STOP_PATIENCE:
            print(f"[FINETUNE] early stop @ {ep}, best={best_ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_epoch": best_ep, "best_val_loss": best_val, "history": hist}


def eval_regressor(
    model: EEGNetRegressorMLP,
    X: np.ndarray,
    y_z: np.ndarray,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    ds = EEGDataset(X, y_z)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    crit = torch.nn.MSELoss()
    tot, n = 0.0, 0
    preds, trues = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            out = model(Xb)
            loss = crit(out, yb)
            bs = Xb.size(0)
            tot += float(loss.item()) * bs
            n += bs
            preds.append(out.cpu().numpy())
            trues.append(yb.cpu().numpy())
    loss = tot / n
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return loss, preds, trues


# ============================================================
# Main
# ============================================================

def main():
    set_seed(SEED)
    device = get_device()

    # 1) 공통 채널 계산 (R1~R5 전체)
    all_releases = sorted(list(RELEASE_ROOTS.keys()))
    common_chans = get_common_eeg_channels(all_releases)

    # 2) CCD 메타 로딩 (R1~R5, clean_data 기준)
    used_releases = TRAIN_RELEASES + VAL_RELEASES + TEST_RELEASES
    ccd_df = load_ccd_trials_all_releases(used_releases)
    print(f"[INFO] CCD trials after RT/press_onset filter: {ccd_df.shape}")

    # 3) CCD epoch 추출 (response-locked, R1~R5 모두)
    X_ccd_list, meta_ccd_list = [], []
    for rel in used_releases:
        X_rel, meta_rel = extract_ccd_epochs_for_release(ccd_df, rel, common_chans)
        X_ccd_list.append(X_rel)
        meta_ccd_list.append(meta_rel)

    X_ccd_all = np.concatenate(X_ccd_list, axis=0)
    meta_ccd_all = pd.concat(meta_ccd_list, ignore_index=True)
    print(f"[INFO] CCD all epochs (resp-locked): {X_ccd_all.shape}, meta: {meta_ccd_all.shape}")

    # 4) split 마스크
    rel_arr = meta_ccd_all["release"].values
    train_mask = np.isin(rel_arr, TRAIN_RELEASES)
    val_mask   = np.isin(rel_arr, VAL_RELEASES)
    test_mask  = np.isin(rel_arr, TEST_RELEASES)

    # 5) split별 raw X / meta
    X_tr_raw   = X_ccd_all[train_mask]
    meta_tr    = meta_ccd_all[train_mask].reset_index(drop=True)

    X_val_raw  = X_ccd_all[val_mask]
    meta_val   = meta_ccd_all[val_mask].reset_index(drop=True)

    X_test_raw = X_ccd_all[test_mask]
    meta_test  = meta_ccd_all[test_mask].reset_index(drop=True)

    # 6) split별 EEG subject-wise robust z-score
    X_tr   = robust_zscore_per_subject(X_tr_raw,   meta_tr)
    X_val  = robust_zscore_per_subject(X_val_raw,  meta_val)
    X_test = robust_zscore_per_subject(X_test_raw, meta_test)

    # 7) RT 배열 (초 단위, 나중에 복원용)
    rt_tr   = meta_tr["rt"].values.astype(np.float32)
    rt_val  = meta_val["rt"].values.astype(np.float32)
    rt_test = meta_test["rt"].values.astype(np.float32)

    # (정보용) global train mean/std
    rt_mean_global = float(rt_tr.mean())
    rt_std_global  = float(rt_tr.std())
    print(f"[INFO] RT (train only, global) mean={rt_mean_global:.4f}, std={rt_std_global:.4f}")

    # 8) RT subject-wise z-score (split별)
    y_tr_z,  rt_tr_mean,  rt_tr_std   = subjectwise_zscore_rt(meta_tr,  rt_tr)
    y_val_z, rt_val_mean, rt_val_std  = subjectwise_zscore_rt(meta_val, rt_val)
    y_test_z, rt_test_mean, rt_test_std = subjectwise_zscore_rt(meta_test, rt_test)

    print(f"[INFO] CCD Train={X_tr.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # 9) 모델 구성 (Encoder + MLP head)
    n_chans = X_tr.shape[1]
    n_times = X_tr.shape[2]

    encoder = EEGNetV4Encoder(n_chans=n_chans, n_times=n_times).to(device)
    reg = EEGNetRegressorMLP(encoder, hidden=128, dropout=0.25).to(device)

    print("[INFO] ===== CCD-only RT regression (EEGNet + MLP head, resp-locked, subj-wise RT z-score) =====")
    train_info = train_regressor(reg, X_tr, y_tr_z, X_val, y_val_z, device)
    print(f"[INFO] CCD-only best_epoch={train_info['best_epoch']} | "
          f"best_val_loss={train_info['best_val_loss']:.4f}")

    # 10) 평가 (train / val / test)
    tr_loss_z, tr_pred_z, tr_true_z = eval_regressor(reg, X_tr,  y_tr_z,  device)
    va_loss_z, va_pred_z, va_true_z = eval_regressor(reg, X_val, y_val_z, device)
    te_loss_z, te_pred_z, te_true_z = eval_regressor(reg, X_test, y_test_z, device)

    # 11) z-space → RT(초) 복원 (subject-wise mean/std 사용)
    tr_pred = tr_pred_z * rt_tr_std + rt_tr_mean
    tr_true = rt_tr

    va_pred = va_pred_z * rt_val_std + rt_val_mean
    va_true = rt_val

    te_pred = te_pred_z * rt_test_std + rt_test_mean
    te_true = rt_test

    print("\n=== Train metrics (CCD-only, resp-locked, strict split) ===")
    print(f"MAE={mae(tr_pred, tr_true):.4f} | RMSE={rmse(tr_pred, tr_true):.4f} | "
          f"nRMSE={nrmse(tr_pred, tr_true):.4f} | r={corr(tr_pred, tr_true):.4f}")

    print("\n=== Val metrics (CCD-only, resp-locked, strict split, Val releases) ===")
    print(f"MAE={mae(va_pred, va_true):.4f} | RMSE={rmse(va_pred, va_true):.4f} | "
          f"nRMSE={nrmse(va_pred, va_true):.4f} | r={corr(va_pred, va_true):.4f}")

    print("\n=== Test (zero-shot CCD, resp-locked, strict split) metrics ===")
    print(f"MAE={mae(te_pred, te_true):.4f} | RMSE={rmse(te_pred, te_true):.4f} | "
          f"nRMSE={nrmse(te_pred, te_true):.4f} | r={corr(te_pred, te_true):.4f}")

    # 12) 예측 / 모델 저장
    pred_df = pd.DataFrame({
        "release": np.concatenate([meta_tr["release"], meta_val["release"], meta_test["release"]]),
        "sub":     np.concatenate([meta_tr["sub"],     meta_val["sub"],     meta_test["sub"]]),
        "run":     np.concatenate([meta_tr["run"],     meta_val["run"],     meta_test["run"]]),
        "trial_index": np.concatenate(
            [meta_tr["trial_index"], meta_val["trial_index"], meta_test["trial_index"]]
        ),
        "split":   np.array(["train"] * len(meta_tr)
                            + ["val"] * len(meta_val)
                            + ["test"] * len(meta_test)),
        "rt_true": np.concatenate([tr_true, va_true, te_true]),
        "rt_pred": np.concatenate([tr_pred, va_pred, te_pred]),
    })
    pred_df.to_csv(OUT_DIR / "ccd_only_predictions_mlp_strict_respLocked_subjRTz.csv", index=False)
    print(f"[OK] Saved predictions -> {OUT_DIR / 'ccd_only_predictions_mlp_strict_respLocked_subjRTz.csv'}")

    torch.save(
        {
            "model_state": reg.state_dict(),
            "encoder_state": encoder.state_dict(),
            "train_info": train_info,
            "train_releases": TRAIN_RELEASES,
            "val_releases": VAL_RELEASES,
            "test_releases": TEST_RELEASES,
        },
        OUT_DIR / "eegnet_ch5_ccd_only_mlp_strict_respLocked_subjRTz.pt",
    )
    print(f"[OK] Saved model -> {OUT_DIR / 'eegnet_ch5_ccd_only_mlp_strict_respLocked_subjRTz.pt'}")


if __name__ == "__main__":
    main()
