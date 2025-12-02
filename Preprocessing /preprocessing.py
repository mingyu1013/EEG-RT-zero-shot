import numpy as np
import pandas as pd
import mne
from pathlib import Path

# =========================
# 0. 기본 설정
# =========================
# 이 파일은 ./EEG 폴더 안에서 실행한다고 가정
ROOT = Path(".")  # == ./EEG

RELEASE_DIRS = [
    "R1_mini_L100_bdf",
    "R2_mini_L100_bdf",
    "R3_mini_L100_bdf",
    "R4_mini_L100_bdf",
    "R5_mini_L100_bdf",
]

TASK_NAME = "contrastChangeDetection"

SHIFT_AFTER_STIM = 0.5   # 자극 이후 0.5초 지점 anchor
WINDOW_LEN       = 0.8   # 윈도우 길이 (예: 0.8초)

PREPROC_ROOT = ROOT / "Preproc_ccd_trialwin"  # 출력 위치

print("===== CCD trial window 전처리 시작 =====")
print("ROOT:", ROOT.resolve())
print("PREPROC_ROOT:", PREPROC_ROOT.resolve())

# =========================
# 1. 릴리즈별 clean CSV 읽어서 처리
# =========================
for rel_dir in RELEASE_DIRS:
    clean_csv = ROOT / rel_dir / "clean_data" / "ccd_rt_all_subjects_clean.csv"
    if not clean_csv.exists():
        print(f"[WARN] clean CSV not found: {clean_csv}")
        continue

    df = pd.read_csv(clean_csv)

    # CCD task만 사용 (혹시라도 다른 task가 섞였을 경우 대비)
    df = df[df["task"] == TASK_NAME].copy()

    if df.empty:
        print(f"[INFO] {rel_dir}: CCD trial 없음, 스킵")
        continue

    rel_short = rel_dir.split("_")[0]  # "R1"
    print(f"\n===== {rel_dir} ({rel_short}) =====")
    print("rows:", len(df))
    print(df["rt"].describe())

    # =========================
    # 2. (sub, run)별로 BDF 찾고, EEG 윈도우 자르기
    # =========================
    rel_root = ROOT / rel_dir   # 예: ./EEG/R1_mini_L100_bdf

    for (sub, run), g in df.groupby(["sub", "run"]):
        run_id   = int(run)
        subj_key = str(sub)  # "NDARFW972KFQ" 같은 값

        # ---- 핵심: target_onset 기준으로 항상 정렬 ----
        g = g.sort_values("target_onset").reset_index(drop=True)

        # 2-1) BDF 파일 찾기
        # 실제 파일명 예시: sub-NDARFW972KFQ_task-contrastChangeDetection_run-1_eeg.bdf
        exact_name = f"sub-{subj_key}_task-{TASK_NAME}_run-{run_id}_eeg.bdf"
        candidates = list(rel_root.rglob(exact_name))

        if not candidates:
            # 혹시 이름이 약간 다른 경우 대비
            patterns = [
                f"*sub-{subj_key}*{TASK_NAME}*run-{run_id}*eeg.bdf",
                f"*{subj_key}*{TASK_NAME}*run-{run_id}*eeg.bdf",
            ]
            for pat in patterns:
                cand = list(rel_root.rglob(pat))
                if cand:
                    candidates = cand
                    break

        if not candidates:
            print(f"[WARN] BDF not found for {rel_short} | sub={subj_key} | run={run_id} under {rel_root}")
            continue

        bdf_path = sorted(candidates)[0]
        print(f"[LOAD] {rel_short} | sub={subj_key} | run={run_id} -> {bdf_path}")

        # 2-2) BDF 로드
        raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose="ERROR")
        sfreq = raw.info["sfreq"]
        print(f"  sfreq = {sfreq}")

        # raw.filter(0.5, 30., fir_design="firwin")

        # 2-3) anchor 시점 (초 -> 샘플)
        anchor_times  = g["target_onset"].to_numpy(dtype=float) + SHIFT_AFTER_STIM
        onset_samples = (anchor_times * sfreq).astype(int)

        events = np.column_stack([
            onset_samples,
            np.zeros(len(onset_samples), dtype=int),
            np.ones(len(onset_samples), dtype=int),
        ])
        event_id = {"ccd_trial": 1}

        # 2-4) Epochs 생성 (anchor 이후 WINDOW_LEN초)
        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=event_id,
            tmin=0.0,
            tmax=WINDOW_LEN,
            baseline=None,
            preload=True,
            proj=False,
            verbose="ERROR",
        )

        X = epochs.get_data().astype("float32")  # (n_trials, n_chans, n_times)

        # X = (X - X.mean(axis=-1, keepdims=True)) / (X.std(axis=-1, keepdims=True) + 1e-6)

        # 2-5) y, meta (정렬된 g 기준으로)
        y = g["rt"].to_numpy(dtype="float32")

        meta = g[["release", "sub", "run", "trial_index", "target_onset", "rt"]].copy()
        meta = meta.reset_index(drop=True)

        if X.shape[0] != len(y) or len(y) != len(meta):
            print(f"[WARN] trial mismatch: X={X.shape[0]}, y={len(y)}, meta={len(meta)} in {rel_short} {subj_key} run-{run_id}")
            continue

        print(f"  -> windows shape: {X.shape}, y shape: {y.shape}")

        # 2-6) 저장
        out_dir = PREPROC_ROOT / rel_dir / f"sub-{subj_key}" / f"run-{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(out_dir / "ccd_trial_windows.npz", X=X, y=y)
        meta.to_csv(out_dir / "metadata.csv", index=False)

        print(f"  [SAVE] {out_dir}")

print("\n===== CCD 전처리 완료 =====")
