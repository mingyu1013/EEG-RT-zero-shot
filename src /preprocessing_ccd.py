# preprocessing_ccd_resp.py
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import mne


def preprocess_raw(
    raw: mne.io.BaseRaw,
    hp_freq: float,
    lp_freq: float,
    notch_freq: float,
    resample_sfreq: float,
) -> mne.io.BaseRaw:
    """
    공통 필터 / notch / resample.
    """
    raw = raw.copy().load_data()
    raw.filter(l_freq=hp_freq, h_freq=lp_freq, n_jobs=1)

    sfreq = raw.info["sfreq"]
    if sfreq >= 2 * notch_freq:
        raw.notch_filter(freqs=[notch_freq], n_jobs=1)

    if abs(raw.info["sfreq"] - resample_sfreq) > 1e-6:
        raw.resample(resample_sfreq)

    return raw


def robust_zscore_per_subject(
    X: np.ndarray,
    meta: pd.DataFrame,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    subject별 median / IQR 기반 robust z-score.
    X: (N, C, T)
    meta: 반드시 'sub' 컬럼 포함.
    """
    X_norm = X.copy()
    subs = meta["sub"].unique()
    for sub in subs:
        idx = np.where(meta["sub"].values == sub)[0]
        X_sub = X_norm[idx]  # (n_sub, C, T)

        med = np.median(X_sub, axis=(0, 2))  # (C,)
        q1 = np.percentile(X_sub, 25, axis=(0, 2))
        q3 = np.percentile(X_sub, 75, axis=(0, 2))
        iqr = q3 - q1
        robust_std = iqr / 1.349
        robust_std[robust_std < eps] = eps

        X_norm[idx] = (X_sub - med[None, :, None]) / robust_std[None, :, None]

    return X_norm


def subjectwise_zscore_rt(
    meta: pd.DataFrame,
    rt: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RT를 subject-wise z-score로 변환.
    반환:
      z    : (N,)  각 trial의 z-scored RT
      mean : (N,)  각 trial이 속한 subject의 RT mean
      std  : (N,)  각 trial이 속한 subject의 RT std
    """
    subs = meta["sub"].values
    z = np.zeros_like(rt, dtype=np.float32)
    means = np.zeros_like(rt, dtype=np.float32)
    stds = np.zeros_like(rt, dtype=np.float32)

    unique_subs = np.unique(subs)
    for sub in unique_subs:
        idx = np.where(subs == sub)[0]
        rt_s = rt[idx]
        m = float(rt_s.mean())
        sd = float(rt_s.std())
        if sd < eps:
            sd = eps
        z[idx] = (rt_s - m) / sd
        means[idx] = m
        stds[idx] = sd

    return z, means, stds


def get_common_eeg_channels(release_roots: Dict[str, Path]) -> List[str]:
    """
    모든 release에서 공통으로 존재하는 EEG 채널 교집합 계산.
    release_roots: {"R1": Path(...), ...}
    """
    common_chans = None
    for rel, root in release_roots.items():
        for sub_dir in root.glob("sub-*"):
            eeg_dir = sub_dir / "eeg"
            if not eeg_dir.is_dir():
                continue
            sub_id = sub_dir.name.replace("sub-", "")
            cands = list(eeg_dir.glob(f"sub-{sub_id}_task-*_*eeg.bdf"))
            if not cands:
                continue
            bdf_path = cands[0]
            try:
                raw = mne.io.read_raw_bdf(bdf_path, preload=False, verbose=False)
            except Exception:
                continue

            picks = mne.pick_types(raw.info, eeg=True)
            ch_names = [raw.ch_names[i] for i in picks]
            if common_chans is None:
                common_chans = set(ch_names)
            else:
                common_chans &= set(ch_names)

    if not common_chans:
        raise RuntimeError("No common EEG channels found")

    ch_list = sorted(list(common_chans))
    print(f"[INFO] Common EEG channels ({len(ch_list)}): {ch_list[:10]} ...")
    return ch_list


def load_ccd_trials_all_releases(
    releases: List[str],
    release_roots: Dict[str, Path],
    ccd_task: str,
    rt_min: float,
    rt_max: float,
) -> pd.DataFrame:
    """
    각 release의 clean_data/ccd_rt_all_subjects_clean.csv 로부터
    CCD trial 메타 로딩. RT 범위 [rt_min, rt_max] 필터 적용.
    """
    all_df = []
    for rel in releases:
        root = release_roots[rel]
        csv_path = root / "clean_data" / "ccd_rt_all_subjects_clean.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        df = pd.read_csv(csv_path)

        # CCD task만 사용 (안전하게)
        if "task" in df.columns:
            df = df[df["task"] == ccd_task].copy()

        # press_onset 있는 trial만 사용 (response-locked epoch를 위해)
        if "press_onset" in df.columns:
            df = df[df["press_onset"].notna()].copy()

        # release 컬럼을 현재 rel로 강제 세팅
        df["release"] = rel

        # RT 범위 필터
        df = df[df["rt"].between(rt_min, rt_max)]

        all_df.append(df)

    df_all = pd.concat(all_df, ignore_index=True)
    return df_all


def find_bdf_ccd(root: Path, ccd_task: str, sub: str, run: float) -> Path:
    """
    CCD BDF path 찾기.
    """
    run_int = int(run)
    eeg_dir = root / f"sub-{sub}" / "eeg"
    pattern = f"sub-{sub}_task-{ccd_task}_run-{run_int}_eeg.bdf"
    cands = list(eeg_dir.glob(pattern))
    if not cands:
        raise FileNotFoundError(f"CCD BDF not found: {pattern}")
    return cands[0]


def extract_ccd_epochs_for_release(
    df_ccd: pd.DataFrame,
    rel: str,
    release_roots: Dict[str, Path],
    common_chans: List[str],
    hp_freq: float,
    lp_freq: float,
    notch_freq: float,
    resample_sfreq: float,
    epoch_tmin: float,
    epoch_tmax: float,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    한 release의 CCD trial들에 대해 epoch (C,T) 추출.
    press_onset 기준 [epoch_tmin, epoch_tmax] (response-locked)
    """
    root = release_roots[rel]
    X_list = []
    meta_rows = []

    df_rel = df_ccd[df_ccd["release"] == rel]

    for (sub, run), grp in df_rel.groupby(["sub", "run"]):
        bdf_path = find_bdf_ccd(root, "contrastChangeDetection", sub, run)
        print(f"[INFO][{rel}] CCD raw: {bdf_path}")
        raw = mne.io.read_raw_bdf(bdf_path, preload=False, verbose=False)
        raw = preprocess_raw(raw, hp_freq, lp_freq, notch_freq, resample_sfreq)
        raw.pick_channels(common_chans)

        sfreq = raw.info["sfreq"]
        n_samples = raw.n_times
        win_samples = int((epoch_tmax - epoch_tmin) * sfreq)
        data = raw.get_data()  # (C, T)

        for _, row in grp.iterrows():
            if "press_onset" not in row or pd.isna(row["press_onset"]):
                continue
            t_press = float(row["press_onset"])
            t0 = t_press + epoch_tmin  # 예: press_onset - 1.0
            start = int(round(t0 * sfreq))
            end = start + win_samples
            if start < 0 or end > n_samples:
                continue
            epoch = data[:, start:end]
            if epoch.shape[1] != win_samples:
                continue

            X_list.append(epoch)
            meta_rows.append(
                {
                    "sub": row["sub"],
                    "run": row["run"],
                    "trial_index": row["trial_index"],
                    "release": row["release"],
                    "rt": float(row["rt"]),
                }
            )

    if not X_list:
        raise RuntimeError(f"No CCD epochs for release {rel}")

    X = np.stack(X_list, axis=0)
    meta = pd.DataFrame(meta_rows).reset_index(drop=True)
    print(f"[INFO][{rel}] CCD epochs shape: {X.shape}")
    return X, meta
