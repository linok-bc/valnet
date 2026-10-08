"""Aggregate per-frame evaluation rows into accuracy / robustness / timing metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

ALONG_BINS = (0, 500, 1000, 1500, 2000, 3000)
SUCCESS_YAW_DEG = 0.5
SUCCESS_LAT_M = 5.0
GROSS_YAW_DEG = 2.0
GROSS_LAT_M = 20.0
CHI2_2DOF_95 = 5.991


def summarize(df: pd.DataFrame) -> dict:
    n = len(df)
    ok_all = df["yaw_err"].notna() & df["lat_err"].notna()
    # Accuracy is measured on frames with a vision measurement; frames where the
    # tracker only holds / extrapolates are reported separately (``*_all``).
    ok = ok_all & df["yaw_meas"].notna() if "yaw_meas" in df else ok_all
    e_yaw = df.loc[ok, "yaw_err"].abs().to_numpy()
    e_lat = df.loc[ok, "lat_err"].abs().to_numpy()
    out: dict = {"n_frames": n, "availability": float(ok.mean()) if n else np.nan}
    if "yaw_meas" in df:
        out["meas_rate"] = float(df["yaw_meas"].notna().mean()) if n else np.nan
    if ok.sum() == 0:
        return out
    for key, e in (("yaw", e_yaw), ("lat", e_lat)):
        out[f"{key}_mae"] = float(e.mean())
        out[f"{key}_rmse"] = float(np.sqrt(np.mean(e**2)))
        out[f"{key}_med"] = float(np.median(e))
        out[f"{key}_p95"] = float(np.percentile(e, 95))
    out["yaw_mae_all"] = float(df.loc[ok_all, "yaw_err"].abs().mean())
    out["lat_mae_all"] = float(df.loc[ok_all, "lat_err"].abs().mean())
    succ = (e_yaw < SUCCESS_YAW_DEG) & (e_lat < SUCCESS_LAT_M)
    gross = (e_yaw > GROSS_YAW_DEG) | (e_lat > GROSS_LAT_M)
    out["success"] = float(succ.sum() / n)  # unavailable frames count as failures
    out["gross"] = float(gross.mean())
    out["runtime_ms"] = float(df["runtime_ms"].mean())
    out["runtime_p95_ms"] = float(df["runtime_ms"].quantile(0.95))
    nees = df.loc[ok, "nees"].dropna()
    out["cov95"] = float((nees < CHI2_2DOF_95).mean()) if len(nees) else np.nan
    if "seg_iou" in df:
        out["seg_iou"] = float(df["seg_iou"].mean())
    if "status" in df:
        out["rejected"] = float((df["status"] == "rejected").mean())
    return out


def summarize_by(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for k, g in df.groupby(keys, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        rows.append({**dict(zip(keys, k)), **summarize(g)})
    return pd.DataFrame(rows)


def add_along_bin(df: pd.DataFrame) -> pd.DataFrame:
    labels = [f"{a}-{b}" for a, b in zip(ALONG_BINS[:-1], ALONG_BINS[1:])]
    df = df.copy()
    df["along_bin"] = pd.cut(df["along_m"], ALONG_BINS, labels=labels, right=False)
    return df
