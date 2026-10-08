"""Per-frame table for one episode: partner's heading + our edge-line cross-track.

Truth. The logs are the rotation-FIXED ones (datasets/xp12_dataset/
rotation_fix_manifest.json, 2026-09-25): the per-episode runway-axis rotation is
removed from along/right/yaw, so the logged values are the truth directly.

    heading     yaw_gt / yaw_est from Restart's runner, with Samuel's per-flight
                geometry REFITTED on the fixed logs (CALIB; his own file was
                fitted on the old logs and would double-correct)
    cross-track truth_m = logged right, from edge_lines/eval.py --truth logged
                on the fixed logs (TILT_EVAL); xt_est recomputed here on the
                cached binary mask -- checked against it by verify()
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bev.camera import CameraModel
from experiments.bev.episodes import load_runways
from experiments.edge_lines.geometry import (crosstrack, edge_points, fit_line,
                                             line_from_fit)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STOCK = HERE / "cache/tilt_eval.json"      # edge_lines.eval --truth logged, fixed logs
CALIB = HERE / "cache/calibration.json"    # Restart calibrate_xp12.py on the fixed logs
YOLO_KEY = "YOLO MASK (baseline/runs/stock/weights/best.pt)"
HEADING_FROM_M = 1500.0
XTRACK_FROM_M = 1000.0


@lru_cache(None)
def camera():
    cfg = json.loads((ROOT / "experiments/bev_yolo/camera.json").read_text())
    return CameraModel(**cfg["camera"])


@lru_cache(None)
def _stock():
    d = pd.DataFrame(json.loads(STOCK.read_text())[YOLO_KEY])
    d["frame"] = d["stem"].astype(int)
    return d


def episodes():
    return sorted(p.stem for p in (HERE / "cache/masks").glob("*.npz"))


def meta(ep):
    airport, rwy = (ROOT / "datasets/xp12_dataset" / ep / "meta.txt").read_text().split()[:2]
    return airport, rwy


def _edge_lines(mask, cam, roll, pitch, height, width):
    """Our free_width estimate plus the two fitted image lines (homogeneous)."""
    left, right, D = edge_points(mask, cam, roll, pitch)
    if D is None or len(left) < 5 or len(right) < 5:
        return np.nan, None, None
    fl, fr = fit_line(left), fit_line(right)
    if fl is None or fr is None:
        return np.nan, None, None
    lL, lR = line_from_fit(fl[0], fl[1], D), line_from_fit(fr[0], fr[1], D)
    out = crosstrack(lL, lR, cam, roll, pitch, height, width)
    return out["free_width"], lL, lR


def load(ep):
    """(df, masks, lines): df one row per frame sorted by range descending."""
    h = pd.read_csv(HERE / "cache/heading" / f"{ep}.csv", dtype={"seq": str})
    s = _stock()
    s = s[s.episode == ep][["frame", "range_m", "truth_m", "free_width"]]
    df = h.merge(s, on="frame", how="inner").sort_values("range_m", ascending=False)
    df = df.reset_index(drop=True)

    # Decompress each array ONCE: every z["masks"] access on an npz re-decompresses
    # the whole (N, H, W) array, and a per-frame view keeps that copy alive --
    # indexing z[...] inside the loop costs N full copies (~100 GB per flight).
    with np.load(HERE / "cache/masks" / f"{ep}.npz") as z:
        indices, soft_all, binary_all = z["indices"], z["masks"], z["binary"]
    k = {int(i): n for n, i in enumerate(indices)}
    soft = {f: soft_all[k[f]] for f in df.frame}
    binary = {f: binary_all[k[f]] for f in df.frame}

    cam = camera()
    width = load_runways()[meta(ep)].width_m
    ep_dir = ROOT / "datasets/xp12_dataset" / ep
    xt, lines = [], {}
    for f in df.frame:
        yaw, pitch, roll = np.loadtxt(ep_dir / "pose" / f"{f:06d}.txt")[:3]
        height = np.loadtxt(ep_dir / "position" / f"{f:06d}.txt")[1]
        est, lL, lR = _edge_lines(binary[f], cam, roll, pitch, height, width)
        xt.append(est)
        lines[f] = (lL, lR)
    df["xt_est"] = xt
    df["xt_err"] = df.xt_est - df.truth_m
    df["heading_on"] = df.range_m <= HEADING_FROM_M
    df["xtrack_on"] = df.range_m <= XTRACK_FROM_M
    return df, soft, lines


def full_flight(ep):
    """Whole approach from the (fixed) logs: range, right, height, yaw."""
    d = ROOT / "datasets/xp12_dataset" / ep
    pos = np.array([np.loadtxt(p) for p in sorted((d / "position").glob("*.txt"))])
    yaw = np.array([np.loadtxt(p)[0] for p in sorted((d / "pose").glob("*.txt"))])
    rng = -pos[:, 0]
    return pd.DataFrame({"range_m": rng, "right_m": pos[:, 2],
                         "height_m": pos[:, 1], "yaw_deg": yaw})


def verify(df):
    """Max |difference| between our recomputed estimate and edge_lines.eval's."""
    return float(np.nanmax(np.abs(df.xt_est - df.free_width)))
