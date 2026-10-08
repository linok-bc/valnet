#!/usr/bin/env python3
"""
Run an estimator on one XP12 approach and render a replay video.

    PYTHONPATH=. python scripts/render_xp12_video.py --seq 000018 \\
        --method maskfit_analytic_lm --source yolo --kf

Shows the camera view with the GT runway (green) vs the runway re-projected
from the estimated (yaw, lateral) + known DOF (red), the segmentation mask,
yaw / lateral time series (GT vs estimate ± 2σ) and a top view of the
trajectory. Also writes the per-frame CSV next to the video.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluation.configs import make_method
from evaluation.observations import YoloMaskCache
from evaluation.runner import RunConfig, run_sequence
from evaluation.video import render_video
from runway_detection.xp12 import load_calibrations, load_sequence


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--method", default="maskfit_analytic_lm")
    ap.add_argument("--source", default="yolo", choices=["gt", "corrupt", "yolo"])
    ap.add_argument("--mode", default="track", choices=["track", "cold"])
    ap.add_argument("--kf", action="store_true", help="kinematic EKF + NIS gating")
    ap.add_argument("--stride", type=int, default=2, help="frame stride (YOLO cache is every 2nd frame)")
    ap.add_argument("--root", type=Path, default=Path("data/xp12/xp12_dataset"))
    ap.add_argument("--zip", type=Path, default=Path("xp12_dataset.zip"))
    ap.add_argument("--yolo-dir", type=Path, default=Path("data/xp12/masks/yolo"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/xp12/videos"))
    ap.add_argument("--fps", type=float, default=15.0)
    args = ap.parse_args()

    calib = load_calibrations("data/xp12/calibration.json")[args.seq]
    seq = load_sequence(args.root, args.seq, zip_path=args.zip)
    yolo = YoloMaskCache(args.yolo_dir)
    frames = yolo.available_indices(args.seq) if args.source == "yolo" else None
    frames = frames[:: max(1, args.stride // 2)] if frames else [f.index for f in seq.frames[:: args.stride]]
    cfg = RunConfig(source=args.source, mode=args.mode, kf=args.kf)
    rows = run_sequence(seq, calib, make_method(args.method), cfg, yolo=yolo, frame_indices=frames)
    df = pd.DataFrame(rows)

    masks = None
    if args.source == "yolo":
        masks = {i: m for i in df["frame"] if (m := yolo.get(args.seq, int(i))) is not None}
    tag = f"{args.seq}_{seq.airport}{seq.runway}_{args.method}{'+ekf' if args.kf else ''}_{args.source}_{args.mode}"
    title = f"{args.method}{' + EKF' if args.kf else ''} | masks: {args.source} | {args.mode}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / f"{tag}.csv", index=False)
    out = render_video(seq, calib, df, args.out_dir / f"{tag}.mp4", masks=masks, title=title, fps=args.fps)
    e = df[["yaw_err", "lat_err"]].abs()
    print(f"{seq.label}: {len(df)} frames  |yaw err| mean {e.yaw_err.mean():.3f}°  |lat err| mean {e.lat_err.mean():.2f} m")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
