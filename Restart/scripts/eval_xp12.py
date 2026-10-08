#!/usr/bin/env python3
"""
Systematic evaluation / ablation of the (yaw, lateral) estimators on XP12 videos.

    PYTHONPATH=. python scripts/eval_xp12.py --suite solver_cold --workers 8
    PYTHONPATH=. python scripts/eval_xp12.py --suite tracking --workers 8
    PYTHONPATH=. python scripts/eval_xp12.py --suite robustness --workers 8

Outputs in ``data/xp12/results/<suite>/``:
- ``frames.csv.gz``   one row per (run, frame): GT, estimate, errors, σ, NEES, timing
- ``summary.csv``     one row per run (MAE / RMSE / p95 / success / gross / runtime / coverage)
- ``by_along.csv``    same, split by along-track distance bins
Requires ``data/xp12/calibration.json`` (scripts/calibrate_xp12.py) and, for
YOLO runs, ``data/xp12/masks/yolo/*.npz`` (scripts/infer_yolo_xp12.py).
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

ROOT = Path("data/xp12/xp12_dataset")
CALIB = Path("data/xp12/calibration.json")


def _init_worker() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    import cv2

    cv2.setNumThreads(1)


def _task(args: tuple) -> list[dict]:
    label, method, cfg, seq_id, yolo_dir, frame_stride, use_yolo_frames = args
    from evaluation.configs import make_method
    from evaluation.observations import YoloMaskCache
    from evaluation.runner import run_sequence
    from runway_detection.xp12 import load_calibrations, load_sequence

    calib = load_calibrations(CALIB)[seq_id]
    seq = load_sequence(ROOT, seq_id)
    yolo = YoloMaskCache(yolo_dir)
    # Same frames for every run of a suite: YOLO-cached indices for YOLO suites
    # (every 2nd frame), otherwise all frames; then the suite stride.
    idx = yolo.available_indices(seq_id) if use_yolo_frames else [f.index for f in seq.frames]
    frames = idx[::frame_stride]
    rows = run_sequence(seq, calib, make_method(method), replace(cfg, stride=1), yolo=yolo, frame_indices=frames)
    for r in rows:
        r["run"] = label
        r["method"] = method
    return rows


def main() -> None:
    from evaluation.configs import SUITES, build_suite
    from evaluation.metrics import add_along_bin, summarize_by
    from runway_detection.xp12 import list_sequences

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True, choices=SUITES)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--yolo-dir", type=Path, default=Path("data/xp12/masks/yolo"))
    ap.add_argument("--out-root", type=Path, default=Path("data/xp12/results"))
    ap.add_argument("--only", nargs="*", help="restrict to run labels containing any of these substrings")
    ap.add_argument("--max-seqs", type=int, default=None)
    ap.add_argument("--merge", action="store_true",
                    help="keep existing frames.csv.gz rows of runs not selected by --only")
    ap.add_argument("--seq-every", type=int, default=1, help="keep every k-th sequence of the suite's set")
    args = ap.parse_args()

    suite = build_suite(args.suite)
    all_seqs = list_sequences(ROOT)
    yolo_seqs = sorted(p.stem for p in args.yolo_dir.glob("*.npz"))
    seqs = {"all": all_seqs, "yolo": yolo_seqs, "yolo_every3": yolo_seqs[::3]}[suite.sequences]
    seqs = seqs[:: args.seq_every]
    if args.max_seqs:
        seqs = seqs[: args.max_seqs]
    runs = [r for r in suite.runs if not args.only or any(s in r.label for s in args.only)]

    use_yolo_frames = suite.sequences.startswith("yolo")
    tasks = [
        (r.label, r.method, r.cfg, s, str(args.yolo_dir), suite.frame_stride, use_yolo_frames)
        for r in runs
        for s in seqs
    ]
    # Put slow runs first so the pool tail is short.
    tasks.sort(key=lambda t: ("orig" not in t[1], "grid" not in t[1]))
    print(f"suite={suite.name}: {len(runs)} runs × {len(seqs)} sequences = {len(tasks)} tasks, {args.workers} workers")

    out_dir = args.out_root / suite.name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t0 = time.time()
    with get_context("spawn").Pool(args.workers, initializer=_init_worker) as pool:
        for k, res in enumerate(pool.imap_unordered(_task, tasks), 1):
            rows.extend(res)
            if k % 50 == 0:  # partial checkpoint so a killed run is not lost
                pd.DataFrame(rows).to_csv(out_dir / "frames_partial.csv.gz", index=False)
            if k % max(1, len(tasks) // 20) == 0 or k == len(tasks):
                print(f"  {k}/{len(tasks)} tasks, {len(rows)} frames, {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    old_path = out_dir / "frames.csv.gz"
    if args.merge and old_path.exists():
        old = pd.read_csv(old_path, dtype={"seq": str})
        old = old[~old["run"].isin(df["run"].unique())]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out_dir / "frames.csv.gz", index=False)
    summary = summarize_by(df, ["run"])
    summary.to_csv(out_dir / "summary.csv", index=False)
    by_along = summarize_by(add_along_bin(df), ["run", "along_bin"])
    by_along.to_csv(out_dir / "by_along.csv", index=False)

    cols = ["run", "n_frames", "availability", "yaw_mae", "yaw_p95", "lat_mae", "lat_p95",
            "success", "gross", "cov95", "runtime_ms"]
    cols = [c for c in cols if c in summary]
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(summary[cols].to_string(index=False))
    print(f"\n→ {out_dir}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
