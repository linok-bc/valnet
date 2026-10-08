#!/usr/bin/env python
"""Partner's joint (heading, lateral) estimator on the joint demo's frames.

Runs Restart's evaluation runner unchanged -- the method and tracker named on
the command line, YOLO masks from cache_masks.py -- starting cold at the first
frame inside 1500 m. One CSV of per-frame rows per episode.

    python -m experiments.joint_demo.run_heading
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

RESTART = Path(__file__).resolve().parents[2] / "Restart"
sys.path.insert(0, str(RESTART))

import pandas as pd  # noqa: E402

from experiments.joint_demo import data  # noqa: E402

from evaluation.configs import make_method  # noqa: E402
from evaluation.observations import YoloMaskCache  # noqa: E402
from evaluation.runner import RunConfig, run_sequence  # noqa: E402
from runway_detection.xp12.calibration import load_calibrations  # noqa: E402
from runway_detection.xp12.loader import load_sequence  # noqa: E402


def one(job):
    ep, args = job
    calib = load_calibrations(data.CALIB)[ep]
    seq = load_sequence(args.root, ep)
    cache = YoloMaskCache(args.masks)
    cfg = RunConfig(source="yolo", mode="track", kf=not args.no_kf)
    rows = run_sequence(seq, calib, make_method(args.method), cfg, yolo=cache,
                        frame_indices=cache.available_indices(ep))
    out = Path(args.out) / f"{ep}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return ep, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/xp12_dataset")
    ap.add_argument("--masks", default="experiments/joint_demo/cache/masks")
    ap.add_argument("--method", default="maskfit_analytic_lm")
    ap.add_argument("--no-kf", action="store_true")
    ap.add_argument("--out", default="experiments/joint_demo/cache/heading")
    ap.add_argument("--episodes", nargs="*")
    ap.add_argument("--workers", type=int, default=13)
    args = ap.parse_args()
    args.root, args.masks = Path(args.root).resolve(), Path(args.masks).resolve()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    eps = args.episodes or sorted(p.stem for p in args.masks.glob("*.npz"))
    with ProcessPoolExecutor(args.workers) as ex:
        for ep, n in ex.map(one, [(e, args) for e in eps]):
            print(f"{ep}: {n} frames", flush=True)


if __name__ == "__main__":
    main()
