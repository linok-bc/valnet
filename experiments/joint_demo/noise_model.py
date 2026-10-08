#!/usr/bin/env python
"""Tilt (free_width) measurement noise vs range, fitted on the VALIDATION split.

The fusion filter (fuse.py) needs the tilt estimate's noise as a function of
range. Fitting it on the 13 test flights would leak the answer into the
comparison, so it is fitted here on split_val -- flights the stock YOLO did not
train on and that are not scored anywhere in the joint demo.

Same pipeline as experiments/edge_lines/eval.py (its own functions): stock YOLO
highest-confidence mask -> row scan -> two robust lines -> free_width, scored
against the logged truth (the logs are rotation-fixed). The model is the per-bin RMS, linearly
interpolated in range (a straight line fits badly: it goes to ~0 m at 150 m
where the measured RMS is 0.4 m). The RMS drops the worst 1% first -- the
filter's NIS gate is what handles those.

    python -m experiments.joint_demo.noise_model
"""

import json
from pathlib import Path

import numpy as np

from experiments.bev.camera import CameraModel
from experiments.bev.episodes import load_runways
from experiments.edge_lines.eval import load_frames, score, yolo_masks
from experiments.joint_demo import data

BINS = np.arange(150, 1501, 150)


def main():
    root = data.ROOT
    cam = CameraModel(**json.loads((root / "experiments/bev_yolo/camera.json").read_text())["camera"])
    runways = load_runways()
    frames = load_frames(root / "datasets/xp12_dataset/split_val.txt", 150, 1500, 1)
    print(f"{len(frames)} val frames, {len({f.episode.path.name for f in frames})} flights")
    rows = score(frames, yolo_masks(str(root / "baseline/runs/stock/weights/best.pt"),
                                    frames, 0.25, None), cam, runways)

    r = np.array([x["range_m"] for x in rows])
    e = np.array([x["err_free_width"] for x in rows], float)
    ok = np.isfinite(e)
    print(f"no estimate on {(~ok).mean():.1%} of frames")
    r, e = r[ok], e[ok]
    cut = np.quantile(np.abs(e), 0.99)
    mid, rms = [], []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (r >= lo) & (r < hi) & (np.abs(e) <= cut)
        mid.append(0.5 * (lo + hi))
        rms.append(float(np.sqrt(np.mean(e[m] ** 2))))
        print(f"  {lo:4.0f}-{hi:4.0f} m  n {m.sum():4d}  bias {e[m].mean():+.2f}  rms {rms[-1]:.2f}")
    out = {"range_m": mid, "rms_m": rms, "fitted_on": "split_val", "n_frames": int(len(e))}
    path = data.HERE / "cache/tilt_noise.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
