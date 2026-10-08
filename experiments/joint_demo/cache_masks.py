#!/usr/bin/env python
"""Cache stock-YOLOv8 runway masks for the joint demo's frames.

The frames are exactly those scored in experiments/edge_lines/runs/stock.json
(the 13 split_test episodes, 150-1500 m), so the partner's heading estimator
and the edge-line cross-track see the same detector on the same images.

Output is the Restart YoloMaskCache format, one <seq>.npz per episode:
    indices  (N,)        frame index
    masks    (N, H, W)   uint8 soft mask, confidence-weighted union of class 0
    binary   (N, H, W)   uint8 highest-confidence instance > 0.5, exactly what
                         experiments/edge_lines/eval.py feeds the edge-line fit

    python -m experiments.joint_demo.cache_masks
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def soft_mask(result, shape):
    h, w = shape
    out = np.zeros((h, w), np.float32)
    if result.masks is None:
        return out
    for m, c in zip(result.masks.data.cpu().numpy(), result.boxes.conf.cpu().numpy()):
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        out = np.maximum(out, np.clip(m * float(c), 0, 1))
    return out


def best_mask(result, shape):
    h, w = shape
    if result.masks is None or len(result.masks) == 0:
        return np.zeros((h, w), np.uint8)
    m = result.masks.data[int(result.boxes.conf.argmax())].cpu().numpy() > 0.5
    if m.shape != (h, w):
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return m.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="experiments/edge_lines/runs/stock.json")
    ap.add_argument("--weights", default="baseline/runs/stock/weights/best.pt")
    ap.add_argument("--root", default="datasets/xp12_dataset")
    ap.add_argument("--out", default="experiments/joint_demo/cache/masks")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    rows = next(iter(json.loads(Path(args.frames).read_text()).values()))
    by_ep = defaultdict(list)
    for r in rows:
        by_ep[r["episode"]].append(int(r["stem"]))

    from ultralytics import YOLO
    model = YOLO(args.weights)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ep, idx in sorted(by_ep.items()):
        path = out_dir / f"{ep}.npz"
        if path.exists():
            continue
        idx = sorted(idx)
        masks, binary = [], []
        for k in range(0, len(idx), 16):
            imgs = [f"{args.root}/{ep}/images/{i:06d}.jpg" for i in idx[k:k + 16]]
            for r in model.predict(imgs, conf=args.conf, retina_masks=True, verbose=False):
                masks.append(np.round(soft_mask(r, r.orig_shape) * 255).astype(np.uint8))
                binary.append(best_mask(r, r.orig_shape))
        np.savez_compressed(path, indices=np.array(idx), masks=np.stack(masks),
                            binary=np.stack(binary))
        print(f"{ep}: {len(idx)} frames -> {path}", flush=True)


if __name__ == "__main__":
    main()
