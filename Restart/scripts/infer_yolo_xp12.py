#!/usr/bin/env python3
"""
Run YOLO-seg on XP12 frames (read from the zip) and cache soft runway masks.

One compressed ``<seq>.npz`` per sequence in ``--output-dir`` with
``indices`` (N,) and ``masks`` (N, H, W) uint8 (soft mask × 255).
Already-cached sequences are skipped, so the script can be resumed.

    PYTHONPATH=. python scripts/infer_yolo_xp12.py --every 4 --stride 2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from runway_detection.xp12 import list_sequences, load_sequence
from runway_detection.yolo.inference import load_yolo_model, union_soft_mask_for_class


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/xp12/xp12_dataset"))
    ap.add_argument("--zip", type=Path, default=Path("xp12_dataset.zip"))
    ap.add_argument("--weights", type=Path, default=Path("weights/best.pt"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/xp12/masks/yolo"))
    ap.add_argument("--sequences", nargs="*", help="explicit sequence ids (default: every --every-th)")
    ap.add_argument("--every", type=int, default=4, help="take every k-th sequence of the sorted list")
    ap.add_argument("--stride", type=int, default=2, help="frame stride within a sequence")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    seq_ids = args.sequences or list_sequences(args.root)[:: args.every]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_yolo_model(args.weights)

    for k, seq_id in enumerate(seq_ids):
        out = args.output_dir / f"{seq_id}.npz"
        if out.exists():
            continue
        seq = load_sequence(args.root, seq_id, zip_path=args.zip)
        frames = seq.frames[:: args.stride]
        t0 = time.time()
        indices, masks = [], []
        for f in frames:
            img = seq.read_image(f)
            res = model.predict(
                source=img, conf=args.conf, imgsz=args.imgsz, verbose=False, device=args.device
            )[0]
            soft = union_soft_mask_for_class(res, 0, img.shape[:2])
            indices.append(f.index)
            masks.append(np.round(soft * 255).astype(np.uint8))
        np.savez_compressed(out, indices=np.array(indices), masks=np.stack(masks))
        dt = time.time() - t0
        print(f"[{k + 1}/{len(seq_ids)}] {seq.label}: {len(frames)} frames in {dt:.0f}s → {out}", flush=True)


if __name__ == "__main__":
    main()
