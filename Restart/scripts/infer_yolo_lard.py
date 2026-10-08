#!/usr/bin/env python3
"""Run YOLO-seg on LARD images and save soft runway masks (.npy)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pose_estimation.benchmark import mask_path_for_image
from runway_detection.lard.loader import iter_samples
from runway_detection.yolo.inference import (
    load_yolo_model,
    mask_has_foreground,
    predict_soft_mask,
)


def infer_lard_masks(
    data_root: Path,
    weights: Path,
    output_dir: Path,
    *,
    source: str = "flsim",
    split: str = "train",
    max_samples: int | None = None,
    cls_id: int = 0,
    conf: float = 0.25,
    imgsz: int = 640,
    device: str | int | None = None,
) -> Path:
    data_root = data_root.resolve()
    output_dir = output_dir.resolve()
    model = load_yolo_model(weights)

    manifest_path = output_dir / f"manifest_{source}_{split}.csv"
    rows: list[dict] = []
    n_ok = 0
    n_skip = 0

    for sample in iter_samples(data_root, source=source, split=split, require_image=True):
        out_path = mask_path_for_image(output_dir, data_root, sample.image_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            mask, result = predict_soft_mask(
                model,
                sample.image_path,
                cls_id=cls_id,
                conf=conf,
                imgsz=imgsz,
                device=device,
            )
        except FileNotFoundError:
            n_skip += 1
            continue

        np.save(out_path, mask)
        detected = mask_has_foreground(mask)
        n_detections = 0
        if result is not None and result.boxes is not None and result.boxes.cls is not None:
            import torch

            cls = result.boxes.cls
            if isinstance(cls, torch.Tensor):
                cls_np = cls.detach().cpu().numpy().astype(int)
            else:
                cls_np = np.asarray(cls, dtype=int)
            n_detections = int((cls_np == cls_id).sum())

        rel_image = str(sample.image_path.relative_to(data_root))
        rel_mask = str(out_path.relative_to(output_dir))
        rows.append(
            {
                "row_index": sample.row_index,
                "image": rel_image,
                "mask": rel_mask,
                "airport": sample.airport,
                "runway": sample.runway,
                "detected": int(detected),
                "n_detections": n_detections,
                "mask_max": float(mask.max()),
                "mask_mean": float(mask.mean()),
            }
        )
        if detected:
            n_ok += 1
        else:
            n_skip += 1

        if len(rows) % 50 == 0:
            print(f"  processed {len(rows)} images ({n_ok} with runway mask)...")

        if max_samples is not None and len(rows) >= max_samples:
            break

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(
        f"Saved {len(rows)} masks to {output_dir} "
        f"({n_ok} with foreground, {n_skip} empty/missing)"
    )
    print(f"Manifest: {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/LARD"))
    parser.add_argument("--weights", type=Path, default=Path("weights/best.pt"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/LARD/masks/yolo_runway"),
    )
    parser.add_argument("--source", default="flsim")
    parser.add_argument("--split", default="train")
    parser.add_argument("-n", "--max-samples", type=int, default=None)
    parser.add_argument("--cls-id", type=int, default=0, help="0=runway in best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    infer_lard_masks(
        args.data_root.resolve(),
        args.weights.resolve(),
        args.output_dir.resolve(),
        source=args.source,
        split=args.split,
        max_samples=args.max_samples,
        cls_id=args.cls_id,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )


if __name__ == "__main__":
    main()
