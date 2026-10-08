#!/usr/bin/env python3
"""
Interactive viewer for sequential g_dof mask fitting.

Scroll frames with the slider or ← / → keys. Each frame shows:
  - GT runway (green), init t−1 (yellow), estimated template (red)
  - YOLO mask (blue), mask tiles (GT · Init · Est · YOLO)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pose_estimation.sequence_viz import (
    interactive_sequence_viewer,
    save_frame_png,
)
from scripts.validate_mask_fit_sequence import (
    collect_sequence_with_masks,
    track_sequence_viz,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/LARD"))
    parser.add_argument("--mask-dir", type=Path, default=Path("data/LARD/masks/yolo_runway"))
    parser.add_argument("--source", default="flsim")
    parser.add_argument("--split", default="train")
    parser.add_argument("--airport", default="CYEG")
    parser.add_argument("--runway", default="20")
    parser.add_argument(
        "--mode",
        choices=("sequential", "gt", "cold"),
        default="sequential",
    )
    parser.add_argument("--min-row", type=int, default=None, help="Only frames with row_index ≥ this")
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Save all frames as PNGs (optional)",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export PNGs, do not open interactive window",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    frames = collect_sequence_with_masks(
        data_root,
        args.mask_dir.resolve(),
        args.airport,
        args.runway,
        source=args.source,
        split=args.split,
    )
    if args.min_row is not None:
        frames = [f for f in frames if f[0].row_index >= args.min_row]
    if not frames:
        raise SystemExit("No YOLO frames found for this sequence.")

    print(f"Tracking {len(frames)} frames ({args.airport} {args.runway}, mode={args.mode})…")
    viz_frames = track_sequence_viz(frames, mode=args.mode)

    if args.export_dir is not None:
        out = args.export_dir.resolve()
        for frame in viz_frames:
            path = out / f"row{frame.sample.row_index:05d}.png"
            save_frame_png(frame, path)
        print(f"Exported {len(viz_frames)} PNGs to {out}")

    if args.export_only:
        return

    tag = f"{args.airport}_{args.runway} — {args.mode}"
    interactive_sequence_viewer(viz_frames, title=f"g_dof tracking · {tag}")


if __name__ == "__main__":
    main()
