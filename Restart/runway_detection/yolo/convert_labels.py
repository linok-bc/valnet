"""CLI: convert runway polygons to centerline triangles in YOLO label files."""

from __future__ import annotations

import argparse
from pathlib import Path

from line_extraction.geometric.centerline import DEFAULT_HALF_WIDTH
from runway_detection.yolo.labels import convert_labels_tree


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Insert geometric centerline into YOLO-seg label files (in-place)"
    )
    p.add_argument(
        "--labels-root",
        type=Path,
        required=True,
        help="Root folder containing .txt labels (recursive)",
    )
    p.add_argument(
        "--half-width",
        type=float,
        default=DEFAULT_HALF_WIDTH,
        help=f"Triangle base half-width in normalized coords (default {DEFAULT_HALF_WIDTH})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print converted labels without writing",
    )
    args = p.parse_args(argv)

    stats = convert_labels_tree(
        args.labels_root,
        dry_run=args.dry_run,
        half_width=args.half_width,
    )
    print(
        f"Found {stats['files_found']} files; "
        f"converted={stats['converted']}, skipped={stats['skipped']}"
    )
    if stats["no_runway"]:
        print(f"No runway (class 0) in {len(stats['no_runway'])} file(s)")


if __name__ == "__main__":
    main()
