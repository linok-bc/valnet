"""YOLO label conversion and class filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from line_extraction.geometric.centerline import (
    DEFAULT_HALF_WIDTH,
    centerline_triangle,
    format_triangle_line,
    is_centerline,
    parse_polygon,
)


def process_txt(
    filepath: str | Path,
    *,
    dry_run: bool = True,
    half_width: float = DEFAULT_HALF_WIDTH,
    skip_if_centerline: bool = True,
) -> list[str] | None:
    """
    Convert a YOLO-seg label file: class 0 runway polygon → centerline (0) + runway (1);
    other classes shifted by +1.

    Returns:
        New label lines, or None if the file was skipped (already has a centerline triangle).
    """
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    if skip_if_centerline:
        for line in lines:
            parts = line.split()
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
            if cls == 0 and is_centerline(coords):
                print(f"Already converted, skipped: {filepath}")
                return None

    new_lines: list[str] = []
    centerline_line = None

    for line in lines:
        parts = line.split()
        cls = int(parts[0])
        coords = list(map(float, parts[1:]))

        if cls == 0:
            pts = parse_polygon(coords)
            A, B, C = centerline_triangle(pts, half_width=half_width)
            centerline_line = format_triangle_line(A, B, C, class_id=0)
            new_lines.append("1 " + " ".join(f"{v:.6f}" for v in coords))
        else:
            new_lines.append(f"{cls + 1} " + " ".join(f"{v:.6f}" for v in coords))

    output = [centerline_line] + new_lines if centerline_line else new_lines

    if dry_run:
        print(f"\n=== {filepath} ===")
        for l in output:
            print(l)
    else:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(output) + "\n")

    return output


def convert_labels_tree(
    labels_root: str | Path,
    *,
    dry_run: bool = False,
    half_width: float = DEFAULT_HALF_WIDTH,
) -> dict:
    """Apply process_txt to all *.txt under labels_root. Returns summary stats."""
    labels_root = Path(labels_root)
    txt_files = list(labels_root.rglob("*.txt"))
    no_runway: list[str] = []
    skipped = 0
    converted = 0

    for fp in txt_files:
        result = process_txt(fp, dry_run=dry_run, half_width=half_width)
        if result is None:
            skipped += 1
            continue
        converted += 1
        if not any(l.startswith("0 ") for l in result):
            no_runway.append(str(fp))

    return {
        "files_found": len(txt_files),
        "converted": converted,
        "skipped": skipped,
        "no_runway": no_runway,
    }


def filter_classes(
    src_root: str | Path,
    dst_root: str | Path,
    keep_classes: Sequence[int],
    *,
    remap: dict[int, int] | None = None,
    splits: Iterable[str] | None = None,
) -> dict:
    """
    Copy label tree keeping only selected class ids.

    If `splits` is set (e.g. train/val/test), only those subfolders are processed.
    If `remap` is provided, class ids are rewritten (e.g. {1: 0} to collapse runway→0).
    """
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    keep = {int(c) for c in keep_classes}
    remap = {int(k): int(v) for k, v in (remap or {}).items()}

    kept_lines = 0
    removed_lines = 0
    files_total = 0
    files_empty = 0

    if splits:
        file_iter: list[tuple[Path, Path]] = []
        for split in splits:
            src_split = src_root / split
            dst_split = dst_root / split
            if not src_split.exists():
                print(f"Missing split, skipped: {src_split}")
                continue
            dst_split.mkdir(parents=True, exist_ok=True)
            for src_txt in src_split.rglob("*.txt"):
                rel = src_txt.relative_to(src_split)
                file_iter.append((src_txt, dst_split / rel))
    else:
        dst_root.mkdir(parents=True, exist_ok=True)
        file_iter = [
            (src_txt, dst_root / src_txt.relative_to(src_root))
            for src_txt in src_root.rglob("*.txt")
        ]

    for src_txt, dst_txt in file_iter:
        dst_txt.parent.mkdir(parents=True, exist_ok=True)
        out: list[str] = []
        with src_txt.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    cls = int(parts[0])
                except ValueError:
                    continue
                if cls not in keep:
                    removed_lines += 1
                    continue
                new_cls = remap.get(cls, cls)
                out.append(f"{new_cls} " + " ".join(parts[1:]))
                kept_lines += 1

        dst_txt.write_text(("\n".join(out) + ("\n" if out else "")), encoding="utf-8")
        files_total += 1
        if not out:
            files_empty += 1

    return {
        "files_total": files_total,
        "kept_lines": kept_lines,
        "removed_lines": removed_lines,
        "files_empty": files_empty,
        "dst": str(dst_root),
    }
