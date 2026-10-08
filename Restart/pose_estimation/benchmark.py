"""Helpers for pose benchmark stratification and mask I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from runway_detection.lard.loader import LardSample
from runway_detection.lard.projection import CORNER_NAMES


def annotated_runway_bbox_area_px(sample: LardSample) -> float:
    """Axis-aligned bbox area of annotated runway corners (px²)."""
    xs = [sample.annotated_corners[name][0] for name in CORNER_NAMES]
    ys = [sample.annotated_corners[name][1] for name in CORNER_NAMES]
    return float((max(xs) - min(xs)) * (max(ys) - min(ys)))


def height_agl_meters(sample: LardSample, *, feet_in_csv: bool = True) -> float | None:
    """
  Return height above runway in meters.

  LARD V2 CSV stores ``height_above_runway`` in feet (label_export); convert unless
  ``feet_in_csv=False``.
  """
    if sample.height_above_runway is None:
        return None
    h = float(sample.height_above_runway)
    if feet_in_csv:
        return h * 0.3048
    return h


def mask_path_for_image(mask_dir: Path, data_root: Path, image_path: Path) -> Path:
    rel = image_path.relative_to(data_root)
    return mask_dir / rel.with_suffix(".npy")


def load_saved_mask(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)


def assign_tertile_bin(values: list[float], value: float) -> str:
    """Assign low / mid / high given empirical tertile thresholds."""
    if not values:
        return "unknown"
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    if value <= q1:
        return "low"
    if value <= q2:
        return "mid"
    return "high"


def summarize_by_bin(results: list[dict], bin_key: str) -> None:
    from collections import defaultdict

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        groups[row.get(bin_key, "unknown")].append(row)

    print(f"\n=== Stratification by {bin_key} ===")
    for bin_name in sorted(groups):
        rows = groups[bin_name]
        yaw = [abs(r["yaw_err_deg"]) for r in rows]
        lat = [abs(r["lat_err_m"]) for r in rows]
        dice = [r["dice"] for r in rows]
        print(
            f"  {bin_name:>8s}  n={len(rows):3d}  "
            f"|Δyaw|={np.mean(yaw):.3f}°  |Δlat|={np.mean(lat):.1f}m  "
            f"dice={np.mean(dice):.3f}"
        )
