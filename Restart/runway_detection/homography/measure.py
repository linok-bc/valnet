"""Measure heading error and lateral offset in rectified runway plane coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from runway_detection.homography.plane_homography import PlaneHomography, image_points_to_plane
from runway_detection.lard.projection import CORNER_NAMES


@dataclass(frozen=True)
class PlanePoseEstimate:
    """Runway pose read from geometry in the rectified plane."""

    heading_error_deg: float
    lateral_offset_m: float
    centerline_angle_deg: float
    threshold_y_m: float
    obb_center_xy: tuple[float, float]
    obb_size_wh: tuple[float, float]
    obb_angle_deg: float


def _normalize_angle_deg(angle: float) -> float:
    """Wrap to [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def measure_from_corners_plane(plane_corners: np.ndarray) -> PlanePoseEstimate:
    """
    Estimate pose from four runway corners in plane coordinates (meters).

    ``plane_corners``: (4, 2) ordered TR, TL, BL, BR.

    - ``heading_error_deg``: angle of centerline vs +X (along-runway) axis.
    - ``lateral_offset_m``: Y of threshold midpoint in the prior plane frame.
      Apply :func:`runway_detection.homography.readout.correct_plane_residuals`
      before using as metric cross-track (removes ``tan(HE) * along_track`` coupling).
    """
    tr, tl, bl, br = plane_corners
    far_mid = 0.5 * (tr + tl)
    near_mid = 0.5 * (bl + br)
    vec = far_mid - near_mid
    angle_rad = math.atan2(float(vec[1]), float(vec[0]))
    heading_deg = math.degrees(angle_rad)

    pts = plane_corners.astype(np.float32)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), obb_angle = rect

    return PlanePoseEstimate(
        heading_error_deg=_normalize_angle_deg(heading_deg),
        lateral_offset_m=float(near_mid[1]),
        centerline_angle_deg=_normalize_angle_deg(heading_deg),
        threshold_y_m=float(near_mid[1]),
        obb_center_xy=(float(cx), float(cy)),
        obb_size_wh=(float(w), float(h)),
        obb_angle_deg=float(obb_angle),
    )


def measure_from_image_corners(
    corners_px: dict[str, tuple[float, float]],
    homography: PlaneHomography,
) -> PlanePoseEstimate:
    """Map annotated/projected image corners through H⁻¹ and measure in plane."""
    pts = np.array([corners_px[name] for name in CORNER_NAMES], dtype=np.float64)
    plane_pts = image_points_to_plane(pts, homography)
    return measure_from_corners_plane(plane_pts)
