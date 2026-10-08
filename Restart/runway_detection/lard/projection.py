"""
Runway corner projection — port of LARD V2 ``label_export.computeLabels`` geometry.

Reference: deel-ai/LARD (branch LARD_V2), ``src/labeling/label_export.py``.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pyproj

from runway_detection.lard.intrinsics import CameraIntrinsics, DEFAULT_FOV_X_DEG, DEFAULT_FOV_Y_DEG

# A=TR, B=TL, C=BL, D=BR (NEW_CORNERS_NAMES in LARD).
CORNER_NAMES = ("TR", "TL", "BL", "BR")
CORNER_KEYS = ("A", "B", "C", "D")

_GEOD = pyproj.Geod(ellps="WGS84")


def get_forward_azimuth(p1: Iterable[float], p2: Iterable[float]) -> float:
    """Azimuth in degrees from p1 to p2 (inputs as lat, lon, alt)."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    azimuth, _, _ = _GEOD.inv(p1[1], p1[0], p2[1], p2[0])
    if azimuth < 0:
        azimuth += 360.0
    return azimuth


def geodetic_to_cartesian(
    lat: float,
    lon: float,
    alt: float,
    ref_lat: float,
    ref_lon: float,
    ref_alt: float,
    runway_azimuth_deg: float,
) -> np.ndarray:
    """Local runway frame: X along runway, Y left, Z up (LARD convention)."""
    azimuth, _, distance = _GEOD.inv(ref_lon, ref_lat, lon, lat)
    x = distance * math.cos(math.radians(azimuth - runway_azimuth_deg))
    y = -distance * math.sin(math.radians(azimuth - runway_azimuth_deg))
    z = alt - ref_alt
    return np.array([x, y, z], dtype=float)


def _rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    q = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)
    i = np.eye(3)
    p = np.outer(axis, axis)
    return p + cos_theta * (i - p) + sin_theta * q


def apply_rotations_intrinsic(yaw_rad: float, pitch_rad: float, roll_rad: float):
    """Yaw (Z) → Pitch (Y') → Roll (X'') intrinsic rotations; returns camera axes."""
    x_init = np.array([1.0, 0.0, 0.0])
    y_init = np.array([0.0, 1.0, 0.0])
    z_init = np.array([0.0, 0.0, 1.0])

    r_yaw = _rotation_matrix(z_init, yaw_rad)
    x_prim = r_yaw @ x_init
    y_prim = r_yaw @ y_init
    z_prim = z_init

    r_pitch = _rotation_matrix(y_prim, pitch_rad)
    x_ter = r_pitch @ x_prim
    y_ter = y_prim
    z_ter = r_pitch @ z_prim

    r_roll = _rotation_matrix(x_ter, roll_rad)
    x_quart = x_ter
    y_quart = r_roll @ y_ter
    z_quart = r_roll @ z_ter
    return x_quart, y_quart, z_quart


def point_in_camera_frame(point: np.ndarray, x_cam, y_cam, z_cam) -> np.ndarray:
    return np.array(
        [np.dot(point, x_cam), np.dot(point, y_cam), np.dot(point, z_cam)],
        dtype=float,
    )


def runway_corners_wgs84(runway_entry: dict) -> list[np.ndarray]:
    """Return [TR, TL, BL, BR] as (lat, lon, alt) arrays from a runways DB entry."""
    return [
        np.array(
            [
                runway_entry[key]["coordinate"]["latitude"],
                runway_entry[key]["coordinate"]["longitude"],
                runway_entry[key]["coordinate"]["altitude"],
            ],
            dtype=float,
        )
        for key in CORNER_KEYS
    ]


def project_runway_corners(
    *,
    runway_corners_latlon: list[np.ndarray],
    lat_cam: float,
    lon_cam: float,
    alt_cam: float,
    yaw_cam_deg: float,
    pitch_cam_deg: float,
    roll_cam_deg: float,
    intrinsics: CameraIntrinsics | None = None,
) -> dict[str, tuple[float, float]]:
    """
    Project the four runway corners to pixel coordinates.

    Uses the same formulas as LARD V2 ``computeLabels``.
    """
    intrinsics = intrinsics or CameraIntrinsics.from_fov()
    a, b, c, d = runway_corners_latlon
    opp_runway_azimuth = get_forward_azimuth((d + c) / 2.0, (a + b) / 2.0)
    origin = (c + d) / 2.0  # LTP

    cam_orientation_rad = np.radians(
        np.array(
            [
                opp_runway_azimuth - yaw_cam_deg,
                90.0 - pitch_cam_deg,
                roll_cam_deg,
            ],
            dtype=float,
        )
    )
    cam_pos_cart = geodetic_to_cartesian(
        lat_cam, lon_cam, alt_cam, origin[0], origin[1], origin[2], opp_runway_azimuth
    )
    x_cam, y_cam, z_cam = apply_rotations_intrinsic(*cam_orientation_rad)

    projected: dict[str, tuple[float, float]] = {}
    for corner_name, pt in zip(CORNER_NAMES, runway_corners_latlon):
        point_cart = geodetic_to_cartesian(
            pt[0], pt[1], pt[2], origin[0], origin[1], origin[2], opp_runway_azimuth
        )
        point_trans = point_cart - cam_pos_cart
        point_cam = point_in_camera_frame(point_trans, x_cam, y_cam, z_cam)
        u, v = intrinsics.project_camera_point(point_cam)
        projected[corner_name] = (u, v)
    return projected


def corners_dict_to_array(corners: dict[str, tuple[float, float]]) -> np.ndarray:
    return np.array([corners[name] for name in CORNER_NAMES], dtype=float)


def corner_reprojection_error(
    projected: dict[str, tuple[float, float]],
    annotated: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Per-corner Euclidean error in pixels (TR, TL, BL, BR)."""
    diffs = []
    for name in CORNER_NAMES:
        px, py = projected[name]
        ax, ay = annotated[name]
        diffs.append(math.hypot(px - ax, py - ay))
    return np.array(diffs, dtype=float)
