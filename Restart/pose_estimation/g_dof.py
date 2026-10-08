"""
Forward runway mask renderer g_dof(yaw, lateral_offset).

Given fixed along-track distance, height, pitch, and roll, renders the runway
rectangle mask as a function of LARD yaw and cross-track offset in the local
runway frame.

Reference: Restart/spec.md §4.1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from pose_estimation.runway_model import CameraPoseLocal, RunwayScene
from runway_detection.lard.intrinsics import CameraIntrinsics
from runway_detection.lard.projection import (
    CORNER_NAMES,
    apply_rotations_intrinsic,
    point_in_camera_frame,
)


@dataclass(frozen=True)
class GDofParams:
    """Fixed extrinsics/intrinsics held constant while sweeping the 2 DOF."""

    along_track_m: float
    height_m: float
    pitch_cam_deg: float
    roll_cam_deg: float
    intrinsics: CameraIntrinsics
    runway_azimuth_deg: float

    @classmethod
    def from_pose(cls, pose: CameraPoseLocal, scene: RunwayScene) -> GDofParams:
        return cls(
            along_track_m=pose.along_track_m,
            height_m=pose.height_m,
            pitch_cam_deg=pose.pitch_cam_deg,
            roll_cam_deg=pose.roll_cam_deg,
            intrinsics=scene.intrinsics,
            runway_azimuth_deg=scene.runway_azimuth_deg,
        )


@dataclass(frozen=True)
class GDofState:
    """
    The two free variables for render-and-compare.

    ``yaw_cam_deg``: LARD simulator yaw (same convention as metadata CSV).
    ``lateral_offset_m``: cross-track offset; +Y = left of centerline (meters).
    """

    yaw_cam_deg: float
    lateral_offset_m: float


def camera_pose_from_g_dof(params: GDofParams, state: GDofState) -> CameraPoseLocal:
    return CameraPoseLocal(
        along_track_m=params.along_track_m,
        lateral_offset_m=state.lateral_offset_m,
        height_m=params.height_m,
        yaw_cam_deg=state.yaw_cam_deg,
        pitch_cam_deg=params.pitch_cam_deg,
        roll_cam_deg=params.roll_cam_deg,
    )


def project_corners_local(
    scene: RunwayScene,
    pose: CameraPoseLocal,
) -> dict[str, tuple[float, float]]:
    """Project runway corners to pixel coordinates in the local runway frame."""
    cam_pos = pose.position_local
    cam_orientation_rad = np.radians(
        np.array(
            [
                scene.runway_azimuth_deg - pose.yaw_cam_deg,
                90.0 - pose.pitch_cam_deg,
                pose.roll_cam_deg,
            ],
            dtype=float,
        )
    )
    x_cam, y_cam, z_cam = apply_rotations_intrinsic(*cam_orientation_rad)
    intrinsics = scene.intrinsics

    projected: dict[str, tuple[float, float]] = {}
    for corner_name, pt in zip(CORNER_NAMES, scene.corner_points_local):
        point_trans = np.asarray(pt, dtype=float) - cam_pos
        point_cam = point_in_camera_frame(point_trans, x_cam, y_cam, z_cam)
        projected[corner_name] = intrinsics.project_camera_point(point_cam)
    return projected


def corners_dict_to_polygon(corners: dict[str, tuple[float, float]]) -> np.ndarray:
    """Return (4, 2) float array TR → TL → BL → BR."""
    return np.array([corners[name] for name in CORNER_NAMES], dtype=float)


def render_runway_mask(
    corners_px: np.ndarray,
    width: int,
    height: int,
    *,
    dtype: Literal["binary", "float"] = "binary",
) -> np.ndarray:
    """Fill the runway quadrilateral; output shape (H, W)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.round(corners_px).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    if dtype == "float":
        return mask.astype(np.float32)
    return mask.astype(bool)


def g_dof(
    scene: RunwayScene,
    params: GDofParams,
    state: GDofState,
    *,
    return_corners: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, tuple[float, float]]]:
    """
    Render the runway mask for ``state = (yaw_cam_deg, lateral_offset_m)``.

    Returns a binary mask of shape (H, W).
    """
    pose = camera_pose_from_g_dof(params, state)
    corners = project_corners_local(scene, pose)
    corners_px = corners_dict_to_polygon(corners)
    mask = render_runway_mask(
        corners_px,
        params.intrinsics.width,
        params.intrinsics.height,
    )
    if return_corners:
        return mask, corners
    return mask


def g_dof_corners(
    scene: RunwayScene,
    params: GDofParams,
    state: GDofState,
) -> dict[str, tuple[float, float]]:
    pose = camera_pose_from_g_dof(params, state)
    return project_corners_local(scene, pose)


def g_dof_jacobian_corners(
    scene: RunwayScene,
    params: GDofParams,
    state: GDofState,
    *,
    eps_yaw_deg: float = 0.01,
    eps_lateral_m: float = 0.01,
) -> np.ndarray:
    """
    Finite-difference Jacobian of stacked corner pixels w.r.t. (yaw, lateral_offset).

    Returns array shape (8, 2): [u_TR, v_TR, u_TL, v_TL, ...] × [∂/∂yaw, ∂/∂lat].
    """
    base = corners_dict_to_polygon(g_dof_corners(scene, params, state)).reshape(-1)

    state_yaw_p = GDofState(state.yaw_cam_deg + eps_yaw_deg, state.lateral_offset_m)
    state_yaw_m = GDofState(state.yaw_cam_deg - eps_yaw_deg, state.lateral_offset_m)
    state_lat_p = GDofState(state.yaw_cam_deg, state.lateral_offset_m + eps_lateral_m)
    state_lat_m = GDofState(state.yaw_cam_deg, state.lateral_offset_m - eps_lateral_m)

    corners_yaw_p = corners_dict_to_polygon(g_dof_corners(scene, params, state_yaw_p)).reshape(-1)
    corners_yaw_m = corners_dict_to_polygon(g_dof_corners(scene, params, state_yaw_m)).reshape(-1)
    corners_lat_p = corners_dict_to_polygon(g_dof_corners(scene, params, state_lat_p)).reshape(-1)
    corners_lat_m = corners_dict_to_polygon(g_dof_corners(scene, params, state_lat_m)).reshape(-1)

    jac = np.zeros((8, 2), dtype=float)
    jac[:, 0] = (corners_yaw_p - corners_yaw_m) / (2.0 * eps_yaw_deg)
    jac[:, 1] = (corners_lat_p - corners_lat_m) / (2.0 * eps_lateral_m)
    return jac


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def corner_pixel_error(
    projected: dict[str, tuple[float, float]],
    reference: dict[str, tuple[float, float]],
) -> np.ndarray:
    diffs = []
    for name in CORNER_NAMES:
        px, py = projected[name]
        rx, ry = reference[name]
        diffs.append(math.hypot(px - rx, py - ry))
    return np.array(diffs, dtype=float)
