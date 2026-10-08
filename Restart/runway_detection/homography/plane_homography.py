"""
Planar homography between the runway plane (local X,Y in meters) and image pixels.

With the four ``known'' DOF fixed (along-track, height, pitch, roll), the
homography depends only on yaw and lateral offset through the camera pose.
Rectifying with the nominal (yaw-aligned, zero lateral) homography exposes
the two unknowns as a 2D rotation + translation in the plane.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pose_estimation.g_dof import project_corners_local
from pose_estimation.runway_model import CameraPoseLocal, RunwayScene
from runway_detection.homography.heading import heading_error_ground_deg
from runway_detection.lard.projection import CORNER_NAMES


@dataclass(frozen=True)
class PlaneHomography:
    """Homography mapping runway plane coords (m) → homogeneous image pixels."""

    plane_to_image: np.ndarray  # 3×3
    image_to_plane: np.ndarray  # 3×3

    @property
    def H(self) -> np.ndarray:
        return self.plane_to_image


def runway_plane_xy(scene: RunwayScene) -> np.ndarray:
    """Corner plane coordinates (4, 2) in meters — TR, TL, BL, BR."""
    return scene.corner_points_local[:, :2].astype(np.float64)


def corners_px_array(corners: dict[str, tuple[float, float]]) -> np.ndarray:
    return np.array([corners[name] for name in CORNER_NAMES], dtype=np.float64)


def homography_from_pose(scene: RunwayScene, pose: CameraPoseLocal) -> PlaneHomography:
    """
    Estimate plane→image homography from projected runway corners (DLT).

    Uses the true corner plane coordinates and their pinhole projections.
    """
    plane_pts = runway_plane_xy(scene)
    image_pts = corners_px_array(project_corners_local(scene, pose))
    h = cv2.getPerspectiveTransform(plane_pts.astype(np.float32), image_pts.astype(np.float32))
    h_inv = np.linalg.inv(h)
    return PlaneHomography(plane_to_image=h, image_to_plane=h_inv)


def nominal_pose_four_dof(pose_gt: CameraPoseLocal, scene: RunwayScene) -> CameraPoseLocal:
    """
    Nominal pose: same along-track / height / pitch / roll, zero lateral offset,
    yaw chosen so the intrinsic runway-frame yaw angle is zero.

    LARD projection uses intrinsic yaw = runway_azimuth - yaw_cam; setting
    yaw_cam = runway_azimuth aligns the camera with the runway axis in the
    local frame (heading error = 0 reference).
    """
    return CameraPoseLocal(
        along_track_m=pose_gt.along_track_m,
        lateral_offset_m=0.0,
        height_m=pose_gt.height_m,
        yaw_cam_deg=scene.runway_azimuth_deg,
        pitch_cam_deg=pose_gt.pitch_cam_deg,
        roll_cam_deg=pose_gt.roll_cam_deg,
    )


def nominal_pose_with_prior(
    pose_gt: CameraPoseLocal,
    scene: RunwayScene,
    *,
    delta_he_deg: float = 0.0,
    delta_lat_m: float = 0.0,
) -> CameraPoseLocal:
    """
  Prior pose for warm-started homography (simulates t-1 estimates).

  Keeps the four known DOF from the current frame; perturbs heading error and
  lateral offset by ``delta_*`` relative to GT (positive delta = prior is
  right-of / clockwise of truth in the same sign convention as GT pose fields).
  """
    he_gt = heading_error_ground_deg(pose_gt, scene)
    he_prior = he_gt + delta_he_deg
    # Small-angle: decrease yaw_cam to increase HE in the ground frame.
    yaw_prior = pose_gt.yaw_cam_deg - (he_prior - he_gt)
    return CameraPoseLocal(
        along_track_m=pose_gt.along_track_m,
        lateral_offset_m=pose_gt.lateral_offset_m + delta_lat_m,
        height_m=pose_gt.height_m,
        yaw_cam_deg=yaw_prior,
        pitch_cam_deg=pose_gt.pitch_cam_deg,
        roll_cam_deg=pose_gt.roll_cam_deg,
    )


def pose_with_he_lat(
    pose_four_dof: CameraPoseLocal,
    scene: RunwayScene,
    *,
    heading_error_deg: float,
    lateral_offset_m: float,
    yaw_reference: CameraPoseLocal,
) -> CameraPoseLocal:
    """
    Build a full 6-DOF pose from known 4-DOF instruments + estimated HE / lateral.

    ``yaw_reference`` supplies the yaw↔HE calibration for the current frame (typically GT).
    """
    he_ref = heading_error_ground_deg(yaw_reference, scene)
    yaw_cam = yaw_reference.yaw_cam_deg - (heading_error_deg - he_ref)
    return CameraPoseLocal(
        along_track_m=pose_four_dof.along_track_m,
        lateral_offset_m=lateral_offset_m,
        height_m=pose_four_dof.height_m,
        yaw_cam_deg=yaw_cam,
        pitch_cam_deg=pose_four_dof.pitch_cam_deg,
        roll_cam_deg=pose_four_dof.roll_cam_deg,
    )


def intrinsic_yaw_deg(pose: CameraPoseLocal, scene: RunwayScene) -> float:
    """First LARD intrinsic rotation component (degrees) — proxy for heading error."""
    return scene.runway_azimuth_deg - pose.yaw_cam_deg


def warp_image_to_plane(
    image: np.ndarray,
    homography: PlaneHomography,
    *,
    plane_roi_xy: np.ndarray | None = None,
    margin_m: float = 150.0,
    meters_per_pixel: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Warp an image into a bird's-eye plane view.

    ``plane_roi_xy`` — (N, 2) runway corner plane coords (m); ROI is their
    bounding box + margin. Without it, the warp often misses the runway (black).
    """
    if plane_roi_xy is not None:
        xs = plane_roi_xy[:, 0]
        ys = plane_roi_xy[:, 1]
        x0 = float(xs.min() - margin_m)
        x1 = float(xs.max() + margin_m)
        y0 = float(ys.min() - margin_m)
        y1 = float(ys.max() + margin_m)
    else:
        x0, x1 = -100.0, 3500.0
        y0, y1 = -150.0, 150.0

    width_px = max(1, int((x1 - x0) / meters_per_pixel))
    height_px = max(1, int((y1 - y0) / meters_per_pixel))

    scale = 1.0 / meters_per_pixel
    plane_to_out = np.array(
        [[scale, 0, -x0 * scale], [0, -scale, y1 * scale], [0, 0, 1]],
        dtype=np.float64,
    )
    # warpPerspective maps destination pixels → source image; use plane→out then out→image⁻¹.
    out_to_image = plane_to_out @ homography.image_to_plane

    warped = cv2.warpPerspective(
        image,
        out_to_image.astype(np.float32),
        (width_px, height_px),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return warped, out_to_image


def image_points_to_plane(
    points_xy: np.ndarray,
    homography: PlaneHomography,
) -> np.ndarray:
    """Map image points (N, 2) to plane coordinates (N, 2) in meters."""
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, homography.image_to_plane.astype(np.float32))
    return out.reshape(-1, 2)
