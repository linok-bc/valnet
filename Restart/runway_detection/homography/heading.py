"""Ground-frame heading error from camera orientation."""

from __future__ import annotations

import math

import numpy as np

from pose_estimation.runway_model import CameraPoseLocal, RunwayScene
from runway_detection.lard.projection import apply_rotations_intrinsic


def heading_error_ground_deg(pose: CameraPoseLocal, scene: RunwayScene) -> float:
    """
    Heading error in the runway local frame: angle between camera forward
    (projected on the ground plane) and +X (along-runway toward threshold).
    """
    orient = np.radians(
        [
            scene.runway_azimuth_deg - pose.yaw_cam_deg,
            90.0 - pose.pitch_cam_deg,
            pose.roll_cam_deg,
        ]
    )
    x_cam, _, _ = apply_rotations_intrinsic(*orient)
    forward = np.asarray(x_cam, dtype=float)
    forward[2] = 0.0
    norm = np.linalg.norm(forward[:2])
    if norm < 1e-9:
        return 0.0
    forward /= norm
    return math.degrees(math.atan2(forward[1], forward[0]))
