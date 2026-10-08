"""LARD V2 camera intrinsics (FOV-based, matching label_export.computeLabels)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# LARD V2 scenario default (see LARD 2.0 paper / scenario YAML).
DEFAULT_FOV_X_DEG = 60.0
DEFAULT_FOV_Y_DEG = 60.0
DEFAULT_IMAGE_SIZE = 1024


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics plus LARD-style FOV projection helpers."""

    width: int
    height: int
    fov_x_deg: float
    fov_y_deg: float
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fov(
        cls,
        width: int = DEFAULT_IMAGE_SIZE,
        height: int = DEFAULT_IMAGE_SIZE,
        fov_x_deg: float = DEFAULT_FOV_X_DEG,
        fov_y_deg: float = DEFAULT_FOV_Y_DEG,
    ) -> CameraIntrinsics:
        fx = width / (2.0 * math.tan(math.radians(fov_x_deg / 2.0)))
        fy = height / (2.0 * math.tan(math.radians(fov_y_deg / 2.0)))
        return cls(
            width=width,
            height=height,
            fov_x_deg=fov_x_deg,
            fov_y_deg=fov_y_deg,
            fx=fx,
            fy=fy,
            cx=width / 2.0,
            cy=height / 2.0,
        )

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=float,
        )

    def project_camera_point(self, point_cam: np.ndarray) -> tuple[float, float]:
        """
        Project a point in the LARD camera frame (x forward, y left, z up)
        using the same perspective model as ``label_export.pointcam_to_pix``.
        """
        p = np.asarray(point_cam, dtype=float).reshape(3)
        if p[0] <= 0:
            raise ValueError("Point is behind the camera (x <= 0)")

        fov_h_rad = math.radians(self.fov_x_deg)
        fov_v_rad = math.radians(self.fov_y_deg)
        world_width = 2.0 * p[0] * math.tan(fov_h_rad * 0.5)
        world_height = 2.0 * p[0] * math.tan(fov_v_rad * 0.5)

        x_norm = 1.0 - ((p[1] / world_width) + 0.5)
        y_norm = 1.0 - ((p[2] / world_height) + 0.5)
        return x_norm * self.width, y_norm * self.height
