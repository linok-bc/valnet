"""Pinhole camera model for the BEV transform."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    """Intrinsics plus the camera's fixed mounting tilt.

    Attributes:
        width, height: image size in pixels.
        fx, fy: focal length in pixels.
        cx, cy: principal point in pixels.
        tilt_deg: fixed downward tilt of the camera relative to the airframe,
            POSITIVE MEANING NOSE-DOWN (the camera looks below the body x axis).
            Ryu et al. call this theta_c and note it is negative in their body
            frame; we flip the sign so "more positive = looking further down"
            everywhere in this package.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    tilt_deg: float = 0.0

    @classmethod
    def from_fov(cls, width, height, fov_x_deg, fov_y_deg, tilt_deg=0.0):
        """Build from horizontal/vertical field of view, principal point centred."""
        fx = (width / 2) / math.tan(math.radians(fov_x_deg) / 2)
        fy = (height / 2) / math.tan(math.radians(fov_y_deg) / 2)
        return cls(width, height, fx, fy, width / 2, height / 2, tilt_deg)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    def scaled(self, width, height) -> "CameraModel":
        """Intrinsics for a resized image."""
        sx, sy = width / self.width, height / self.height
        return CameraModel(width, height, self.fx * sx, self.fy * sy,
                           self.cx * sx, self.cy * sy, self.tilt_deg)
