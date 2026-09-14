"""
Bird's-eye-view transform from aircraft attitude, after Ryu et al. (IJRR 2024),
"An estimation method for vision-based autonomous landing system for fixed-wing
aircraft", section 3.1.2.4.

Two departures from the paper, both deliberate:

1. The paper picks four source points from vanishing-point geometry (their
   Figures 8-10) and warps them to the image corners. We instead compute the
   exact ground-plane homography from the intrinsics and attitude. It is the
   same idea with a closed form: testable, and it gives a BEV canvas with known
   ground coordinates instead of an arbitrary one.

2. YAW IS NOT USED. The paper warps using roll and pitch only (section
   3.1.2.4), and so do we. That is the point: whatever heading error the
   aircraft has against the runway SURVIVES into the BEV image as a rotation of
   the runway rectangle. For an oriented-box model that rotation is the signal,
   not a nuisance -- it is the heading error, read directly off the box angle.

Frames (all right-handed, OpenCV camera convention):

    camera: x right, y down, z forward along the optical axis
    ground: X forward (horizontal, camera azimuth), Y right, Z down

The world origin sits AT the camera, so the ground plane is Z = h where h is
height above ground. A ground point is (X, Y, h).
"""

import math

import cv2
import numpy as np

from experiments.bev.camera import CameraModel

# camera axes when depression and roll are both zero (looking level, forward):
#   x_cam = Y_ground, y_cam = Z_ground, z_cam = X_ground
_R_LEVEL = np.array([[0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0],
                     [1.0, 0.0, 0.0]], dtype=np.float64)


def depression_deg(pitch_deg: float, tilt_deg: float) -> float:
    """Total downward look angle of the optical axis.

    Positive means looking at the ground. A nose-up pitch raises the camera, so
    it subtracts from the fixed downward mounting tilt.
    """
    return tilt_deg - pitch_deg


def rotation_ground_to_camera(roll_deg: float, depression: float) -> np.ndarray:
    """R such that p_camera = R @ p_ground.

    `roll_deg` follows the collector's airframe convention: POSITIVE IS RIGHT
    WING DOWN. Rolling the aircraft right rotates the camera clockwise about its
    own optical axis, which rotates the scene anticlockwise in the image, hence
    the negated angle below. Fitting the camera against the logs (see
    experiments/bev_yolo/calibrate.py) picks this sign out cleanly: it lands at a
    1.4 px median corner residual against 5.6 px for the other one.
    """
    d = math.radians(depression)
    r = -math.radians(roll_deg)
    # pitch the camera down about its own x (right) axis
    rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, math.cos(d), -math.sin(d)],
                   [0.0, math.sin(d), math.cos(d)]], dtype=np.float64)
    # roll about the optical axis
    rz = np.array([[math.cos(r), -math.sin(r), 0.0],
                   [math.sin(r), math.cos(r), 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ rx @ _R_LEVEL


def ground_to_image(cam: CameraModel, roll_deg: float, pitch_deg: float,
                    height_m: float) -> np.ndarray:
    """Homography mapping ground (X, Y, 1) -> image (u, v, 1).

    Ground points are (X forward, Y right) in metres; the plane sits `height_m`
    below the camera.
    """
    R = rotation_ground_to_camera(roll_deg, depression_deg(pitch_deg, cam.tilt_deg))
    M = cam.K @ R
    # columns for X, Y, and the constant term h * Z
    return np.column_stack([M[:, 0], M[:, 1], height_m * M[:, 2]])


def horizon_v(cam: CameraModel, roll_deg: float, pitch_deg: float):
    """Image row of the horizon at the principal column, or None if off-frame.

    Sanity anchor: with zero roll and zero depression the horizon sits exactly
    on the principal point; looking down pushes it ABOVE centre (smaller v).
    """
    R = rotation_ground_to_camera(roll_deg, depression_deg(pitch_deg, cam.tilt_deg))
    # the vanishing line of the ground plane is the image of its points at
    # infinity: directions (X, Y, 0)
    l = np.linalg.inv(cam.K).T @ R @ np.array([0.0, 0.0, 1.0])
    if abs(l[1]) < 1e-12:
        return None
    return float(-(l[0] * cam.cx + l[2]) / l[1])


class BEVGrid:
    """A metric raster of the ground plane.

    Pixel (u, v) of the BEV image corresponds to the ground point

        X = x_near + (height - 1 - v) * mpp        (forward, metres)
        Y = (u - (width - 1) / 2)    * mpp         (right,   metres)

    so the aircraft sits at the bottom-centre looking up the image, X grows
    upward and Y grows rightward.
    """

    def __init__(self, width: int, height: int, mpp: float, x_near: float = 0.0):
        self.width = int(width)
        self.height = int(height)
        self.mpp = float(mpp)
        self.x_near = float(x_near)

    @property
    def size(self):
        return (self.width, self.height)

    @property
    def A(self) -> np.ndarray:
        """Homography mapping BEV pixel (u, v, 1) -> ground (X, Y, 1)."""
        return np.array([
            [0.0, -self.mpp, self.x_near + (self.height - 1) * self.mpp],
            [self.mpp, 0.0, -(self.width - 1) / 2.0 * self.mpp],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def extent_m(self):
        """(forward_min, forward_max, lateral_half_width) in metres."""
        return (self.x_near,
                self.x_near + (self.height - 1) * self.mpp,
                (self.width - 1) / 2.0 * self.mpp)


def bev_from_image(cam: CameraModel, grid: BEVGrid, roll_deg: float,
                   pitch_deg: float, height_m: float) -> np.ndarray:
    """Homography mapping source image pixels -> BEV pixels.

    This is what cv2.warpPerspective and cv2.perspectiveTransform both want.
    """
    h_gi = ground_to_image(cam, roll_deg, pitch_deg, height_m)
    image_from_bev = h_gi @ grid.A
    return np.linalg.inv(image_from_bev)


def warp_image(img: np.ndarray, m_bev: np.ndarray, grid: BEVGrid,
               border_value=0) -> np.ndarray:
    """Warp a perspective image into the BEV grid."""
    return cv2.warpPerspective(
        img, m_bev, grid.size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
