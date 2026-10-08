"""Local runway geometry for pose estimation (g_dof)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from runway_detection.lard.intrinsics import CameraIntrinsics
from runway_detection.lard.loader import LardSample, load_runways_database
from runway_detection.lard.projection import (
    CORNER_NAMES,
    geodetic_to_cartesian,
    get_forward_azimuth,
    runway_corners_wgs84,
)


@dataclass(frozen=True)
class RunwayScene:
    """
    Runway rectangle in a local runway frame.

    Axes (LARD convention):
    - X: along runway (threshold → far end)
    - Y: left of centerline
    - Z: up (relative to threshold midpoint altitude)
    """

    corner_points_local: np.ndarray  # (4, 3) — TR, TL, BL, BR
    runway_azimuth_deg: float
    origin_geodetic: tuple[float, float, float]  # threshold midpoint (lat, lon, alt)
    intrinsics: CameraIntrinsics
    airport: str = ""
    runway_id: str = ""

    @property
    def width_m(self) -> float:
        tr, tl, _, _ = self.corner_points_local
        return float(np.linalg.norm(tr - tl))

    @property
    def length_m(self) -> float:
        tr, _, bl, _ = self.corner_points_local
        return float(np.linalg.norm(tr - bl))

    @classmethod
    def from_lard_sample(
        cls,
        sample: LardSample,
        runways_db: dict,
    ) -> RunwayScene:
        runway_entry = runways_db[sample.airport][sample.runway]
        corners_latlon = runway_corners_wgs84(runway_entry)
        a, b, c, d = corners_latlon
        runway_azimuth_deg = get_forward_azimuth((d + c) / 2.0, (a + b) / 2.0)
        origin = (c + d) / 2.0
        corner_points_local = np.array(
            [
                geodetic_to_cartesian(
                    pt[0],
                    pt[1],
                    pt[2],
                    origin[0],
                    origin[1],
                    origin[2],
                    runway_azimuth_deg,
                )
                for pt in corners_latlon
            ],
            dtype=float,
        )
        return cls(
            corner_points_local=corner_points_local,
            runway_azimuth_deg=runway_azimuth_deg,
            origin_geodetic=(float(origin[0]), float(origin[1]), float(origin[2])),
            intrinsics=sample.intrinsics,
            airport=sample.airport,
            runway_id=sample.runway,
        )

    @classmethod
    def from_lard(
        cls,
        sample: LardSample,
        data_root: str,
        *,
        source: str = "flsim",
    ) -> RunwayScene:
        runways_db = load_runways_database(data_root, source)
        return cls.from_lard_sample(sample, runways_db)


@dataclass(frozen=True)
class CameraPoseLocal:
    """
    Camera pose in the runway local frame.

    ``along_track_m``: distance from threshold midpoint along approach (meters).
    ``lateral_offset_m``: cross-track offset; positive = left of centerline (meters).
    ``height_m``: height above runway plane at threshold (meters).
  """

    along_track_m: float
    lateral_offset_m: float
    height_m: float
    yaw_cam_deg: float
    pitch_cam_deg: float
    roll_cam_deg: float

    @property
    def position_local(self) -> np.ndarray:
        return np.array(
            [-self.along_track_m, self.lateral_offset_m, self.height_m],
            dtype=float,
        )

    @classmethod
    def from_lard_sample(cls, sample: LardSample, scene: RunwayScene) -> CameraPoseLocal:
        lat0, lon0, alt0 = scene.origin_geodetic
        cam = geodetic_to_cartesian(
            sample.lat,
            sample.lon,
            sample.alt,
            lat0,
            lon0,
            alt0,
            scene.runway_azimuth_deg,
        )
        return cls(
            along_track_m=float(abs(cam[0])),
            lateral_offset_m=float(cam[1]),
            height_m=float(cam[2]),
            yaw_cam_deg=sample.yaw,
            pitch_cam_deg=sample.pitch,
            roll_cam_deg=sample.roll,
        )


def corners_px_array(corners: dict[str, tuple[float, float]]) -> np.ndarray:
    return np.array([corners[name] for name in CORNER_NAMES], dtype=float)
