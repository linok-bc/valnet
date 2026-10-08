"""Deterministic correction from raw 2D plane readout to metric pose residuals.

Coupling correction (warm-started homography)
---------------------------------------------
Raw lateral readout ``lat_2D`` (threshold Y in the prior plane) mixes true
cross-track offset with a rotation artefact. For heading error ``he_2D`` (deg)
and known along-track distance ``along_track_m``:

    lat_corr = lat_2D - tan(radians(he_2D)) * along_track_m

Sequential update (prior from t-1):

    HE_est  = HE_prior - he_2D
    lat_est = lat_prior - lat_corr

See ``runway_detection/homography/README.md`` and ``scripts/validate_homography_sequence.py``.
"""

from __future__ import annotations

import math

from runway_detection.homography.measure import PlanePoseEstimate


def correct_plane_residuals(
    estimate: PlanePoseEstimate,
    *,
    along_track_m: float,
) -> tuple[float, float]:
    """
    Convert raw plane readout (centerline angle + threshold Y) to metric residuals.

    In a warm-started prior frame, ``estimate`` is the pose offset of the true
    runway w.r.t. the prior homography. Raw ``lateral_offset_m`` (threshold Y)
    couples with heading error: rotating the runway quad by ``he`` shifts the
    threshold midpoint by roughly ``tan(he) * along_track`` in plane Y.

    Returns ``(heading_error_deg, lateral_offset_m)`` corrected residuals.
    """
    he_deg = estimate.heading_error_deg
    he_rad = math.radians(he_deg)
    lat_raw = estimate.lateral_offset_m
    lat_corr = lat_raw - math.tan(he_rad) * along_track_m
    return he_deg, lat_corr


def apply_prior_residual(
    he_prior_deg: float,
    lat_prior_m: float,
    estimate: PlanePoseEstimate,
    *,
    along_track_m: float,
) -> tuple[float, float]:
    """
    Update ground-frame HE / lateral from a prior and plane readout.

    The plane residual points from truth toward the prior alignment; subtract
    it to move the estimate toward the observed runway pose.
    """
    d_he, d_lat = correct_plane_residuals(estimate, along_track_m=along_track_m)
    return he_prior_deg - d_he, lat_prior_m - d_lat


def correct_plane_residuals_small_angle(
    estimate: PlanePoseEstimate,
    *,
    along_track_m: float,
) -> tuple[float, float]:
    """Same as :func:`correct_plane_residuals` but uses ``he_rad * along``."""
    he_deg = estimate.heading_error_deg
    he_rad = math.radians(he_deg)
    lat_corr = estimate.lateral_offset_m - he_rad * along_track_m
    return he_deg, lat_corr
