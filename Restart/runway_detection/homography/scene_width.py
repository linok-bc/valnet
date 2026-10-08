"""Runway scene helpers — scaled width for width calibration."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pose_estimation.runway_model import RunwayScene


def scene_with_runway_width(scene: RunwayScene, width_m: float) -> RunwayScene:
    """
    Return a copy of ``scene`` with runway width set to ``width_m`` (meters).

    Scales corner Y coordinates about the centerline (Y=0); length unchanged.
    """
    if width_m <= 0:
        raise ValueError("width_m must be positive")
    scale = width_m / scene.width_m
    corners = scene.corner_points_local.copy()
    corners[:, 1] *= scale
    return replace(scene, corner_points_local=corners)
