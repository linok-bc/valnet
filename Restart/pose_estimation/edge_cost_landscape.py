"""Cost landscape over (yaw, lateral) for edge-based joint fit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pose_estimation.edge_fit import edge_cost, prepare_edge_observation
from pose_estimation.g_dof import GDofParams, GDofState
from pose_estimation.runway_model import RunwayScene


@dataclass(frozen=True)
class EdgeCostLandscape:
    yaw_deg: np.ndarray
    lateral_m: np.ndarray
    edge_cost: np.ndarray  # (Ny, Nx) mean Huber point-to-line
    valid: np.ndarray

    @property
    def edge_cost_masked(self) -> np.ndarray:
        out = self.edge_cost.copy()
        out[~self.valid] = np.nan
        return out


def evaluate_edge_cost_grid(
    scene: RunwayScene,
    params: GDofParams,
    observed: np.ndarray,
    *,
    center: GDofState,
    assign_state: GDofState | None = None,
    yaw_half_range_deg: float = 3.0,
    lateral_half_range_m: float = 30.0,
    n_yaw: int = 61,
    n_lat: int = 61,
    huber_delta_px: float = 3.0,
    min_tangent_ratio: float = 0.5,
    max_contour_points: int = 400,
) -> EdgeCostLandscape:
    """
    Sample edge joint cost on a regular grid in pose space.

    Contour points are extracted once from the observed mask; left/right
    assignment uses projected edges at ``assign_state`` (defaults to ``center``).
    """
    assign = assign_state or center
    left_pts, right_pts = prepare_edge_observation(
        observed,
        scene,
        params,
        assign,
        min_tangent_ratio=min_tangent_ratio,
        max_points=max_contour_points,
    )

    yaw_deg = np.linspace(
        center.yaw_cam_deg - yaw_half_range_deg,
        center.yaw_cam_deg + yaw_half_range_deg,
        n_yaw,
    )
    lateral_m = np.linspace(
        center.lateral_offset_m - lateral_half_range_m,
        center.lateral_offset_m + lateral_half_range_m,
        n_lat,
    )

    costs = np.full((n_yaw, n_lat), np.nan, dtype=float)
    valid = np.zeros((n_yaw, n_lat), dtype=bool)

    for iy, yaw in enumerate(yaw_deg):
        for ix, lat in enumerate(lateral_m):
            state = GDofState(float(yaw), float(lat))
            try:
                c = edge_cost(
                    scene,
                    params,
                    state,
                    left_pts,
                    right_pts,
                    huber_delta_px=huber_delta_px,
                )
            except (ValueError, FloatingPointError):
                continue
            if not np.isfinite(c):
                continue
            valid[iy, ix] = True
            costs[iy, ix] = c

    return EdgeCostLandscape(
        yaw_deg=yaw_deg,
        lateral_m=lateral_m,
        edge_cost=costs,
        valid=valid,
    )


def edge_cost_at_state(
    scene: RunwayScene,
    params: GDofParams,
    state: GDofState,
    left_pts: np.ndarray,
    right_pts: np.ndarray,
    *,
    huber_delta_px: float = 3.0,
) -> float:
    return edge_cost(
        scene,
        params,
        state,
        left_pts,
        right_pts,
        huber_delta_px=huber_delta_px,
    )
