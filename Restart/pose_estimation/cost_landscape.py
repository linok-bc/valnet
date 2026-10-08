"""Cost landscape over (yaw, lateral) for render-and-compare pose estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pose_estimation.g_dof import GDofParams, GDofState
from pose_estimation.metrics import MaskMetric, dice_loss, l2_loss, soft_dice
from pose_estimation.optimize import OptimizerConfig, _EvalContext, build_context
from pose_estimation.runway_model import RunwayScene


@dataclass(frozen=True)
class CostLandscape:
    """Sampled cost over a 2-D grid in pose space."""

    yaw_deg: np.ndarray  # (Ny,)
    lateral_m: np.ndarray  # (Nx,)
    l2: np.ndarray  # (Ny, Nx) — same metric as LM refinement
    dice_loss: np.ndarray  # (Ny, Nx) — 1 - soft Dice
    dice_score: np.ndarray  # (Ny, Nx) — metric used in coarse search
    valid: np.ndarray  # (Ny, Nx) bool — render succeeded

    @property
    def l2_masked(self) -> np.ndarray:
        out = self.l2.copy()
        out[~self.valid] = np.nan
        return out

    @property
    def dice_loss_masked(self) -> np.ndarray:
        out = self.dice_loss.copy()
        out[~self.valid] = np.nan
        return out


def evaluate_cost_grid(
    scene: RunwayScene,
    params: GDofParams,
    observed: np.ndarray,
    *,
    center: GDofState,
    yaw_half_range_deg: float = 3.0,
    lateral_half_range_m: float = 30.0,
    n_yaw: int = 61,
    n_lat: int = 61,
    config: OptimizerConfig | None = None,
    init_for_roi: GDofState | None = None,
) -> CostLandscape:
    """
    Sample ``dist(g_dof(yaw, lat), mask_yolo)`` on a regular grid.

    Uses the **same preprocessing** as :func:`estimate_pose_mask_space`:
    ROI around observed∪rendered masks, downsample to 48×48, Gaussian blur
  (σ=1.5) on the rendered mask only.
    """
    config = config or OptimizerConfig()
    roi_init = init_for_roi or center
    ctx = build_context(scene, params, observed, config=config, init=roi_init)

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

    l2 = np.full((n_yaw, n_lat), np.nan, dtype=float)
    dice_l = np.full((n_yaw, n_lat), np.nan, dtype=float)
    dice_s = np.full((n_yaw, n_lat), np.nan, dtype=float)
    valid = np.zeros((n_yaw, n_lat), dtype=bool)
    obs = ctx.observed_downsampled()

    for iy, yaw in enumerate(yaw_deg):
        for ix, lat in enumerate(lateral_m):
            state = GDofState(float(yaw), float(lat))
            try:
                pred = ctx.render_downsampled(state)
            except (ValueError, FloatingPointError):
                continue
            valid[iy, ix] = True
            l2[iy, ix] = l2_loss(obs, pred)
            d = soft_dice(obs, pred)
            dice_s[iy, ix] = d
            dice_l[iy, ix] = 1.0 - d

    return CostLandscape(
        yaw_deg=yaw_deg,
        lateral_m=lateral_m,
        l2=l2,
        dice_loss=dice_l,
        dice_score=dice_s,
        valid=valid,
    )


def cost_at_state(ctx: _EvalContext, state: GDofState) -> tuple[float, float, float]:
    """Return (l2, dice_loss, dice_score) at one pose."""
    pred = ctx.render_downsampled(state)
    obs = ctx.observed_downsampled()
    d = soft_dice(obs, pred)
    return l2_loss(obs, pred), 1.0 - d, d


def make_eval_context(
    scene: RunwayScene,
    params: GDofParams,
    observed: np.ndarray,
    *,
    init: GDofState,
    config: OptimizerConfig | None = None,
) -> _EvalContext:
    config = config or OptimizerConfig()
    return build_context(scene, params, observed, config=config, init=init)
