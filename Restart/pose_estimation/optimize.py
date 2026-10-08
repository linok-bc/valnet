"""
Mask-space pose optimization: coarse grid + local refinement (LM or GD).

Minimizes a mask distance metric between an observed (soft) mask Y and
g_dof(yaw, lateral_offset) inside an ROI (spec §4.3–§4.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

RefineMethod = Literal["lm", "gd"]

from pose_estimation.g_dof import GDofParams, GDofState, g_dof
from pose_estimation.metrics import MaskMetric, metric_value, residuals_l2, soft_dice
from pose_estimation.roi import ROIBox, downsample_mask, roi_from_mask
from pose_estimation.runway_model import RunwayScene


@dataclass(frozen=True)
class SearchBounds:
    yaw_min_deg: float
    yaw_max_deg: float
    lateral_min_m: float
    lateral_max_m: float


@dataclass
class PoseEstimate:
    state: GDofState
    cost: float
    dice: float
    n_coarse_evals: int = 0
    n_refine_iters: int = 0
    converged: bool = False
    search_mode: str = "single"
    refine_method: RefineMethod = "lm"
    refine_trace: list[GDofState] | None = None
    covariance: object | None = None  # PoseCovariance when computed
    state_coarse: GDofState | None = None  # grid winner before refine
    seed_search: object | None = None  # SeedSearchResult when relocation ran


@dataclass(frozen=True)
class CoarseLevel:
    """One hierarchical search level: half-ranges and grid steps."""

    yaw_half_range_deg: float
    lateral_half_range_m: float
    yaw_step_deg: float
    lateral_step_m: float


# Wide → medium → fine (zoom window ≥ 2× previous step per axis).
DEFAULT_COARSE_LEVELS: tuple[CoarseLevel, ...] = (
    CoarseLevel(8.0, 80.0, 2.0, 10.0),
    CoarseLevel(2.0, 10.0, 0.5, 2.5),
    CoarseLevel(0.5, 2.5, 0.125, 0.625),
)

# Narrow → wide warm-start around t-1 (hierarchical beam search; see hierarchical_coarse_search).
WARM_START_COARSE_LEVELS: tuple[CoarseLevel, ...] = (
    CoarseLevel(5.0, 25.0, 0.5, 2.5),
    CoarseLevel(1.0, 5.0, 0.125, 0.5),
    CoarseLevel(0.25, 1.25, 0.0625, 0.3125),
    CoarseLevel(0.0625, 0.3125, 0.03125, 0.15625),
)


@dataclass
class OptimizerConfig:
    metric: MaskMetric = MaskMetric.L2
    roi_margin_px: int = 48
    downsample_size: int = 48
    # Legacy single-grid coarse (used when search_mode="single").
    coarse_yaw_step_deg: float = 0.25
    coarse_lateral_step_m: float = 1.0
    search_mode: str = "coarse_to_fine"  # "single" | "coarse_to_fine"
    coarse_levels: tuple[CoarseLevel, ...] = DEFAULT_COARSE_LEVELS
    beam_top_k: int = 3
    render_blur_sigma: float = 1.5
    refine_method: RefineMethod = "lm"
    max_refine_iters: int = 12
    cost_tol: float = 1e-6
    step_tol_deg: float = 0.005
    step_tol_m: float = 0.02
    lm_lambda_init: float = 1e-2
    lm_lambda_up: float = 10.0
    lm_lambda_down: float = 0.3
    lm_min_cost_drop: float = 0.03  # stop when one accepted LM step reduces L2 by less than this
    gd_lr_init: float = 1.0
    gd_lr_min: float = 1e-4
    gd_max_line_search: int = 12
    gd_step_yaw_scale: float = 80.0  # multiplies ∂cost/∂yaw → step in degrees
    gd_step_lat_scale: float = 200.0  # multiplies ∂cost/∂lat → step in meters
    gd_fd_eps_yaw_deg: float = 0.1
    gd_fd_eps_lat_m: float = 1.0
    fd_eps_yaw_deg: float = 0.02
    fd_eps_lateral_m: float = 0.05
    measurement_sigma: float = 0.15  # per-pixel std on downsampled mask [0,1]
    # "raster": full-res fillPoly → crop → blur → area downsample (original path).
    #   Piecewise-constant in (yaw, lat): FD Jacobians vanish for sub-pixel steps.
    # "analytic": per-pixel coverage of the projected quad on the downsampled ROI
    #   grid from signed edge distances (Gaussian-CDF edge profile). Smooth in
    #   (yaw, lat), so LM / Fisher information see sub-pixel motion; ~10× faster.
    renderer: str = "raster"


@dataclass
class _EvalContext:
    scene: RunwayScene
    params: GDofParams
    observed: np.ndarray
    roi: ROIBox
    ds_size: int
    metric: MaskMetric
    render_blur_sigma: float = 0.0
    renderer: str = "raster"
    _obs_ds: np.ndarray | None = None

    def render_patch(self, state: GDofState) -> np.ndarray:
        import cv2

        mask = g_dof(self.scene, self.params, state).astype(np.float32)
        patch = self.roi.crop(mask)
        if self.render_blur_sigma > 0:
            patch = cv2.GaussianBlur(patch, (0, 0), self.render_blur_sigma)
        return patch

    def render_downsampled(self, state: GDofState) -> np.ndarray:
        if self.renderer == "analytic":
            return self._render_analytic(state)
        return downsample_mask(self.render_patch(state), self.ds_size)

    def _render_analytic(self, state: GDofState) -> np.ndarray:
        """
        Soft quad coverage on the (ds × ds) ROI grid.

        Coverage = Π_edges Φ(d_k / σ) with d_k the signed distance (full-res px,
        + inside) from each downsampled pixel center to edge k, and
        σ² = blur² + (pixel footprint)²/12 (box-filter variance of the area
        downsample). Exact away from corners; the quad must be convex.
        """
        from pose_estimation.g_dof import corners_dict_to_polygon, g_dof_corners

        corners = corners_dict_to_polygon(g_dof_corners(self.scene, self.params, state))
        n = self.ds_size
        step_x = self.roi.width / n
        step_y = self.roi.height / n
        # Full-res pixel (i, j) covers [i, i+1); ds pixel centers in the same frame.
        xs = self.roi.x0 + (np.arange(n) + 0.5) * step_x - 0.5
        ys = self.roi.y0 + (np.arange(n) + 0.5) * step_y - 0.5
        gx, gy = np.meshgrid(xs, ys)
        sigma = math.sqrt(self.render_blur_sigma**2 + (step_x * step_y) / 12.0 + 1.0 / 12.0)
        area2 = float(
            np.sum(corners[:, 0] * np.roll(corners[:, 1], -1) - np.roll(corners[:, 0], -1) * corners[:, 1])
        )
        orient = 1.0 if area2 > 0 else -1.0
        cov = np.ones((n, n), dtype=np.float64)
        for k in range(4):
            p = corners[k]
            q = corners[(k + 1) % 4]
            e = q - p
            length = math.hypot(float(e[0]), float(e[1]))
            if length < 1e-9:
                continue
            # cross(e, g − p) > 0 on the interior side of a positively oriented polygon.
            d = orient * (e[0] * (gy - p[1]) - e[1] * (gx - p[0])) / length
            cov *= _gauss_cdf(d / sigma)
        return cov.astype(np.float32)

    def observed_downsampled(self) -> np.ndarray:
        if self._obs_ds is None:
            patch = self.roi.crop(self.observed).astype(np.float32)
            self._obs_ds = downsample_mask(patch, self.ds_size)
        return self._obs_ds

    def cost(self, state: GDofState) -> float:
        pred = self.render_downsampled(state)
        obs = self.observed_downsampled()
        return metric_value(self.metric, obs, pred)

    def residuals(self, state: GDofState) -> np.ndarray:
        return residuals_l2(self.observed_downsampled(), self.render_downsampled(state))


def _gauss_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF (tanh approximation, |err| < 2e-4)."""
    return 0.5 * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))


def _state_vec(state: GDofState) -> np.ndarray:
    return np.array([state.yaw_cam_deg, state.lateral_offset_m], dtype=float)


def _state_from_vec(v: np.ndarray) -> GDofState:
    return GDofState(float(v[0]), float(v[1]))


def bounds_around(
    center: GDofState,
    *,
    yaw_half_range_deg: float = 2.0,
    lateral_half_range_m: float = 25.0,
) -> SearchBounds:
    return SearchBounds(
        yaw_min_deg=center.yaw_cam_deg - yaw_half_range_deg,
        yaw_max_deg=center.yaw_cam_deg + yaw_half_range_deg,
        lateral_min_m=center.lateral_offset_m - lateral_half_range_m,
        lateral_max_m=center.lateral_offset_m + lateral_half_range_m,
    )


def _dice_score(ctx: _EvalContext, state: GDofState) -> float | None:
    try:
        pred = ctx.render_downsampled(state)
        return soft_dice(ctx.observed_downsampled(), pred)
    except (ValueError, FloatingPointError):
        return None


def _grid_candidates(
    ctx: _EvalContext,
    bounds: SearchBounds,
    *,
    yaw_step_deg: float,
    lateral_step_m: float,
) -> list[tuple[GDofState, float]]:
    yaw_grid = np.arange(bounds.yaw_min_deg, bounds.yaw_max_deg + 1e-9, yaw_step_deg)
    lat_grid = np.arange(bounds.lateral_min_m, bounds.lateral_max_m + 1e-9, lateral_step_m)
    scored: list[tuple[GDofState, float]] = []
    for yaw in yaw_grid:
        for lat in lat_grid:
            state = GDofState(float(yaw), float(lat))
            score = _dice_score(ctx, state)
            if score is not None:
                scored.append((state, score))
    return scored


def coarse_search(
    ctx: _EvalContext,
    bounds: SearchBounds,
    *,
    yaw_step_deg: float,
    lateral_step_m: float,
) -> tuple[GDofState, float, int]:
    scored = _grid_candidates(
        ctx, bounds, yaw_step_deg=yaw_step_deg, lateral_step_m=lateral_step_m
    )
    if not scored:
        center = GDofState(
            0.5 * (bounds.yaw_min_deg + bounds.yaw_max_deg),
            0.5 * (bounds.lateral_min_m + bounds.lateral_max_m),
        )
        return center, 1.0, 0

    best_state, best_score = max(scored, key=lambda item: item[1])
    return best_state, 1.0 - best_score, len(scored)


def _dedupe_states(
    scored: list[tuple[GDofState, float]],
    *,
    yaw_tol_deg: float = 0.05,
    lat_tol_m: float = 0.25,
) -> list[tuple[GDofState, float]]:
    """Merge near-duplicate states, keeping the best score."""
    scored = sorted(scored, key=lambda item: item[1], reverse=True)
    kept: list[tuple[GDofState, float]] = []
    for state, score in scored:
        if any(
            abs(state.yaw_cam_deg - k.yaw_cam_deg) < yaw_tol_deg
            and abs(state.lateral_offset_m - k.lateral_offset_m) < lat_tol_m
            for k, _ in kept
        ):
            continue
        kept.append((state, score))
    return kept


def hierarchical_coarse_search(
    ctx: _EvalContext,
    center: GDofState,
    levels: tuple[CoarseLevel, ...],
    *,
    top_k: int = 3,
) -> tuple[GDofState, int]:
    """
    Coarse-to-fine beam search (TSS / multi-resolution style).

    At each level, evaluate a grid around every beam candidate, keep global
  top-K, then zoom. Next-level half-range should be ≥ 2× previous step so
    cell-boundary optima are not lost.
    """
    beam = [center]
    n_evals = 0
    level_scored: list[tuple[GDofState, float]] = []

    for level in levels:
        level_scored: list[tuple[GDofState, float]] = []
        for cand in beam:
            bounds = bounds_around(
                cand,
                yaw_half_range_deg=level.yaw_half_range_deg,
                lateral_half_range_m=level.lateral_half_range_m,
            )
            scored = _grid_candidates(
                ctx,
                bounds,
                yaw_step_deg=level.yaw_step_deg,
                lateral_step_m=level.lateral_step_m,
            )
            level_scored.extend(scored)
            n_evals += len(scored)

        if not level_scored:
            break

        level_scored = _dedupe_states(level_scored)
        beam = [s for s, _ in level_scored[:top_k]]

    if not level_scored:
        return center, n_evals

    best_state, _ = max(level_scored, key=lambda item: item[1])
    return best_state, n_evals


def _jacobian_fd(
    ctx: _EvalContext,
    state: GDofState,
    *,
    eps_yaw_deg: float,
    eps_lateral_m: float,
) -> np.ndarray:
    r0 = ctx.residuals(state)
    n = r0.size
    j = np.zeros((n, 2), dtype=float)

    sy_p = GDofState(state.yaw_cam_deg + eps_yaw_deg, state.lateral_offset_m)
    sy_m = GDofState(state.yaw_cam_deg - eps_yaw_deg, state.lateral_offset_m)
    sl_p = GDofState(state.yaw_cam_deg, state.lateral_offset_m + eps_lateral_m)
    sl_m = GDofState(state.yaw_cam_deg, state.lateral_offset_m - eps_lateral_m)

    try:
        j[:, 0] = (ctx.residuals(sy_p) - ctx.residuals(sy_m)) / (2.0 * eps_yaw_deg)
        j[:, 1] = (ctx.residuals(sl_p) - ctx.residuals(sl_m)) / (2.0 * eps_lateral_m)
    except (ValueError, FloatingPointError):
        pass
    return j


def _cost_gradient_fd(
    ctx: _EvalContext,
    state: GDofState,
    *,
    eps_yaw_deg: float,
    eps_lat_m: float,
) -> np.ndarray:
    """Central FD of scalar L2 cost (more stable than Jᵀr on rasterized masks)."""
    c_yaw_p = ctx.cost(GDofState(state.yaw_cam_deg + eps_yaw_deg, state.lateral_offset_m))
    c_yaw_m = ctx.cost(GDofState(state.yaw_cam_deg - eps_yaw_deg, state.lateral_offset_m))
    c_lat_p = ctx.cost(GDofState(state.yaw_cam_deg, state.lateral_offset_m + eps_lat_m))
    c_lat_m = ctx.cost(GDofState(state.yaw_cam_deg, state.lateral_offset_m - eps_lat_m))
    return np.array(
        [
            (c_yaw_p - c_yaw_m) / (2.0 * eps_yaw_deg),
            (c_lat_p - c_lat_m) / (2.0 * eps_lat_m),
        ],
        dtype=float,
    )


def refine_lm(
    ctx: _EvalContext,
    init: GDofState,
    *,
    config: OptimizerConfig,
    trace: list[GDofState] | None = None,
) -> tuple[GDofState, float, int, bool]:
    x = _state_vec(init)
    lam = config.lm_lambda_init
    cost = ctx.cost(init)
    n_iters = 0
    converged = False
    if trace is not None:
        trace.append(init)

    for _ in range(config.max_refine_iters):
        state = _state_from_vec(x)
        try:
            r = ctx.residuals(state)
            j = _jacobian_fd(
                ctx,
                state,
                eps_yaw_deg=config.fd_eps_yaw_deg,
                eps_lateral_m=config.fd_eps_lateral_m,
            )
        except (ValueError, FloatingPointError):
            break

        if not np.all(np.isfinite(j)):
            break

        jtj = j.T @ j
        g = j.T @ r

        accepted = False
        for _attempt in range(8):
            a = jtj + lam * np.eye(2)
            try:
                delta = np.linalg.solve(a, g)
            except np.linalg.LinAlgError:
                lam *= config.lm_lambda_up
                continue

            x_new = x - delta
            state_new = _state_from_vec(x_new)
            try:
                cost_new = ctx.cost(state_new)
            except (ValueError, FloatingPointError):
                lam *= config.lm_lambda_up
                continue

            if cost_new < cost:
                cost_drop = cost - cost_new
                step_yaw = abs(delta[0])
                step_lat = abs(delta[1])
                x = x_new
                cost = cost_new
                lam = max(lam * config.lm_lambda_down, 1e-12)
                accepted = True
                n_iters += 1
                state_new = _state_from_vec(x)
                if trace is not None:
                    trace.append(state_new)
                if (
                    cost_drop < config.lm_min_cost_drop
                    or (
                        cost_drop < config.cost_tol
                        and step_yaw < config.step_tol_deg
                        and step_lat < config.step_tol_m
                    )
                ):
                    converged = True
                    return state_new, cost, n_iters, converged
                break
            lam *= config.lm_lambda_up

        if not accepted:
            break

    return _state_from_vec(x), cost, n_iters, converged


def refine_gd(
    ctx: _EvalContext,
    init: GDofState,
    *,
    config: OptimizerConfig,
    trace: list[GDofState] | None = None,
) -> tuple[GDofState, float, int, bool]:
    """
    Preconditioned gradient descent on L2 mask cost.

    Uses diagonal curvature scaling from ``J^T J`` and backtracking line search
    to handle the elongated (yaw, lateral) bowl.
    """
    x = _state_vec(init)
    cost = ctx.cost(init)
    n_iters = 0
    converged = False
    if trace is not None:
        trace.append(init)

    for _ in range(config.max_refine_iters):
        state = _state_from_vec(x)
        try:
            grad = _cost_gradient_fd(
                ctx,
                state,
                eps_yaw_deg=config.gd_fd_eps_yaw_deg,
                eps_lat_m=config.gd_fd_eps_lat_m,
            )
        except (ValueError, FloatingPointError):
            break

        if not np.all(np.isfinite(grad)):
            break
        if float(np.linalg.norm(grad)) < 1e-14:
            break

        direction = np.array(
            [
                config.gd_step_yaw_scale * grad[0],
                config.gd_step_lat_scale * grad[1],
            ],
            dtype=float,
        )

        accepted = False
        alpha = config.gd_lr_init
        for _attempt in range(config.gd_max_line_search):
            if alpha < config.gd_lr_min:
                break
            x_new = x - alpha * direction
            state_new = _state_from_vec(x_new)
            try:
                cost_new = ctx.cost(state_new)
            except (ValueError, FloatingPointError):
                alpha *= 0.5
                continue

            if cost_new < cost - 1e-10:
                step_yaw = abs(x_new[0] - x[0])
                step_lat = abs(x_new[1] - x[1])
                cost_drop = cost - cost_new
                x = x_new
                cost = cost_new
                n_iters += 1
                accepted = True
                state_new = _state_from_vec(x)
                if trace is not None:
                    trace.append(state_new)
                if (
                    cost_drop < config.cost_tol
                    and step_yaw < config.step_tol_deg
                    and step_lat < config.step_tol_m
                ):
                    converged = True
                    return state_new, cost, n_iters, converged
                break
            alpha *= 0.5

        if not accepted:
            break

    return _state_from_vec(x), cost, n_iters, converged


def refine_pose(
    ctx: _EvalContext,
    init: GDofState,
    *,
    config: OptimizerConfig,
    trace: list[GDofState] | None = None,
) -> tuple[GDofState, float, int, bool]:
    if config.refine_method == "gd":
        return refine_gd(ctx, init, config=config, trace=trace)
    return refine_lm(ctx, init, config=config, trace=trace)


def build_context(
    scene: RunwayScene,
    params: GDofParams,
    observed: np.ndarray,
    *,
    config: OptimizerConfig,
    roi: ROIBox | None = None,
    init: GDofState | None = None,
) -> _EvalContext:
    if roi is None:
        obs_roi = roi_from_mask(observed > 0.25, margin_px=config.roi_margin_px)
        if init is not None:
            try:
                init_mask = g_dof(scene, params, init)
                pred_roi = roi_from_mask(init_mask, margin_px=config.roi_margin_px)
                y0 = min(obs_roi.y0, pred_roi.y0)
                x0 = min(obs_roi.x0, pred_roi.x0)
                y1 = max(obs_roi.y1, pred_roi.y1)
                x1 = max(obs_roi.x1, pred_roi.x1)
                roi = ROIBox(y0, y1, x0, x1)
            except (ValueError, FloatingPointError):
                roi = obs_roi
        else:
            roi = obs_roi
    return _EvalContext(
        scene=scene,
        params=params,
        observed=observed,
        roi=roi,
        ds_size=config.downsample_size,
        metric=config.metric,
        render_blur_sigma=config.render_blur_sigma,
        renderer=config.renderer,
    )


def estimate_pose_mask_space(
    scene: RunwayScene,
    params: GDofParams,
    observed: np.ndarray,
    *,
    init: GDofState,
    bounds: SearchBounds | None = None,
    config: OptimizerConfig | None = None,
    use_coarse: bool = True,
    roi: ROIBox | None = None,
    search_center: GDofState | None = None,
    record_refine_trace: bool = False,
    P_pred: np.ndarray | None = None,
    compute_covariance: bool = True,
    relocate_invalid_seed: bool = True,
) -> PoseEstimate:
    """
    Estimate (yaw, lateral_offset) from an observed soft mask.

    Pipeline: coarse search (single grid or coarse-to-fine beam) → refine.

    ``init`` seeds ROI union; ``search_center`` anchors the hierarchical search
    (defaults to ``init``). Refinement starts from the coarse winner, not ``init``.
    """
    config = config or OptimizerConfig()
    center = search_center or init
    ctx = build_context(scene, params, observed, config=config, roi=roi, init=init)

    seed_info = None
    if relocate_invalid_seed:
        from pose_estimation.seed_search import find_valid_seed_expanding_rings, has_mask_overlap

        if not has_mask_overlap(ctx, center):
            seed_info = find_valid_seed_expanding_rings(ctx, center)
            if seed_info.relocated:
                center = seed_info.seed
                init = seed_info.seed
                ctx = build_context(scene, params, observed, config=config, roi=None, init=init)

    state = center if use_coarse else init
    n_coarse = 0
    mode = config.search_mode
    if use_coarse:
        if mode == "coarse_to_fine":
            state, n_coarse = hierarchical_coarse_search(
                ctx,
                center,
                config.coarse_levels,
                top_k=config.beam_top_k,
            )
        else:
            b = bounds or bounds_around(center)
            state, _, n_coarse = coarse_search(
                ctx,
                b,
                yaw_step_deg=config.coarse_yaw_step_deg,
                lateral_step_m=config.coarse_lateral_step_m,
            )
            mode = "single"
    else:
        mode = "refine_only"

    state_coarse = state
    trace: list[GDofState] | None = [] if record_refine_trace else None
    state, cost, n_refine, converged = refine_pose(ctx, state, config=config, trace=trace)

    cov = None
    if compute_covariance:
        from pose_estimation.covariance import compute_pose_covariance, default_prior_covariance

        P_prior = P_pred if P_pred is not None else default_prior_covariance()
        cov = compute_pose_covariance(ctx, state, P_pred=P_prior, config=config)

    pred = ctx.render_downsampled(state)
    obs = ctx.observed_downsampled()
    dice = soft_dice(obs, pred)

    return PoseEstimate(
        state=state,
        cost=cost,
        dice=dice,
        n_coarse_evals=n_coarse,
        n_refine_iters=n_refine,
        converged=converged,
        search_mode=mode,
        refine_method=config.refine_method,
        refine_trace=trace,
        covariance=cov,
        state_coarse=state_coarse,
        seed_search=seed_info,
    )
