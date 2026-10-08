"""
Uniform wrappers around the repo's (yaw, lateral) estimators for evaluation.

Every estimator maps ``(FrameInput, prior)`` → ``Estimate``. ``prior`` is the
tracker's predicted state (``None`` on cold start / re-acquisition); the
estimators never see ground truth.

| name        | method                                                          |
|-------------|-----------------------------------------------------------------|
| maskfit     | render-and-compare on the soft mask (``estimate_pose_mask_space``) |
| edgefit     | Huber point-to-line LM on lateral contour points (spec_edges)    |
| cornerfit   | 2-DOF LM on the 4 quad corners extracted from the mask           |
| homography  | warm-started plane readout (``runway_detection.homography``)     |
| pnp         | OpenCV IPPE 6-DOF PnP on the 4 corners (ignores known DOF)       |
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from evaluation.observations import corners_dict, extract_quad, order_quad_like
from pose_estimation.edge_fit import (
    edge_residuals,
    extract_lateral_contour_points,
    prepare_edge_observation,
)
from pose_estimation.g_dof import (
    GDofParams,
    GDofState,
    camera_pose_from_g_dof,
    corners_dict_to_polygon,
    g_dof_corners,
)
from pose_estimation.optimize import CoarseLevel, OptimizerConfig, estimate_pose_mask_space
from pose_estimation.runway_model import RunwayScene
from runway_detection.homography.measure import measure_from_image_corners
from runway_detection.homography.plane_homography import homography_from_pose
from runway_detection.homography.readout import apply_prior_residual

# Cold search must cover the XP12 envelope (|yaw| ≤ 17°, |lat| ≤ 132 m).
COLD_WIDE_LEVELS: tuple[CoarseLevel, ...] = (
    CoarseLevel(18.0, 140.0, 2.0, 10.0),
    CoarseLevel(2.0, 10.0, 0.5, 2.5),
    CoarseLevel(0.5, 2.5, 0.125, 0.625),
)


@dataclass
class FrameInput:
    mask: np.ndarray | None
    scene: RunwayScene  # estimator's runway model (width prior may be wrong)
    params: GDofParams  # known 4 DOF as measured (possibly noisy)
    _quad: np.ndarray | None = field(default=None, repr=False)
    _quad_done: bool = False

    def quad(self) -> np.ndarray | None:
        """Unordered cyclic quad extracted from the mask (cached)."""
        if not self._quad_done:
            self._quad = extract_quad(self.mask) if self.mask is not None else None
            self._quad_done = True
        return self._quad

    def ordered_corners(self, reference_state: GDofState) -> np.ndarray | None:
        q = self.quad()
        if q is None:
            return None
        try:
            ref = corners_dict_to_polygon(g_dof_corners(self.scene, self.params, reference_state))
        except ValueError:
            return None
        return order_quad_like(q, ref)


@dataclass
class Estimate:
    state: GDofState | None
    ok: bool
    P: np.ndarray | None = None  # measurement covariance [yaw, lat] (data only)
    n_evals: int = 0
    n_iters: int = 0
    runtime_ms: float = 0.0
    extra: dict = field(default_factory=dict)


def _vec(s: GDofState) -> np.ndarray:
    return np.array([s.yaw_cam_deg, s.lateral_offset_m], dtype=float)


def _state(v: np.ndarray) -> GDofState:
    return GDofState(float(v[0]), float(v[1]))


def _lm(
    residual_fn,
    x0: np.ndarray,
    *,
    eps: tuple[float, float] = (1e-3, 1e-2),
    max_iters: int = 30,
    lam0: float = 1e-3,
    tol: float = 1e-9,
) -> tuple[np.ndarray, int, np.ndarray | None, np.ndarray | None]:
    """Small 2-D Levenberg–Marquardt with central-FD Jacobian. Returns x, iters, r, J."""
    x = np.asarray(x0, dtype=float)
    try:
        r = residual_fn(x)
    except ValueError:
        return x, 0, None, None
    cost = float(r @ r)
    lam = lam0
    J = None
    it = 0
    for it in range(1, max_iters + 1):
        J = np.zeros((r.size, 2))
        try:
            for k in range(2):
                dx = np.zeros(2)
                dx[k] = eps[k]
                J[:, k] = (residual_fn(x + dx) - residual_fn(x - dx)) / (2 * eps[k])
        except ValueError:
            break
        A = J.T @ J
        g = J.T @ r
        accepted = False
        while lam < 1e8:
            try:
                step = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), g)
                rn = residual_fn(x + step)
            except (np.linalg.LinAlgError, ValueError):
                lam *= 10
                continue
            cn = float(rn @ rn)
            if cn < cost:
                drop = cost - cn
                x, r, cost = x + step, rn, cn
                lam = max(lam * 0.3, 1e-9)
                accepted = True
                break
            lam *= 10
        if not accepted or drop < tol * max(cost, 1e-12):
            break
    return x, it, r, J


def _gauss_newton_cov(r: np.ndarray | None, J: np.ndarray | None) -> np.ndarray | None:
    """σ̂² (JᵀJ)⁻¹ with σ̂² from the residual — data-only covariance."""
    if r is None or J is None or r.size <= 2:
        return None
    dof = max(r.size - 2, 1)
    s2 = float(r @ r) / dof
    try:
        return s2 * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        return None


class Estimator:
    name = "base"

    def estimate(self, inp: FrameInput, prior: GDofState | None, P_prior: np.ndarray | None) -> Estimate:
        t0 = time.perf_counter()
        if inp.mask is None or not np.any(inp.mask > 0.5):
            est = Estimate(None, False, extra={"fail": "no_mask"})
        else:
            try:
                est = self._estimate(inp, prior, P_prior)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                est = Estimate(None, False, extra={"fail": type(exc).__name__})
        est.runtime_ms = 1e3 * (time.perf_counter() - t0)
        return est

    def _estimate(self, inp, prior, P_prior) -> Estimate:  # pragma: no cover
        raise NotImplementedError


# ------------------------------------------------------------------ maskfit


@dataclass
class MaskFit(Estimator):
    """``estimate_pose_mask_space`` with separate cold / warm search settings."""

    config: OptimizerConfig = field(default_factory=OptimizerConfig)
    cold_levels: tuple[CoarseLevel, ...] = COLD_WIDE_LEVELS
    warm_levels: tuple[CoarseLevel, ...] | None = None  # None → refine only from prior
    name: str = "maskfit"

    def _estimate(self, inp, prior, P_prior) -> Estimate:
        if prior is None:
            cfg = replace(self.config, coarse_levels=self.cold_levels)
            center, use_coarse = GDofState(0.0, 0.0), True
        elif self.warm_levels is not None:
            cfg = replace(self.config, coarse_levels=self.warm_levels)
            center, use_coarse = prior, True
        else:
            cfg = self.config
            center, use_coarse = prior, False
        e = estimate_pose_mask_space(
            inp.scene,
            inp.params,
            inp.mask,
            init=center,
            config=cfg,
            use_coarse=use_coarse,
            search_center=center,
            P_pred=P_prior,
            compute_covariance=True,
        )
        P = e.covariance.P_data if e.covariance is not None else None
        return Estimate(
            e.state,
            True,
            P=P,
            n_evals=e.n_coarse_evals,
            n_iters=e.n_refine_iters,
            extra={"dice": e.dice, "cost": e.cost},
        )


# ---------------------------------------------------------------- cornerfit


@dataclass
class CornerFit(Estimator):
    """LM on the 8 corner coordinates with the 4 known DOF fixed."""

    name: str = "cornerfit"

    def _estimate(self, inp, prior, P_prior) -> Estimate:
        x0 = GDofState(0.0, 0.0) if prior is None else prior
        obs = inp.ordered_corners(x0)
        if obs is None:
            return Estimate(None, False, extra={"fail": "no_quad"})

        def res(v):
            return (corners_dict_to_polygon(g_dof_corners(inp.scene, inp.params, _state(v))) - obs).ravel()

        x, it, r, J = _lm(res, _vec(x0), eps=(1e-3, 1e-2))
        # Re-label corners at the solution (cold start labels from a rough reference).
        obs2 = inp.ordered_corners(_state(x))
        if obs2 is not None and not np.allclose(obs2, obs):
            obs = obs2
            x, it2, r, J = _lm(res, x, eps=(1e-3, 1e-2))
            it += it2
        return Estimate(_state(x), True, P=_gauss_newton_cov(r, J), n_iters=it,
                        extra={"corner_rms_px": float(np.sqrt(np.mean(r**2))) if r is not None else np.nan})


# ------------------------------------------------------------------ edgefit


def _split_by_observed_axis(inp: FrameInput, state: GDofState) -> tuple[np.ndarray, np.ndarray]:
    """
    Left/right contour split robust to prior *offset* errors.

    Uses only the prior's projected centerline *direction* (insensitive to small
    yaw/lat errors) through the observed mask centroid; the original
    nearest-predicted-edge rule fails once the prior is off by more than half
    the runway width in pixels (long range).
    """
    pts = extract_lateral_contour_points(inp.mask)
    if pts.size == 0:
        z = np.zeros((0, 2))
        return z, z
    c = corners_dict_to_polygon(g_dof_corners(inp.scene, inp.params, state))  # TR, TL, BL, BR
    far_mid, near_mid = 0.5 * (c[0] + c[1]), 0.5 * (c[2] + c[3])
    d = far_mid - near_mid
    d /= max(np.linalg.norm(d), 1e-9)
    ys, xs = np.nonzero(inp.mask > 0.5)
    centroid = np.array([xs.mean(), ys.mean()])
    rel = pts - centroid
    side = d[0] * rel[:, 1] - d[1] * rel[:, 0]
    # Sign convention: the predicted left edge (TL–BL) must fall on the "left" side.
    left_ref = 0.5 * (c[1] + c[2]) - 0.5 * (far_mid + near_mid)
    left_sign = np.sign(d[0] * left_ref[1] - d[1] * left_ref[0]) or 1.0
    is_left = side * left_sign > 0
    return pts[is_left], pts[~is_left]


@dataclass
class EdgeFit(Estimator):
    """
    spec_edges.md joint Huber fit; cold start initialised by CornerFit.

    ``assign="prior"``: spec §4 rule (nearest edge projected at the prior).
    ``assign="axis"``: split by the observed centroid + prior centerline direction.
    """

    huber_delta_px: float = 3.0
    rounds: int = 2  # re-assign contour points to edges between rounds
    assign: str = "axis"
    name: str = "edgefit"

    def _estimate(self, inp, prior, P_prior) -> Estimate:
        n_extra = 0
        if prior is None:
            init = CornerFit()._estimate(inp, None, None)
            if not init.ok:
                return init
            x = _vec(init.state)
            n_extra = init.n_iters
        else:
            x = _vec(prior)
        it_total = 0
        r = J = None
        for _ in range(self.rounds):
            if self.assign == "prior":
                left, right = prepare_edge_observation(inp.mask, inp.scene, inp.params, _state(x))
            else:
                left, right = _split_by_observed_axis(inp, _state(x))
            if len(left) < 3 or len(right) < 3:
                return Estimate(None, False, extra={"fail": "few_edge_points"})

            def res(v, left=left, right=right):
                return edge_residuals(inp.scene, inp.params, _state(v), left, right,
                                      huber_delta_px=self.huber_delta_px)

            x, it, r, J = _lm(res, x, eps=(1e-3, 1e-2))
            it_total += it
        return Estimate(_state(x), True, P=_gauss_newton_cov(r, J), n_iters=it_total + n_extra)


# --------------------------------------------------------------- homography


@dataclass
class Homography(Estimator):
    """
    Plane readout warm-started at the prior (README ``runway_detection/homography``).

    ``iterations`` > 1 re-linearises the homography at the updated estimate
    (1 = the original single-shot sequential update). Cold start uses (0, 0).
    """

    iterations: int = 1
    name: str = "homography"

    def _estimate(self, inp, prior, P_prior) -> Estimate:
        x = GDofState(0.0, 0.0) if prior is None else prior
        obs = inp.ordered_corners(x)
        if obs is None:
            return Estimate(None, False, extra={"fail": "no_quad"})
        corners = corners_dict(obs)
        for _ in range(self.iterations):
            pose_prior = camera_pose_from_g_dof(inp.params, x)
            H = homography_from_pose(inp.scene, pose_prior)
            plane = measure_from_image_corners(corners, H)
            # The centerline is undirected: a near/far corner swap reads as ±180°.
            he_2d = (plane.heading_error_deg + 90.0) % 180.0 - 90.0
            plane = replace(plane, heading_error_deg=he_2d, centerline_angle_deg=he_2d)
            # Runway azimuth is 0 in the local frame → HE_ground = −yaw_cam (Z-Y-X Euler).
            he, lat = apply_prior_residual(
                -x.yaw_cam_deg, x.lateral_offset_m, plane, along_track_m=inp.params.along_track_m
            )
            x = GDofState(-he, lat)
        return Estimate(x, True, n_iters=self.iterations)


# ---------------------------------------------------------------------- pnp


@dataclass
class PnP(Estimator):
    """Full 6-DOF IPPE PnP on the 4 corners — classic baseline, ignores known DOF."""

    name: str = "pnp"

    def _estimate(self, inp, prior, P_prior) -> Estimate:
        ref = GDofState(0.0, 0.0) if prior is None else prior
        obs = inp.ordered_corners(ref)
        if obs is None:
            return Estimate(None, False, extra={"fail": "no_quad"})
        K = inp.scene.intrinsics.K
        obj = inp.scene.corner_points_local.astype(np.float64)
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            obj, obs.astype(np.float64), K, None, flags=cv2.SOLVEPNP_IPPE
        )
        if not n:
            return Estimate(None, False, extra={"fail": "pnp"})
        # OpenCV cam (x right, y down, z fwd) → LARD cam (x fwd, y left, z up)
        M = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)
        cands = []
        for rvec, tvec, err in zip(rvecs, tvecs, np.ravel(errs)):
            R_cv, _ = cv2.Rodrigues(rvec)
            C = (-R_cv.T @ tvec).ravel()  # camera center in runway frame
            fwd = (R_cv.T @ M)[:, 0]
            yaw_cam = -math.degrees(math.atan2(fwd[1], fwd[0]))
            # IPPE returns the planar mirror ambiguity: keep camera above the
            # runway and before the threshold, then lowest reprojection error.
            plausible = C[2] > 0 and C[0] < 0 and abs(yaw_cam) < 90
            cands.append((not plausible, float(err), yaw_cam, C))
        _, _, yaw_cam, C = min(cands, key=lambda c: (c[0], c[1]))
        return Estimate(
            GDofState(yaw_cam, float(C[1])),
            True,
            extra={"pnp_along_m": float(-C[0]), "pnp_height_m": float(C[2])},
        )
