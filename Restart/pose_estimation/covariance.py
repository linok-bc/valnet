"""
Posterior covariance for (yaw, lateral_offset) from mask Fisher information.

See Restart/spec_covariance.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

from pose_estimation.g_dof import GDofState

if TYPE_CHECKING:
    from pose_estimation.optimize import OptimizerConfig, _EvalContext

CHI2_95_2DOF = 5.991  # scipy.stats.chi2.ppf(0.95, df=2)


@dataclass(frozen=True)
class PoseCovariance:
    """Covariance of ``[yaw_cam_deg, lateral_offset_m]``."""

    P_data: np.ndarray  # (2, 2) data-only inverse Fisher
    P_posterior: np.ndarray  # (2, 2) fused with temporal prior
    I_data: np.ndarray
    I_total: np.ndarray
    eigvals: np.ndarray  # ascending
    eigvecs: np.ndarray  # columns, same order as eigvals
    measurement_sigma: float
    n_residuals: int

    @property
    def sigma_min(self) -> float:
        return float(math.sqrt(max(self.eigvals[0], 0.0)))

    @property
    def sigma_max(self) -> float:
        return float(math.sqrt(max(self.eigvals[1], 0.0)))

    @property
    def uncertain_direction(self) -> np.ndarray:
        """Unit eigenvector along the larger stddev (ill-conditioned axis)."""
        return self.eigvecs[:, 1].copy()


def default_prior_covariance(
    *,
    yaw_std_deg: float = 2.0,
    lat_std_m: float = 10.0,
) -> np.ndarray:
    """Diagonal ``P_pred`` for the first frame or cold start."""
    return np.diag([yaw_std_deg**2, lat_std_m**2]).astype(float)


def process_noise(
    *,
    yaw_std_deg: float = 0.1,
    lat_std_m: float = 0.5,
) -> np.ndarray:
    """Simple diagonal motion noise between frames."""
    return np.diag([yaw_std_deg**2, lat_std_m**2]).astype(float)


def _state_vec(state: GDofState) -> np.ndarray:
    return np.array([state.yaw_cam_deg, state.lateral_offset_m], dtype=float)


def P_to_plot_coords(P: np.ndarray) -> np.ndarray:
    """Map covariance from [yaw, lat] to plot axes [lat, yaw]."""
    s = np.array([[0.0, 1.0], [1.0, 0.0]])
    return s @ P @ s.T


def fisher_information_data(
    H: np.ndarray,
    *,
    sigma: float,
    residual_mean: bool = True,
) -> np.ndarray:
    """
    ``I_data = Hᵀ R⁻¹ H`` for diagonal pixel noise ``R = σ² I``.

    ``H`` is ∂r/∂x for mask residuals. When the cost is ``mean(r²)``, the
    Gauss–Newton Hessian carries a ``2/n`` factor.
    """
    n = H.shape[0]
    if n == 0 or sigma <= 0:
        return np.zeros((2, 2), dtype=float)
    scale = 2.0 / n if residual_mean else 1.0
    return scale * (H.T @ H) / (sigma**2)


def fuse_information(
    I_data: np.ndarray,
    P_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``I_total = I_data + P_pred⁻¹``, ``P_t = I_total⁻¹``."""
    I_pred = np.linalg.inv(P_pred)
    I_total = I_data + I_pred
    P_post = np.linalg.inv(I_total)
    return P_post, I_total


def numerical_cost_hessian(
    cost_fn: Callable[[np.ndarray], float],
    x: np.ndarray,
    *,
    eps: float = 1e-3,
) -> np.ndarray:
    """Central finite-difference Hessian of a scalar cost."""
    Hn = np.zeros((2, 2), dtype=float)
    for i in range(2):
        for j in range(2):
            ei = np.eye(2)[i] * eps
            ej = np.eye(2)[j] * eps
            Hn[i, j] = (
                cost_fn(x + ei + ej)
                - cost_fn(x + ei - ej)
                - cost_fn(x - ei + ej)
                + cost_fn(x - ei - ej)
            ) / (4.0 * eps**2)
    return Hn


def compute_pose_covariance(
    ctx: _EvalContext,
    state: GDofState,
    *,
    P_pred: np.ndarray | None = None,
    config: OptimizerConfig,
    measurement_sigma: float | None = None,
    data_only: bool = False,
) -> PoseCovariance:
    """
    Data Fisher information + temporal prior fusion at a pose estimate.

    Returns ``P_posterior`` for downstream fusion and eigen-decomposition for
    diagnostics (valley-aligned uncertainty).
    """
    from pose_estimation.optimize import _jacobian_fd

    sigma = measurement_sigma
    if sigma is None:
        sigma = config.measurement_sigma

    H = _jacobian_fd(
        ctx,
        state,
        eps_yaw_deg=config.fd_eps_yaw_deg,
        eps_lateral_m=config.fd_eps_lateral_m,
    )
    I_data = fisher_information_data(H, sigma=sigma, residual_mean=True)
    P_data = _safe_inv(I_data)
    if data_only:
        P_post = P_data
        I_total = I_data
    else:
        prior = P_pred if P_pred is not None else default_prior_covariance()
        P_post, I_total = fuse_information(I_data, prior)
    eigvals, eigvecs = np.linalg.eigh(P_post)

    return PoseCovariance(
        P_data=P_data,
        P_posterior=P_post,
        I_data=I_data,
        I_total=I_total,
        eigvals=eigvals,
        eigvecs=eigvecs,
        measurement_sigma=sigma,
        n_residuals=int(H.shape[0]),
    )


def laplace_check(
    ctx: _EvalContext,
    state: GDofState,
    P_fused: np.ndarray,
    *,
    eps: float = 1e-3,
) -> tuple[np.ndarray, float]:
    """
    Numerical Hessian cross-check (spec §5).

    Returns ``(P_hessian, relative_frobenius_error)`` vs ``P_fused``.
    """
    x = _state_vec(state)

    def cost_fn(v: np.ndarray) -> float:
        return ctx.cost(GDofState(float(v[0]), float(v[1])))

    Hn = numerical_cost_hessian(cost_fn, x, eps=eps)
    P_check = _safe_inv(0.5 * Hn)
    denom = np.linalg.norm(P_fused) + 1e-12
    rel_err = float(np.linalg.norm(P_check - P_fused) / denom)
    return P_check, rel_err


def _safe_inv(M: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M)


def confidence_ellipse_xy(
    state: GDofState,
    P: np.ndarray,
    *,
    chi2: float = CHI2_95_2DOF,
    n_points: int = 72,
) -> tuple[np.ndarray, np.ndarray]:
    """
    95% confidence ellipse in plot coordinates (lateral, yaw).

    ``P`` is expressed in ``[yaw_deg, lateral_m]``; returns ``(lateral, yaw)`` arrays.
    """
    center = np.array([state.lateral_offset_m, state.yaw_cam_deg], dtype=float)
    P_plot = P_to_plot_coords(P)
    eigvals, eigvecs = np.linalg.eigh(P_plot)
    eigvals = np.maximum(eigvals, 0.0)
    axes = np.sqrt(chi2 * eigvals)
    t = np.linspace(0.0, 2.0 * math.pi, n_points, endpoint=True)
    circle = np.vstack([np.cos(t), np.sin(t)])
    pts = center[:, None] + eigvecs @ (axes[:, None] * circle)
    return pts[0], pts[1]
