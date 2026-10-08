"""
Per-sequence runway geometry for XP12 approaches, fitted on GT corner labels.

The dataset gives no runway database, so width / length and small frame offsets
are fitted once per sequence by Levenberg–Marquardt on the labelled corners,
with the camera model fixed (HFOV 65°) and the sim pose taken as truth. This
plays the role of the runway database + instrument calibration that a real
system would have; the fitted width is the "width prior" used by estimators.

Parameters ``θ = (W, L, d0, h0, c0, ψ0)``:

- ``W``, ``L``: runway width / length (m)
- ``d0``: along-track origin offset (m)
- ``h0``: height offset — height = y + h0
- ``c0``: centerline offset (m, +left)
- ``ψ0``: runway axis azimuth w.r.t. the sim x axis (deg, + = runway turns left)

The sim frame is (x fwd, y up, z right). The runway frame used by ``g_dof`` is
the sim frame translated by (d0, c0, h0) and rotated by ψ0 about the vertical,
so GT heading / lateral are expressed w.r.t. the *true* runway axis. Only
``d0`` and ``ψ0`` are fitted by default: ``h0`` / ``c0`` trade off against
``d0`` (ill-conditioned) for negligible RMS gain.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from pose_estimation.runway_model import CameraPoseLocal, RunwayScene
from runway_detection.xp12.loader import XP12Frame, XP12Sequence, xp12_intrinsics


@dataclass(frozen=True)
class XP12Calibration:
    seq_id: str
    airport: str
    runway: str
    width_m: float
    length_m: float
    d0_m: float
    h0_m: float
    c0_m: float
    psi0_deg: float
    rms_px: float  # GT-corner reprojection RMS after fit (oracle floor)
    p95_px: float
    n_frames: int


def _rot_world_from_cam(heading_deg, pitch_deg, roll_deg):
    """Vectorized LARD intrinsic Z→Y'→X'' rotation (runway azimuth = 0)."""
    y = np.radians(-np.asarray(heading_deg))
    p = np.radians(-np.asarray(pitch_deg))
    r = np.radians(np.asarray(roll_deg))
    cy, sy, cp, sp, cr, sr = np.cos(y), np.sin(y), np.cos(p), np.sin(p), np.cos(r), np.sin(r)
    n = y.shape[0]
    R = np.empty((n, 3, 3))
    R[:, 0, 0] = cy * cp
    R[:, 0, 1] = cy * sp * sr - sy * cr
    R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp
    R[:, 1, 1] = sy * sp * sr + cy * cr
    R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr
    return R


def corner_points_local(width_m: float, length_m: float) -> np.ndarray:
    """TR, TL, BL, BR in the runway frame (X along, Y left, Z up)."""
    hw = 0.5 * width_m
    return np.array(
        [[length_m, -hw, 0.0], [length_m, hw, 0.0], [0.0, hw, 0.0], [0.0, -hw, 0.0]],
        dtype=float,
    )


def sim_to_runway(
    theta: np.ndarray,
    pose: np.ndarray,
    position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sim pose/position (N,3) → runway-frame camera position (N,3: X along, Y left, Z up)
    and heading relative to the runway axis (N,).
    """
    _, _, d0, h0, c0, psi0 = theta
    c, s = math.cos(math.radians(psi0)), math.sin(math.radians(psi0))
    x = position[:, 0] + d0
    y = -position[:, 2] - c0
    # rotate by -ψ0 so the runway axis becomes +X
    cam = np.stack([c * x + s * y, -s * x + c * y, position[:, 1] + h0], axis=1)
    return cam, pose[:, 0] + psi0


def project_batch(
    theta: np.ndarray,
    pose: np.ndarray,
    position: np.ndarray,
) -> np.ndarray:
    """Project the 4 corners for N frames → (N, 4, 2) pixels (TR, TL, BL, BR)."""
    W, L = theta[0], theta[1]
    K = xp12_intrinsics()
    cam, heading = sim_to_runway(theta, pose, position)
    R = _rot_world_from_cam(heading, pose[:, 1], pose[:, 2])
    pts = corner_points_local(W, L)[None] - cam[:, None, :]  # (N,4,3)
    pc = np.einsum("nkj,nji->nki", pts, R)  # R^T · p  (camera x fwd, y left, z up)
    u = K.cx - K.fx * pc[..., 1] / pc[..., 0]
    v = K.cy - K.fy * pc[..., 2] / pc[..., 0]
    return np.stack([u, v], axis=-1)


def fit_calibration(
    seq: XP12Sequence,
    *,
    fit_offsets: tuple[str, ...] = ("d0", "psi0"),
    max_iters: int = 100,
) -> XP12Calibration:
    arr = seq.arrays
    pose, position, obs = arr["pose"], arr["position"], arr["corners"]
    names = ("W", "L", "d0", "h0", "c0", "psi0")
    free = [0, 1] + [names.index(n) for n in fit_offsets]
    theta = np.array([45.0, 2500.0, 0.0, 0.0, 0.0, 0.0])

    def residuals(t: np.ndarray) -> np.ndarray:
        return (project_batch(t, pose, position) - obs).ravel()

    r = residuals(theta)
    cost = r @ r
    lam = 1e-3
    for _ in range(max_iters):
        J = np.zeros((r.size, len(free)))
        for k, idx in enumerate(free):
            eps = 1e-4 * max(1.0, abs(theta[idx]))
            tp = theta.copy()
            tp[idx] += eps
            J[:, k] = (residuals(tp) - r) / eps
        A = J.T @ J
        g = J.T @ r
        improved = False
        while lam < 1e10:
            step = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), g)
            cand = theta.copy()
            cand[free] += step
            rn = residuals(cand)
            cn = rn @ rn
            if cn < cost:
                rel = (cost - cn) / max(cost, 1e-12)
                theta, r, cost = cand, rn, cn
                lam = max(lam / 3.0, 1e-12)
                improved = True
                break
            lam *= 4.0
        if not improved or rel < 1e-10:
            break

    err = np.linalg.norm(r.reshape(-1, 2), axis=1)
    return XP12Calibration(
        seq_id=seq.seq_id,
        airport=seq.airport,
        runway=seq.runway,
        width_m=float(theta[0]),
        length_m=float(theta[1]),
        d0_m=float(theta[2]),
        h0_m=float(theta[3]),
        c0_m=float(theta[4]),
        psi0_deg=float(theta[5]),
        rms_px=float(math.sqrt(np.mean(err**2))),
        p95_px=float(np.percentile(err, 95)),
        n_frames=len(seq),
    )


def save_calibrations(calibs: dict[str, XP12Calibration], path: str | Path) -> None:
    Path(path).write_text(json.dumps({k: asdict(v) for k, v in calibs.items()}, indent=1))


def load_calibrations(path: str | Path) -> dict[str, XP12Calibration]:
    raw = json.loads(Path(path).read_text())
    return {k: XP12Calibration(**v) for k, v in raw.items()}


def scene_from_calibration(
    calib: XP12Calibration,
    *,
    width_scale: float = 1.0,
) -> RunwayScene:
    """``RunwayScene`` in a local frame with azimuth 0 (``width_scale`` ≠ 1 simulates a wrong DB width)."""
    return RunwayScene(
        corner_points_local=corner_points_local(calib.width_m * width_scale, calib.length_m),
        runway_azimuth_deg=0.0,
        origin_geodetic=(0.0, 0.0, 0.0),
        intrinsics=xp12_intrinsics(),
        airport=calib.airport,
        runway_id=calib.runway,
    )


def calibration_theta(calib: XP12Calibration) -> np.ndarray:
    return np.array(
        [calib.width_m, calib.length_m, calib.d0_m, calib.h0_m, calib.c0_m, calib.psi0_deg]
    )


def pose_from_frame(frame: XP12Frame, calib: XP12Calibration) -> CameraPoseLocal:
    """GT camera pose in the runway-aligned local frame used by ``g_dof``."""
    cam, heading = sim_to_runway(
        calibration_theta(calib),
        np.array([[frame.heading_rel_deg, frame.pitch_deg, frame.roll_deg]]),
        np.array([[frame.x_m, frame.y_m, frame.z_m]]),
    )
    return CameraPoseLocal(
        along_track_m=float(-cam[0, 0]),
        lateral_offset_m=float(cam[0, 1]),
        height_m=float(cam[0, 2]),
        yaw_cam_deg=float(heading[0]),
        pitch_cam_deg=90.0 + frame.pitch_deg,
        roll_cam_deg=frame.roll_deg,
    )
