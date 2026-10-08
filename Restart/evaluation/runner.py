"""
Run one estimator over one XP12 sequence and log per-frame errors vs GT.

Modes
- ``cold``  : every frame is estimated from scratch (no temporal prior).
- ``track`` : frame t is warm-started from the estimate at t−1 (spec §4.4, as in
  ``scripts/validate_mask_fit_sequence.py``). If the estimator fails with a
  prior, the frame is retried cold (re-acquisition).
- ``track`` + ``kf=True``: spec §4.5–4.6 EKF on ``s = [yaw, yaw_rate, lat]``:

      yaw'  = yaw + yaw_rate·Δk
      lat'  = lat + Δalong · tan(yaw_mid)      (kinematics, Δalong from the known DOF)

  The *predicted* state warm-starts the estimator; its output is fused with a
  fixed measurement noise R, gated by NIS (χ²₂ 99.9 %), and after
  ``reacq_after`` consecutive rejections the filter is re-initialised cold.

Perturbations (robustness ablations)
- ``dof_noise``: per-frame Gaussian noise on the 4 "known" DOF fed to the
  estimator (``pitch_deg``, ``roll_deg``, ``height_m``, ``along_m``).
- ``width_scale``: runway width prior error (1.05 = DB says 5 % wider).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from evaluation.estimators import Estimate, Estimator, FrameInput
from evaluation.observations import YoloMaskCache, gt_mask, mask_iou, observed_mask
from pose_estimation.g_dof import GDofParams, GDofState
from runway_detection.xp12.calibration import XP12Calibration, pose_from_frame, scene_from_calibration
from runway_detection.xp12.loader import XP12Sequence

CHI2_2DOF_999 = 13.816
_D2R = math.pi / 180.0


@dataclass
class RunConfig:
    source: str = "gt"  # gt | corrupt | yolo
    mode: str = "track"  # cold | track
    stride: int = 1
    kf: bool = False
    kf_R_diag: tuple[float, float] = (0.15**2, 2.0**2)  # measurement noise (deg², m²)
    # process noise per source frame on (yaw, yaw_rate, lat)
    kf_Q_diag: tuple[float, float, float] = (0.02**2, 0.01**2, 0.1**2)
    nis_gate: float = CHI2_2DOF_999
    reacq_after: int = 3
    init_P_diag: tuple[float, float] = (2.0**2, 10.0**2)
    dof_noise: dict = field(default_factory=dict)
    width_scale: float = 1.0
    seed: int = 0
    max_frames: int | None = None


def _noisy_params(pose, scene, noise: dict, rng: np.random.Generator) -> GDofParams:
    p = GDofParams.from_pose(pose, scene)
    if not noise:
        return p
    return GDofParams(
        along_track_m=p.along_track_m + rng.normal(0, noise.get("along_m", 0.0)),
        height_m=p.height_m + rng.normal(0, noise.get("height_m", 0.0)),
        pitch_cam_deg=p.pitch_cam_deg + rng.normal(0, noise.get("pitch_deg", 0.0)),
        roll_cam_deg=p.roll_cam_deg + rng.normal(0, noise.get("roll_deg", 0.0)),
        intrinsics=p.intrinsics,
        runway_azimuth_deg=p.runway_azimuth_deg,
    )


class KinematicEKF:
    """EKF on s = [yaw (deg), yaw_rate (deg/frame), lat (m)] with z = [yaw, lat]."""

    H = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.s: np.ndarray | None = None
        self.P = np.zeros((3, 3))
        self.R = np.diag(cfg.kf_R_diag)
        self.n_reject = 0

    def init(self, z: np.ndarray) -> None:
        self.s = np.array([z[0], 0.0, z[1]])
        self.P = np.diag([self.R[0, 0], 0.1**2, self.R[1, 1]])
        self.n_reject = 0

    def predict(self, dk: float, d_along: float) -> None:
        yaw, rate, lat = self.s
        yaw_mid = (yaw + 0.5 * rate * dk) * _D2R
        sec2 = 1.0 / math.cos(yaw_mid) ** 2
        self.s = np.array([yaw + rate * dk, rate, lat + d_along * math.tan(yaw_mid)])
        F = np.array(
            [[1.0, dk, 0.0], [0.0, 1.0, 0.0], [d_along * sec2 * _D2R, d_along * sec2 * _D2R * 0.5 * dk, 1.0]]
        )
        self.P = F @ self.P @ F.T + np.diag(self.cfg.kf_Q_diag) * dk

    @property
    def z_pred(self) -> np.ndarray:
        return self.H @ self.s

    @property
    def P_z(self) -> np.ndarray:
        return self.H @ self.P @ self.H.T

    def update(self, z: np.ndarray) -> tuple[float, bool]:
        S = self.P_z + self.R
        innov = z - self.z_pred
        nis = float(innov @ np.linalg.solve(S, innov))
        if nis > self.cfg.nis_gate:
            self.n_reject += 1
            return nis, False
        self.n_reject = 0
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.s = self.s + K @ innov
        self.P = (np.eye(3) - K @ self.H) @ self.P
        return nis, True


def _as_vec(est: Estimate) -> np.ndarray | None:
    return None if est.state is None else np.array([est.state.yaw_cam_deg, est.state.lateral_offset_m])


def run_sequence(
    seq: XP12Sequence,
    calib: XP12Calibration,
    estimator: Estimator,
    cfg: RunConfig,
    *,
    yolo: YoloMaskCache | None = None,
    frame_indices: list[int] | None = None,
    log_seg_iou: bool = True,
) -> list[dict]:
    rng = np.random.default_rng(cfg.seed + int(seq.seq_id))
    est_scene = scene_from_calibration(calib, width_scale=cfg.width_scale)

    frames = seq.frames
    if frame_indices is not None:
        wanted = set(frame_indices)
        frames = [f for f in frames if f.index in wanted]
    else:
        frames = frames[:: cfg.stride]
    if cfg.max_frames:
        frames = frames[: cfg.max_frames]

    x: np.ndarray | None = None  # plain tracking state [yaw, lat]
    P: np.ndarray | None = None
    ekf = KinematicEKF(cfg) if cfg.kf else None
    prev_index: int | None = None
    prev_along: float | None = None
    rows: list[dict] = []

    for f in frames:
        pose = pose_from_frame(f, calib)
        gt = np.array([pose.yaw_cam_deg, pose.lateral_offset_m])
        params = _noisy_params(pose, est_scene, cfg.dof_noise, rng)
        mask = observed_mask(cfg.source, f, yolo=yolo, rng=rng)
        if cfg.source == "yolo" and mask is None:
            continue  # frame not in the YOLO cache
        inp = FrameInput(mask, est_scene, params)
        dk = float(f.index - prev_index) if prev_index is not None else 1.0
        d_along = params.along_track_m - prev_along if prev_along is not None else 0.0
        prev_index, prev_along = f.index, params.along_track_m

        # ---- prior
        prior, P_prior = None, np.diag(cfg.init_P_diag)
        if cfg.mode == "track":
            if ekf is not None and ekf.s is not None:
                ekf.predict(dk, d_along)
                prior, P_prior = GDofState(*map(float, ekf.z_pred)), ekf.P_z
            elif ekf is None and x is not None:
                prior, P_prior = GDofState(float(x[0]), float(x[1])), P

        # ---- measurement (retry cold if a warm-started estimate fails)
        est = estimator.estimate(inp, prior, P_prior)
        status = "ok"
        if est.state is None and prior is not None:
            cold = estimator.estimate(inp, None, None)
            cold.runtime_ms += est.runtime_ms
            est = cold
            status = "reacquired"
        z = _as_vec(est)
        if z is None:
            status = "fail"

        # ---- state update
        nis = math.nan
        if cfg.mode == "cold":
            x_out, P_out = z, est.P
        elif ekf is None:
            if z is not None:
                x, P = z, est.P
            x_out, P_out = x, P
        else:
            if ekf.s is None:
                if z is not None:
                    ekf.init(z)
            elif z is not None:
                nis, accepted = ekf.update(z)
                if not accepted:
                    status = "rejected"
                    if ekf.n_reject >= cfg.reacq_after:
                        cold = estimator.estimate(inp, None, None)
                        est.runtime_ms += cold.runtime_ms
                        if cold.state is not None:
                            ekf.init(_as_vec(cold))
                            status = "reacquired"
            if ekf.s is not None:
                x_out, P_out = ekf.z_pred, ekf.P_z
            else:
                x_out, P_out = None, None

        err = (x_out - gt) if x_out is not None else np.array([math.nan, math.nan])
        nees = math.nan
        if x_out is not None and P_out is not None and np.all(np.isfinite(P_out)):
            try:
                nees = float(err @ np.linalg.solve(P_out, err))
            except np.linalg.LinAlgError:
                pass
        have_P = x_out is not None and P_out is not None and P_out[0, 0] > 0 and P_out[1, 1] > 0
        row = {
            "seq": seq.seq_id,
            "airport": seq.airport,
            "runway": seq.runway,
            "frame": f.index,
            "along_m": pose.along_track_m,
            "height_m": pose.height_m,
            "roll_deg": pose.roll_cam_deg,
            "yaw_gt": gt[0],
            "lat_gt": gt[1],
            "yaw_est": x_out[0] if x_out is not None else math.nan,
            "lat_est": x_out[1] if x_out is not None else math.nan,
            "yaw_meas": z[0] if z is not None else math.nan,
            "lat_meas": z[1] if z is not None else math.nan,
            "yaw_err": err[0],
            "lat_err": err[1],
            "sigma_yaw": math.sqrt(P_out[0, 0]) if have_P else math.nan,
            "sigma_lat": math.sqrt(P_out[1, 1]) if have_P else math.nan,
            "rho": P_out[0, 1] / math.sqrt(P_out[0, 0] * P_out[1, 1]) if have_P else math.nan,
            "nees": nees,
            "nis": nis,
            "status": status,
            "runtime_ms": est.runtime_ms,
            "n_evals": est.n_evals,
            "n_iters": est.n_iters,
            "fail": est.extra.get("fail", ""),
        }
        if log_seg_iou and cfg.source != "gt" and mask is not None:
            row["seg_iou"] = mask_iou(mask, gt_mask(f))
        rows.append(row)
    return rows
