#!/usr/bin/env python
"""Fuse our edge-line tilt cross-track into Samuel's kinematic EKF.

Samuel's filter (Restart/evaluation/runner.py, KinematicEKF) tracks
s = [yaw, yaw_rate, lat] and takes his mask fit's z = [yaw, lat] each frame.
Fusion adds a second, scalar measurement of lat per frame:

    z_tilt = -free_width            (Restart's lat is + left, ours is + right)
    R_tilt = sigma(range)^2         (noise_model.py, fitted on split_val)

applied after his update, with its own NIS gate (chi2_1 99.9% = 10.83). Because
the kinematic model couples lat to yaw, the tilt update also corrects yaw
through the covariance. The loop is run_sequence's `track + kf` path copied
unchanged apart from that update, so his estimator is still warm-started from
the (now fused) prediction -- closed loop, as it would run in flight.

Scored against the shared truth (the rotation-fixed logs, via data.STOCK)
and Samuel's own truth. `samuel` must reproduce cache/heading exactly.

    python -m experiments.joint_demo.fuse
"""

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from experiments.joint_demo import data

sys.path.insert(0, str(data.ROOT / "Restart"))
from evaluation.configs import make_method  # noqa: E402
from evaluation.estimators import FrameInput  # noqa: E402
from evaluation.observations import YoloMaskCache, observed_mask  # noqa: E402
from evaluation.runner import KinematicEKF, RunConfig, _as_vec  # noqa: E402
from pose_estimation.g_dof import GDofParams, GDofState  # noqa: E402
from runway_detection.xp12.calibration import (load_calibrations,  # noqa: E402
                                               pose_from_frame,
                                               scene_from_calibration)
from runway_detection.xp12.loader import load_sequence  # noqa: E402

CHI2_1DOF_999 = 10.828


def tilt_sigma(range_m, model):
    return float(np.interp(range_m, model["range_m"], model["rms_m"]))


def tilt_update(ekf, z, R):
    """Scalar update of lat; returns (nis, accepted)."""
    H = np.array([0.0, 0.0, 1.0])
    S = float(H @ ekf.P @ H) + R
    innov = z - ekf.s[2]
    nis = innov * innov / S
    if nis > CHI2_1DOF_999:
        return nis, False
    K = ekf.P @ H / S
    ekf.s = ekf.s + K * innov
    ekf.P = (np.eye(3) - np.outer(K, H)) @ ekf.P
    return nis, True


def run(ep, fuse, model):
    calib = load_calibrations(data.CALIB)[ep]
    seq = load_sequence(data.ROOT / "datasets/xp12_dataset", ep)
    cache = YoloMaskCache(data.HERE / "cache/masks")
    estimator = make_method("maskfit_analytic_lm")
    cfg = RunConfig(source="yolo", mode="track", kf=True)
    scene = scene_from_calibration(calib)
    df, _, _ = data.load(ep)
    tilt = dict(zip(df.frame, -df.xt_est))
    rng_m = dict(zip(df.frame, df.range_m))

    wanted = set(cache.available_indices(ep))
    frames = [f for f in seq.frames if f.index in wanted]
    ekf = KinematicEKF(cfg)
    prev_index = prev_along = None
    rows = []
    for f in frames:
        pose = pose_from_frame(f, calib)
        params = GDofParams.from_pose(pose, scene)
        inp = FrameInput(observed_mask("yolo", f, yolo=cache), scene, params)
        dk = float(f.index - prev_index) if prev_index is not None else 1.0
        d_along = params.along_track_m - prev_along if prev_along is not None else 0.0
        prev_index, prev_along = f.index, params.along_track_m

        prior, P_prior = None, np.diag(cfg.init_P_diag)
        if ekf.s is not None:
            ekf.predict(dk, d_along)
            prior, P_prior = GDofState(*map(float, ekf.z_pred)), ekf.P_z
        est = estimator.estimate(inp, prior, P_prior)
        if est.state is None and prior is not None:
            est = estimator.estimate(inp, None, None)
        z = _as_vec(est)
        if ekf.s is None:
            if z is not None:
                ekf.init(z)
        elif z is not None:
            _, accepted = ekf.update(z)
            if not accepted and ekf.n_reject >= cfg.reacq_after:
                cold = estimator.estimate(inp, None, None)
                if cold.state is not None:
                    ekf.init(_as_vec(cold))

        tilt_ok = math.nan
        zt = tilt.get(f.index, math.nan)
        if fuse and ekf.s is not None and np.isfinite(zt):
            _, tilt_ok = tilt_update(ekf, zt, tilt_sigma(rng_m[f.index], model) ** 2)
        x = ekf.z_pred if ekf.s is not None else np.array([math.nan, math.nan])
        rows.append({"frame": f.index, "yaw_gt": pose.yaw_cam_deg, "lat_gt": pose.lateral_offset_m,
                     "yaw": x[0], "lat": x[1], "tilt_accepted": tilt_ok})
    return pd.DataFrame(rows)


def one(job):
    ep, model = job
    s = run(ep, False, model).add_prefix("samuel_").rename(columns={"samuel_frame": "frame"})
    f = run(ep, True, model).add_prefix("fused_").rename(columns={"fused_frame": "frame"})
    out = s.merge(f, on="frame")
    out.to_csv(data.HERE / "cache/fusion" / f"{ep}.csv", index=False)
    return ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="*")
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()
    model = json.loads((data.HERE / "cache/tilt_noise.json").read_text())
    (data.HERE / "cache/fusion").mkdir(parents=True, exist_ok=True)
    eps = args.episodes or data.episodes()
    with ProcessPoolExecutor(args.workers) as ex:
        for ep in ex.map(one, [(e, model) for e in eps]):
            print(ep, flush=True)


if __name__ == "__main__":
    main()
