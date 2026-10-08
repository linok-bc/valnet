#!/usr/bin/env python
"""Data for the "how Samuel's method works" explainer page.

Uses Restart's own code for every number -- the `maskfit_analytic_lm` config,
build_context (ROI + 48x48 resample), the analytic renderer and the L2 cost --
on two real frames of held-out flight 000145 (far, ~1300 m; near, ~400 m):

    grid       cost over (yaw, cross-track) around his estimate
    patches    the 48x48 rendered patch at every grid cell (uint8, base64)
    observed   the 48x48 observed patch (YOLO soft mask, same ROI)
    quads      the rendered runway corners in the camera crop, per cell
    crop       the camera image under the ROI (JPEG, base64)

plus the per-frame EKF track of the whole flight from cache/heading.
Cross-track is reported + RIGHT (his code stores + left).

    python -m experiments.joint_demo.explain_samuel
"""

import base64
import json
import sys
from dataclasses import replace

import cv2
import numpy as np
import pandas as pd

from experiments.joint_demo import data

sys.path.insert(0, str(data.ROOT / "Restart"))
from evaluation.configs import _AN_LM  # noqa: E402
from evaluation.observations import YoloMaskCache  # noqa: E402
from pose_estimation.g_dof import (GDofParams, GDofState,  # noqa: E402
                                   corners_dict_to_polygon, g_dof_corners)
from pose_estimation.optimize import build_context  # noqa: E402
from runway_detection.xp12.calibration import (load_calibrations,  # noqa: E402
                                               pose_from_frame,
                                               scene_from_calibration)
from runway_detection.xp12.loader import load_sequence  # noqa: E402

EP = "000145"
TARGETS = {"far": 1300.0, "near": 400.0}
N = 25                                        # grid cells per axis
SPAN = {"far": (1.2, 12.0), "near": (1.2, 6.0)}   # half-range (deg, m) around the estimate


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def frame_payload(key, seq, calib, scene, heading, cache):
    row = heading.iloc[(heading.along_m - TARGETS[key]).abs().argmin()]
    f = next(x for x in seq.frames if x.index == int(row.frame))
    pose = pose_from_frame(f, calib)
    params = GDofParams.from_pose(pose, scene)
    observed = cache.get(EP, f.index)
    est = GDofState(float(row.yaw_meas), float(row.lat_meas))   # this frame's own fit
    ctx = build_context(scene, params, observed, config=_AN_LM, init=est)
    roi = ctx.roi
    n = ctx.ds_size

    hy, hl = SPAN[key]
    yaws = est.yaw_cam_deg + np.linspace(-hy, hy, N)
    # + LEFT in his convention; descending, so the displayed + RIGHT axis ascends
    lats = est.lateral_offset_m + np.linspace(hl, -hl, N)
    cost = np.zeros((N, N))
    patches = np.zeros((N, N, n, n), np.uint8)
    quads = np.zeros((N, N, 4, 2), np.float32)
    for i, y in enumerate(yaws):
        for j, l in enumerate(lats):
            s = GDofState(float(y), float(l))
            p = ctx.render_downsampled(s)
            cost[i, j] = ctx.cost(s)
            patches[i, j] = np.clip(np.round(p * 255), 0, 255)
            q = corners_dict_to_polygon(g_dof_corners(scene, params, s))
            quads[i, j] = q - [roi.x0, roi.y0]
    obs = np.clip(np.round(ctx.observed_downsampled() * 255), 0, 255).astype(np.uint8)

    img = seq.read_image(f)
    crop = img[roi.y0:roi.y1, roi.x0:roi.x1]
    scale = 480 / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    ok, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    truth_q = np.array([f.corners_px[c] for c in ("TR", "TL", "BL", "BR")]) - [roi.x0, roi.y0]

    # lat axis flipped to + right for display: index j -> -lats[j]
    return {
        "frame": int(f.index), "range_m": float(row.along_m), "height_m": float(pose.height_m),
        "roi": {"x0": roi.x0, "y0": roi.y0, "w": roi.width, "h": roi.height}, "n": n,
        "crop_scale": scale, "crop_jpg": base64.b64encode(jpg.tobytes()).decode(),
        "yaws": yaws.tolist(), "rights": (-lats).tolist(),
        "cost": cost.tolist(), "patches": b64(patches), "observed": b64(obs),
        "quads": quads.round(2).tolist(), "truth_quad": truth_q.round(2).tolist(),
        "est": {"yaw": est.yaw_cam_deg, "right": -est.lateral_offset_m},
        "truth": {"yaw": float(row.yaw_gt), "right": -float(row.lat_gt)},
        "known": {"pitch_deg": float(pose.pitch_cam_deg), "roll_deg": float(pose.roll_cam_deg),
                  "height_m": float(pose.height_m), "along_m": float(pose.along_track_m),
                  "width_m": float(calib.width_m)},
    }


def main():
    calib = load_calibrations(data.CALIB)[EP]
    seq = load_sequence(data.ROOT / "datasets/xp12_dataset", EP)
    scene = scene_from_calibration(calib)
    heading = pd.read_csv(data.HERE / "cache/heading" / f"{EP}.csv", dtype={"seq": str})
    cache = YoloMaskCache(data.HERE / "cache/masks")
    out = {"flight": EP, "site": " ".join(data.meta(EP)), "grid_n": N,
           "frames": {k: frame_payload(k, seq, calib, scene, heading, cache) for k in TARGETS}}
    h = heading.sort_values("along_m", ascending=False)
    out["track"] = {
        "range_m": h.along_m.round(1).tolist(),
        "meas_right": (-h.lat_meas).round(3).tolist(), "ekf_right": (-h.lat_est).round(3).tolist(),
        "truth_right": (-h.lat_gt).round(3).tolist(),
        "meas_yaw": h.yaw_meas.round(4).tolist(), "ekf_yaw": h.yaw_est.round(4).tolist(),
        "truth_yaw": h.yaw_gt.round(4).tolist(), "sigma_right": h.sigma_lat.round(3).tolist(),
        "status": h.status.tolist(),
    }
    path = data.HERE / "report/samuel_explainer.json"
    path.write_text(json.dumps(out))
    for k, fr in out["frames"].items():
        c = np.array(fr["cost"])
        print(k, "frame", fr["frame"], "range %.0f" % fr["range_m"], "roi", fr["roi"],
              "cost min/max %.4f/%.4f" % (c.min(), c.max()),
              "est", fr["est"], "truth", fr["truth"])
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
