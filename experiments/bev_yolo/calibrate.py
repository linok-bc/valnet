#!/usr/bin/env python
"""Fit the camera model from the collector's own logs, and check the sign conventions.

The BEV warp is only as good as the camera model behind it, and two things about
the xp12 camera are not written down anywhere: its focal length in pixels (we
have the X-Plane FOV preference, which may not be what was rendered) and its tilt
relative to the airframe. Both are recoverable, because every frame carries
enough ground truth to predict where the runway corners SHOULD land:

    runways.dat        runway width and length
    position/*.txt     where the aircraft is, in runway coordinates
    pose/*.txt         how it is oriented
    labels/*.txt       where the corners actually landed

so the camera parameters are whatever makes those agree. This script fits them
by least squares and prints the residual in pixels.

It also runs the sign conventions as a hypothesis test. Flipping the sign of
roll, pitch or yaw, or swapping the pose column order, gives a different model
that would silently produce a plausible-but-wrong BEV; only the right one fits
to a small residual. Run this FIRST on any regenerated dataset. A residual of a
few pixels means the geometry is trustworthy; tens of pixels means something in
the chain disagrees and no amount of training will fix it.

    python -m experiments.bev_yolo.calibrate --data /path/to/xp12_dataset
"""

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from experiments.bev.camera import CameraModel
from experiments.bev.episodes import find_episodes, load_runways, runway_to_ground
from experiments.bev.transform import ground_to_image

# (yaw, pitch, roll) multipliers. The collector documents +yaw = nose right,
# +pitch = nose up, +roll = right wing down; this is the hypothesis to confirm.
SIGN_HYPOTHESES = {
    "documented (+yaw +pitch +roll)": (1, 1, 1),
    "roll flipped": (1, 1, -1),
    "pitch flipped": (1, -1, 1),
    "yaw flipped": (-1, 1, 1),
    "roll+yaw flipped": (-1, 1, -1),
}


def label_corners(poly_norm: np.ndarray) -> np.ndarray:
    """Reduce a label polygon to 4 corners in near-left, far-left, far-right,
    near-right order.

    The current collector writes exactly those four. The older 42-point format
    walked the right edge outward (0..20) and the left edge back (21..41), so
    its corners live at known indices.
    """
    n = len(poly_norm)
    if n == 4:
        return poly_norm
    if n == 42:
        return poly_norm[[41, 21, 20, 0]]
    raise ValueError(f"cannot pick corners from a {n}-point label")


def collect_observations(root, runways, max_frames, stride, seed=0):
    """(observed corners px, predicted-input tuple) for a sample of labelled frames."""
    rng = np.random.default_rng(seed)
    episodes = find_episodes(root)
    if not episodes:
        raise SystemExit(f"no episodes under {root}")
    rows = []
    for ep in episodes:
        rw = runways.get((ep.airport, ep.runway_name))
        if rw is None:
            continue
        corners_rw = rw.corners_runway_frame()
        for fr in ep.frames(stride=stride):
            if not fr.labelled or fr.height_m <= 1.0:
                continue
            try:
                obs = label_corners(fr.polygon_norm)
            except ValueError:
                continue
            rows.append((obs, corners_rw, fr))
    if not rows:
        raise SystemExit("no usable labelled frames found")
    if len(rows) > max_frames:
        rows = [rows[i] for i in rng.choice(len(rows), max_frames, replace=False)]
    return rows


def residuals(params, rows, size, signs, fit_principal):
    """Pixel residuals of every corner of every sampled frame."""
    fx, fy, tilt = params[0], params[1], params[2]
    cx = params[3] if fit_principal else size[0] / 2
    cy = params[4] if fit_principal else size[1] / 2
    cam = CameraModel(size[0], size[1], fx, fy, cx, cy, tilt_deg=tilt)
    sy, sp, sr = signs

    out = []
    for obs_norm, corners_rw, fr in rows:
        ground = runway_to_ground(corners_rw, fr.along_m, fr.right_m, sy * fr.yaw_deg)
        H = ground_to_image(cam, sr * fr.roll_deg, sp * fr.pitch_deg, fr.height_m)
        p = np.column_stack([ground, np.ones(len(ground))]) @ H.T
        behind = p[:, 2] <= 1e-6
        uv = np.where(behind[:, None], 0.0, p[:, :2] / np.where(behind, 1.0, p[:, 2])[:, None])
        obs = obs_norm * np.array(size, dtype=np.float64)
        r = uv - obs
        r[behind] = 1e4                     # a corner behind the camera is a hard miss
        out.append(r.reshape(-1))
    return np.concatenate(out)


def fit(rows, size, signs, fit_principal, x0=None):
    fx0 = (size[0] / 2) / math.tan(math.radians(65.0) / 2)
    fy0 = (size[1] / 2) / math.tan(math.radians(39.43) / 2)
    p0 = list(x0 or [fx0, fy0, 0.0])
    if fit_principal:
        p0 += [size[0] / 2, size[1] / 2]
    sol = least_squares(residuals, p0, args=(rows, size, signs, fit_principal),
                        loss="soft_l1", f_scale=5.0, max_nfev=200)
    r = residuals(sol.x, rows, size, signs, fit_principal).reshape(-1, 2)
    err = np.linalg.norm(r, axis=1)
    return sol.x, {
        "median_px": float(np.median(err)),
        "mean_px": float(err.mean()),
        "p90_px": float(np.percentile(err, 90)),
        "max_px": float(err.max()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="root of the xp12 dataset")
    ap.add_argument("--runways", default=None, help="path to runways.dat")
    ap.add_argument("--image-size", type=int, nargs=2, default=[1280, 720])
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--fit-principal", action="store_true",
                    help="also fit the principal point (leave off unless the "
                         "residual stays high with it centred)")
    ap.add_argument("--out", default="experiments/bev_yolo/camera.json")
    ap.add_argument("--all-signs", action="store_true",
                    help="fit every sign hypothesis, not just the documented one")
    args = ap.parse_args()

    runways = load_runways(args.runways)
    rows = collect_observations(args.data, runways, args.max_frames, args.stride)
    size = tuple(args.image_size)
    print(f"{len(rows)} labelled frames sampled from {args.data}\n")

    hypotheses = SIGN_HYPOTHESES if args.all_signs else {
        "documented (+yaw +pitch +roll)": (1, 1, 1)}

    results = {}
    for name, signs in hypotheses.items():
        x, stats = fit(rows, size, signs, args.fit_principal)
        results[name] = (x, stats, signs)
        fovx = 2 * math.degrees(math.atan(size[0] / 2 / x[0]))
        fovy = 2 * math.degrees(math.atan(size[1] / 2 / x[1]))
        print(f"{name:32s} median {stats['median_px']:8.2f} px   p90 {stats['p90_px']:8.2f} px"
              f"   fx {x[0]:7.1f} fy {x[1]:7.1f} ({fovx:.2f}/{fovy:.2f} deg) tilt {x[2]:+.3f} deg")

    best = min(results, key=lambda k: results[k][1]["median_px"])
    x, stats, signs = results[best]
    print(f"\nbest: {best}")
    if stats["median_px"] > 10.0:
        print("  WARNING: median residual above 10 px. The labels, the logs and the\n"
              "  camera model do not agree -- do not build a BEV dataset on this.")
    elif args.all_signs and best != "documented (+yaw +pitch +roll)":
        print("  WARNING: a flipped sign fits better than the documented convention.\n"
              "  Check the collector before trusting anything downstream.")

    cam = dict(width=size[0], height=size[1], fx=float(x[0]), fy=float(x[1]),
               cx=float(x[3]) if args.fit_principal else size[0] / 2,
               cy=float(x[4]) if args.fit_principal else size[1] / 2,
               tilt_deg=float(x[2]))
    payload = dict(camera=cam, signs=dict(zip(("yaw", "pitch", "roll"), map(int, signs))),
                   residual_px=stats, n_frames=len(rows), data=str(args.data))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
