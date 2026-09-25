#!/usr/bin/env python
"""Does the logged cross-track error follow `range x angle` on EVERY episode?

check_rotation.py showed, on three episodes, that the logged `right` is
measured against a runway direction rotated a few tenths of a degree from the
rendered runway, pivoting at the threshold. If that is the whole story, then on
every episode

    error(range) = estimate - logged = angle * range + ~0

that is: a straight line in range, through ~zero at the threshold, with a
per-episode slope. This fits that line per episode against two independent
estimates of the true cross-track:

    label   the edge-line estimator on the label corners (every frame)
    image   the same estimator on painted edges found from image gradients
            alone -- no label involved (every `--image-stride`th frame)

and reports, per episode: the angle, the intercept (the error at the threshold)
and the residual left once the line is removed. Three things must hold for the
claim to stand: small residuals, intercepts near zero, and label and image
agreeing on the angle.

    python -m experiments.edge_lines.verify_rotation
"""

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.bev.camera import CameraModel
from experiments.bev.episodes import Episode, find_episodes, load_runways
from experiments.edge_lines.check_rotation import (image_crosstrack, label_crosstrack,
                                                   log_quad, measure_edges)


def line_fit(r, err):
    """err = a + b r by least squares, then by a robust re-fit that ignores the
    worst 5% -- the image edges occasionally lock onto clutter. Returns
    (angle_deg, intercept_m, residual_rms_m, n)."""
    ok = np.isfinite(err)
    r, err = r[ok], err[ok]
    if len(r) < 10:
        return (np.nan, np.nan, np.nan, len(r))
    b, a = np.polyfit(r, err, 1)
    res = err - (a + b * r)
    keep = np.abs(res) <= np.quantile(np.abs(res), 0.95)
    b, a = np.polyfit(r[keep], err[keep], 1)
    res = err[keep] - (a + b * r[keep])
    return (float(np.degrees(np.arctan(b))), float(a), float(np.sqrt(np.mean(res ** 2))),
            int(keep.sum()))


def one_episode(job):
    path, lo, hi, image_stride, cam_kw, runways_path = job
    cam = CameraModel(**cam_kw)
    ep = Episode(path)
    rw = load_runways(runways_path).get((ep.airport, ep.runway_name))
    if rw is None:
        return None
    rows = []
    for k, stem in enumerate(ep.stems):
        f = ep.frame(stem)
        if not f.labelled or not lo <= f.range_m <= hi:
            continue
        lab = label_crosstrack(f, cam, rw.width_m)
        img = np.nan
        if k % image_stride == 0:
            gray = cv2.imread(str(f.image_path), cv2.IMREAD_GRAYSCALE)
            est = image_crosstrack(measure_edges(gray, f.polygon_px(cam.width, cam.height),
                                                 log_quad(f, rw, cam)), f, cam, rw.width_m)
            img = np.nan if est is None else est
        rows.append((f.range_m, f.right_m, lab, img))
    if len(rows) < 10:
        return None
    a = np.array(rows)
    r, logged, lab, img = a.T
    return {"episode": ep.path.name, "airport": ep.airport, "runway": ep.runway_name,
            "lat": rw.r1_lat, "lon": rw.r1_lon, "n": len(r),
            "label": line_fit(r, lab - logged), "image": line_fit(r, img - logged),
            "curve": {"r": r.tolist(), "label_err": (lab - logged).tolist(),
                      "image_err": (img - logged).tolist()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="datasets/xp12_dataset")
    ap.add_argument("--camera", default="experiments/bev_yolo/camera.json")
    ap.add_argument("--range", default="150,1500")
    ap.add_argument("--image-stride", type=int, default=4)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="experiments/edge_lines/runs/rotation_all")
    args = ap.parse_args()

    cam_kw = json.loads(Path(args.camera).read_text())["camera"]
    lo, hi = map(float, args.range.split(","))
    jobs = [(ep.path, lo, hi, args.image_stride, cam_kw, None)
            for ep in find_episodes(args.root)]
    with ProcessPoolExecutor(args.workers) as pool:
        res = [r for r in pool.map(one_episode, jobs) if r is not None]
    res.sort(key=lambda x: x["episode"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["episode", "airport", "runway", "n",
                    "label_angle_deg", "label_intercept_m", "label_resid_m",
                    "image_angle_deg", "image_intercept_m", "image_resid_m"])
        for x in res:
            w.writerow([x["episode"], x["airport"], x["runway"], x["n"],
                        *[f"{v:.4f}" for v in x["label"][:3]],
                        *[f"{v:.4f}" for v in x["image"][:3]]])

    L = np.array([x["label"][:3] for x in res])
    I = np.array([x["image"][:3] for x in res])
    ok = np.all(np.isfinite(I), axis=1)
    q = lambda v: f"median {np.median(v):.3f}, p90 {np.quantile(v, 0.9):.3f}, max {np.max(v):.3f}"
    print(f"{len(res)} episodes, range {lo:.0f}-{hi:.0f} m")
    print(f"\nLABEL vs logs (every frame)")
    print(f"  angle deg      : {L[:, 0].min():+.3f} .. {L[:, 0].max():+.3f}, "
          f"|angle| {q(np.abs(L[:, 0]))}")
    print(f"  intercept m    : |a| {q(np.abs(L[:, 1]))}, mean {L[:, 1].mean():+.3f}")
    print(f"  residual rms m : {q(L[:, 2])}")
    print(f"\nIMAGE vs logs (every {args.image_stride}th frame, {ok.sum()} episodes fit)")
    print(f"  angle deg      : |angle| {q(np.abs(I[ok, 0]))}")
    print(f"  intercept m    : |a| {q(np.abs(I[ok, 1]))}, mean {I[ok, 1].mean():+.3f}")
    print(f"  residual rms m : {q(I[ok, 2])}")
    d = I[ok, 0] - L[ok, 0]
    print(f"\nIMAGE vs LABEL angle: corr {np.corrcoef(I[ok, 0], L[ok, 0])[0, 1]:.4f}, "
          f"slope {np.polyfit(L[ok, 0], I[ok, 0], 1)[0]:.3f}, "
          f"|diff| {q(np.abs(d))}")
    worst = sorted(res, key=lambda x: -x["label"][2])[:5]
    print("\nworst label residuals: " + ", ".join(
        f"{x['episode']} {x['airport']} {x['runway']} {x['label'][2]:.3f} m" for x in worst))

    # ------------------------------------------------------------ figure --
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    for x in res:
        c = x["curve"]
        r = np.array(c["r"])
        ax[0].plot(r, np.array(c["label_err"]), lw=0.8, alpha=0.55)
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].invert_xaxis()
    ax[0].set_xlabel("range to threshold (m)")
    ax[0].set_ylabel("label-derived minus logged cross-track (m)")
    ax[0].set_title(f"Every episode ({len(res)}): straight lines through ~0 at the threshold",
                    fontsize=10)
    ax[0].grid(alpha=0.3)

    for x in res:
        c = x["curve"]
        r = np.array(c["r"])
        b = np.tan(np.radians(x["label"][0]))
        ax[1].plot(r, np.array(c["label_err"]) - b * r, lw=0.8, alpha=0.55)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].invert_xaxis()
    ax[1].set_ylim(-1.5, 1.5)
    ax[1].set_xlabel("range to threshold (m)")
    ax[1].set_ylabel("same, minus angle x range (m)")
    ax[1].set_title(f"What is left once range x angle is removed "
                    f"(median rms {np.median(L[:, 2]) * 100:.1f} cm)", fontsize=10)
    ax[1].grid(alpha=0.3)

    ax[2].plot(L[ok, 0], I[ok, 0], "o", ms=4, alpha=0.7)
    lim = np.nanmax(np.abs(np.r_[L[:, 0], I[ok, 0]])) * 1.1
    ax[2].plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="y = x")
    ax[2].set_xlim(-lim, lim), ax[2].set_ylim(-lim, lim)
    ax[2].set_aspect("equal")
    ax[2].set_xlabel("angle from label corners (deg)")
    ax[2].set_ylabel("angle from painted edges in the image (deg)")
    ax[2].set_title(f"Two independent measurements of each episode's angle "
                    f"(r = {np.corrcoef(I[ok, 0], L[ok, 0])[0, 1]:.3f})", fontsize=10)
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}.png", dpi=130)
    Path(f"{out}.json").write_text(json.dumps(res))
    print(f"\nwrote {out}.csv / .png / .json")


if __name__ == "__main__":
    main()
