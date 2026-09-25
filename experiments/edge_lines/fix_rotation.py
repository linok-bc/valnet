#!/usr/bin/env python
"""Correct the logged yaw and cross-track for the rotated runway axis -- into a BUFFER.

The collector logs `right`, `along` and `yaw` against a runway axis that is
rotated a few tenths of a degree from the rendered runway, a fixed angle per
episode (verify_rotation.py; confirmed against the image by check_rotation.py).
The recorded polygon is correct, so each episode's angle is measured from it:

    label-derived cross-track - logged cross-track = a + tan(angle) * range

fitted over 150-1500 m (median residual 5 mm). The correction then rotates the
logged (along, right) about the threshold by that angle and subtracts it from
the logged yaw:

    along' =  cos(t) * along + sin(t) * right
    right' = -sin(t) * along + cos(t) * right        (~ right + tan(t) * range)
    yaw'   =  yaw - t

Height, pitch and roll are untouched: a rotation about the vertical cannot move
them. The constant ~-0.2 m intercept `a` is NOT corrected -- it is a separate,
unexplained effect, and fitting it away would hide it.

NOTHING IN THE DATASET IS OVERWRITTEN. Output goes to `--out`, mirroring the
episode layout:

    <out>/<episode>/pose/<stem>.txt        yaw pitch roll       (yaw corrected)
    <out>/<episode>/position/<stem>.txt    along height right   (along, right corrected)
    <out>/manifest.json                    per-episode angle, intercept, residual

    python -m experiments.edge_lines.fix_rotation
    python -m experiments.edge_lines.fix_rotation --figures-only
"""

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.bev.camera import CameraModel
from experiments.bev.episodes import Episode, find_episodes, load_runways
from experiments.edge_lines.check_rotation import (image_crosstrack, log_quad,
                                                   measure_edges)
from experiments.edge_lines.geometry import crosstrack, line_through


def label_estimate(f, cam, width_m):
    """Cross-track and runway heading (relative to the nose) from the polygon alone."""
    q = f.polygon_px(cam.width, cam.height)
    o = crosstrack(line_through(q[0], q[1]), line_through(q[3], q[2]), cam,
                   f.roll_deg, f.pitch_deg, f.height_m, width_m, f.yaw_deg)
    return o["free_width"], o["yaw_fit_deg"]


def corrected(f, t_rad):
    """(along, right, yaw) of a frame with the axis rotation removed."""
    c, s = math.cos(t_rad), math.sin(t_rad)
    return (c * f.along_m + s * f.right_m,
            -s * f.along_m + c * f.right_m,
            f.yaw_deg - math.degrees(t_rad))


def fit_episode(job):
    """Measure one episode's angle from its polygons and write the corrected logs."""
    path, cam_kw, out_root, lo, hi, write = job
    cam = CameraModel(**cam_kw)
    ep = Episode(path)
    rw = load_runways()[(ep.airport, ep.runway_name)]
    frames = [ep.frame(s) for s in ep.stems]
    r, err = [], []
    for f in frames:
        if f.labelled and lo <= f.range_m <= hi:
            est, _ = label_estimate(f, cam, rw.width_m)
            r.append(f.range_m), err.append(est - f.right_m)
    r, err = np.array(r), np.array(err)
    b, a = np.polyfit(r, err, 1)
    resid = float(np.sqrt(np.mean((err - (a + b * r)) ** 2)))
    t = math.atan(b)

    if write:
        d = Path(out_root) / ep.path.name
        (d / "pose").mkdir(parents=True, exist_ok=True)
        (d / "position").mkdir(parents=True, exist_ok=True)
        for f in frames:
            along, right, yaw = corrected(f, t)
            (d / "pose" / f"{f.stem}.txt").write_text(f"{yaw} {f.pitch_deg} {f.roll_deg}\n")
            (d / "position" / f"{f.stem}.txt").write_text(f"{along} {f.height_m} {right}\n")
    return ep.path.name, {"airport": ep.airport, "runway": ep.runway_name,
                          "angle_deg": math.degrees(t), "intercept_m": float(a),
                          "fit_resid_m": resid, "n_fit": int(len(r)),
                          "n_frames": len(frames),
                          "max_right_change_m": float(max(abs(corrected(f, t)[1] - f.right_m)
                                                          for f in frames
                                                          if lo <= f.range_m <= hi))}


def read_fixed(out_root, f):
    """(along, right, yaw) for a frame, from the buffer."""
    d = Path(out_root) / f.episode.path.name
    yaw = float((d / "pose" / f"{f.stem}.txt").read_text().split()[0])
    along, _, right = map(float, (d / "position" / f"{f.stem}.txt").read_text().split())
    return along, right, yaw


def verify(out_root, eps, cam, lo, hi):
    """Refit on the BUFFER: every angle should now be ~0 and yaw should match the polygon."""
    T = load_runways()
    ang, dyaw_before, dyaw_after = [], [], []
    for ep in eps:
        rw = T[(ep.airport, ep.runway_name)]
        r, e, db, da = [], [], [], []
        for f in ep.frames(2):
            if not f.labelled or not lo <= f.range_m <= hi:
                continue
            est, hdg = label_estimate(f, cam, rw.width_m)
            _, right, yaw = read_fixed(out_root, f)
            r.append(f.range_m), e.append(est - right)
            db.append(hdg - f.yaw_deg), da.append(hdg - yaw)
        b, _ = np.polyfit(r, e, 1)
        ang.append(math.degrees(math.atan(b)))
        dyaw_before.append(np.median(db)), dyaw_after.append(np.median(da))
    ang, dyaw_before, dyaw_after = map(np.abs, map(np.array, (ang, dyaw_before, dyaw_after)))
    print(f"\nVERIFY on the buffer ({len(eps)} episodes):")
    print(f"  remaining cross-track angle  |deg|: median {np.median(ang):.4f}, max {ang.max():.4f}")
    print(f"  polygon heading - yaw        |deg|: before median {np.median(dyaw_before):.3f} "
          f"max {dyaw_before.max():.3f}  ->  after median {np.median(dyaw_after):.4f} "
          f"max {dyaw_after.max():.4f}")


def before_after(out_root, manifest, picks, cam, title, path, lo, hi, show_range=1400):
    """One row per episode: image overlay before | after, and cross-track vs range."""
    T = load_runways()
    fig, axes = plt.subplots(len(picks), 3, figsize=(19, 3.9 * len(picks)),
                             gridspec_kw={"width_ratios": [1, 1, 1.35]})
    for i, e in enumerate(picks):
        m = manifest[e]
        ep = Episode(Path("datasets/xp12_dataset") / e)
        rw = T[(ep.airport, ep.runway_name)]
        frames = [ep.frame(s) for s in ep.stems]
        f = min(frames, key=lambda f: abs(f.range_m - show_range))
        img = cv2.cvtColor(cv2.imread(str(f.image_path)), cv2.COLOR_BGR2RGB)
        lab = f.polygon_px(cam.width, cam.height)
        before = log_quad(f, rw, cam)
        along, right, yaw = read_fixed(out_root, f)
        fixed = log_quad(type("F", (), dict(along_m=along, right_m=right, yaw_deg=yaw,
                                             roll_deg=f.roll_deg, pitch_deg=f.pitch_deg,
                                             height_m=f.height_m))(), rw, cam)
        pts = np.vstack([lab, before, fixed]); w = np.ptp(pts[:, 0])
        x0, x1 = int(pts[:, 0].min() - 0.12 * w - 6), int(pts[:, 0].max() + 0.12 * w + 6)
        y0, y1 = int(pts[:, 1].min() - 6), int(pts[:, 1].max() + 8)
        for j, (q, col, name) in enumerate(((before, "#ff3030", "BEFORE: from logged values"),
                                            (fixed, "#1f77ff", "AFTER: from corrected values"))):
            ax = axes[i, j]
            ax.imshow(img[y0:y1, x0:x1], extent=(x0 - .5, x1 - .5, y1 - .5, y0 - .5),
                      interpolation="nearest")
            for qq, c, lw, nm in ((lab, "#22dd44", 2.2, "recorded polygon"), (q, col, 1.2, name)):
                qq = np.vstack([qq, qq[:1]])
                ax.plot(qq[:, 0], qq[:, 1], color=c, lw=lw, label=nm)
            far = np.linalg.norm((lab[1] + lab[2]) / 2 - (q[1] + q[2]) / 2)
            ax.set_title(f"{name.split(':')[0]}  -  far end {far:.1f} px from polygon", fontsize=9)
            ax.set_xticks([]), ax.set_yticks([])
            ax.legend(loc="upper left", fontsize=7, framealpha=0.85)
        axes[i, 0].set_ylabel(f"{e} {m['airport']} {m['runway']}\n"
                              f"heading fix {m['angle_deg']:+.3f} deg\n{f.range_m:.0f} m out",
                              fontsize=10)

        r, lg, fx, lb, im = [], [], [], [], []
        for f in frames[::2]:
            if not f.labelled or not lo <= f.range_m <= hi:
                continue
            gray = cv2.cvtColor(cv2.imread(str(f.image_path)), cv2.COLOR_BGR2GRAY)
            est = image_crosstrack(measure_edges(gray, f.polygon_px(cam.width, cam.height),
                                                 log_quad(f, rw, cam)), f, cam, rw.width_m)
            r.append(f.range_m), lg.append(f.right_m), fx.append(read_fixed(out_root, f)[1])
            lb.append(label_estimate(f, cam, rw.width_m)[0])
            im.append(np.nan if est is None else est)
        ax = axes[i, 2]
        ax.plot(r, lg, color="#ff3030", lw=1.8, label="BEFORE: logged")
        ax.plot(r, fx, color="#1f77ff", lw=1.8, label="AFTER: corrected")
        ax.plot(r, lb, color="#22aa44", lw=1.0, ls="--", label="from recorded polygon")
        ax.plot(r, im, ".", color="#333", ms=3, label="from painted edges in image")
        ax.invert_xaxis(); ax.grid(alpha=0.3)
        ax.set_xlabel("range to threshold (m)"); ax.set_ylabel("cross-track (m right)")
        ax.set_title(f"largest change {m['max_right_change_m']:.1f} m", fontsize=9)
        if i == 0:
            ax.legend(fontsize=7.5)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=95)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="datasets/xp12_dataset")
    ap.add_argument("--out", default="datasets/xp12_corrections")
    ap.add_argument("--camera", default="experiments/bev_yolo/camera.json")
    ap.add_argument("--range", default="150,1500", help="fit window, metres")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args()

    cam_kw = json.loads(Path(args.camera).read_text())["camera"]
    cam = CameraModel(**cam_kw)
    lo, hi = map(float, args.range.split(","))
    eps = find_episodes(args.root)
    out = Path(args.out)

    if args.figures_only:
        manifest = json.loads((out / "manifest.json").read_text())["episodes"]
    else:
        jobs = [(ep.path, cam_kw, out, lo, hi, True) for ep in eps]
        with ProcessPoolExecutor(args.workers) as pool:
            manifest = dict(sorted(pool.map(fit_episode, jobs)))
        (out / "manifest.json").write_text(json.dumps({
            "what": "per-episode runway-axis rotation removed from logged along/right/yaw; "
                    "see experiments/edge_lines/fix_rotation.py",
            "source": str(Path(args.root).resolve()), "fit_range_m": [lo, hi],
            "camera": cam_kw, "episodes": manifest}, indent=1))
        print(f"wrote corrected pose/position for {len(manifest)} episodes to {out}/")
        verify(out, eps, cam, lo, hi)

    by_heading = sorted(manifest, key=lambda e: -abs(manifest[e]["angle_deg"]))[:5]
    by_metres = sorted(manifest, key=lambda e: -manifest[e]["max_right_change_m"])[:5]
    print("\ntop 5 by heading correction : " + ", ".join(
        f"{e} {manifest[e]['airport']} {manifest[e]['runway']} {manifest[e]['angle_deg']:+.3f} deg"
        for e in by_heading))
    print("top 5 by cross-track change : " + ", ".join(
        f"{e} {manifest[e]['airport']} {manifest[e]['runway']} {manifest[e]['max_right_change_m']:.2f} m"
        for e in by_metres))
    media = Path("experiments/edge_lines/media")
    media.mkdir(parents=True, exist_ok=True)
    before_after(out, manifest, by_heading, cam,
                 "Top 5 heading corrections: logged vs corrected, against the recorded polygon",
                 media / "fix_top_heading.jpg", lo, hi)
    if by_metres != by_heading:
        before_after(out, manifest, by_metres, cam,
                     "Top 5 cross-track changes: logged vs corrected, against the recorded polygon",
                     media / "fix_top_crosstrack.jpg", lo, hi)


if __name__ == "__main__":
    main()
