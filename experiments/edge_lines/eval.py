#!/usr/bin/env python
"""Edge-line cross-track on YOLO masks against the same on ground-truth masks.

THE QUESTION. The edge-line estimator (geometry.py) reads cross-track from the
tilt of the runway's two edges, so it needs no heading and no along-track
distance. How much of its accuracy survives a real segmentation mask?

THREE SOURCES, one pipeline, the same frames:

    EXACT LINES   lines through the label's four corners -- no mask at all. The
                  floor: what the labels and camera model allow.
    GT MASK       the label quad rasterised to a mask, then row-scanned and
                  fitted exactly like a prediction. What a perfect segmenter gets.
    YOLO MASK     the detector's highest-confidence mask, full resolution.

The gap GT MASK -> YOLO MASK is what the segmenter costs. EXACT -> GT MASK is
what pixel quantisation of a thin runway costs, before any detector is involved.

FOUR VARIANTS per source, keyed <direction>_<scale> (see geometry.py):

    free_width     no heading, no height -- only roll, pitch and runway width.
                   The headline: the fewest known quantities.
    free_height    no heading; uses the logged height instead of the width
    known_width    heading from the log, a control
    known_height

TWO TRUTHS. The logged `right` is rotated off the true runway axis by a small
per-episode angle (-0.21 .. +0.31 deg on this split): the collector turns
X-Plane's LOCAL x/z offsets into runway axes using the TRUE-north bearing, but
the local frame is only aligned with true north at its own origin. The label
corners come from lat/lon via worldToLocal and do not have the error. Against
the logged truth even EXACT LINES errs by range * angle -- up to 8 m -- with a
per-episode residual of ~5 mm once that line is removed.

So every table is printed against the `corrected` truth: the logged value plus
the per-episode rotation measured from the label corners (EXACT LINES). That
makes EXACT LINES ~0 by construction; it is there to report the angle, and the
GT MASK -> YOLO MASK comparison is what the correction is for. `--truth logged`
prints the raw version.

THE SPLIT. Scored on the stock YOLOv8 model's OWN held-out test split
(datasets/xp12_dataset/split_test.txt). The bev_toy test split is not usable
here: 11 of its 15 episodes are in this model's training split.

    python -m experiments.edge_lines.eval
    python -m experiments.edge_lines.eval --no-yolo          # GT only, no GPU
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from experiments.bev.camera import CameraModel
from experiments.bev.episodes import Episode, load_runways
from experiments.bev.ryu import kalman_track
from experiments.bev_yolo.ryu_eval import summarise
from experiments.edge_lines.geometry import (crosstrack, estimate_from_mask,
                                             line_through, rasterize)

VARIANTS = ("free_width", "free_height", "known_width", "known_height")


def load_frames(split_file, lo, hi, stride):
    """Every labelled frame of the split's episodes inside the range window."""
    eps = sorted({re.search(r"/(\d{6})/images/", l).group(1)
                  for l in Path(split_file).read_text().splitlines() if l.strip()})
    root = Path(split_file).parent
    frames = []
    for e in eps:
        ep = Episode(root / e)
        for f in ep.frames(stride):
            if f.labelled and lo <= f.range_m < hi:
                frames.append(f)
    return frames


def yolo_masks(weights, frames, conf, device, batch=16, imgsz=None):
    """Highest-confidence runway mask per frame at image resolution, or None."""
    from ultralytics import YOLO
    model = YOLO(weights)
    out = []
    for i in range(0, len(frames), batch):
        paths = [str(f.image_path) for f in frames[i:i + batch]]
        kw = {} if imgsz is None else {"imgsz": imgsz}
        for r in model.predict(paths, conf=conf, retina_masks=True, verbose=False,
                               device=device, **kw):
            if r.masks is None or len(r.masks) == 0:
                out.append(None)
                continue
            j = int(r.boxes.conf.argmax())
            m = r.masks.data[j].cpu().numpy() > 0.5
            h, w = r.orig_shape
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            out.append(m.astype(np.uint8))
        print(f"  yolo {min(i + batch, len(frames))}/{len(frames)}", end="\r")
    print()
    return out


def score(frames, source, cam, runways):
    """One row per frame: every variant's estimate and error."""
    rows = []
    for k, f in enumerate(frames):
        W = runways[(f.episode.airport, f.episode.runway_name)].width_m
        row = {"episode": f.episode.path.name, "stem": f.stem,
               "range_m": f.range_m, "along_m": f.along_m, "truth_m": f.right_m}
        args = (cam, f.roll_deg, f.pitch_deg, f.height_m, W, f.yaw_deg)
        if source == "exact":
            q = f.polygon_px(cam.width, cam.height)
            out = crosstrack(line_through(q[0], q[1]), line_through(q[3], q[2]), *args)
        else:
            mask = (rasterize(f.polygon_px(cam.width, cam.height), cam.width, cam.height)
                    if source == "gt" else source[k])
            out = None if mask is None else estimate_from_mask(mask, *args)
        for v in VARIANTS:
            est = np.nan if out is None else out[v]
            row[v] = est
            row[f"err_{v}"] = est - f.right_m
        if out is not None:
            row["width_err_m"] = out["free_width_m"] - W * 1.0
            row["rows"] = min(out.get("rows_left", 0), out.get("rows_right", 0))
        rows.append(row)
    return rows


def fit_rotation(exact_rows):
    """Per-episode (angle_rad, intercept_m) of EXACT-LINES error against range."""
    by_ep = defaultdict(list)
    for row in exact_rows:
        if np.isfinite(row["err_free_width"]):
            by_ep[row["episode"]].append(row)
    out = {}
    for ep, rs in by_ep.items():
        x = np.array([r["range_m"] for r in rs])
        y = np.array([r["err_free_width"] for r in rs])
        b, a = np.polyfit(x, y, 1)
        out[ep] = (float(b), float(a))
    return out


def retarget(rows, rot):
    """Re-express every error against the rotation-corrected truth."""
    for row in rows:
        b = rot.get(row["episode"], (0.0, 0.0))[0]
        row["truth_logged_m"] = row["truth_m"]
        row["truth_m"] = row["truth_logged_m"] + b * row["range_m"]
        for v in VARIANTS:
            row[f"err_{v}"] = row[v] - row["truth_m"]


def kalman(rows, key, q, r, gate=None, tag="kf"):
    """Forward (causal) Kalman per episode along distance flown; adds err_<tag>_<key>."""
    by_ep = defaultdict(list)
    for row in rows:
        by_ep[row["episode"]].append(row)
    for ep_rows in by_ep.values():
        ep_rows.sort(key=lambda x: x["along_m"])
        z = np.array([x[key] for x in ep_rows])
        s = np.array([x["along_m"] for x in ep_rows])
        for row, f in zip(ep_rows, kalman_track(z, s, q=q, r=r, gate=gate)):
            row[f"err_{tag}_{key}"] = f - row["truth_m"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-file", default="datasets/xp12_dataset/split_test.txt")
    ap.add_argument("--weights", default="baseline/runs/stock/weights/best.pt")
    ap.add_argument("--camera", default="experiments/bev_yolo/camera.json")
    ap.add_argument("--range", default="150,1500", help="lo,hi metres")
    ap.add_argument("--bins", default="150,300,600,1000,1500")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-yolo", action="store_true")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="YOLO inference size; default is the size it was trained at")
    ap.add_argument("--variant", default="free_width,free_height,known_width",
                    help=f"which to print, from {', '.join(VARIANTS)}")
    ap.add_argument("--truth", default="corrected", choices=("corrected", "logged"))
    ap.add_argument("--kf-q", type=float, default=1e-5)
    ap.add_argument("--kf-r", type=float, default=4.0)
    ap.add_argument("--kf-gate", type=float, default=9.0,
                    help="innovation gate in sigma^2 for the gated filter (9 = 3 sigma)")
    ap.add_argument("--out", default="experiments/edge_lines/runs/results.json")
    args = ap.parse_args()

    cam = CameraModel(**json.loads(Path(args.camera).read_text())["camera"])
    runways = load_runways()
    lo, hi = map(float, args.range.split(","))
    bins = [float(b) for b in args.bins.split(",")]
    frames = load_frames(args.split_file, lo, hi, args.stride)
    print(f"{len(frames)} frames, {len({f.episode.path.name for f in frames})} episodes, "
          f"{lo:.0f}-{hi:.0f} m, camera fx {cam.fx:.1f}")

    sources = {"EXACT LINES (label corners, no mask)": "exact",
               "GT MASK (label rasterised)": "gt"}
    if not args.no_yolo:
        sources[f"YOLO MASK ({args.weights})"] = yolo_masks(
            args.weights, frames, args.conf, args.device, imgsz=args.imgsz)

    results, rot = {}, None
    for label, src in sources.items():
        rows = score(frames, src, cam, runways)
        if src == "exact":
            rot = fit_rotation(rows)
            print("\nlogged-truth rotation per episode (deg, intercept m): "
                  + ", ".join(f"{e} {np.degrees(b):+.3f}/{a:+.2f}"
                              for e, (b, a) in sorted(rot.items())))
        if args.truth == "corrected":
            retarget(rows, rot)
        print(f"\n{'=' * 78}\n{label}   [truth: {args.truth}]")
        for v in args.variant.split(","):
            summarise(rows, bins, v, key=f"err_{v}")
        kalman(rows, "free_width", args.kf_q, args.kf_r)
        summarise(rows, bins, f"free_width + Kalman (q={args.kf_q}, r={args.kf_r})",
                  key="err_kf_free_width")
        kalman(rows, "free_width", args.kf_q, args.kf_r, gate=args.kf_gate, tag="kfg")
        summarise(rows, bins, f"free_width + GATED Kalman (gate {args.kf_gate:g} sigma^2)",
                  key="err_kfg_free_width")
        results[label] = rows

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
