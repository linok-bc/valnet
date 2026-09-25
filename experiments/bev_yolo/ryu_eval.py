#!/usr/bin/env python
"""Score BEV runway segmentation with Ryu et al.'s estimator instead of ours.

THE QUESTION. The BEV toy experiment found that segmenting from a bird's-eye
view does NOT beat segmenting the perspective image (README.md), and that the
residual error is almost entirely HEADING error levered by range: cross-track is
read off the threshold midpoint projected onto a normal that comes from a fitted
box angle, so an angular error dpsi enters as `range * sin dpsi`.

Ryu et al. do not have that lever. They take heading from the INS (their eq 22)
and never estimate it. They also read only two points off the mask -- the front
edge -- and get their metric scale by measuring the runway's apparent width
against its known true width (eq 18), rather than from a metric raster.

So: does the paper's estimator, on the SAME BEV segmentation, beat the
minimum-area-rectangle estimator the experiment has been using? And which of the
three differences is doing the work?

THE ABLATION. `experiments.bev.ryu` implements the three swaps as independent
flags, so the variants below span them. Read the table bottom-up:

    rect        scale=grid  yaw=fitted  edge=rect    the existing estimator
    front       scale=grid  yaw=fitted  edge=front   + two points only
    fitted_yaw  scale=width yaw=fitted  edge=front   + the measured scale
    paper       scale=width yaw=known   edge=front   + the INS heading = Ryu

`paper` vs `fitted_yaw` is the value of knowing yaw. `fitted_yaw` vs `front` is
what the measured scale costs. `front` vs `rect` is what throwing away all but
two points costs. And `paper` vs `rect` is the headline.

THE SHAPE LEG. The paper's error is worst where it should be easiest: 2.50 m MAE
inside 300 m against 1.69 m at 1000-1500 m. Measured on the test split, the
MEDIAN threshold-midpoint lateral error is flat with range (-0.9 to -1.4 m) but
its p90 grows from 2.25 m at 600-1000 m to 6.21 m inside 300 m, with worst cases
at 13-23 m -- about half a runway width. That is the two-point rule latching onto
a PARTIAL front edge: read two points, and a clipped edge or a rounded corner
moves their midpoint by half a width. Three variants replace those two points
with something fitted to the whole mask:

    paper_rect   the near edge of the minimum-area rectangle
    paper_axis   the polygon's AREA CENTROID -- also on the centre line, but an
                 average rather than an extremum, and unbiased by the far-end
                 clipping that every frame here has
    axis_fitted  the same, camera-only (yaw fitted), to compare against fitted_yaw

`paper_axis` estimates cross-track only: its `along_m` is the centroid's, not the
threshold's. That is the trade -- give up the range estimate the paper also
wants, keep the lateral one it is judged on.

CAVEAT, AND IT IS THE WHOLE INTERPRETATION. `yaw="known"` solves an EASIER
problem. The BEV experiment's premise is that yaw and cross-track are the two
unknowns; handing the estimator the true yaw removes one of them. That is
legitimate for Ryu et al., who have an INS on the aircraft and are not claiming
to estimate heading from the camera -- but it means `paper` and `rect` are not
answering the same question. `fitted_yaw` is the like-for-like row. Both are
printed for exactly this reason; do not quote `paper` against a camera-only
baseline without saying it used the INS.

REPORTING follows the paper's Table 3 -- MAE, RMSE, MAX -- so the numbers here
can be put next to its 1.779 / 2.380 / 6.618 m approach-phase figures. The
existing eval.py reports medians instead, which are not comparable; both are
printed.

    python -m experiments.bev_yolo.ryu_eval --arm bev_seg --oracle-only
    python -m experiments.bev_yolo.ryu_eval --arm bev_seg,bev_seg_alt
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.bev.camera import CameraModel
from experiments.bev.ryu import RyuEstimator, kalman_track, simplify
from experiments.bev_yolo.arms import load_config
from experiments.bev_yolo.eval import (label_polygon, polygon_to_ground,
                                       predicted_polygon)

# The ablation ladder. Order matters: each row adds one of the paper's three
# departures to the row above, so a column read downward decomposes the total.
VARIANTS = {
    "rect":       dict(scale="grid",  yaw="fitted", edge="rect"),
    "front":      dict(scale="grid",  yaw="fitted", edge="front"),
    "fitted_yaw": dict(scale="width", yaw="fitted", edge="front"),
    "paper":      dict(scale="width", yaw="known",  edge="front"),
    # --- the shape leg: same estimator, a sturdier point on the centre line ---
    "paper_rect": dict(scale="width", yaw="known",  edge="rect"),
    "paper_axis": dict(scale="width", yaw="known",  edge="axis"),
    "axis_fitted": dict(scale="width", yaw="fitted", edge="axis"),
}


def score_rows(polys_px, records, cam, arm, est, ground_window=True,
               dp_frac=0.01):
    """Run one estimator variant over a split. One row per frame, in order."""
    rows = []
    for poly_px, rec in zip(polys_px, records):
        row = {"name": rec["name"], "episode": rec["episode"],
               "range_m": rec["range_m"], "along_m": rec["along_m"],
               "truth_m": rec["right_m"], "ok": False, "err": np.nan,
               "est_m": np.nan, "width_err": np.nan}
        rows.append(row)
        if poly_px is None:
            continue
        # Douglas-Peucker in the DETECTOR's pixels (paper section 3.1.1), before
        # back-projection, so the tolerance means the same thing at every range.
        poly_g = polygon_to_ground(simplify(poly_px, dp_frac), rec, cam, arm,
                                   ground_window)
        out = est(poly_g, cam, rec["roll_deg"], rec["pitch_deg"], rec["height_m"],
                  rec["runway_width_m"], yaw_deg=rec["yaw_deg"])
        if out is None:
            continue
        row.update(ok=True, est_m=out["right_m"],
                   err=out["right_m"] - rec["right_m"],
                   width_err=out["width_m"] - rec["runway_width_m"])
    return rows


def apply_filter(rows, q, r):
    """Section 3.1.2.5: Kalman-filter each approach separately, in along-track order.

    Episodes are filtered independently because they are different flights --
    carrying state across the boundary would smooth one approach's error into
    the next one's first frames.
    """
    by_ep = defaultdict(list)
    for row in rows:
        by_ep[row["episode"]].append(row)
    for ep_rows in by_ep.values():
        ep_rows.sort(key=lambda x: x["along_m"])     # negative on approach, rising
        z = np.array([x["est_m"] if x["ok"] else np.nan for x in ep_rows])
        s = np.array([x["along_m"] for x in ep_rows])
        for row, f in zip(ep_rows, kalman_track(z, s, q=q, r=r)):
            row["err_kf"] = f - row["truth_m"] if np.isfinite(f) else np.nan
    return rows


def summarise(rows, bins, label, key="err"):
    """Per-range-bin errors in the paper's Table 3 columns, plus the median.

    MAE and RMSE are what the paper reports. The median is what eval.py reports,
    and the gap between them is informative on its own: a median far below the
    MAE means a few blown-up frames, which is the failure mode a Kalman filter
    is there to absorb.
    """
    print(f"\n  {label}")
    print(f"    {'range (m)':>13}  {'n':>5}  {'det':>5}  {'MAE':>7}  {'RMSE':>7}"
          f"  {'MAX':>7}  {'med':>7}  {'bias':>7}")
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = [x for x in rows if lo <= x["range_m"] < hi]
        hit = [x for x in sel if np.isfinite(x.get(key, np.nan))]
        if not sel:
            continue
        if not hit:
            print(f"    {lo:5.0f}-{hi:6.0f}  {len(sel):5d}  {0.0:5.2f}"
                  + "  " + "  ".join([f"{'-':>7}"] * 5))
            continue
        e = np.array([x[key] for x in hit])
        print(f"    {lo:5.0f}-{hi:6.0f}  {len(sel):5d}  {len(hit)/len(sel):5.2f}"
              f"  {np.abs(e).mean():7.3f}  {np.sqrt((e**2).mean()):7.3f}"
              f"  {np.abs(e).max():7.3f}  {np.median(np.abs(e)):7.3f}  {e.mean():7.3f}")
    hit = [x for x in rows if np.isfinite(x.get(key, np.nan))]
    if hit:
        e = np.array([x[key] for x in hit])
        print(f"    {'ALL':>12}  {len(rows):5d}  {len(hit)/len(rows):5.2f}"
              f"  {np.abs(e).mean():7.3f}  {np.sqrt((e**2).mean()):7.3f}"
              f"  {np.abs(e).max():7.3f}  {np.median(np.abs(e)):7.3f}  {e.mean():7.3f}")


def evaluate(arm, cfg, weights, split, device, oracle_only, variants,
             ground_window, dp_frac, q, r, no_filter, pivot):
    root = Path(cfg["dataset"]["out"])
    cam = CameraModel(**json.loads((root / "meta" / "camera.json").read_text())["camera"])
    meta = {json.loads(l)["name"]: json.loads(l)
            for l in (root / "meta" / f"{split}.jsonl").read_text().splitlines()}

    img_dir = root / "arms" / arm.name / "images" / split
    lbl_dir = root / "arms" / arm.name / "labels" / split
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    if not images:
        raise SystemExit(f"no images in {img_dir}")
    records = [meta[p.stem] for p in images]

    def size_of(rec):
        if not arm.is_bev:
            return (cam.width, cam.height)
        g = rec["grids"][arm.grid]
        return (g["width"], g["height"])

    sources = {"ORACLE (ground-truth label)":
               [label_polygon(lbl_dir / f"{p.stem}.txt", *size_of(meta[p.stem]))
                for p in images]}

    if not oracle_only:
        from ultralytics import YOLO
        model = YOLO(weights)
        preds = []
        for i in range(0, len(images), 64):
            batch = images[i:i + 64]
            res = model.predict(batch, conf=cfg["eval"]["conf"], verbose=False,
                                device=device,
                                retina_masks=(arm.target == "segment"))
            preds += [predicted_polygon(x, arm.target) for x in res]
        sources[f"PREDICTED ({Path(weights).name})"] = preds

    grid_note = f", {arm.grid} grid" if arm.is_bev else ""
    print(f"\n{'='*78}\n=== {arm.name}  ({arm.view} image, {arm.target} target"
          f"{grid_note})  split={split} n={len(images)} ===")

    out = {"arm": arm.name, "split": split, "results": {}}
    bins = cfg["eval"]["range_bins_m"]
    for src_label, polys in sources.items():
        print(f"\n{'-'*78}\n{src_label}")
        for vname in variants:
            spec = VARIANTS[vname]
            est = RyuEstimator(pivot=pivot, **spec)
            rows = score_rows(polys, records, cam, arm, est,
                              ground_window, dp_frac)
            tag = (f"{vname}  (scale={spec['scale']} yaw={spec['yaw']} "
                   f"edge={spec['edge']} pivot={pivot})")
            summarise(rows, bins, tag)
            if not no_filter:
                apply_filter(rows, q, r)
                summarise(rows, bins, f"{vname}  + Kalman (q={q}, r={r})",
                          key="err_kf")
            out["results"][f"{src_label}|{vname}"] = rows
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="experiments/bev_yolo/config.yaml")
    ap.add_argument("--arm", default="bev_seg",
                    help="comma-separated arm names, or 'all'")
    ap.add_argument("--variant", default=",".join(VARIANTS),
                    help=f"comma-separated, from: {', '.join(VARIANTS)}")
    ap.add_argument("--weights", default=None,
                    help="default: <project>/<arm>/weights/best.pt")
    ap.add_argument("--split", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--oracle-only", action="store_true",
                    help="skip the network: the floor this estimator can reach")
    ap.add_argument("--no-filter", action="store_true",
                    help="skip the Kalman filter (paper section 3.1.2.5)")
    ap.add_argument("--kf-q", type=float, default=1e-5,
                    help="process noise, m^2 per metre flown; the paper tunes "
                         "this against flight data (section 3.1.2.5)")
    ap.add_argument("--kf-r", type=float, default=4.0,
                    help="measurement noise, m^2 (4.0 = a 2 m one-sigma frame)")
    ap.add_argument("--dp-frac", type=float, default=0.01,
                    help="Douglas-Peucker epsilon, as a fraction of the perimeter")
    ap.add_argument("--pivot", default="auto", choices=("auto", "principal", "nadir"),
                    help="the point eqs 19-21 rescale about. 'principal' is the "
                         "paper's eq 12 and is UNDEFINED on 98.8%% of these "
                         "frames, because this camera looks along the body axis "
                         "while the approach is flown nose-up; see ryu.py")
    ap.add_argument("--no-ground-window", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    cfg, arms, _ = load_config(args.config)
    names = list(arms) if args.arm == "all" else args.arm.split(",")
    for n in names:
        if n not in arms:
            raise SystemExit(f"unknown arm {n!r}; config has {', '.join(arms)}")
    variants = args.variant.split(",")
    for v in variants:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v!r}; have {', '.join(VARIANTS)}")

    split = args.split or cfg["eval"]["split"]
    project = Path(cfg["train"]["project"]).resolve()

    out = []
    for n in names:
        weights = args.weights or str(project / n / "weights" / "best.pt")
        if not args.oracle_only and not Path(weights).exists():
            print(f"\n=== {n}: no weights at {weights}, skipping ===")
            continue
        out.append(evaluate(arms[n], cfg, weights, split, args.device,
                            args.oracle_only, variants,
                            not args.no_ground_window, args.dp_frac,
                            args.kf_q, args.kf_r, args.no_filter, args.pivot))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2, default=float))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
