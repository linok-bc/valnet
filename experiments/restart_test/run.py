#!/usr/bin/env python
"""Samuel's Restart estimator (yaw + cross-track) on the new xp12 TEST split.

Frames: the visibility-filtered combined test list (datasets/xp12_splits/
combined/test.txt), on approach, 150-1500 m before the threshold -- the window
the joint demo used. Restart's runner starts cold at each flight's first such
frame and tracks with its kinematic EKF (`track + kf`, method
maskfit_analytic_lm), exactly as experiments/joint_demo/run_heading.py.

Stages (each cached, rerun skips finished work):

  calib   Restart's per-flight calibration (W, L, d0, psi0) fitted on the GT
          corners. The new data clips labels at the image border (5-6 vertex
          polygons) and zeroes out-of-frame corners in corners/, so the fit uses
          only frames whose four corners/ points are flagged visible and lie
          strictly inside the image. Restart's own loader is left unchanged.
  masks   soft YOLO masks (Restart's YoloMaskCache format) from the 1280-px
          YOLOv8s-seg trained on the new data, for the window frames.
  run     Restart's run_sequence per flight -> per-frame CSV.
  score   yaw and cross-track error, overall / by range / by weather.

Excluded: weather/000209 (CYWK 36). Its logged height is ~165 m too high (a
height offset fitted on its corners is -164.5 m; every other test flight fits
~0), so any method fed the logged height is wrong there. weather/000654 (ZMUB
14) is kept but reported separately: its far corners sit 16-26 px off a flat
runway at every range (a sloped runway), which Restart's flat model cannot fit.

Truth is Restart's own (pose_from_frame with the fitted calibration); with
d0 ~ 0 and psi0 ~ 0 on this data it equals the rotation-fixed logs.
Restart's lateral is + LEFT; scores use |error| so the sign does not matter.

    python -m experiments.restart_test.run
    python -m experiments.restart_test.run --stages score
    python -m experiments.restart_test.run --stages run score --filter old    # or none
"""

import argparse
import csv
import dataclasses
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Restart"))

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
LO, HI = 150.0, 1500.0
BINS = [150, 300, 600, 1000, 1500]
# Tracking filter variants (Restart RunConfig): "crab" = EKF with the crab-angle state
# (kf_crab=True, the fix), "old" = the original 3-state EKF, "none" = warm start only.
ROWS = {"crab": "rows_crab", "old": "rows", "none": "rows_nokf"}
SUMMARY = {"crab": "summary_crab.json", "old": "summary.json", "none": "summary_nokf.json"}


def window_frames(test_list):
    """{(dataset, episode): [frame index]} for scored test frames 150-1500 m out."""
    out = defaultdict(list)
    for line in Path(test_list).read_text().splitlines():
        if not line.strip():
            continue
        p = ROOT / line.strip()
        along = float((p.parents[1] / "position" / f"{p.stem}.txt").read_text().split()[0])
        if -HI <= along <= -LO:
            out[(p.parents[2].name, p.parents[1].name)].append(int(p.stem))
    return {k: sorted(v) for k, v in sorted(out.items())}


def corners_inside(ep_dir, index, margin=0.002):
    v = np.array((ep_dir / "corners" / f"{index:06d}.txt").read_text().split(), float).reshape(4, 3)
    return bool((v[:, 2] == 1).all() and ((v[:, :2] > margin) & (v[:, :2] < 1 - margin)).all())


def stage_calib(flights):
    from runway_detection.xp12 import fit_calibration, load_sequence
    from runway_detection.xp12.calibration import save_calibrations
    for ds in sorted({ds for ds, _ in flights}):
        path = CACHE / f"calibration_{ds}.json"
        if path.exists():
            continue
        calibs = {}
        for d, ep in flights:
            if d != ds:
                continue
            root = ROOT / "datasets" / ds
            seq = load_sequence(root, ep)
            seq = dataclasses.replace(seq, frames=[f for f in seq.frames
                                                   if corners_inside(root / ep, f.index)])
            calibs[ep] = c = fit_calibration(seq)
            print(f"calib {ds} {ep} {c.airport} {c.runway}  W={c.width_m:.1f} L={c.length_m:.0f} "
                  f"d0={c.d0_m:+.2f} psi0={c.psi0_deg:+.4f}  rms={c.rms_px:.2f}px n={c.n_frames}", flush=True)
        save_calibrations(calibs, path)


def soft_mask(result, shape):
    """Confidence-weighted union of class-0 masks, as joint_demo/cache_masks.py."""
    import cv2
    h, w = shape
    out = np.zeros((h, w), np.float32)
    if result.masks is None:
        return out
    for m, c in zip(result.masks.data.cpu().numpy(), result.boxes.conf.cpu().numpy()):
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        out = np.maximum(out, np.clip(m * float(c), 0, 1))
    return out


def stage_masks(flights, weights, imgsz, conf):
    from ultralytics import YOLO
    model = YOLO(weights)
    for (ds, ep), idx in flights.items():
        path = CACHE / "masks" / ds / f"{ep}.npz"
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        masks = []
        for k in range(0, len(idx), 8):
            imgs = [str(ROOT / "datasets" / ds / ep / "images" / f"{i:06d}.jpg") for i in idx[k:k + 8]]
            for r in model.predict(imgs, imgsz=imgsz, conf=conf, half=True, retina_masks=True, verbose=False):
                masks.append(np.round(soft_mask(r, r.orig_shape) * 255).astype(np.uint8))
        np.savez_compressed(path, indices=np.array(idx), masks=np.stack(masks))
        print(f"masks {ds} {ep}: {len(idx)} frames", flush=True)


def run_one(job):
    ds, ep, method, filt = job
    from evaluation.configs import make_method
    from evaluation.observations import YoloMaskCache
    from evaluation.runner import RunConfig, run_sequence
    from runway_detection.xp12.calibration import load_calibrations
    from runway_detection.xp12.loader import load_sequence
    out = CACHE / ROWS[filt] / ds / f"{ep}.csv"
    if out.exists():
        return ds, ep, None
    calib = load_calibrations(CACHE / f"calibration_{ds}.json")[ep]
    seq = load_sequence(ROOT / "datasets" / ds, ep)
    cache = YoloMaskCache(CACHE / "masks" / ds)
    rows = run_sequence(seq, calib, make_method(method), RunConfig(source="yolo", mode="track", kf=filt != "none",
                                                                  kf_crab=filt == "crab"),
                        yolo=cache, frame_indices=cache.available_indices(ep))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return ds, ep, len(rows)


def stage_run(flights, method, workers, filt):
    jobs = [(ds, ep, method, filt) for ds, ep in flights]
    with ProcessPoolExecutor(workers) as ex:
        for ds, ep, n in ex.map(run_one, jobs):
            if n is not None:
                print(f"run {ds} {ep}: {n} frames", flush=True)


def stats(e):
    e = np.abs(np.asarray(e, float))
    e = e[np.isfinite(e)]
    if not len(e):
        return {"n": 0}
    return {"n": int(len(e)), "mae": float(e.mean()), "median": float(np.median(e)),
            "p95": float(np.percentile(e, 95)), "max": float(e.max())}


def stage_score(flights, filt):
    regime = {}
    for r in csv.DictReader(open(ROOT / "datasets/xp12_splits/flights.csv")):
        regime[(r["dataset"], r["episode"])] = r["weather_type"] or "easy"
    rows = []
    for ds, ep in flights:
        for r in csv.DictReader(open(CACHE / ROWS[filt] / ds / f"{ep}.csv")):
            rng = abs(float(r["along_m"]))
            rows.append({"ds": ds, "ep": ep, "regime": regime[(ds, ep)], "range": rng,
                         "yaw_err": float(r["yaw_err"]), "lat_err": float(r["lat_err"]),
                         "status": r["status"]})
    nan = sum(not np.isfinite(r["yaw_err"]) for r in rows)

    def table(groups):
        return {k: {"yaw_deg": stats([r["yaw_err"] for r in g]), "xtrack_m": stats([r["lat_err"] for r in g])}
                for k, g in groups.items()}

    by_bin = defaultdict(list)
    for r in rows:
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            if lo <= r["range"] < hi or (hi == BINS[-1] and r["range"] == hi):
                by_bin[f"{lo}-{hi}"].append(r)
    by_regime, by_flight = defaultdict(list), defaultdict(list)
    for r in rows:
        by_regime[r["regime"]].append(r)
        by_flight[f"{r['ds']}/{r['ep']}"].append(r)
    flight_mae = {k: (v["yaw_deg"].get("mae"), v["xtrack_m"].get("mae")) for k, v in table(by_flight).items()}
    summary = {"window_m": [LO, HI], "flights": len(flights), "frames": len(rows), "no_estimate": nan,
               "status": dict(sorted(defaultdict(int, {s: sum(r["status"] == s for r in rows)
                                                       for s in {r["status"] for r in rows}}).items())),
               "overall": table({"all": rows})["all"], "by_range": table(dict(sorted(by_bin.items(),
                                                                                    key=lambda kv: int(kv[0].split("-")[0])))),
               "by_weather": table(dict(sorted(by_regime.items()))), "by_flight_mae": flight_mae}
    (HERE / SUMMARY[filt]).write_text(json.dumps(summary, indent=1))

    def show(name, t):
        print(f"\n{name}")
        print(f"  {'':16s} {'n':>6s} | {'yaw MAE':>8s} {'p95':>6s} {'max':>6s} (deg) | {'xt MAE':>7s} {'p95':>6s} {'max':>6s} (m)")
        for k, v in t.items():
            y, x = v["yaw_deg"], v["xtrack_m"]
            print(f"  {k:16s} {y['n']:6d} | {y['mae']:8.3f} {y['p95']:6.3f} {y['max']:6.2f}       | "
                  f"{x['mae']:7.2f} {x['p95']:6.2f} {x['max']:6.1f}")
    print(f"{len(flights)} flights, {len(rows)} frames, {nan} without an estimate; status {summary['status']}")
    show("overall (150-1500 m)", {"all": summary["overall"]})
    show("by range to threshold (m)", summary["by_range"])
    show("by weather", summary["by_weather"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", default=["calib", "masks", "run", "score"])
    ap.add_argument("--test-list", default=str(ROOT / "datasets/xp12_splits/combined/test.txt"))
    ap.add_argument("--weights", default=str(ROOT / "baseline/runs/xp12_combined_yolo/weights/best.pt"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--method", default="maskfit_analytic_lm")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--filter", choices=list(ROWS), default="crab",
                    help="crab: EKF with crab-angle state (the fix); old: original 3-state EKF, which "
                         "biases cross-track 3-5 m under crosswind; none: warm start only")
    ap.add_argument("--exclude", nargs="*", default=["xp12_dataset_weather/000209"])
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    flights = {k: v for k, v in window_frames(args.test_list).items() if "/".join(k) not in args.exclude}
    print(f"{len(flights)} flights, {sum(map(len, flights.values()))} frames in {LO:.0f}-{HI:.0f} m", flush=True)
    if "calib" in args.stages:
        stage_calib(flights)
    if "masks" in args.stages:
        stage_masks(flights, args.weights, args.imgsz, args.conf)
    if "run" in args.stages:
        stage_run(flights, args.method, args.workers, args.filter)
    if "score" in args.stages:
        stage_score(flights, args.filter)


if __name__ == "__main__":
    main()
