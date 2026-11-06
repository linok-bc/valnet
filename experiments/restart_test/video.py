#!/usr/bin/env python
"""Replay one test flight with Samuel's estimator, with and without his EKF.

Shows why the kinematic EKF biases cross-track under crosswind: the aircraft
crabs (nose into the wind, ground track along the runway), and the EKF's
motion model moves the aircraft along its nose.

Layout (1920 x 1080):

    +-------------------------------+------------------------+
    | camera: YOLO mask (violet),   | cross-track vs range   |
    | truth runway (aqua), runway   |  truth / EKF / per-frame|
    | drawn from each estimate      +------------------------+
    | (red = original EKF, second   | yaw vs range           |
    |  estimate per --second)       |                        |
    +-------------------------------+------------------------+
    | readout: range, crosswind, crab angle, both errors      |
    +---------------------------------------------------------+

Needs run.py's cache (calibration, masks, and the rows of each filter shown).
The second estimate (--second) is the fixed EKF with the crab-angle state
(run.py --filter crab, the default) or the per-frame estimate without a filter.

    python -m experiments.restart_test.video --flight xp12_dataset_weather/000556
"""

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "capstone-dof-estimation"))

from evaluation.video import _quad  # noqa: E402
from pose_estimation.g_dof import GDofParams  # noqa: E402
from runway_detection.xp12.calibration import (load_calibrations, pose_from_frame,  # noqa: E402
                                               scene_from_calibration)
from runway_detection.xp12.loader import load_sequence  # noqa: E402

CACHE = Path(__file__).resolve().parent / "cache"
W, H, CAM_W, CAM_H = 1920, 1080, 1280, 720
VIOLET, AQUA, RED, AMBER = (230, 90, 170), (230, 220, 60), (70, 70, 235), (40, 180, 255)
GREEN = (90, 210, 90)
SECOND = {"crab": ("rows_crab", "fixed EKF (crab state)", GREEN),
          "none": ("rows_nokf", "per frame (no EKF)", AMBER)}
BLUE, TEAL = (214, 120, 42), (122, 175, 27)   # summary page's YOLO and DeepLab colours
# --pair masks: the fixed EKF fed by each segmenter (compare_masks.py rows)
MASK_PAIR = (("rows_yolo_crab", "YOLOv8 masks", BLUE), ("rows_deeplab_crab", "DeepLabV3+ masks", TEAL))
WHITE, GREY, DARK, BG = (240, 240, 240), (150, 150, 150), (70, 70, 70), (24, 24, 24)


def text(img, s, xy, scale=0.6, color=WHITE, thick=1):
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def read_rows(kind, ds, ep):
    return {int(r["frame"]): r for r in csv.DictReader(open(CACHE / kind / ds / f"{ep}.csv"))}


class Plot:
    """Series vs range (km), range decreasing to the right."""

    def __init__(self, x0, y0, w, h, title, unit, lo, hi, rmax=1.5):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.title, self.unit, self.lo, self.hi, self.rmax = title, unit, lo, hi, rmax

    def X(self, r):
        return int(self.x0 + (self.rmax - r / 1000) / self.rmax * self.w)

    def Y(self, v):
        v = min(max(v, self.lo), self.hi)
        return int(self.y0 + self.h - (v - self.lo) / (self.hi - self.lo) * self.h)

    def draw(self, img, series):
        text(img, f"{self.title} ({self.unit})", (self.x0, self.y0 - 12), 0.6, WHITE)
        cv2.rectangle(img, (self.x0, self.y0), (self.x0 + self.w, self.y0 + self.h), DARK, 1)
        for v in np.linspace(self.lo, self.hi, 5):
            cv2.line(img, (self.x0, self.Y(v)), (self.x0 + self.w, self.Y(v)), (45, 45, 45), 1)
            text(img, f"{v:+.1f}" if abs(v) < 10 else f"{v:+.0f}", (self.x0 - 52, self.Y(v) + 5), 0.42, GREY)
        for r in (1.5, 1.0, 0.5, 0.15):
            text(img, f"{r:g}", (self.X(r * 1000) - 8, self.y0 + self.h + 18), 0.42, GREY)
        text(img, "range to threshold (km)", (self.x0 + self.w - 190, self.y0 + self.h + 38), 0.42, GREY)
        for pts, color in series:
            if len(pts) > 1:
                p = np.array([(self.X(r), self.Y(v)) for r, v in pts], np.int32)
                cv2.polylines(img, [p], False, color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flight", required=True, help="<dataset>/<episode>")
    ap.add_argument("--out", default=str(ROOT / "results/xp12_combined/demos"))
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--second", choices=list(SECOND), default="crab")
    ap.add_argument("--pair", choices=["filters", "masks"], default="filters",
                    help="filters: original EKF vs --second; masks: fixed EKF on YOLO vs DeepLab masks")
    args = ap.parse_args()

    ds, ep = args.flight.split("/")
    seq = load_sequence(ROOT / "datasets" / ds, ep)
    calib = load_calibrations(CACHE / f"calibration_{ds}.json")[ep]
    scene = scene_from_calibration(calib)
    if args.pair == "masks":
        (rows1, label1, C1), (rows2, label2, C2) = MASK_PAIR
        masks, tag = {}, "masks"          # two segmenters: draw only the estimated runways
    else:
        rows1, label1, C1 = "rows", "original EKF", RED
        rows2, label2, C2 = SECOND[args.second]
        z = np.load(CACHE / "masks" / ds / f"{ep}.npz")
        masks, tag = dict(zip(z["indices"].tolist(), z["masks"])), args.second
    ekf, raw = read_rows(rows1, ds, ep), read_rows(rows2, ds, ep)
    CT = (245, 245, 245) if args.pair == "masks" else AQUA   # truth colour; white keeps it apart from teal
    frames = {f.index: f for f in seq.frames}
    wj = ROOT / "datasets" / ds / ep / "weather.json"
    w = json.loads(wj.read_text()) if wj.exists() else {"regime": "easy", "crosswind_msc": 0.0}

    idx = sorted(ekf)
    # Restart's lateral is + LEFT; shown here as + RIGHT like the rest of the project.
    truth = {i: (abs(float(ekf[i]["along_m"])), -float(ekf[i]["lat_gt"]), float(ekf[i]["yaw_gt"])) for i in idx}
    est_e = {i: (-float(ekf[i]["lat_est"]), float(ekf[i]["yaw_est"])) for i in idx}
    est_r = {i: (-float(raw[i]["lat_est"]), float(raw[i]["yaw_est"])) for i in idx}
    xt_all = [v for i in idx for v in (truth[i][1], est_e[i][0], est_r[i][0])]
    yaw_all = [v for i in idx for v in (truth[i][2], est_e[i][1], est_r[i][1])]
    pad = lambda v, m: (min(v) - m, max(v) + m)
    xt_lo, xt_hi = pad(xt_all, 1.0)
    yw_lo, yw_hi = pad(yaw_all, 0.2)
    p_xt = Plot(CAM_W + 70, 60, 560, 380, "cross-track, + right", "m", xt_lo, xt_hi)
    p_yaw = Plot(CAM_W + 70, 540, 560, 380, "yaw relative to runway", "deg", yw_lo, yw_hi)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"samuel_{tag}_{ds.replace('xp12_dataset_', '')}_{ep}.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264",
                           "-pix_fmt", "yuv420p", "-crf", "22", str(dst)], stdin=subprocess.PIPE)
    prev = None
    for k, i in enumerate(idx):
        f = frames[i]
        img = cv2.imread(str(seq.image_path(f))).astype(np.float32)
        if i in masks:
            m = (masks[i].astype(np.float32) / 255)[..., None]
            img = img * (1 - 0.45 * m) + np.array(VIOLET, np.float32) * 0.45 * m
        img = img.astype(np.uint8)
        params = GDofParams.from_pose(pose_from_frame(f, calib), scene)
        rng, xt_t, yaw_t = truth[i]
        for (lat_r, yaw), color, th in [((-xt_t, yaw_t), CT, 2), ((-est_r[i][0], est_r[i][1]), C2, 2),
                                        ((-est_e[i][0], est_e[i][1]), C1, 2)]:
            q = _quad(scene, params, yaw, lat_r)
            if q is not None:
                cv2.polylines(img, [np.round(q).astype(np.int32)], True, color, th, cv2.LINE_AA)

        canvas = np.full((H, W, 3), BG, np.uint8)
        canvas[:CAM_H, :CAM_W] = img
        done = idx[:k + 1]
        p_xt.draw(canvas, [([(truth[j][0], truth[j][1]) for j in done], CT),
                           ([(truth[j][0], est_r[j][0]) for j in done], C2),
                           ([(truth[j][0], est_e[j][0]) for j in done], C1)])
        p_yaw.draw(canvas, [([(truth[j][0], truth[j][2]) for j in done], CT),
                            ([(truth[j][0], est_r[j][1]) for j in done], C2),
                            ([(truth[j][0], est_e[j][1]) for j in done], C1)])

        # crab angle: nose (yaw) minus ground-track direction, from consecutive truth positions
        crab = math.nan
        if prev is not None and prev[0] - rng > 1:
            track = math.degrees(math.atan2(xt_t - prev[1], prev[0] - rng))
            crab = yaw_t - track
        prev = (rng, xt_t)

        y0 = CAM_H + 50
        text(canvas, f"Samuel's estimator (Restart, maskfit_analytic_lm) on {ds.replace('xp12_dataset_', '')}/{ep}"
                     f"  {seq.airport} {seq.runway}", (30, y0), 0.75, WHITE, 2)
        text(canvas, f"weather: {w['regime']}   crosswind {abs(w['crosswind_msc']):.1f} m/s   "
                     f"range {rng:,.0f} m   height {params.height_m:.0f} m", (30, y0 + 38), 0.65, GREY)
        crab_s = f"{crab:+.2f} deg" if math.isfinite(crab) else "..."
        text(canvas, f"crab angle (nose minus ground track): {crab_s}", (30, y0 + 72), 0.65, WHITE)
        rows = [("truth", CT, xt_t, yaw_t, None, None),
                (label1, C1, est_e[i][0], est_e[i][1], est_e[i][0] - xt_t, est_e[i][1] - yaw_t),
                (label2, C2, est_r[i][0], est_r[i][1], est_r[i][0] - xt_t, est_r[i][1] - yaw_t)]
        for n, (name, color, xt, yw, exs, eyw) in enumerate(rows):
            yy = y0 + 120 + 36 * n
            cv2.line(canvas, (30, yy - 6), (70, yy - 6), color, 3)
            text(canvas, name, (82, yy), 0.62, WHITE)
            text(canvas, f"cross-track {xt:+6.2f} m", (330, yy), 0.62, WHITE)
            text(canvas, f"yaw {yw:+.3f} deg", (580, yy), 0.62, WHITE)
            if exs is not None:
                text(canvas, f"error {exs:+5.2f} m / {eyw:+.3f} deg", (800, yy), 0.62,
                     (110, 110, 255) if abs(exs) > 2 else WHITE)
        text(canvas, "outlines = runway drawn from each estimate" if not masks else
                     "camera: violet = YOLO mask; outlines = runway drawn from each estimate",
             (30, H - 20), 0.5, GREY)
        ff.stdin.write(canvas.tobytes())
    ff.stdin.close()
    ff.wait()
    print(dst, len(idx), "frames")


if __name__ == "__main__":
    main()
