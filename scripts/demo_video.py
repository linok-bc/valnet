#!/usr/bin/env python
"""Replay one test flight with a model's saved runway predictions.

Reads results/.../predictions_yolo/<dataset>/<episode>.json (predict_flights.py)
and draws, per frame:

    camera   predicted runway (violet fill, highest confidence), label (aqua outline);
             frames beyond half the visibility (dropped from scoring) are flagged
    panel    flight, weather, range, confidence, IoU, and IoU vs range so far

    python scripts/demo_video.py --flight xp12_dataset_weather/000197
"""

import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

SM = 1609.344
VIOLET, AQUA, AMBER, WHITE, GREY = (230, 90, 170), (230, 220, 60), (40, 180, 255), (240, 240, 240), (150, 150, 150)
PANEL_W = 480


def poly_px(poly):
    return np.round(np.asarray(poly, float)).astype(np.int32)


def iou(a, b, shape):
    ma, mb = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
    if len(a) >= 3:
        cv2.fillPoly(ma, [a], 1)
    if len(b) >= 3:
        cv2.fillPoly(mb, [b], 1)
    u = (ma | mb).sum()
    return (ma & mb).sum() / u if u else 0.0


def text(img, s, xy, scale=0.6, color=WHITE, thick=1):
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def plot(panel, pts, vis_half, top, h):
    """IoU vs range (km, decreasing to the right), drawn into panel[top:top+h]."""
    x0, x1, y0, y1 = 50, PANEL_W - 20, top + h - 46, top + 10
    rmax = 3.2
    cv2.rectangle(panel, (x0, y1), (x1, y0), GREY, 1)
    X = lambda r: int(x0 + (rmax - min(r, rmax)) / rmax * (x1 - x0))
    Y = lambda v: int(y0 - v * (y0 - y1))
    if vis_half / 1000 < rmax:
        xv = X(vis_half / 1000)
        cv2.rectangle(panel, (x0, y1), (xv, y0), (60, 60, 60), -1)
        text(panel, "not scored", (x0 + 4, y1 + 16), 0.45, GREY)
    for v in (0.5, 1.0):
        cv2.line(panel, (x0, Y(v)), (x1, Y(v)), (80, 80, 80), 1)
        text(panel, f"{v:.1f}", (x0 - 34, Y(v) + 5), 0.45, GREY)
    for r in (3, 2, 1, 0):
        text(panel, f"{r}", (X(r) - 4, y0 + 16), 0.45, GREY)
    text(panel, "IoU vs range to threshold (km)", (x0, top + h - 2), 0.45, GREY)
    for r, v in pts:
        cv2.circle(panel, (X(r / 1000), Y(v)), 2, VIOLET, -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flight", required=True, help="<dataset>/<episode>")
    ap.add_argument("--pred", default="results/xp12_combined/predictions_yolo")
    ap.add_argument("--out", default="results/xp12_combined/demos")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--model-name", default="YOLOv8s-seg, 1280 px")
    ap.add_argument("--tag", default="", help="filename prefix, e.g. the model, so videos of one flight do not collide")
    args = ap.parse_args()

    ds, ep = args.flight.split("/")
    root = Path("datasets") / ds / ep
    pred = json.loads((Path(args.pred) / ds / f"{ep}.json").read_text())
    meta = root.joinpath("meta.txt").read_text().split("\n")
    airport_rwy = meta[0].strip()
    vis = float(meta[1].split()[3]) * SM
    wj = root / "weather.json"
    regime = json.loads(wj.read_text())["regime"] if wj.exists() else "easy (clear)"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{args.tag + '_' if args.tag else ''}{ds.replace('xp12_dataset_', '')}_{ep}.mp4"
    W, H = 1280, 720
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{W + PANEL_W}x{H}", "-r", str(args.fps), "-i", "-",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", str(dst)],
                          stdin=subprocess.PIPE)
    pts = []
    for fr in pred["frames"]:
        s = fr["stem"]
        img = cv2.imread(str(root / "images" / f"{s}.jpg"))
        lab = np.array(open(root / "labels" / f"{s}.txt").read().split()[1:], float).reshape(-1, 2) * [W, H]
        g = poly_px(lab)
        along, height, right = map(float, open(root / "position" / f"{s}.txt").read().split())
        rng = math.hypot(along, height) if along < 0 else 0.0

        det = fr["dets"][0] if fr["dets"] else None
        if det and len(det["poly"]) >= 3:
            p = poly_px(det["poly"])
            fill = img.copy()
            cv2.fillPoly(fill, [p], VIOLET)
            img = cv2.addWeighted(fill, 0.4, img, 0.6, 0)
            cv2.polylines(img, [p], True, VIOLET, 2, cv2.LINE_AA)
            v = iou(g, p, (H, W))
        else:
            v = 0.0
        cv2.polylines(img, [g], True, AQUA, 2, cv2.LINE_AA)
        if fr["scored"]:
            pts.append((rng, v))
        else:
            cv2.rectangle(img, (0, H - 44), (W, H), (30, 30, 30), -1)
            text(img, "beyond half the visibility: dropped from scoring", (16, H - 15), 0.75, AMBER, 2)

        panel = np.full((H, PANEL_W, 3), 24, np.uint8)
        y = 40
        for s_, sc, c in [(args.model_name, 0.7, WHITE),
                          (f"{ds.replace('xp12_dataset_', '')}/{ep}   {airport_rwy}", 0.6, GREY),
                          (f"weather: {regime}   visibility {vis / 1000:.2f} km", 0.6, GREY)]:
            text(panel, s_, (20, y), sc, c, 2 if sc > 0.65 else 1)
            y += 32
        y += 10
        rows = [("range to threshold", f"{rng:,.0f} m" if along < 0 else "past threshold"),
                ("height", f"{height:,.0f} m"),
                ("confidence", f"{det['conf']:.2f}" if det else "no detection"),
                ("IoU with label", f"{v:.3f}"),
                ("scored", "yes" if fr["scored"] else "no")]
        for k, val in rows:
            text(panel, k, (20, y), 0.6, GREY)
            text(panel, val, (250, y), 0.6, WHITE)
            y += 30
        if pts:
            text(panel, f"mean IoU so far (scored): {np.mean([q for _, q in pts]):.3f}", (20, y + 10), 0.6, WHITE)
        cv2.line(panel, (20, H - 300), (60, H - 300), VIOLET, 3)
        text(panel, "prediction", (68, H - 295), 0.5, GREY)
        cv2.line(panel, (190, H - 300), (230, H - 300), AQUA, 3)
        text(panel, "label", (238, H - 295), 0.5, GREY)
        plot(panel, pts, vis / 2, H - 280, 270)

        ff.stdin.write(np.hstack([img, panel]).tobytes())
    ff.stdin.close()
    ff.wait()
    print(dst, len(pred["frames"]), "frames")


if __name__ == "__main__":
    main()
