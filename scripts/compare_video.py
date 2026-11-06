#!/usr/bin/env python
"""Side-by-side replay of one test flight: YOLOv8-seg, VALNet and DeepLabV3+.

Reads the saved predictions of each model (scripts/predict_flights.py) and draws
a 2x2 grid at 1920x1080:

    +----------------------+----------------------+
    | YOLOv8s-seg          | VALNet               |
    +----------------------+----------------------+
    | DeepLabV3+           | flight, weather, range|
    |                      | IoU vs range, all 3  |
    +----------------------+----------------------+

Each camera panel: that model's highest-confidence mask (filled), the label
outline (white) and the frame's IoU. Frames beyond half the visibility (dropped
from scoring) are flagged and left out of the plot.

    python scripts/compare_video.py --flight xp12_dataset_weather/000197
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from demo_video import iou, poly_px, text

SM = 1609.344
W, H, PW, PH = 1920, 1080, 960, 540
AQUA, AMBER, WHITE, GREY, BG = (230, 220, 60), (40, 180, 255), (240, 240, 240), (150, 150, 150), (24, 24, 24)
MODELS = [  # (name, predictions dir, BGR color)
    ("YOLOv8s-seg", "predictions_yolo_widened", (214, 120, 42)),  # #2a78d6, the summary page's colors
    ("VALNet", "predictions_valnet", (52, 104, 235)),  # #eb6834
    ("DeepLabV3+", "predictions_deeplab", (122, 175, 27)),  # #1baf7a
]
LABEL = (245, 245, 245)  # label outline: white, distinct from every model color


def camera_panel(img, label, det, color, name, v, scored):
    out = img.copy()
    if det is not None:
        fill = out.copy()
        cv2.fillPoly(fill, [det], color)
        out = cv2.addWeighted(fill, 0.4, out, 0.6, 0)
        cv2.polylines(out, [det], True, color, 2, cv2.LINE_AA)
    cv2.polylines(out, [label], True, LABEL, 2, cv2.LINE_AA)
    out = cv2.resize(out, (PW, PH), interpolation=cv2.INTER_AREA)
    cv2.rectangle(out, (0, 0), (PW, 46), (20, 20, 20), -1)
    cv2.line(out, (16, 23), (52, 23), color, 4)
    text(out, name, (64, 31), 0.8, WHITE, 2)
    text(out, f"IoU {v:.3f}" if det is not None else "no detection", (PW - 190, 31), 0.75, WHITE if scored else GREY, 2)
    if not scored:
        cv2.rectangle(out, (0, PH - 36), (PW, PH), (30, 30, 30), -1)
        text(out, "beyond half the visibility: not scored", (14, PH - 11), 0.6, AMBER, 2)
    return out


def info_panel(flight, airport, regime, vis, rng, height, past, curves):
    p = np.full((PH, PW, 3), BG, np.uint8)
    text(p, f"{flight}   {airport}", (30, 46), 0.8, WHITE, 2)
    text(p, f"weather: {regime}   visibility {vis / 1000:.2f} km", (30, 84), 0.65, GREY)
    text(
        p,
        ("past threshold" if past else f"range to threshold {rng:,.0f} m") + f"   height {height:,.0f} m",
        (30, 116),
        0.65,
        WHITE,
    )
    # IoU vs range, range decreasing to the right
    x0, x1, y0, y1, rmax = 70, PW - 30, PH - 60, 200, 3.2
    X = lambda m: int(x0 + (rmax - min(m / 1000, rmax)) / rmax * (x1 - x0))  # m -> px
    Y = lambda v: int(y0 - (v - 0.5) / 0.5 * (y0 - y1))
    cv2.rectangle(p, (x0, y1), (x1, y0), (70, 70, 70), 1)
    if vis / 2 / 1000 < rmax:
        cv2.rectangle(p, (x0, y1), (X(vis / 2), y0), (45, 45, 45), -1)
        text(p, "not scored", (x0 + 6, y1 + 20), 0.5, GREY)
    for v in (0.5, 0.75, 1.0):
        cv2.line(p, (x0, Y(v)), (x1, Y(v)), (55, 55, 55), 1)
        text(p, f"{v:.2f}", (x0 - 52, Y(v) + 5), 0.5, GREY)
    for r in (3, 2, 1, 0):
        text(p, f"{r}", (X(r * 1000) - 5, y0 + 22), 0.5, GREY)
    text(p, "IoU vs range to threshold (km)", (x0, PH - 12), 0.55, GREY)
    for (name, _, color), pts in zip(MODELS, curves):
        for r, v in pts:
            cv2.circle(p, (X(r), Y(max(v, 0.5))), 2, color, -1)
    for k, ((name, _, color), pts) in enumerate(zip(MODELS, curves)):
        m = f"{np.mean([v for _, v in pts]):.3f}" if pts else "-"
        cv2.line(p, (30 + 300 * k, 158), (66 + 300 * k, 158), color, 4)
        text(p, f"{name} {m}", (74 + 300 * k, 164), 0.55, WHITE)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flight", required=True, help="<dataset>/<episode>")
    ap.add_argument("--results", default=str(REPO / "results/xp12_combined"))
    ap.add_argument("--out", default=str(REPO / "results/xp12_combined/demos"))
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    ds, ep = args.flight.split("/")
    root = REPO / "datasets" / ds / ep
    preds = []
    for _, d, _ in MODELS:
        frames = json.loads((Path(args.results) / d / ds / f"{ep}.json").read_text())["frames"]
        preds.append({f["stem"]: f for f in frames})
    stems = [f for f in preds[0]]
    meta = (root / "meta.txt").read_text().split("\n")
    vis = float(meta[1].split()[3]) * SM
    wj = root / "weather.json"
    regime = json.loads(wj.read_text())["regime"] if wj.exists() else "easy (clear)"
    name = f"{ds.replace('xp12_dataset_', '')}/{ep}"

    dst = Path(args.out) / f"compare_{ds.replace('xp12_dataset_', '')}_{ep}.mp4"
    dst.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{W}x{H}",
            "-r",
            str(args.fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            str(dst),
        ],
        stdin=subprocess.PIPE,
    )
    curves = [[] for _ in MODELS]
    for s in stems:
        img = cv2.imread(str(root / "images" / f"{s}.jpg"))
        lab = poly_px(
            np.array((root / "labels" / f"{s}.txt").read_text().split()[1:], float).reshape(-1, 2) * [1280, 720]
        )
        along, height, _ = map(float, (root / "position" / f"{s}.txt").read_text().split())
        rng = math.hypot(along, height) if along < 0 else 0.0
        scored = preds[0][s]["scored"]
        panels = []
        for k, ((mname, _, color), pr) in enumerate(zip(MODELS, preds)):
            dets = pr[s]["dets"]
            det = poly_px(dets[0]["poly"]) if dets and len(dets[0]["poly"]) >= 3 else None
            v = iou(lab, det, (720, 1280)) if det is not None else 0.0
            if scored:
                curves[k].append((rng, v))
            panels.append(camera_panel(img, lab, det, color, mname, v, scored))
        panels.append(info_panel(name, meta[0].strip(), regime, vis, rng, height, along >= 0, curves))
        frame = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()
    print(dst, len(stems), "frames")


if __name__ == "__main__":
    main()
