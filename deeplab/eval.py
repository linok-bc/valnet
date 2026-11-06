#!/usr/bin/env python
"""Per-frame runway-mask IoU at full 1280x720, overall / by weather / by range.

One metric for every model, so they compare on equal terms:

    --weights   score a DeepLabV3+ checkpoint (mask = sigmoid > 0.5)
    --yolo-json score saved YOLO predictions (scripts/predict_flights.py JSON;
                mask = highest-confidence polygon, filled)

Both are scored on the same frame list (default: the visibility-filtered
combined test split) against the same rasterised labels (deeplab/data.py).
Writes a per-frame CSV and a summary JSON next to --out.

    python deeplab/eval.py --weights deeplab/runs/xp12_combined/best.pt
    python deeplab/eval.py --yolo-json results/xp12_combined/predictions_yolo --out results/xp12_combined/iou_yolo
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from deeplab.data import XP12Seg, label_mask, read_list  # noqa: E402

BINS = [("past threshold", None, 0), ("0-150", 0, 150), ("150-300", 150, 300), ("300-600", 300, 600),
        ("600-1000", 600, 1000), ("1000-1500", 1000, 1500), ("1500-3000", 1500, 3000), (">3000", 3000, 1e9)]


def frame_iou(pred, gt):
    i = np.logical_and(pred, gt).sum()
    u = np.logical_or(pred, gt).sum()
    return i, u


@torch.no_grad()
def evaluate(model, loader, dev="cuda", amp=True, per_frame=False):
    """Mean per-frame IoU, dataset IoU (sum I / sum U) and frames with an empty prediction."""
    model.eval()
    rows, I, U, empty = [], 0, 0, 0
    for img, mask, paths in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(img.to(dev).to(memory_format=torch.channels_last))
        pred = (logits[:, 0] > 0).cpu().numpy()
        gt = mask.numpy().astype(bool)
        for p, g, path in zip(pred, gt, paths):
            i, u = frame_iou(p, g)
            I, U, empty = I + i, U + u, empty + (not p.any())
            rows.append({"path": path, "iou": i / u if u else 1.0, "pred_px": int(p.sum()), "gt_px": int(g.sum())})
    model.train()
    out = {"mean_iou": float(np.mean([r["iou"] for r in rows])), "dataset_iou": float(I / U), "empty_pred": int(empty)}
    if per_frame:
        out["rows"] = rows
    return out


def yolo_rows(pred_dir, files):
    """Same per-frame rows from saved YOLO polygons (highest-confidence detection)."""
    by_flight = defaultdict(dict)
    for f in Path(pred_dir).glob("*/*.json"):
        for fr in json.loads(f.read_text())["frames"]:
            by_flight[(f.parent.name, f.stem)][fr["stem"]] = fr["dets"]
    rows = []
    for path in files:
        p = Path(path)
        dets = by_flight[(p.parents[2].name, p.parents[1].name)][p.stem]
        gt = label_mask(path, 720, 1280).astype(bool)
        pred = np.zeros((720, 1280), np.uint8)
        if dets and len(dets[0]["poly"]) >= 3:
            cv2.fillPoly(pred, [np.round(np.array(dets[0]["poly"])).astype(np.int32)], 1)
        i, u = frame_iou(pred.astype(bool), gt)
        rows.append({"path": path, "iou": i / u if u else 1.0, "pred_px": int(pred.sum()), "gt_px": int(gt.sum())})
    return rows


def summarize(rows):
    weather = {}
    for r in csv.DictReader(open(REPO / "datasets/xp12_splits/flights.csv")):
        weather[(r["dataset"], r["episode"])] = r["weather_type"] or "easy"
    groups = defaultdict(list)
    for r in rows:
        p = REPO / r["path"]
        along = float((p.parents[1] / "position" / f"{p.stem}.txt").read_text().split()[0])
        rng = -along
        b = next(n for n, lo, hi in BINS if (lo is None and rng <= 0) or (lo is not None and lo < rng <= hi))
        r["weather"], r["range_bin"] = weather[(p.parents[2].name, p.parents[1].name)], b
        groups[("all", "all")].append(r)
        groups[("weather", r["weather"])].append(r)
        groups[("range", b)].append(r)

    def stat(g):
        iou = np.array([r["iou"] for r in g])
        return {"n": len(g), "mean_iou": float(iou.mean()), "median_iou": float(np.median(iou)),
                "p5_iou": float(np.percentile(iou, 5)), "empty_pred": int(sum(r["pred_px"] == 0 for r in g))}

    order = [n for n, _, _ in BINS]
    return {"overall": stat(groups[("all", "all")]),
            "by_weather": {k: stat(groups[("weather", k)]) for k in sorted({k for t, k in groups if t == "weather"})},
            "by_range": {k: stat(groups[("range", k)]) for k in order if ("range", k) in groups}}


def show(summary, name):
    print(f"\n{name}")
    for title, t in [("overall", {"all": summary["overall"]}), ("by weather", summary["by_weather"]),
                     ("by range to threshold (m)", summary["by_range"])]:
        print(f"  {title}")
        for k, v in t.items():
            print(f"    {k:16s} n={v['n']:6d}  mean IoU {v['mean_iou']:.4f}  median {v['median_iou']:.4f}  "
                  f"p5 {v['p5_iou']:.4f}  empty {v['empty_pred']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights")
    ap.add_argument("--yolo-json")
    ap.add_argument("--list", default="datasets/xp12_splits/combined/test.txt")
    ap.add_argument("--out", default="deeplab/runs/xp12_combined/test")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    if args.weights:
        from torch.utils.data import DataLoader
        from deeplab.model import DeepLabV3Plus
        model = DeepLabV3Plus(pretrained=False).cuda().to(memory_format=torch.channels_last)
        model.load_state_dict(torch.load(args.weights, map_location="cuda"))
        dl = DataLoader(XP12Seg(args.list), args.batch, num_workers=args.workers, pin_memory=True)
        rows, name = evaluate(model, dl, per_frame=True)["rows"], f"DeepLabV3+ {args.weights}"
    elif args.yolo_json:
        rows, name = yolo_rows(args.yolo_json, read_list(args.list)), f"YOLO predictions {args.yolo_json}"
    else:
        ap.error("give --weights or --yolo-json")

    s = summarize(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out.with_suffix(".json").write_text(json.dumps(s, indent=1))
    show(s, name)


if __name__ == "__main__":
    main()
