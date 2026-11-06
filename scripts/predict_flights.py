#!/usr/bin/env python
r"""Save a segmentation model's runway predictions for every frame of a split's flights.

For demos and per-frame scoring (deeplab/eval.py --yolo-json): one JSON per
flight with each frame's detections as polygons in source-image pixels, so
overlays, videos and IoU can be computed later without the model. Runs on ALL
frames of each flight in <split dir>/unfiltered/<split>.txt (including those
the visibility filter dropped); `scored` marks frames in the filtered list,
i.e. the ones the test metrics use.

    <out>/<dataset>/<episode>.json
        {"weights": ..., "arch": ..., "frames": [
            {"stem": "000123", "scored": true,
             "dets": [{"conf": 0.97, "poly": [[x, y], ...]}, ...]}, ...]}

Architectures, all scored the same way downstream:

    yolo     Ultralytics YOLO-seg checkpoint
    valnet   VALNet state_dict (configs/main.yaml architecture), Ultralytics'
             val preprocessing: letterbox to imgsz x imgsz, RGB, /255
    deeplab  DeepLabV3+ state_dict (deeplab/model.py); each connected component
             of sigmoid > 0.5 is one detection, largest first, conf = its peak

The 960-px box cap (yolo, valnet). YOLOv8's box head reaches at most
15 bins x stride 32 = 480 px per side, so at imgsz 1280 no box is wider than
960 px, and masks are cropped to their box: a runway seen from the ground
(up to 1280 px wide) gets vertical cut sides. By default a box at the cap is
widened to the image edges before the crop (--no-widen restores the plain
crop). On test frames with runways wider than 1000 px this raised mean IoU
from 0.764 to 0.892 and left other frames unchanged; removing the crop
altogether does not work (0.37), the masks are untrained outside the box.

    python scripts/predict_flights.py --arch yolo --weights baseline/runs/xp12_combined_yolo/weights/best.pt
    python scripts/predict_flights.py --arch valnet --weights checkpoints/xp12_combined/valnet_best.pt \\
        --out results/xp12_combined/predictions_valnet
    python scripts/predict_flights.py --arch deeplab --weights deeplab/runs/xp12_combined/best.pt \\
        --out results/xp12_combined/predictions_deeplab
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CAP_FRACTION = 0.74  # box width / mask width at or above which the box is at the 960-px cap


def widen_capped_crop():
    """Make Ultralytics crop a capped box's mask to the full image width instead of the box."""
    from ultralytics.utils import ops

    crop = ops.crop_mask

    def widened(masks, boxes):
        b = boxes.clone()
        W = masks.shape[-1]
        capped = (b[:, 2] - b[:, 0]) >= CAP_FRACTION * W
        b[capped, 0], b[capped, 2] = 0, W
        return crop(masks, b)

    ops.crop_mask = widened


def mask_polys(mask):
    """Largest outer contour of a binary mask as an (N, 2) polygon, or [] if empty."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return []
    return max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)


class YoloRunner:
    def __init__(self, weights, imgsz, conf, max_det):
        from ultralytics import YOLO

        self.model, self.kw = (
            YOLO(weights),
            {"imgsz": imgsz, "half": True, "conf": conf, "max_det": max_det, "retina_masks": True, "verbose": False},
        )

    def __call__(self, paths):
        out = []
        for r in self.model.predict(paths, **self.kw):
            dets = []
            if r.masks is not None:
                for c, poly in zip(r.boxes.conf.tolist(), r.masks.xy):
                    dets.append({"conf": round(c, 4), "poly": poly.round(1).tolist()})
            out.append(dets)
        return out


class ValnetRunner:
    def __init__(self, weights, imgsz, conf, max_det):
        from omegaconf import OmegaConf
        from ultralytics.data.augment import LetterBox

        from valnet.valnet import VALNetModel

        cfg = OmegaConf.load(REPO / "configs/main.yaml")
        self.model = VALNetModel.from_ultralytics(
            REPO / cfg.yolov8_checkpoint, ch=(128, 256, 512), nc=cfg.data.num_classes, **cfg.model
        )
        # scripts/train.py saves with this alias (for v8SegmentationLoss), so the
        # checkpoint has every weight under model.* too; scripts/test.py does the same.
        self.model.model = torch.nn.ModuleList(
            [self.model.backbone_p3, self.model.backbone_p4, self.model.backbone_p5, self.model.neck, self.model.head]
        )
        self.model.load_state_dict(torch.load(weights, map_location="cuda"))
        self.model = self.model.cuda().eval()
        self.letterbox = LetterBox(new_shape=(imgsz, imgsz), scaleup=False)  # as Ultralytics' val dataset
        self.imgsz, self.conf, self.max_det = imgsz, conf, max_det

    @torch.no_grad()
    def __call__(self, paths):
        from ultralytics.utils.nms import non_max_suppression
        from ultralytics.utils.ops import process_mask

        srcs = [cv2.imread(p) for p in paths]
        lb = [self.letterbox(image=im) for im in srcs]
        x = torch.from_numpy(np.stack([im[..., ::-1].transpose(2, 0, 1) for im in lb]).copy()).cuda().float() / 255
        with torch.autocast("cuda", dtype=torch.float16):
            (decoded, proto), _ = self.model(x)
        decoded, proto = decoded.float(), proto.float()
        nc = decoded.shape[1] - 4 - proto.shape[1]
        nms = non_max_suppression(
            decoded, conf_thres=self.conf, iou_thres=0.7, nc=nc, classes=[0], multi_label=False, max_det=self.max_det
        )
        out = []
        for i, (det, src) in enumerate(zip(nms, srcs)):
            h, w = src.shape[:2]
            r = min(self.imgsz / h, self.imgsz / w, 1.0)
            nh, nw = round(h * r), round(w * r)
            top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
            dets = []
            if len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], (self.imgsz, self.imgsz), upsample=True)
                masks = (masks[:, top : top + nh, left : left + nw] > 0.5).cpu().numpy()
                for c, m in sorted(zip(det[:, 4].tolist(), masks), key=lambda t: -t[0]):
                    if (nh, nw) != (h, w):
                        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    poly = mask_polys(m)
                    if len(poly) >= 3:
                        dets.append({"conf": round(c, 4), "poly": poly.round(1).tolist()})
            out.append(dets)
        return out


class DeeplabRunner:
    def __init__(self, weights, *_):
        from deeplab.model import DeepLabV3Plus

        self.model = DeepLabV3Plus(pretrained=False).cuda().to(memory_format=torch.channels_last).eval()
        self.model.load_state_dict(torch.load(weights, map_location="cuda"))

    @torch.no_grad()
    def __call__(self, paths):
        from deeplab.data import MEAN, STD

        ims = [(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD for p in paths]
        x = torch.from_numpy(np.stack([im.transpose(2, 0, 1) for im in ims])).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            prob = torch.sigmoid(self.model(x.to(memory_format=torch.channels_last))[:, 0].float()).cpu().numpy()
        out = []
        for p in prob:
            n, lab, stats, _ = cv2.connectedComponentsWithStats((p > 0.5).astype(np.uint8))
            dets = []
            for k in sorted(range(1, n), key=lambda k: -stats[k, cv2.CC_STAT_AREA]):
                comp = lab == k
                poly = mask_polys(comp)
                if len(poly) >= 3:
                    dets.append({"conf": round(float(p[comp].max()), 4), "poly": poly.round(1).tolist()})
            out.append(dets)
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--arch", choices=["yolo", "valnet", "deeplab"], default="yolo")
    ap.add_argument("--dir", default="datasets/xp12_splits/combined")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="results/xp12_combined/predictions_yolo")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--max-det", type=int, default=5)
    ap.add_argument("--no-widen", action="store_true", help="plain box crop, i.e. keep the 960-px cut")
    ap.add_argument(
        "--gpu-fraction",
        type=float,
        default=None,
        help="cap this process's GPU memory, e.g. when sharing the GPU with training",
    )
    args = ap.parse_args()

    if args.gpu_fraction:
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
    if args.arch != "deeplab" and not args.no_widen:
        widen_capped_crop()

    d = Path(args.dir)
    scored = {l.strip() for l in (d / f"{args.split}.txt").read_text().splitlines() if l.strip()}
    flights = defaultdict(list)
    for l in (d / "unfiltered" / f"{args.split}.txt").read_text().splitlines():
        if l.strip():
            flights[Path(l.strip()).parents[1]].append(l.strip())

    runner = {"yolo": YoloRunner, "valnet": ValnetRunner, "deeplab": DeeplabRunner}[args.arch](
        args.weights, args.imgsz, args.conf, args.max_det
    )
    out = Path(args.out)
    for i, (ep, imgs) in enumerate(sorted(flights.items())):
        dst = out / ep.parent.name / f"{ep.name}.json"
        if dst.exists():
            continue
        frames = []
        imgs = sorted(imgs)
        # Chunk by hand: given a list, YOLO.predict() loads the whole list as one batch.
        for k in range(0, len(imgs), args.batch):
            chunk = imgs[k : k + args.batch]
            for dets, img in zip(runner(chunk), chunk):
                frames.append({"stem": Path(img).stem, "scored": img in scored, "dets": dets})
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(
            json.dumps(
                {
                    "weights": args.weights,
                    "arch": args.arch,
                    "imgsz": args.imgsz,
                    "widened": args.arch != "deeplab" and not args.no_widen,
                    "frames": frames,
                }
            )
        )
        print(f"[{i + 1}/{len(flights)}] {dst}  {len(frames)} frames", flush=True)


if __name__ == "__main__":
    main()
