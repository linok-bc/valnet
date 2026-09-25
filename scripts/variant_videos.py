"""
Per-flight segmentation videos for the four focus variants.

1. Runs every variant over every xp12 TEST flight and records, per frame, the
   top-1 runway detection's mask IoU against the ground-truth quad at the
   native 1280x720 resolution.
2. Picks one flight — the one whose mean IoU (averaged over the variants) is
   closest to the median flight, i.e. a typical approach, not a flattering one.
3. Renders one H.264 video per variant of that flight: prediction filled in
   the variant's colour, ground truth outlined in white, IoU/confidence and
   range to threshold per frame.

All variants go through the same letterbox, forward pass, NMS and mask
decoding (the path valnet/evaluate_map.py uses), so the videos differ only by
model. Detections below `--conf` count as "no detection" (IoU 0).

Usage:
    python scripts/variant_videos.py                    # -> results/variant_report/
    python scripts/variant_videos.py --flight 000140    # force a flight
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression

from valnet.valnet import VALNetModel

REPO = Path(__file__).resolve().parents[1]

# name, kind, checkpoint, overlay colour (RGB; categorical slots 1-4)
VARIANTS = [
    ("YOLOv8s-seg stock", "yolo", "baseline/runs/stock/weights/best.pt", (42, 120, 214)),
    ("VALNet", "valnet", "checkpoints/valnet_best.pt", (235, 104, 52)),
    ("VALNet + COCO pretraining", "valnet", "checkpoints/valnet_coco_ft/valnet_best.pt", (27, 175, 122)),
    ("YOLO26s-seg stock", "yolo", "baseline/runs/yolo26_stock/weights/best.pt", (237, 161, 0)),
]


def load_model(kind, ckpt, device):
    if kind == "yolo":
        return YOLO(str(REPO / ckpt)).model.to(device).float().eval()
    state = torch.load(REPO / ckpt, map_location=device)
    arch = dict(
        cem_depthwise=state["neck.cem_p3.lowpass_conv.weight"].shape[1] == 1,
        afpn_stage1="neck.afpn.layer1_1.conv.weight" in state,
    )
    nc = state["head.cv3.0.2.weight"].shape[0]
    m = VALNetModel.from_ultralytics(REPO / "checkpoints/yolov8s-seg.pt", ch=(128, 256, 512), nc=nc, **arch)
    m.model = nn.ModuleList([m.backbone_p3, m.backbone_p4, m.backbone_p5, m.neck, m.head])
    m.load_state_dict(state)
    return m.to(device).eval()


def gt_mask(label_file, h, w):
    """Rasterize the label polygon(s) at native resolution."""
    mask = np.zeros((h, w), np.uint8)
    for line in label_file.read_text().split("\n"):
        v = line.split()
        if len(v) < 7:
            continue
        pts = np.array(v[1:], float).reshape(-1, 2) * [w, h]
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask


@torch.no_grad()
def predict(model, frames, device, conf):
    """Top-1 runway mask per frame at native resolution, plus its confidence."""
    lb = LetterBox((640, 640), auto=False, scaleup=False)
    x = np.stack([lb(image=f)[..., ::-1] for f in frames])                 # BGR -> RGB
    x = torch.from_numpy(x.copy()).permute(0, 3, 1, 2).to(device).float() / 255
    (decoded, proto), _ = model(x)
    end2end = decoded.shape[-1] == 6 + proto.shape[1]
    dets = non_max_suppression(decoded, conf_thres=conf, iou_thres=0.7, nc=decoded.shape[1] - 4 - proto.shape[1],
                               classes=[0], multi_label=False, max_det=300, end2end=end2end)
    out = []
    for i, d in enumerate(dets):
        h, w = frames[i].shape[:2]
        if not len(d):
            out.append((np.zeros((h, w), np.uint8), 0.0))
            continue
        d = d[d[:, 4].argmax()][None]
        m = ops.process_mask(proto[i], d[:, 6:], d[:, :4], shape=(640, 640), upsample=True)   # [1,640,640]
        m = ops.scale_masks(m[None].float(), (h, w))[0, 0]                                       # undo letterbox
        out.append(((m > 0.5).cpu().numpy().astype(np.uint8), float(d[0, 4])))
    return out


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def flights():
    files = (REPO / "datasets/xp12_dataset/split_test.txt").read_text().split()
    by = {}
    for f in files:
        by.setdefault(Path(f).parents[1].name, []).append(Path(f))
    return {k: sorted(v) for k, v in sorted(by.items())}


def score_all(models, device, conf, batch):
    """Per-frame IoU for every variant on every test flight."""
    scores = {}
    for fid, imgs in flights().items():
        scores[fid] = {name: [] for name, *_ in VARIANTS}
        for s in range(0, len(imgs), batch):
            chunk = imgs[s:s + batch]
            frames = [cv2.imread(str(p)) for p in chunk]
            gts = [gt_mask(p.parents[1] / "labels" / (p.stem + ".txt"), *f.shape[:2]) for p, f in zip(chunk, frames)]
            for (name, *_), model in zip(VARIANTS, models):
                for (pm, _), g in zip(predict(model, frames, device, conf), gts):
                    scores[fid][name].append(iou(pm, g))
        print(fid, "  ".join(f"{n}: {np.mean(v):.3f}" for n, v in scores[fid].items()))
    return scores


def font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf"):
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def render(fid, name, model, color, device, conf, out_path, fps, width):
    imgs = flights()[fid]
    ep = imgs[0].parents[1]
    runway = (ep / "meta.txt").read_text().split("\n")[0].strip()
    f_big, f_small = font(26), font(20)
    h_out = int(round(width * 720 / 1280 / 2) * 2)
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{h_out}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "26",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)],
        stdin=subprocess.PIPE)
    ious = []
    for s in range(0, len(imgs), 16):
        chunk = imgs[s:s + 16]
        frames = [cv2.imread(str(p)) for p in chunk]
        for k, (p, f, (pm, c)) in enumerate(zip(chunk, frames, predict(model, frames, device, conf))):
            g = gt_mask(ep / "labels" / (p.stem + ".txt"), *f.shape[:2])
            v = iou(pm, g)
            ious.append(v)
            rgb = f[..., ::-1].astype(np.float32)
            sel = pm.astype(bool)
            rgb[sel] = 0.5 * rgb[sel] + 0.5 * np.array(color, np.float32)
            img = rgb.astype(np.uint8).copy()
            cnt, _ = cv2.findContours(pm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnt, -1, color, 2, cv2.LINE_AA)
            gc, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, gc, -1, (255, 255, 255), 2, cv2.LINE_AA)
            pos = np.array((ep / "position" / (p.stem + ".txt")).read_text().split(), float)
            im = Image.fromarray(img)
            d = ImageDraw.Draw(im, "RGBA")
            d.rectangle((0, 0, 1280, 78), fill=(12, 14, 18, 170))
            d.rectangle((18, 18, 34, 34), fill=color)
            d.text((44, 12), name, font=f_big, fill=(255, 255, 255))
            d.text((18, 46), f"{runway}   ·   frame {s + k + 1}/{len(imgs)}   ·   {-pos[0]:,.0f} m to threshold"
                             f"   ·   {pos[1]:.0f} m above runway", font=f_small, fill=(210, 212, 216))
            right = f"IoU {v:.2f}   conf {c:.2f}" if c > 0 else "no detection"
            d.text((1262, 30), right, font=f_big, fill=(255, 255, 255), anchor="ra")
            d.rectangle((0, 690, 1280, 720), fill=(12, 14, 18, 150))
            d.text((18, 694), "filled: prediction   ·   white outline: ground truth", font=f_small,
                   fill=(210, 212, 216))
            enc.stdin.write(np.asarray(im.resize((width, h_out), Image.LANCZOS)).tobytes())
    enc.stdin.close()
    enc.wait()
    return ious


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flight", default=None, help="test flight id (default: the median flight)")
    ap.add_argument("--conf", type=float, default=0.25, help="detection threshold (Ultralytics predict default)")
    ap.add_argument("--out", default="results/variant_report")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--width", type=int, default=960)
    a = ap.parse_args()
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = [load_model(kind, ckpt, device) for _, kind, ckpt, _ in VARIANTS]

    scores = score_all(models, device, a.conf, batch=16)
    flight_mean = {f: float(np.mean([np.mean(v) for v in s.values()])) for f, s in scores.items()}
    median = float(np.median(list(flight_mean.values())))
    fid = a.flight or min(flight_mean, key=lambda f: abs(flight_mean[f] - median))
    print(f"flight {fid}: mean IoU {flight_mean[fid]:.3f} (median flight {median:.3f})")

    videos = []
    for (name, kind, ckpt, color), model in zip(VARIANTS, models):
        path = out / f"{name.lower().replace(' + ', '_').replace(' ', '_').replace('-', '')}.mp4"
        ious = render(fid, name, model, color, device, a.conf, path, a.fps, a.width)
        videos.append(dict(name=name, file=path.name, checkpoint=ckpt, mean_iou=float(np.mean(ious)),
                           frames_iou_ge_075=float(np.mean(np.array(ious) >= 0.75)), iou=ious))
        print(f"  {name}: {path.name}  mean IoU {np.mean(ious):.3f}")

    ep = REPO / "datasets/xp12_dataset" / fid
    ranges = [-float((ep / "position" / (p.stem + ".txt")).read_text().split()[0]) for p in flights()[fid]]
    json.dump(dict(flight=fid, runway=(ep / "meta.txt").read_text().split("\n")[0].strip(), frames=len(ranges),
                   range_m=ranges, flight_mean_iou=flight_mean, median_flight_iou=median, conf=a.conf,
                   videos=videos, test_mean_iou={n: float(np.mean([x for s in scores.values() for x in s[n]]))
                                                 for n, *_ in VARIANTS}),
              open(out / "videos.json", "w"))


if __name__ == "__main__":
    main()
