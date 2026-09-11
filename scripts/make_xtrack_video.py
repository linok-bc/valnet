"""
Generate a side-by-side video showing predicted vs ground-truth cross-track
distance for a landing sequence.

Expects a sequence folder structured like:
    <seq_dir>/
        images/    *.png   (or *.jpg)
        labels/    *.txt   (YOLO-format segmentation labels)
        position/  *.txt   "<x> <y> <z>"
        pose/      *.txt   "<roll> <pitch> <yaw>" in degrees

For each frame, the model is run, and an output frame is composed showing:
  - Original image with predicted/GT centerline offsets annotated
  - Time-series plot of predicted vs GT cross-track up to current frame

Usage:
    python make_xtrack_video.py \
        --checkpoint checkpoints/xtrack_best.pt \
        --sequence /path/to/run_dir \
        --output xtrack_video.mp4 \
        --fps 15
"""

import os
import sys
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xtracknet.xtracknet import XTrackNet


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def load_label_mask(label_path: Path, h: int, w: int) -> np.ndarray:
    """Load YOLO-format polygon labels and rasterize into a binary mask.

    YOLO seg label format per line:
        class_id x1 y1 x2 y2 ... xN yN     (normalized)
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    if not label_path.exists():
        return mask
    text = label_path.read_text().strip()
    if not text:
        return mask
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        coords = list(map(float, parts[1:]))
        pts = np.array([
            [coords[i] * w, coords[i + 1] * h]
            for i in range(0, len(coords), 2)
        ], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def letterbox(img: np.ndarray, target_size: int = 640) -> np.ndarray:
    """Resize + pad to square (matches YOLO's letterbox)."""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_h = target_size - nh
    pad_w = target_size - nw
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    if img.ndim == 3:
        return cv2.copyMakeBorder(resized, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=(114, 114, 114))
    else:
        return cv2.copyMakeBorder(resized, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=0)


def load_sequence(seq_dir: Path):
    """Yield (frame_idx, rgb_orig, rgb_letterboxed, mask_letterboxed, roll, pitch, alt, xtrack_gt)."""
    img_dir = seq_dir / "images"
    label_dir = seq_dir / "labels"
    pose_dir = seq_dir / "pose"
    pos_dir = seq_dir / "position"

    image_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTS])
    if not image_files:
        raise FileNotFoundError(f"No images found in {img_dir}")

    for i, img_path in enumerate(image_files[250:]):
        stem = img_path.stem

        # Image
        img = cv2.imread(str(img_path))  # BGR
        if img is None:
            print(f"  [skip] could not read {img_path}")
            continue
        h, w = img.shape[:2]
        rgb_orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Mask
        label_path = label_dir / f"{stem}.txt"
        mask = load_label_mask(label_path, h, w)

        # Letterbox both
        rgb_lb = letterbox(rgb_orig, 640)
        mask_lb = letterbox(mask, 640)

        # Pose / position
        pose_path = pose_dir / f"{stem}.txt"
        pos_path = pos_dir / f"{stem}.txt"
        if not pose_path.exists() or not pos_path.exists():
            print(f"  [skip] missing pose/position for {stem}")
            continue
        rpy = [float(v) for v in pose_path.read_text().split()]
        position = [float(v) for v in pos_path.read_text().split()]

        roll = math.radians(rpy[0])
        pitch = math.radians(rpy[1])
        alt = position[1]
        xtrack_gt = position[2]

        yield i, rgb_orig, rgb_lb, mask_lb, roll, pitch, alt, xtrack_gt


def render_frame(rgb_orig: np.ndarray, mask_lb: np.ndarray, frame_idx: int, n_total: int,
                 pred: float, gt: float, alt: float, history_pred, history_gt,
                 fig_size=(12, 6)) -> np.ndarray:
    """Compose a single output frame: image (left) + plot (right)."""
    fig, (ax_img, ax_plot) = plt.subplots(1, 2, figsize=fig_size,
                                          gridspec_kw={"width_ratios": [1.5, 1]})

    # --- Left: image with overlay ---
    ax_img.imshow(rgb_orig)
    ax_img.set_xticks([]); ax_img.set_yticks([])

    # Mask overlay (resize letterboxed mask back conceptually — just show on top of orig)
    # For simplicity, we display original image without mask overlay since the letterbox
    # transform makes it awkward to map back. Instead, annotate with text.
    text = (
        f"Frame {frame_idx + 1}/{n_total}\n"
        f"Altitude:    {alt:7.1f} m\n"
        f"Pred xtrack: {pred:+7.2f} m\n"
        f"GT   xtrack: {gt:+7.2f} m\n"
        f"Error:       {pred - gt:+7.2f} m"
    )
    ax_img.text(0.02, 0.98, text, transform=ax_img.transAxes,
                fontsize=11, fontfamily="monospace", verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.7),
                color="white")

    # --- Right: time-series plot ---
    frames = list(range(len(history_pred)))
    ax_plot.plot(frames, history_gt, label="GT", color="tab:green", linewidth=2)
    ax_plot.plot(frames, history_pred, label="Pred", color="tab:orange", linewidth=2)
    ax_plot.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax_plot.set_xlabel("Frame")
    ax_plot.set_ylabel("Cross-track (m)")
    ax_plot.set_title("Cross-track over sequence")
    ax_plot.legend(loc="upper left")
    ax_plot.grid(True, alpha=0.3)

    # Auto-ranged y-axis with a small margin, but include both pred and gt
    if history_pred and history_gt:
        all_vals = history_pred + history_gt
        ymin, ymax = min(all_vals), max(all_vals)
        margin = max(1.0, 0.2 * (ymax - ymin))
        ax_plot.set_ylim(ymin - margin, ymax + margin)
    ax_plot.set_xlim(0, n_total)

    plt.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    frame = buf[..., :3].copy()
    plt.close(fig)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sequence", required=True, help="Folder with images/labels/pose/position subdirs")
    parser.add_argument("--output", default="xtrack_video.mp4")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--device", default=None, help="cuda or cpu (autodetect by default)")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = XTrackNet().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    xtrack_std = ckpt["xtrack_std"]
    print(f"Loaded checkpoint epoch {ckpt.get('epoch', '?')}, xtrack_std = {xtrack_std:.2f}")

    seq_dir = Path(args.sequence)

    # Pre-count for progress + plot x-axis
    n_total = sum(1 for f in (seq_dir / "images").iterdir() if f.suffix.lower() in IMG_EXTS) - 250
    print(f"Sequence: {n_total} frames")

    # Determine output frame size by rendering one dummy frame
    sample_frame = None
    history_pred, history_gt = [], []

    writer = None

    with torch.no_grad():
        for i, rgb_orig, rgb_lb, mask_lb, roll, pitch, alt, gt in load_sequence(seq_dir):
            # To tensor (BCHW), normalized
            rgb_t  = torch.from_numpy(rgb_lb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            mask_t = torch.from_numpy(mask_lb).unsqueeze(0).unsqueeze(0).float().to(device)

            roll_t  = torch.tensor([roll],  device=device)
            pitch_t = torch.tensor([pitch], device=device)
            alt_t   = torch.tensor([alt],   device=device)

            pred_norm = model(rgb_t, mask_t, roll_t, pitch_t, alt_t).item()
            pred_m = pred_norm * xtrack_std

            history_pred.append(pred_m)
            history_gt.append(gt)

            frame = render_frame(rgb_orig, mask_lb, i, n_total, pred_m, gt, alt,
                                 history_pred, history_gt)

            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open {args.output} for writing")

            # OpenCV expects BGR
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if (i + 1) % 20 == 0 or (i + 1) == n_total:
                print(f"  {i + 1}/{n_total}  pred={pred_m:+6.2f}  gt={gt:+6.2f}  "
                      f"err={pred_m - gt:+6.2f}")

    if writer is not None:
        writer.release()

    # Final summary
    if history_pred:
        errs = np.abs(np.array(history_pred) - np.array(history_gt))
        print(f"\nDone. Wrote {args.output}")
        print(f"Sequence MAE:    {errs.mean():.2f} m")
        print(f"Sequence median: {np.median(errs):.2f} m")
        print(f"Sequence max AE: {errs.max():.2f} m")


if __name__ == "__main__":
    main()
