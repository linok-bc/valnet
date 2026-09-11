import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
import torch
from torch.utils.tensorboard import SummaryWriter

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
import ultralytics.data.build as build_module

from xtracknet.xtrackdataset import XTrackDataset
from xtracknet.xtracknet import XTrackNet


def evaluate_xtrack(model, loader, xtrack_std, device):
    """Run model over loader, return per-sample errors and metadata."""
    model.eval()
    all_pred_m = []
    all_gt_m   = []
    all_alt    = []
    all_roll   = []
    all_pitch  = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing", file=sys.stdout):
            rgb   = batch["img"].to(device).float() / 255.0
            masks = batch["masks"].to(device).float()
            mask = (masks.unsqueeze(1) > 0).float() if masks.dim() == 3 else (masks.sum(1, keepdim=True) > 0).float()
            roll  = batch["roll"].to(device)
            pitch = batch["pitch"].to(device)
            alt   = batch["alt"].to(device)
            gt_m  = batch["xtrack"].to(device)

            pred_m = model(rgb, mask, roll, pitch, alt).squeeze(-1) * xtrack_std

            all_pred_m.append(pred_m.cpu())
            all_gt_m.append(gt_m.cpu())
            all_alt.append(alt.cpu())
            all_roll.append(roll.cpu())
            all_pitch.append(pitch.cpu())

    return {
        "pred":  torch.cat(all_pred_m),
        "gt":    torch.cat(all_gt_m),
        "alt":   torch.cat(all_alt),
        "roll":  torch.cat(all_roll),
        "pitch": torch.cat(all_pitch),
    }


def report_metrics(results, alt_buckets=(0, 50, 100, float('inf'))):
    """Print overall and altitude-bucketed metrics."""
    pred = results["pred"]
    gt   = results["gt"]
    alt  = results["alt"]

    err = pred - gt
    abs_err = err.abs()

    # Overall
    print("\n=== Overall ===")
    print(f"  N samples : {len(err)}")
    print(f"  MAE       : {abs_err.mean():.2f} m")
    print(f"  Median AE : {abs_err.median():.2f} m")
    print(f"  RMSE      : {(err ** 2).mean().sqrt():.2f} m")
    print(f"  Bias      : {err.mean():+.2f} m  (predicted - gt)")
    print(f"  P90 AE    : {abs_err.quantile(0.90):.2f} m")
    print(f"  P95 AE    : {abs_err.quantile(0.95):.2f} m")
    print(f"  Max AE    : {abs_err.max():.2f} m")

    # Per-altitude buckets
    print("\n=== By altitude ===")
    print(f"  {'range (m)':>15}  {'N':>5}  {'MAE':>7}  {'Median':>7}  {'RMSE':>7}  {'Bias':>7}")
    for lo, hi in zip(alt_buckets[:-1], alt_buckets[1:]):
        m = (alt >= lo) & (alt < hi)
        n = m.sum().item()
        if n == 0:
            continue
        e = err[m]
        ae = abs_err[m]
        label = f"[{lo:.0f}, {hi:.0f})" if hi != float('inf') else f"[{lo:.0f}, inf)"
        print(f"  {label:>15}  {n:>5}  {ae.mean():>7.2f}  {ae.median():>7.2f}  "
              f"{(e**2).mean().sqrt():>7.2f}  {e.mean():>+7.2f}")

def write_csv(results, path):
    """Dump per-sample results for later analysis."""
    import csv
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["pred_m", "gt_m", "abs_err_m", "alt_m", "roll_rad", "pitch_rad"])
        for i in range(len(results["pred"])):
            w.writerow([
                f"{results['pred'][i]:.4f}",
                f"{results['gt'][i]:.4f}",
                f"{abs(results['pred'][i] - results['gt'][i]):.4f}",
                f"{results['alt'][i]:.4f}",
                f"{results['roll'][i]:.6f}",
                f"{results['pitch'][i]:.6f}",
            ])
    print(f"\nPer-sample results written to {path}")


def write_tb_plots(results, writer):
    """Log diagnostic plots/histograms to TensorBoard."""
    err = results["pred"] - results["gt"]
    writer.add_histogram("test/err_signed_m", err, 0)
    writer.add_histogram("test/err_abs_m", err.abs(), 0)
    writer.add_histogram("test/gt_m", results["gt"], 0)
    writer.add_histogram("test/pred_m", results["pred"], 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to xtrack checkpoint .pt file")
    parser.add_argument("--csv", default=None, help="Optional path to write per-sample CSV")
    parser.add_argument("--split", default="test", choices=["test", "val"],
                        help="Which split to evaluate on")
    args = parser.parse_args()

    main_dir = Path(os.path.dirname(os.path.realpath(__file__)), '..').absolute()
    omegaCfg = OmegaConf.load(Path(main_dir, 'configs/main.yaml'))

    cfg = get_cfg()
    cfg.data = omegaCfg.data.config
    cfg.imgsz = omegaCfg.data.image_size
    cfg.batch = omegaCfg.test.batch_size
    cfg.task = "segment"
    # Disable all augmentations for testing
    cfg.fliplr = 0.0
    cfg.mosaic = 0.0
    cfg.translate = 0.0
    cfg.scale = 0.0
    cfg.degrees = 0.0
    cfg.shear = 0.0
    cfg.perspective = 0.0
    cfg.erasing = 0.0
    cfg.hsv_h = 0.0
    cfg.hsv_s = 0.0
    cfg.hsv_v = 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    build_module.YOLODataset = XTrackDataset
    data_dict = check_det_dataset(cfg.data)

    test_dataset = build_yolo_dataset(
        cfg=cfg, img_path=data_dict[args.split], batch=cfg.batch,
        data=data_dict, mode="val", rect=False,
    )
    test_loader = build_dataloader(
        dataset=test_dataset,
        batch=omegaCfg.test.batch_size,
        workers=omegaCfg.test.num_workers,
        shuffle=False,
    )

    # Load model + saved normalization
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = XTrackNet().to(device)
    model.load_state_dict(ckpt["model"])
    xtrack_std = ckpt["xtrack_std"]
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    print(f"Cross-track std (from training): {xtrack_std:.2f} m")
    print(f"Evaluating on '{args.split}' split: {len(test_dataset)} samples")

    results = evaluate_xtrack(model, test_loader, xtrack_std, device)
    report_metrics(results)

    # TensorBoard diagnostics
    logdir = Path(main_dir, omegaCfg.train.logdir, "xtrack_test")
    writer = SummaryWriter(log_dir=logdir)
    write_tb_plots(results, writer)
    writer.close()
    print(f"\nDiagnostic histograms written to {logdir}")

    if args.csv:
        write_csv(results, args.csv)


if __name__ == "__main__":
    main()
