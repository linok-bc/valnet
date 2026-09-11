import os
import sys
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
import ultralytics.data.build as build_module

from xtracknet.xtrackdataset import XTrackDataset
from xtracknet.xtracknet import XTrackNet

main_dir = Path(os.path.dirname(os.path.realpath(__file__)), '..').absolute()
omegaCfg = OmegaConf.load(Path(main_dir, 'configs/xtrack_main.yaml'))

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

cfg = get_cfg()
cfg.data = omegaCfg.data.config
cfg.imgsz = omegaCfg.data.image_size
cfg.batch = omegaCfg.train.batch_size
cfg.task = "segment"
cfg.fliplr = 0.0   # would invalidate cross-track sign
cfg.mosaic = 0.0
cfg.translate = 0.0
cfg.scale = 0.0
cfg.degrees = 0.0
cfg.shear = 0.0
cfg.perspective = 0.0
cfg.erasing = 0.0
cfg.hsv_h = 0.015
cfg.hsv_s = 0.4
cfg.hsv_v = 0.4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Patch dataloader to use XTrackDataset
build_module.YOLODataset = XTrackDataset

data_dict = check_det_dataset(cfg.data)
train_dataset = build_yolo_dataset(cfg=cfg, img_path=data_dict["train"], batch=cfg.batch,
                                    data=data_dict, mode="train", rect=False)
val_dataset = build_yolo_dataset(cfg=cfg, img_path=data_dict["val"], batch=cfg.batch,
                                  data=data_dict, mode="val", rect=False)

# Compute target normalization from training data
xtrack_vals = []
for f in train_dataset.im_files:
    p = Path(f).parents[1] / "position" / f"{Path(f).stem}.txt"
    xtrack_vals.append(float(p.read_text().split()[2]))
xtrack_std = torch.tensor(xtrack_vals).std().item()
print(f"Cross-track std: {xtrack_std:.2f} m  (over {len(xtrack_vals)} samples)")

weights = [max(abs(x), 0.5) ** 0.5 for x in xtrack_vals]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.batch,
    sampler=sampler,
    num_workers=omegaCfg.train.num_workers,
    pin_memory=True,
    collate_fn=train_dataset.collate_fn,  # critical — uses XTrackDataset.collate_fn
    drop_last=True,
)
val_loader   = build_dataloader(val_dataset,   batch=cfg.batch, workers=omegaCfg.train.num_workers, shuffle=False)
os.makedirs("checkpoints", exist_ok=True)

model = XTrackNet().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=omegaCfg.train.epochs)

logdir = Path(main_dir, omegaCfg.train.logdir, "xtrack")
writer = SummaryWriter(log_dir=logdir)

best_mae = float('inf')

for epoch in tqdm(range(omegaCfg.train.epochs), desc = f'Total epochs = {omegaCfg.train.epochs}'):
    # --- Train ---
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch #{epoch}", file=sys.stdout, leave=False):
        rgb   = batch["img"].to(device).float() / 255.0
        # Build single-channel binary mask from per-instance masks: union over instances
        masks = batch["masks"].to(device).float()         # [B, H, W]
        mask = (masks.unsqueeze(1) > 0).float() if masks.dim() == 3 else (masks.sum(1, keepdim=True) > 0).float()
        roll  = batch["roll"].to(device)
        pitch = batch["pitch"].to(device)
        alt   = batch["alt"].to(device)
        gt    = batch["xtrack"].to(device).unsqueeze(-1) / xtrack_std

        pred = model(rgb, mask, roll, pitch, alt)
        loss = F.smooth_l1_loss(pred, gt)

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        train_loss += loss.item()

    sched.step()
    writer.add_scalar("train/loss", train_loss / len(train_loader), epoch)

    # --- Val ---
    if (epoch + 1) % omegaCfg.train.val_freq == 0:
        model.eval()
        errs = []
        with torch.no_grad():
            for batch in val_loader:
                rgb   = batch["img"].to(device).float() / 255.0
                masks = batch["masks"].to(device).float()
                mask = (masks.unsqueeze(1) > 0).float() if masks.dim() == 3 else (masks.sum(1, keepdim=True) > 0).float()
                roll  = batch["roll"].to(device)
                pitch = batch["pitch"].to(device)
                alt   = batch["alt"].to(device)
                gt_m  = batch["xtrack"].to(device)

                pred_m = model(rgb, mask, roll, pitch, alt).squeeze(-1) * xtrack_std
                errs.append((pred_m - gt_m).abs().cpu())
        errs = torch.cat(errs)
        writer.add_scalar("val/mae_m", errs.mean().item(), epoch)
        writer.add_scalar("val/median_m", errs.median().item(), epoch)
        writer.add_scalar("val/pred_std", pred_m.std().item(), epoch)
        writer.add_scalar("val/pred_mean", pred_m.mean().item(), epoch)

        val_mae = errs.mean().item()
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save({"model": model.state_dict(), "xtrack_std": xtrack_std, "epoch": epoch}, "checkpoints/xtrack_best.pt")

    if (epoch + 1) % omegaCfg.train.save_freq == 0:
        torch.save({
            "model": model.state_dict(),
            "xtrack_std": xtrack_std,
            "epoch": epoch,
        }, f"checkpoints/xtrack_epoch{epoch+1}.pt")

torch.save({
    "model": model.state_dict(),
    "xtrack_std": xtrack_std,
    "epoch": omegaCfg.train.epochs - 1,
}, "checkpoints/xtrack_final.pt")

writer.close()
