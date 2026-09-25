import os
import sys
import math
from copy import copy
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.cfg import get_cfg

from valnet.valnet import VALNetModel, resize_head_classes
from valnet.evaluate_map import evaluate_valnet
from valnet.ema import ModelEMA

"""
# 1. Build model
"""
main_dir = Path(os.path.dirname(os.path.realpath(__file__)), '..').absolute()
omegaCfgPath = Path(main_dir, 'configs/main.yaml')
# dotlist overrides, e.g. `python scripts/train.py train.epochs=1 train.fraction=0.05`
omegaCfg = OmegaConf.merge(OmegaConf.load(omegaCfgPath), OmegaConf.from_cli())

cfg = get_cfg() # from Ultralytics
cfg.data = omegaCfg.data.config
cfg.imgsz = omegaCfg.data.image_size
cfg.batch = omegaCfg.train.batch_size
cfg.task = "segment"
cfg.fliplr = omegaCfg.train.fliplr  # 0 for xp12: a flip mirrors cross-track
cfg.mosaic = omegaCfg.train.mosaic  # composites 4 images; see configs/main.yaml
cfg.fraction = omegaCfg.train.fraction  # of the train split; <1 only for smoke tests

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
init_from = omegaCfg.train.init_from
# When starting from a VALNet checkpoint, build the head at the checkpoint's
# class count first (80 after COCO pretraining) and resize after loading.
valnet = VALNetModel.from_ultralytics(
    Path(main_dir, omegaCfg.yolov8_checkpoint), ch=(128, 256, 512),
    nc=None if init_from else omegaCfg.data.num_classes,
    **omegaCfg.model,
)
valnet = valnet.to(device)

# for v8SegmentationLoss; there are weights that can be adjusted
valnet.args = get_cfg()
valnet.model = nn.ModuleList([valnet.backbone_p3, valnet.backbone_p4, valnet.backbone_p5, valnet.neck, valnet.head])

if init_from:
    valnet.load_state_dict(torch.load(Path(main_dir, init_from), map_location=device))
    if valnet.head.nc != omegaCfg.data.num_classes:
        resize_head_classes(valnet.head, omegaCfg.data.num_classes)
    print(f"initialized from {init_from}")

# Freeze whole parts, e.g. [backbone, head] to pretrain only CEM+AFPN+OAM.
# Their BatchNorm layers are also held in eval mode every epoch (see the
# training loop), so running statistics stay at their pretrained values.
FREEZABLE = {
    "backbone": [valnet.backbone_p3, valnet.backbone_p4, valnet.backbone_p5],
    "neck": [valnet.neck, valnet.oams],
    "head": [valnet.head],
}
frozen = [m for part in omegaCfg.train.freeze for m in FREEZABLE[part]]
for m in frozen:
    m.requires_grad_(False)
if frozen:
    print(f"frozen: {list(omegaCfg.train.freeze)}")

"""
# 2. Build datasets, dataloaders, and validators using Ultralytics
"""
logdir = Path(main_dir, omegaCfg.train.logdir)
os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter(log_dir=logdir)

data_dict = check_det_dataset(cfg.data)

train_dataset = build_yolo_dataset(
    cfg=cfg, img_path=data_dict["train"], batch=cfg.batch,
    data=data_dict, mode="train", rect=False,
)
val_dataset = build_yolo_dataset(
    cfg=cfg, img_path=data_dict["val"], batch=cfg.batch,
    data=data_dict, mode="val", rect=False,
)
train_loader = build_dataloader(
    dataset=train_dataset,
    batch=cfg.batch,
    workers=omegaCfg.train.num_workers,
    shuffle=True,
)
val_loader = build_dataloader(
    dataset=val_dataset,
    batch=cfg.batch,
    workers=omegaCfg.train.num_workers,
    shuffle=False,
)

"""
# 3. Training setup (matching Table 5 in paper)
"""
# Parameter groups split on two axes:
#   - which part of the network (each gets its own LR, see configs/main.yaml)
#   - whether the tensor should be weight-decayed. Norm weights and biases are
#     1-D; decaying them fights the normalization, so they get wd=0. This is
#     what Ultralytics' own optimizer builder does.
LR_MULT = {
    "backbone": omegaCfg.optim.backbone_lr_mult,
    "neck":     omegaCfg.optim.neck_lr_mult,
    "head":     omegaCfg.optim.head_lr_mult,
}


def part_of(param_name):
    """Map a parameter name onto one of the three LR groups."""
    if param_name.startswith("backbone"):
        return "backbone"
    if param_name.startswith(("neck", "oams")):
        return "neck"
    if param_name.startswith("head"):
        return "head"
    return None


# named_parameters() deduplicates, so the backbone_p3/p4/p5 slices and the
# `valnet.model` ModuleList do not re-list tensors already seen under
# `backbone.*` — every parameter lands in exactly one group.
buckets = {}
for pname, param in valnet.named_parameters():
    if not param.requires_grad:
        continue
    part = part_of(pname)
    if part is None:
        raise ValueError(f"parameter not assigned to an LR group: {pname}")
    buckets.setdefault((part, param.ndim > 1), []).append(param)

param_groups = [
    {
        "params": params,
        "lr": omegaCfg.optim.initial_lr * LR_MULT[part],
        "weight_decay": omegaCfg.optim.weight_decay if decay else 0.0,
        "group_name": f"{part}/{'decay' if decay else 'no_decay'}",
    }
    for (part, decay), params in sorted(buckets.items())
]

grouped = sum(p.numel() for g in param_groups for p in g["params"])
trainable = sum(p.numel() for p in valnet.parameters() if p.requires_grad)
assert grouped == trainable, f"param groups cover {grouped} of {trainable} trainable params"

print("Optimizer parameter groups:")
for g in param_groups:
    print(f"  {g['group_name']:<18s} lr={g['lr']:.2e}  wd={g['weight_decay']:.1e}  "
          f"{sum(p.numel() for p in g['params']):>10,d} params")

optimizer = torch.optim.SGD(
    param_groups,
    lr=omegaCfg.optim.initial_lr,
    momentum=omegaCfg.optim.momentum,
)

# LR schedule: linear warmup then cosine decay, stepped once per ITERATION.
# LambdaLR returns a multiplier on each group's own base LR, so the per-part
# ratios above are preserved across the whole run.
iters_per_epoch = len(train_loader)
total_iters = omegaCfg.train.epochs * iters_per_epoch
warmup_iters = int(omegaCfg.optim.warmup_epochs * iters_per_epoch)
lr_floor = omegaCfg.optim.final_lr / omegaCfg.optim.initial_lr


def lr_multiplier(it):
    if warmup_iters > 0 and it < warmup_iters:
        return (it + 1) / warmup_iters
    progress = (it - warmup_iters) / max(1, total_iters - warmup_iters)
    return lr_floor + (1.0 - lr_floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def momentum_at(it):
    """Ramp momentum across the warmup window, as Ultralytics does."""
    if warmup_iters <= 0 or it >= warmup_iters:
        return omegaCfg.optim.momentum
    f = (it + 1) / warmup_iters
    return omegaCfg.optim.warmup_momentum + f * (
        omegaCfg.optim.momentum - omegaCfg.optim.warmup_momentum
    )


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
criterion = v8SegmentationLoss(valnet)

# Mixed precision. GradScaler keeps fp16 gradients from underflowing; it is a
# no-op when disabled, so the CPU path is unchanged.
use_amp = bool(omegaCfg.train.amp) and device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

# EMA of the weights. This is what gets validated and checkpointed — the raw
# weights are only ever used to compute gradients.
ema = ModelEMA(valnet, decay=omegaCfg.train.ema_decay)

ckpt_dir = Path(main_dir, omegaCfg.train.ckpt_dir)
os.makedirs(ckpt_dir, exist_ok=True)

print(f"{iters_per_epoch} iters/epoch, {total_iters} total, "
      f"{warmup_iters} warmup | AMP={use_amp}")

"""
# 4. Training loop
"""
epochs = [0]
val_mAP = [0]
val_AP50 = [0]
val_AP75 = [0]
best_map = -1.0
global_iter = 0

for epoch in tqdm(range(omegaCfg.train.epochs), desc="Training epoch", file=sys.stdout):
    valnet.train()
    # Frozen parts keep their BN statistics. Only the BN layers go to eval:
    # the Segment head's forward switches output format on .training.
    for m in frozen:
        for bn in m.modules():
            if isinstance(bn, nn.BatchNorm2d):
                bn.eval()

    # Finish on un-mosaicked images, as Ultralytics' trainer does (close_mosaic)
    if cfg.mosaic > 0 and epoch == omegaCfg.train.epochs - omegaCfg.train.close_mosaic:
        train_dataset.close_mosaic(hyp=copy(cfg))
        train_loader.reset()
        tqdm.write(f"epoch {epoch + 1}: mosaic off for the last {omegaCfg.train.close_mosaic} epochs")
    epoch_loss = 0.0

    for batch_i, batch in tqdm(
            enumerate(train_loader), 
            leave=True if epoch==omegaCfg.train.epochs else False, 
            desc="Training batch", 
            file=sys.stdout, 
            total=len(train_loader)
        ):
        images = batch["img"].to(device).float() / 255.0
        for k in ("bboxes", "cls", "batch_idx", "masks"):
            if k in batch:
                batch[k] = batch[k].to(device)

        # Momentum warmup (LR warmup is handled by the scheduler)
        momentum = momentum_at(global_iter)
        for g in optimizer.param_groups:
            g["momentum"] = momentum

        # Forward pass. The loss is computed inside autocast so its
        # intermediates stay half-precision too.
        with torch.amp.autocast("cuda", enabled=use_amp):
            preds = valnet(images)
            # Compute loss using the head's built-in loss
            loss, _ = criterion(preds, batch)
            loss = loss.sum()

        # Backward
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if omegaCfg.optim.grad_clip > 0:
            # Undo the scaler's multiplier first, so we clip true gradients
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(valnet.parameters(), max_norm=omegaCfg.optim.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(valnet)

        epoch_loss += loss.item()
        global_iter += 1

    avg_loss = epoch_loss / len(train_loader)
    writer.add_scalar("Loss/total_epoch", avg_loss, epoch)
    for g in optimizer.param_groups:
        writer.add_scalar(f"LR/{g['group_name']}", g["lr"], epoch)
    writer.add_scalar("Optim/momentum", optimizer.param_groups[0]["momentum"], epoch)

    # Save checkpoint (EMA weights — those are the ones we actually deploy)
    if (epoch + 1) % omegaCfg.train.save_freq == 0:
        torch.save(ema.ema.state_dict(), Path(ckpt_dir, f"valnet_epoch{epoch+1}.pt"))

    # Perform evaluation at set intervals
    if (epoch + 1) % omegaCfg.train.val_freq == 0:
        metrics = evaluate_valnet(ema.ema, val_loader)
        val_map, val_ap50, val_ap75 = metrics['mAP'], metrics['AP50'], metrics['AP75']
        tqdm.write(f"Validation mAP: {val_map}, AP@50: {val_ap50}, AP@75: {val_ap75} "
                   f"(proto-res mAP {metrics['mAP_proto']:.4f})")

        epochs.append(epoch+1)
        val_mAP.append(val_map)
        val_AP50.append(val_ap50)
        val_AP75.append(val_ap75)
        
        writer.add_scalar("Val/mAP", val_map, epoch)
        writer.add_scalar("Val/AP50", val_ap50, epoch)
        writer.add_scalar("Val/mAP75", val_ap75, epoch)
        writer.add_scalar("Val/mAP_proto", metrics["mAP_proto"], epoch)
        writer.add_scalar("Val/AP50_proto", metrics["AP50_proto"], epoch)

        # Track the best epoch — without this the only recoverable weights are
        # whatever the last periodic save happened to catch.
        if val_map > best_map:
            best_map = val_map
            torch.save(ema.ema.state_dict(), Path(ckpt_dir, "valnet_best.pt"))
            tqdm.write(f"  new best mAP {best_map:.4f} -> valnet_best.pt")
    
    writer.flush()

print(f"Training complete — best val mAP {best_map:.4f} ({Path(ckpt_dir, 'valnet_best.pt')})")
writer.close()

plt.plot(epochs, val_mAP, label = "val mAP")
plt.plot(epochs, val_AP50, label = "val AP50")
plt.plot(epochs, val_AP75, label = "val AP75")
plt.legend()
plt.savefig('validation.jpg')
plt.show()
