#!/usr/bin/env python
"""Fine-tune DeepLabV3+ on xp12 runway segmentation.

Writes <run_dir>/last.pt (resumable: model, optimizer, scaler, epoch),
<run_dir>/best.pt (model only, best validation mean IoU) and
<run_dir>/results.csv. Rerunning with the same run_dir resumes from last.pt.

    python -m tools.deeplab.train
    python -m tools.deeplab.train train.epochs=1 train.max_iters=50     # smoke test
"""

import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from e2e_runway_dof_estimation.segment import DeepLabV3Plus  # noqa: E402
from tools.deeplab.data import XP12Seg  # noqa: E402
from tools.deeplab.eval import evaluate  # noqa: E402


def loss_fn(logits, mask, dice_weight):
    logits, target = logits[:, 0], mask.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2)) + target.sum((1, 2)) + 1)
    return bce + dice_weight * dice.mean()


def main():
    cfg = OmegaConf.merge(OmegaConf.load(REPO / "tools/deeplab/config.yaml"), OmegaConf.from_cli())
    t = cfg.train
    run = REPO / t.run_dir
    run.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run / "config.yaml")
    dev = "cuda"

    root = REPO / cfg.data.root
    train_ds = XP12Seg(root / cfg.data.train, train=True, fliplr=cfg.data.fliplr, jitter=cfg.data.jitter)
    val_ds = XP12Seg(root / cfg.data.val)
    train_dl = DataLoader(train_ds, t.batch, shuffle=True, num_workers=t.workers, pin_memory=True,
                          drop_last=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, t.batch, num_workers=t.workers, pin_memory=True)

    model = DeepLabV3Plus().to(dev).to(memory_format=torch.channels_last)
    pre, new = model.param_groups()
    opt = torch.optim.SGD([{"params": pre, "lr_mult": t.pretrained_lr_mult},
                           {"params": new, "lr_mult": 1.0}],
                          lr=t.lr, momentum=t.momentum, weight_decay=t.weight_decay, nesterov=True)
    scaler = torch.amp.GradScaler(enabled=t.amp)
    total = t.epochs * len(train_dl)
    start, best = 0, -1.0
    if (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed from epoch {start}, best mean IoU {best:.4f}")

    print(f"train {len(train_ds)} frames, val {len(val_ds)}; {len(train_dl)} iters/epoch, {t.epochs} epochs")
    max_iters = t.get("max_iters")
    log = run / "results.csv"
    for epoch in range(start, t.epochs):
        model.train()
        t0, run_loss = time.time(), 0.0
        for k, (img, mask, _) in enumerate(train_dl):
            it = epoch * len(train_dl) + k
            lr = t.lr * (1 - it / total) ** 0.9 * min(1.0, (it + 1) / t.warmup_iters)
            for g in opt.param_groups:
                g["lr"] = lr * g["lr_mult"]
            img = img.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            mask = mask.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=t.amp):
                logits = model(img)
            loss = loss_fn(logits.float(), mask, t.dice_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item()
            if k % 200 == 0:
                print(f"epoch {epoch + 1}/{t.epochs} iter {k}/{len(train_dl)} loss {loss.item():.4f} "
                      f"lr {lr:.5f} {(k + 1) * t.batch / (time.time() - t0):.1f} img/s", flush=True)
            if max_iters and k + 1 >= max_iters:
                break
        row = {"epoch": epoch + 1, "train_loss": run_loss / (k + 1), "time_s": round(time.time() - t0, 1)}

        if (epoch + 1) % t.val_freq == 0 or epoch + 1 == t.epochs:
            m = evaluate(model, val_dl, dev, amp=t.amp)
            row.update({"val_mean_iou": m["mean_iou"], "val_dataset_iou": m["dataset_iou"],
                        "val_empty_pred": m["empty_pred"]})
            print(f"  val mean IoU {m['mean_iou']:.4f}  dataset IoU {m['dataset_iou']:.4f}  "
                  f"empty predictions {m['empty_pred']}", flush=True)
            if m["mean_iou"] > best:
                best = m["mean_iou"]
                torch.save(model.state_dict(), run / "best.pt")
                print(f"  new best -> {run / 'best.pt'}", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best": best}, run / "last.pt")
        new_file = not log.exists()
        with open(log, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "time_s", "val_mean_iou",
                                               "val_dataset_iou", "val_empty_pred"])
            if new_file:
                w.writeheader()
            w.writerow(row)
        print(f"epoch {epoch + 1} done in {row['time_s'] / 60:.1f} min, loss {row['train_loss']:.4f}", flush=True)
    print(f"done; best val mean IoU {best:.4f} -> {run / 'best.pt'}")


if __name__ == "__main__":
    main()
