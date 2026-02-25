import sys
from tqdm import tqdm
import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import LOGGER
from ultralytics.utils.ops import process_mask
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.cfg import get_cfg

from pathlib import Path
from valnet.valnet import VALNetModel
from valnet.evaluate_map import evaluate_valnet
from matplotlib import pyplot as plt

"""
# 1. Build model
"""
cfg = get_cfg()
cfg.data = "configs/bars.yaml"
cfg.imgsz = 640
cfg.batch = 24
cfg.task = "segment"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
valnet = VALNetModel.from_ultralytics("yolov8s-seg.pt", ch=(128, 256, 512))
valnet = valnet.to(device)

# for v8SegmentationLoss; there are weights that can be adjusted
valnet.args = get_cfg()
valnet.model = nn.ModuleList([valnet.backbone_p3, valnet.backbone_p4, valnet.backbone_p5, valnet.neck, valnet.head])


"""
# 2. Build datasets, dataloaders, and validators using Ultralytics
"""

data_dict = check_det_dataset(cfg.data)

train_dataset = build_yolo_dataset(
    cfg=cfg,
    img_path=data_dict["train"],
    batch=cfg.batch,
    data=data_dict,
    mode="train",
    rect=False,
)
train_loader = build_dataloader(
    dataset=train_dataset,
    batch=cfg.batch,
    workers=8,
    shuffle=True,
)

val_dataset = build_yolo_dataset(
    cfg=cfg,
    img_path=data_dict["val"],
    batch=cfg.batch,
    data=data_dict,
    mode="val",
    rect=False,
)
val_loader = build_dataloader(
    dataset=val_dataset,
    batch=cfg.batch,
    workers=8,
    shuffle=False,
)

"""
# 3. Training setup (matching Table 5 in paper)
"""
optimizer = torch.optim.SGD(
    valnet.parameters(),
    lr=0.01,
    momentum=0.937,
    weight_decay=0.0005,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50, eta_min=0.0001
)
criterion = v8SegmentationLoss(valnet)


"""
# 4. Training loop
"""
EPOCHS = 50

epochs = [0]
val_mAP = [0]
val_AP50 = [0]
val_AP75 = [0]

for epoch in tqdm(range(EPOCHS), desc="Training epoch", file=sys.stdout):
    valnet.train()
    epoch_loss = 0.0

    for batch_i, batch in tqdm(
            enumerate(train_loader), 
            leave=True if epoch==EPOCHS else False, 
            desc="Training batch", 
            file=sys.stdout, 
            total=len(train_loader)
        ):
        images = batch["img"].to(device).float() / 255.0
        
        # Forward pass
        preds = valnet(images)

        # Compute loss using the head's built-in loss
        # The head expects (predictions, batch_dict) and returns loss
        loss, loss_items = criterion(preds, batch)
        loss = loss.sum()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    scheduler.step()

    avg_loss = epoch_loss / len(train_loader)
    tqdm.write(f"Epoch {epoch+1}/{EPOCHS}  loss: {avg_loss:.4f}")

    # Save checkpoint
    if (epoch + 1) % 10 == 0:
        torch.save(valnet.state_dict(), f"checkpoints/valnet_epoch{epoch+1}.pt")


    # Perform evaluation at set intervals
    if (epoch + 1) % 5 == 0:
        metrics = evaluate_valnet(valnet, val_loader)
        val_map, val_ap50, val_ap75 = metrics['mAP'], metrics['AP50'], metrics['AP75']
        print(f"Validation mAP: {val_map}, AP@50: {val_ap50}, AP@75: {val_ap75}")

        epochs.append(epoch+1)
        val_mAP.append(val_map)
        val_AP50.append(val_ap50)
        val_AP75.append(val_ap75)

print("Training complete")

plt.plot(epochs, val_mAP, label = "val mAP")
plt.plot(epochs, val_AP50, label = "val AP50")
plt.plot(epochs, val_AP75, label = "val AP75")
plt.legend()
plt.savefig('validation.jpg')
plt.show()
