from tqdm import tqdm
import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import LOGGER
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.cfg import get_cfg

from types import SimpleNamespace
from valnet.valnet import VALNetModel

# ──────────────────────────────────────────────
# 1. Build model
# ──────────────────────────────────────────────
cfg = get_cfg()
cfg.data = "configs/bars.yaml"
cfg.imgsz = 640
cfg.batch = 60
cfg.task = "segment"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
valnet = VALNetModel.from_ultralytics("yolov8s-seg.pt", ch=(128, 256, 512))
valnet = valnet.to(device)

# for v8SegmentationLoss; there are weights that can be adjusted
valnet.args = get_cfg()
valnet.model = nn.ModuleList([valnet.backbone_p3, valnet.backbone_p4, valnet.backbone_p5, valnet.neck, valnet.head])

# ──────────────────────────────────────────────
# 2. Build dataset & dataloader using Ultralytics
# ──────────────────────────────────────────────
# You'll need a standard YOLO-seg dataset YAML:
#
#   rld.yaml:
#     path: /path/to/RLD
#     train: images/train
#     val: images/val
#     test: images/test
#     names:
#       0: runway



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

# ──────────────────────────────────────────────
# 3. Training setup (matching Table 5 in paper)
# ──────────────────────────────────────────────
optimizer = torch.optim.SGD(
    valnet.parameters(),
    lr=0.01,
    momentum=0.937,
    weight_decay=0.0005,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50, eta_min=0.0001
)

# ──────────────────────────────────────────────
# 4. Loss — reuse YOLOv8's loss from the head
# ──────────────────────────────────────────────
# The Segment head has a built-in loss computation.
# If using the extracted head, it computes loss when
# given predictions + batch targets during training.

# ──────────────────────────────────────────────
# 5. Training loop
# ──────────────────────────────────────────────
EPOCHS = 50

for epoch in tqdm(range(EPOCHS), desc="Training epoch"):
    valnet.train()
    epoch_loss = 0.0

    for batch_i, batch in tqdm(
            enumerate(train_loader), 
            leave=True if epoch==EPOCHS else False, 
            desc="Training batch", 
            total=len(train_loader)
        ):
        images = batch["img"].to(device).float() / 255.0
        
        # Forward pass
        preds = valnet(images)

        # Compute loss using the head's built-in loss
        # The head expects (predictions, batch_dict) and returns loss
        criterion = v8SegmentationLoss(valnet)
        loss, loss_items = criterion(preds, batch)
        loss = loss.sum()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    scheduler.step()
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}  loss: {avg_loss:.4f}  lr: {scheduler.get_last_lr()[0]:.6f}")

    # Save checkpoint
    if (epoch + 1) % 10 == 0:
        torch.save(valnet.state_dict(), f"valnet_epoch{epoch+1}.pt")

print("Training complete")
