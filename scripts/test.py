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
from valnet.generate_masks import generate_test_masks
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

test_dataset = build_yolo_dataset(
    cfg=cfg,
    img_path=data_dict["test"],
    batch=cfg.batch,
    data=data_dict,
    mode="val",
    rect=False,
)
test_loader = build_dataloader(
    dataset=test_dataset,
    batch=cfg.batch,
    workers=8,
    shuffle=False,
)

"""
# 3. Testing setup
"""

valnet.load_state_dict(torch.load('/home/linok/Downloads/valnet/checkpoints/valnet_epoch50.pt'))

"""
# 4. Mask generation
"""
generate_test_masks(valnet, test_loader, '/home/linok/Downloads/valnet/results/test_masks', save_binary=False)
