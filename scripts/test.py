import os
import torch
from torch import nn
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg

from omegaconf import OmegaConf
from pathlib import Path
from valnet.valnet import VALNetModel
from valnet.generate_masks import generate_test_masks
from valnet.evaluate_map import evaluate_valnet

if __name__ == '__main__':
    
    """
    # 1. Build model
    """
    main_dir = Path(os.path.dirname(os.path.realpath(__file__)), '..').absolute()
    omegaCfgPath = Path(main_dir, 'configs/main.yaml')
    # dotlist overrides, e.g. `python scripts/test.py test.checkpoint=checkpoints/x.pt`
    omegaCfg = OmegaConf.merge(OmegaConf.load(omegaCfgPath), OmegaConf.from_cli())
    
    cfg = get_cfg() # from Ultralytics
    cfg.data = omegaCfg.data.config
    cfg.imgsz = omegaCfg.data.image_size
    cfg.batch = omegaCfg.train.batch_size
    cfg.task = "segment"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    valnet = VALNetModel.from_ultralytics(
        Path(main_dir, omegaCfg.yolov8_checkpoint), ch=(128, 256, 512), nc=omegaCfg.data.num_classes,
        **omegaCfg.model,
    )
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
        batch=omegaCfg.test.batch_size,
        workers=omegaCfg.test.num_workers,
        shuffle=False,
    )
    
    """
    # 3. Testing setup
    """
     
    valnet.load_state_dict(torch.load(Path(main_dir, omegaCfg.test.checkpoint), map_location=device))
    
    """
    # 4. Segmentation metrics, generation
    """
    seg_metrics = evaluate_valnet(valnet, test_loader)
    print(f"  {'':<6s} {'full-res':>9s} {'proto-res':>10s}")
    for k in ("mAP", "AP50", "AP75"):
        print(f"  {k:<6s} {seg_metrics[k]:>9.4f} {seg_metrics[k + '_proto']:>10.4f}")
    
    generate_test_masks(
        valnet,
        test_loader,
        output_dir=Path(main_dir, omegaCfg.test.output_dir),
        device=device,
    )
