"""
Evaluate a trained YOLOv8-seg baseline.

Reports the same metric twice, on purpose:

  1. Ultralytics' own SegmentationValidator — the number to trust, and the one
     that is comparable to published YOLOv8 results.
  2. valnet/evaluate_map.py — the harness VALNet's numbers come from. Only
     these are directly comparable to a VALNet run.

If the two disagree by much, the shared harness is what needs investigating,
not the model. That is worth knowing before it is used to judge VALNet.

Usage:
    python baseline/eval.py --weights baseline/runs/controlled/weights/best.pt
    python baseline/eval.py --weights ... --split test
"""

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset

import sys
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from valnet.evaluate_map import evaluate_valnet  # noqa: E402


def build_loader(data_yaml, split, imgsz, batch, workers):
    """Same dataloader construction as scripts/train.py, so the harness sees
    identical images, letterboxing and mask rasterization."""
    cfg = get_cfg()
    cfg.data = str(data_yaml)
    cfg.imgsz = imgsz
    cfg.batch = batch
    cfg.task = "segment"

    data_dict = check_det_dataset(cfg.data)
    if split not in data_dict or not data_dict[split]:
        raise SystemExit(f"split {split!r} not defined in {data_yaml}")

    dataset = build_yolo_dataset(
        cfg=cfg, img_path=data_dict[split], batch=batch,
        data=data_dict, mode="val", rect=False,
    )
    return build_dataloader(dataset=dataset, batch=batch, workers=workers, shuffle=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-native", action="store_true", help="skip the Ultralytics validator")
    cli = ap.parse_args()

    cfg = OmegaConf.load(Path(__file__).parent / "config.yaml")
    data_yaml = REPO / cfg.data
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLO(cli.weights)

    if not cli.skip_native:
        print("=== Ultralytics SegmentationValidator ===")
        r = model.val(data=str(data_yaml), split=cli.split, imgsz=cfg.imgsz,
                      batch=cli.batch, device=device, verbose=False)
        print(f"  mask mAP50-95 : {r.seg.map:.4f}")
        print(f"  mask mAP50    : {r.seg.map50:.4f}")
        print(f"  mask mAP75    : {r.seg.map75:.4f}")
        print(f"  box  mAP50-95 : {r.box.map:.4f}")

    print("\n=== shared harness (valnet/evaluate_map.py) ===")
    loader = build_loader(data_yaml, cli.split, cfg.imgsz, cli.batch, cli.workers)
    net = model.model.to(device).eval()
    m = evaluate_valnet(net, loader, device=device)
    print(f"  mAP  : {m['mAP']:.4f}")
    print(f"  AP50 : {m['AP50']:.4f}")
    print(f"  AP75 : {m['AP75']:.4f}")
    print("\nCompare the shared-harness numbers against VALNet; the native ones against the literature.")


if __name__ == "__main__":
    main()
