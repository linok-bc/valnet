#!/usr/bin/env python
"""Train one arm (or all of them) of the BEV toy experiment.

Stock Ultralytics, no custom loop: the point of the experiment is the input
representation and the target geometry, so everything else is held fixed and
taken off the shelf.

    python -m experiments.bev_yolo.train --arm bev_obb
    python -m experiments.bev_yolo.train --arm all
"""

import argparse
from pathlib import Path

import yaml

from experiments.bev_yolo.build_dataset import ARMS


def train_arm(arm, cfg, dataset_root, extra):
    from ultralytics import YOLO

    view, target = ARMS[arm]
    data = Path(dataset_root) / "arms" / arm / "data.yaml"
    if not data.exists():
        raise SystemExit(f"{data} missing -- run build_dataset.py first")

    t = dict(cfg["train"])
    weights = t.pop("weights")[target]
    # absolute: Ultralytics re-roots a RELATIVE project under its own settings
    # runs_dir, which would scatter the arms under runs/<task>/... and hide them
    # from eval.py
    project = str(Path(t.pop("project")).resolve())

    print(f"\n=== {arm}: {view} image, {target} target, from {weights} ===")
    model = YOLO(weights)
    return model.train(data=str(data), name=arm, project=project,
                       exist_ok=True, **t, **extra)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="experiments/bev_yolo/config.yaml")
    ap.add_argument("--arm", default="all", choices=["all", *ARMS])
    ap.add_argument("--epochs", type=int, default=None, help="override config")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    extra = {}
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.device is not None:
        extra["device"] = args.device

    arms = list(ARMS) if args.arm == "all" else [args.arm]
    for arm in arms:
        train_arm(arm, cfg, cfg["dataset"]["out"], extra)


if __name__ == "__main__":
    main()
