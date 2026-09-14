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

from experiments.bev_yolo.arms import load_config


def train_arm(arm, cfg, dataset_root, extra):
    from ultralytics import YOLO

    data = Path(dataset_root) / "arms" / arm.name / "data.yaml"
    if not data.exists():
        raise SystemExit(f"{data} missing -- run build_dataset.py first")

    t = dict(cfg["train"])
    weights = t.pop("weights")[arm.target]
    # absolute: Ultralytics re-roots a RELATIVE project under its own settings
    # runs_dir, which would scatter the arms under runs/<task>/... and hide them
    # from eval.py
    project = str(Path(t.pop("project")).resolve())

    grid_note = f", {arm.grid} grid" if arm.is_bev else ""
    print(f"\n=== {arm.name}: {arm.view} image, {arm.target} target{grid_note}, "
          f"from {weights} ===")
    model = YOLO(weights)
    return model.train(data=str(data), name=arm.name, project=project,
                       exist_ok=True, **t, **extra)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="experiments/bev_yolo/config.yaml")
    ap.add_argument("--arm", default="all",
                    help="an arm name from the config, or 'all'")
    ap.add_argument("--epochs", type=int, default=None, help="override config")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg, arms, _ = load_config(args.config)
    if args.arm != "all" and args.arm not in arms:
        raise SystemExit(f"unknown arm {args.arm!r}; config has {', '.join(arms)}")
    extra = {}
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.device is not None:
        extra["device"] = args.device

    for arm in ([arms[a] for a in arms] if args.arm == "all" else [arms[args.arm]]):
        train_arm(arm, cfg, cfg["dataset"]["out"], extra)


if __name__ == "__main__":
    main()
