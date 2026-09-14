"""
Train a stock YOLOv8-seg baseline on the xp12 runway dataset.

Deliberately uses Ultralytics' own trainer rather than the hand-rolled loop in
scripts/train.py. The point of a baseline is that it has no custom code to be
wrong: if this underperforms, it is the data or the task, not our plumbing.

Usage:
    python baseline/train.py                    # uses preset from config.yaml
    python baseline/train.py --preset stock
    python baseline/train.py --name my_run --epochs 100
"""

import argparse
from pathlib import Path

from omegaconf import OmegaConf
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]


def build_args(cfg, preset):
    """Flatten the chosen preset into Ultralytics train() kwargs."""
    if preset not in cfg:
        known = [k for k in cfg if OmegaConf.is_dict(cfg[k])]
        raise SystemExit(f"unknown preset {preset!r}; config has {known}")
    args = {
        "data": str(REPO / cfg.data),
        "imgsz": cfg.imgsz,
        "seed": cfg.seed,
        "deterministic": True,
        "project": str(REPO / cfg.project),
        "task": "segment",
    }
    args.update(OmegaConf.to_container(cfg[preset], resolve=True))
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=None, help="controlled | stock (default: config.yaml)")
    ap.add_argument("--name", default=None, help="run name under baseline/runs/")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None)
    cli = ap.parse_args()

    cfg = OmegaConf.load(Path(__file__).parent / "config.yaml")
    preset = cli.preset or cfg.preset
    args = build_args(cfg, preset)

    if cli.epochs is not None:
        args["epochs"] = cli.epochs
    if cli.batch is not None:
        args["batch"] = cli.batch
        # nbs tracks batch in the controlled preset so accumulate stays 1
        if preset == "controlled":
            args["nbs"] = cli.batch
    if cli.device is not None:
        args["device"] = cli.device
    args["name"] = cli.name or preset

    print(f"preset: {preset}")
    for k in sorted(args):
        print(f"  {k:<18s} {args[k]}")

    model = YOLO(str(REPO / cfg.weights))
    model.train(**args)

    best = Path(args["project"], args["name"], "weights", "best.pt")
    print(f"\nbest weights: {best}")
    print(f"evaluate on the shared harness with:\n"
          f"  python baseline/eval.py --weights {best}")


if __name__ == "__main__":
    main()
