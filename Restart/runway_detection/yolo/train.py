"""Train Ultralytics YOLO-seg from a YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_train_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid train config: {path}")
    return cfg


def train_from_config(config_path: str | Path):
    from ultralytics import YOLO

    cfg = load_train_config(config_path)
    model_name = cfg.pop("model", "yolo26n-seg.pt")
    data = cfg.pop("data")
    # Remaining keys are passed to model.train(...)
    model = YOLO(model_name)
    return model.train(data=data, **cfg)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train YOLO-seg with a config YAML")
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configs/train/quick.yaml or full.yaml",
    )
    args = p.parse_args(argv)
    train_from_config(args.config)


if __name__ == "__main__":
    main()
