"""Build Ultralytics dataset YAML for centerline / centerline+runway."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import yaml

PRESETS: dict[str, dict[int, str]] = {
    "centerline": {0: "centerline"},
    "centerline_runway": {0: "centerline", 1: "runway"},
}


def write_dataset_yaml(
    out_path: str | Path,
    *,
    data_root: str | Path,
    names: Mapping[int, str],
    train: str | None = None,
    val: str | None = None,
    test: str | None = None,
) -> Path:
    """
    Write a YOLO data YAML.

    Relative train/val/test default to images/train, images/val, images/test under data_root.
    """
    data_root = Path(data_root)
    names = {int(k): str(v) for k, v in names.items()}
    cfg = {
        "path": str(data_root.resolve()),
        "train": train or "images/train",
        "val": val or "images/val",
        "test": test or "images/test",
        "names": names,
        "nc": len(names),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Write Ultralytics dataset YAML")
    p.add_argument("--out", type=Path, required=True, help="Output .yaml path")
    p.add_argument("--data-root", type=Path, required=True, help="Dataset root (path:)")
    p.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="centerline_runway",
        help="Class name preset",
    )
    p.add_argument("--train", default=None)
    p.add_argument("--val", default=None)
    p.add_argument("--test", default=None)
    args = p.parse_args(argv)

    path = write_dataset_yaml(
        args.out,
        data_root=args.data_root,
        names=PRESETS[args.preset],
        train=args.train,
        val=args.val,
        test=args.test,
    )
    print(f"Wrote {path}")
    print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
