"""Train / val / test split for paired images and YOLO labels."""

from __future__ import annotations

import argparse
import random
import shutil
from math import floor
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def split_dataset(
    images_src: str | Path,
    labels_src: str | Path,
    images_dst: str | Path,
    labels_dst: str | Path,
    *,
    seed: int = 42,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    move_files: bool = False,
) -> dict:
    """
    Shuffle images and assign to train/val/test; move or copy matching label .txt files.

    test_ratio is implied as 1 - train_ratio - val_ratio.
    """
    images_src = Path(images_src)
    labels_src = Path(labels_src)
    images_dst = Path(images_dst)
    labels_dst = Path(labels_dst)

    if not images_src.exists():
        raise FileNotFoundError(f"Images folder not found: {images_src}")
    if not labels_src.exists():
        raise FileNotFoundError(f"Labels folder not found: {labels_src}")

    images = [
        p for p in images_src.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    images.sort()

    rng = random.Random(seed)
    rng.shuffle(images)

    n = len(images)
    n_train = floor(train_ratio * n)
    n_val = floor(val_ratio * n)
    n_test = n - n_train - n_val

    splits = {
        "train": images[:n_train],
        "val": images[n_train : n_train + n_val],
        "test": images[n_train + n_val :],
    }

    for split in ("train", "val", "test"):
        (images_dst / split).mkdir(parents=True, exist_ok=True)
        (labels_dst / split).mkdir(parents=True, exist_ok=True)

    moved = {"train": 0, "val": 0, "test": 0}
    missing_label: list[str] = []
    dest_exists: list[str] = []
    op = shutil.move if move_files else shutil.copy2

    for split, imgs in splits.items():
        for img_path in imgs:
            label_path = labels_src / f"{img_path.stem}.txt"
            if not label_path.exists():
                missing_label.append(img_path.name)
                continue

            img_dst = images_dst / split / img_path.name
            label_dst = labels_dst / split / label_path.name

            if img_dst.exists() or label_dst.exists():
                dest_exists.append(img_path.name)
                continue

            op(str(img_path), str(img_dst))
            op(str(label_path), str(label_dst))
            moved[split] += 1

    return {
        "n": n,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "moved": moved,
        "missing_label": missing_label,
        "dest_exists": dest_exists,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Split images/labels into train/val/test")
    p.add_argument("--images-src", type=Path, required=True)
    p.add_argument("--labels-src", type=Path, required=True)
    p.add_argument("--images-dst", type=Path, required=True)
    p.add_argument("--labels-dst", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.80)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copy (default: copy)",
    )
    args = p.parse_args(argv)

    stats = split_dataset(
        args.images_src,
        args.labels_src,
        args.images_dst,
        args.labels_dst,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        move_files=args.move,
    )
    print("Split done")
    print(f"Images found: {stats['n']}")
    print(
        f"Target: train={stats['n_train']}, val={stats['n_val']}, test={stats['n_test']}"
    )
    print("Copied/moved:", stats["moved"])
    if stats["missing_label"]:
        print(f"Missing labels: {len(stats['missing_label'])}")
    if stats["dest_exists"]:
        print(f"Already at destination: {len(stats['dest_exists'])}")


if __name__ == "__main__":
    main()
