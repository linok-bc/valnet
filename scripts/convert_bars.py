"""
Convert BARS dataset labelme JSON annotations to YOLO segmentation format.

Usage:
    python convert_bars.py --json_dir /path/to/bars/annotations \
                           --out_dir /path/to/bars/labels \
                           --classes runway aiming threshold
"""

import json
import os
import argparse
from pathlib import Path


def convert_labelme_to_yolo_seg(json_path, out_dir, class_map):
    """
    Convert a single labelme JSON file to YOLO segmentation format.

    Args:
        json_path: path to labelme JSON
        out_dir: output directory for .txt label files
        class_map: dict mapping label string -> class id, e.g. {"runway": 0}
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Derive output filename from imagePath in the JSON
    img_name = Path(data.get("imagePath", Path(json_path).stem + ".jpg")).stem
    txt_path = os.path.join(out_dir, img_name + ".txt")

    lines = []

    img_height = data.get("imageHeight")
    img_width = data.get("imageWidth")

    for shape in data.get("shapes", []):
        label = shape["label"]
        if label not in class_map:
            continue

        class_id = class_map[label]
        points = shape["points"]

        # Normalize coordinates to [0, 1]
        normalized = []
        for x, y in points:
            normalized.append(f"{x / img_width:.6f}")
            normalized.append(f"{y / img_height:.6f}")

        line = f"{class_id} " + " ".join(normalized)
        lines.append(line)

    with open(txt_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Convert BARS labelme JSON to YOLO-seg format")
    parser.add_argument("--json_dir", required=True, help="Directory containing labelme JSON files")
    parser.add_argument("--out_dir", required=True, help="Output directory for YOLO .txt labels")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["runway"],
        help="Which labels to include and their order (first = class 0). "
             "Default: runway only. Example: --classes runway aiming threshold",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Build class map from ordered list
    class_map = {name: i for i, name in enumerate(args.classes)}
    print(f"Class mapping: {class_map}")

    json_files = sorted(Path(args.json_dir).glob("*.json"))
    print(f"Found {len(json_files)} JSON files")

    for jf in json_files:
        convert_labelme_to_yolo_seg(jf, args.out_dir, class_map)

    print(f"Converted {len(json_files)} files -> {args.out_dir}")
    print(f"\nYAML snippet for your dataset config:")
    print(f"names:")
    for name, idx in class_map.items():
        print(f"  {idx}: {name}")

if __name__ == "__main__":
    main()
