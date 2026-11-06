#!/usr/bin/env python
"""Drop frames whose runway is too far away to see from the xp12 split lists.

The runway polygon is projected from geometry, so it is labelled even when fog
or rain hides the runway. A frame is dropped when its slant range to the
threshold exceeds FRACTION of the flight's set visibility (meta.txt, line 2,
field 4, statute miles; same value as weather.json). At half the visibility
the runway keeps roughly 22% of its contrast (Koschmieder, 5% threshold).
Frames past the threshold (along >= 0) are always kept.

The unfiltered lists are kept in <split dir>/unfiltered/ and are always the
input, so the script can be rerun with a different fraction. Writes the
filtered train/val/test.txt in place, plus visibility_filter.json with
per-split and per-regime counts.

    python scripts/filter_visibility.py
    python scripts/filter_visibility.py --fraction 1.0
"""

import argparse
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

SM = 1609.344
SPLITS = ("train", "val", "test")


def visibility_m(ep: Path) -> float:
    return float((ep / "meta.txt").read_text().split("\n")[1].split()[3]) * SM


def regime(ep: Path) -> str:
    w = ep / "weather.json"
    return json.loads(w.read_text())["regime"] if w.exists() else "easy"


def visible(img: str, vis: dict, fraction: float) -> bool:
    p = Path(img)
    ep = p.parents[1]
    along, height, _ = map(float, (ep / "position" / f"{p.stem}.txt").read_text().split())
    if along >= 0:
        return True
    return math.hypot(along, height) <= fraction * vis[ep]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="datasets/xp12_splits")
    ap.add_argument("--fraction", type=float, default=0.5)
    args = ap.parse_args()

    vis = {}
    for d in sorted(p for p in Path(args.root).iterdir() if (p / "data.yaml").exists()):
        raw = d / "unfiltered"
        raw.mkdir(exist_ok=True)
        stats = {"fraction": args.fraction, "splits": {}, "by_regime": defaultdict(lambda: [0, 0])}
        for s in SPLITS:
            if not (raw / f"{s}.txt").exists():
                shutil.copy(d / f"{s}.txt", raw / f"{s}.txt")
            lines = [l.strip() for l in (raw / f"{s}.txt").read_text().splitlines() if l.strip()]
            kept = []
            for img in lines:
                ep = Path(img).parents[1]
                if ep not in vis:
                    vis[ep] = visibility_m(ep)
                ok = visible(img, vis, args.fraction)
                r = stats["by_regime"][regime(ep)]
                r[0] += 1
                r[1] += not ok
                if ok:
                    kept.append(img)
            (d / f"{s}.txt").write_text("\n".join(kept) + "\n")
            stats["splits"][s] = {"frames": len(lines), "kept": len(kept), "dropped": len(lines) - len(kept)}
        stats["by_regime"] = {k: {"frames": n, "dropped": x} for k, (n, x) in sorted(stats["by_regime"].items())}
        (d / "visibility_filter.json").write_text(json.dumps(stats, indent=1) + "\n")
        print(d.name, json.dumps(stats["splits"]))


if __name__ == "__main__":
    main()
