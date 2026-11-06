#!/usr/bin/env python
"""Keep every Nth frame of each flight in an xp12 split list.

Consecutive frames are nearly identical, so training on every frame mostly
costs time. Subsampling is per flight (frames 0, N, 2N, ... of each flight's
kept frames), so every flight stays represented.

Reads <split dir>/<split>.txt (already visibility-filtered, see
filter_visibility.py) and writes <split dir>/<split>_every<N>.txt.

    python scripts/subsample_split.py --stride 5 --splits train val
"""

import argparse
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="datasets/xp12_splits/combined")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    args = ap.parse_args()

    for s in args.splits:
        src = Path(args.dir) / f"{s}.txt"
        flights = defaultdict(list)
        for line in src.read_text().splitlines():
            if line.strip():
                flights[str(Path(line.strip()).parents[1])].append(line.strip())
        kept = [img for f in flights.values() for img in sorted(f)[::args.stride]]
        dst = src.with_name(f"{s}_every{args.stride}.txt")
        dst.write_text("\n".join(kept) + "\n")
        print(f"{dst}: {len(kept)} of {sum(map(len, flights.values()))} frames, {len(flights)} flights")


if __name__ == "__main__":
    main()
