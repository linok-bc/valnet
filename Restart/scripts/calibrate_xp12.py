#!/usr/bin/env python3
"""
Fit per-sequence runway geometry (width, length, d0, ψ0) on XP12 GT corners.

Writes ``data/xp12/calibration.json`` (consumed by every XP12 evaluation script).
Extract metadata first (images stay in the zip)::

    unzip -q xp12_dataset.zip 'xp12_dataset/*/meta.txt' 'xp12_dataset/*/pose/*' \\
        'xp12_dataset/*/position/*' 'xp12_dataset/*/labels/*' -d data/xp12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from runway_detection.xp12 import fit_calibration, list_sequences, load_sequence, save_calibrations


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/xp12/xp12_dataset"))
    ap.add_argument("--output", type=Path, default=Path("data/xp12/calibration.json"))
    args = ap.parse_args()

    calibs = {}
    for seq_id in list_sequences(args.root):
        seq = load_sequence(args.root, seq_id)
        c = fit_calibration(seq)
        calibs[seq_id] = c
        print(
            f"{seq_id} {c.airport:>4} {c.runway:>3}  W={c.width_m:5.1f} L={c.length_m:5.0f} "
            f"d0={c.d0_m:6.1f} ψ0={c.psi0_deg:+.3f}°  rms={c.rms_px:.2f}px p95={c.p95_px:.2f}px  n={c.n_frames}"
        )
    save_calibrations(calibs, args.output)
    rms = np.array([c.rms_px for c in calibs.values()])
    print(f"\n{len(calibs)} sequences → {args.output}")
    print(f"GT-corner RMS: median={np.median(rms):.2f}px  p90={np.percentile(rms, 90):.2f}px  max={rms.max():.2f}px")


if __name__ == "__main__":
    main()
