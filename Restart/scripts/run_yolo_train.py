#!/usr/bin/env python3
"""Thin wrapper: python scripts/run_yolo_train.py --config configs/train/full.yaml"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runway_detection.yolo.train import main

if __name__ == "__main__":
    main()
