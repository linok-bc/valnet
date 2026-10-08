#!/usr/bin/env python3
"""Download DEEL-AI/LARD_V2 subsets into data/LARD.

Examples:
  python scripts/download_lard.py
  python scripts/download_lard.py --source flsim --split train
  python scripts/download_lard.py --budget-gb 10
  python scripts/download_lard.py --scenarios CYEG-2-20-12-30__10-smpl__10h40
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runway_detection.lard.download import main

if __name__ == "__main__":
    main()
