# Segmentation comparison: YOLOv8-seg, VALNet, DeepLabV3+

> **Frozen archive.** This branch holds one experiment and every file it needs, split
> from `main` at tag `pre-restructure` (2026-11-05). It is not kept in sync with `main`,
> where the cleaned demo package lives.

Environment: `conda env create -f environment.yml` (the `ValNet` env). Datasets are not in
git; scripts expect them under `datasets/` at the repo root. The easy + weather XP12 data is
on Hugging Face at `linok-bc/xp12_flights`.

Three segmenters trained on the easy + weather XP12 data (airport-grouped split, frames beyond
half the visibility dropped, every 5th frame for training) at 1280 px for 50 epochs, scored
by per-frame mask IoU at full 1280x720 on 49,792 test frames from 89 unseen-airport flights.

| Model | Mean IoU | Params | ms/frame* |
|---|---|---|---|
| DeepLabV3+ | 0.891 | 40.3M | 49 |
| YOLOv8s-seg | 0.878 | 11.8M | 26 |
| VALNet | 0.836 | 12.8M | 36 |

\*End to end incl. image load and post-processing; DeepLab's network alone is ~16 ms (fp16, RTX 3090 Ti).

DeepLab leads from 600 m out; all three tie on the runway once YOLO/VALNet's 960-px box cap
is worked around (a box at the cap is widened to the image edges before the mask crop:
past-threshold IoU 0.790 -> 0.895).

Run in order: `scripts/filter_visibility.py`, `scripts/subsample_split.py`,
`bash scripts/xp12_combined.sh yolo valnet deeplab` (train + test),
`bash scripts/xp12_predict_all.sh` (predictions, IoU, videos), `scripts/build_summary.py`
(results page). The summary page reads `experiments/restart_test/summary*.json` from
`exp/crab-ekf`.
