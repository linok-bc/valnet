# Edge-line cross-track, and the logged-axis rotation fix

> **Frozen archive.** This branch holds one experiment and every file it needs, split
> from `main` at tag `pre-restructure` (2026-11-05). It is not kept in sync with `main`,
> where the cleaned demo package lives.

Environment: `conda env create -f environment.yml` (the `ValNet` env). Datasets are not in
git; scripts expect them under `datasets/` at the repo root. The easy + weather XP12 data is
on Hugging Face at `linok-bc/xp12_flights`.

Cross-track from how the runway's two edge lines lean in the image (as in car lane keeping),
which needs only roll, pitch and runway width: no heading, no along-track, no warp.

| Mask source | Cross-track MAE |
|---|---|
| Ground-truth mask | 0.40 m |
| YOLOv8s-seg at 1280 px, gated Kalman | 0.85 m |
| YOLOv8s-seg at 1280 px | 0.98 m |
| YOLOv8s-seg at 640 px | 1.54 m |

Along the way: the collector logged yaw and cross-track against a runway axis rotated
-0.46..+0.52 deg per flight (up to 13 m at 1.5 km). `check_rotation.py` / `verify_rotation.py`
establish it against the images; `fix_rotation.py` measures each flight's angle from its
polygons and writes corrected logs.

Details and commands: `experiments/edge_lines/README.md`. Tests:
`python -m experiments.tests.test_edge_lines`.
