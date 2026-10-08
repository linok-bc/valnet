# Capstone Restart

Pipeline for runway detection → centerline extraction → pose (angles), organized by stage then technique.

```text
data/  →  runway_detection/  →  line_extraction/  →  pose_estimation/
                yolo/ | sam/       geometric/ | skeletonization/
```

## Setup

```bash
cd Restart
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# From repo root so imports resolve:
export PYTHONPATH="$(pwd)"
```

## Data

See [`data/README.md`](data/README.md). Put raw dumps under `data/raw/`, processed labels under `data/processed/`, train/val/test under `data/splits/`.

## YOLO workflow (current)

1. **Convert labels** — runway polygon (class 0) → centerline triangle (class 0) + runway (class 1); other classes shifted +1. Uses `line_extraction.geometric`.

```bash
python -m runway_detection.yolo.convert_labels \
  --labels-root data/raw/labels \
  --half-width 0.0005
```

2. **Split** train / val / test:

```bash
python -m runway_detection.yolo.split \
  --images-src data/raw/images \
  --labels-src data/raw/labels \
  --images-dst data/splits/images \
  --labels-dst data/splits/labels
```

3. **Train** (quick or full):

```bash
python -m runway_detection.yolo.train --config configs/train/quick.yaml
# or
python scripts/run_yolo_train.py --config configs/train/full.yaml
```

Copy [`configs/data.example.yaml`](configs/data.example.yaml) and point paths at your splits before training.

## Legacy notebooks

Original Colab / server notebooks are archived under [`notebooks/legacy/`](notebooks/legacy/) (not the source of truth).

## Evaluation on X-Plane 12 videos

Systematic ablation of the (yaw, lateral) estimators on 118 XP12 approaches:
see [`evaluation/README.md`](evaluation/README.md) (pipeline, commands) and the
generated `data/xp12/results/REPORT.md` (tables, figures, key findings).
Replay videos: `scripts/render_xp12_video.py`.
