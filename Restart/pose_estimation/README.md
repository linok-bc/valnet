# Pose estimation

Render-and-compare pose estimation from a runway mask (`spec.md` §4).

## `g_dof` (phase 1 — implemented)

Deterministic forward renderer:

```
mask = g_dof(scene, params, state)
```

- **Unknowns** (`GDofState`): `yaw_cam_deg` (LARD simulator yaw), `lateral_offset_m` (cross-track, +Y = left)
- **Fixed** (`GDofParams`): along-track distance, height AGL, pitch, roll, intrinsics
- **Output**: binary runway mask `(H, W)` via pinhole projection + `cv2.fillPoly`

Built on validated LARD V2 geometry (`runway_detection/lard/projection.py`).

### Validate on local LARD data

```bash
cd Restart
source .venv/bin/activate
MPLCONFIGDIR=.mplconfig PYTHONPATH=. python scripts/validate_g_dof.py \
  --data-root data/LARD -n 5
```

Overlays: `data/LARD/g_dof_overlays/`.

### API

```python
from pathlib import Path
from pose_estimation import RunwayScene, GDofParams, GDofState, g_dof
from pose_estimation.runway_model import CameraPoseLocal
from runway_detection.lard.loader import iter_samples, load_runways_database

data_root = Path("data/LARD")
sample = next(iter_samples(data_root, require_image=True))
db = load_runways_database(data_root, "flsim")
scene = RunwayScene.from_lard_sample(sample, db)
pose = CameraPoseLocal.from_lard_sample(sample, scene)
params = GDofParams.from_pose(pose, scene)
state = GDofState(yaw_cam_deg=pose.yaw_cam_deg, lateral_offset_m=pose.lateral_offset_m)
mask = g_dof(scene, params, state)
```

## Mask-space optimizer (phase 2 — implemented)

Fit `(yaw, lateral_offset)` to an observed soft mask:

```python
from pose_estimation import estimate_pose_mask_space, GDofState, corrupt_mask

est = estimate_pose_mask_space(scene, params, observed_mask, init=init_state)
# est.state, est.dice, est.converged
```

Pipeline: ROI crop → downsample → **coarse grid** → **LM** on L2 residuals (mask-space).

### Benchmark on corrupted synthetic masks

```bash
MPLCONFIGDIR=.mplconfig PYTHONPATH=. python scripts/validate_mask_fit.py \
  --data-root data/LARD -n 10
```

### YOLO masks on LARD (OOD segmentation)

```bash
# 1) Run YOLO on LARD images → soft masks (.npy)
PYTHONPATH=. python scripts/infer_yolo_lard.py \
  --data-root data/LARD --weights weights/best.pt -n 200

# 2) Pose fit on YOLO masks (init at GT isolates mask error)
PYTHONPATH=. python scripts/validate_mask_fit.py \
  --data-root data/LARD \
  --mask-dir data/LARD/masks/yolo_runway \
  --init-gt --stratify -n 100 \
  --output data/LARD/results/mask_fit_yolo.csv
```

`best.pt` classes: `0=runway`, `1=threshold`, `2=aiming` (uses class 0).

Stratification bins: tertiles of **height AGL** (m) and **annotated runway bbox area** (px²).

## Stubs

- `angles.py` — vanishing-point baseline (Approach A)
- Next: temporal warm-start (`x_{t-1}+v·dt`), EKF layer (§4.5)

Pipeline position: **stage 3** (final).
