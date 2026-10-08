# Evaluation on X-Plane 12 approach videos

Systematic evaluation / ablation of the (yaw, lateral-offset) estimators on the
118 XP12 approach sequences (`xp12_dataset.zip`, 50 820 frames, 1280×720).
Results: [`data/xp12/results/REPORT.md`](../data/xp12/results/REPORT.md) (generated).

## 1. Data & ground truth

Only metadata is extracted; images are read straight from the zip.

```bash
unzip -q xp12_dataset.zip 'xp12_dataset/*/meta.txt' 'xp12_dataset/*/pose/*' \
    'xp12_dataset/*/position/*' 'xp12_dataset/*/labels/*' -d data/xp12
PYTHONPATH=. python scripts/calibrate_xp12.py          # → data/xp12/calibration.json
```

Conventions (recovered by fitting the labelled corners, see
`runway_detection/xp12/`):

| file | content |
|---|---|
| `pose/i.txt` | heading rel. runway (°, + nose right), pitch (°, + up), roll (°) |
| `position/i.txt` | x along (m, < 0 before threshold), y height (m), z cross-track (m, + right) |
| `labels/i.txt` | `0` + near-left, far-left, far-right, near-right corners (normalized) |
| camera | pinhole, **HFOV 65°**, centered principal point, square pixels |

No runway database is provided, so per sequence we fit width, length, an
along-track offset `d0` and the **runway azimuth offset ψ0** (±0.5°: the sim
x-axis is not exactly the runway axis — ignoring it biases lateral GT by
~15 m at 2 km). GT-corner reprojection RMS after the fit: median 0.8 px,
p90 1.9 px. GT (yaw, lateral) are expressed in that true runway frame. The
fitted width plays the role of the runway-DB width prior.

## 2. Pipeline

```
mask source ──► estimator(prior) ──► tracker (cold | t−1 | EKF + NIS) ──► per-frame rows ──► metrics / video
 gt | corrupt | yolo
```

| module | role |
|---|---|
| `observations.py` | mask sources, quad extraction + corner labelling from a mask |
| `estimators.py` | uniform wrappers: `MaskFit`, `EdgeFit`, `CornerFit`, `Homography`, `PnP` |
| `runner.py` | cold / tracking / kinematic EKF, known-DOF noise, width-prior error |
| `metrics.py` | MAE / RMSE / p95, success & gross-error rates, NEES coverage, timing |
| `configs.py` | named method variants and ablation suites |
| `video.py` | replay video (GT vs predicted runway, DOF curves ± 2σ, top view) |

### Methods (see `configs.py`)

| name | what |
|---|---|
| `maskfit_orig` | repo default `estimate_pose_mask_space` (raster render, L2, LM w/ early stop) |
| `maskfit_orig_defaultgrid` | same, repo default cold grid (±8°, ±80 m) |
| `maskfit_analytic` | + analytic anti-aliased renderer (`OptimizerConfig.renderer="analytic"`) |
| `maskfit_analytic_lm` | + LM without the absolute early stop (`lm_min_cost_drop=0`), refine-only when tracking |
| `maskfit_analytic_lm_grid` | + warm hierarchical grid before LM when tracking |
| `maskfit_dice` / `_gd` / `_ds32` / `_ds96` / `_noblur` | metric, optimizer, resolution, blur ablations |
| `edgefit` / `edgefit_priorassign` | spec_edges Huber LM; contour split by observed axis / by prior edges (spec §4) |
| `cornerfit` | 2-DOF LM on the 4 extracted corners, known DOF fixed |
| `homography` / `homography_it5` | warm-started plane readout, 1 / 5 re-linearisations |
| `pnp` | OpenCV IPPE 6-DOF PnP on the corners (ignores known DOF) |

### Tracking

- `track`: warm start from the t−1 estimate (as `validate_mask_fit_sequence.py`);
  failed frames are retried cold.
- `track` + `kf`: EKF on `[yaw, yaw_rate, lat]` with kinematic lateral
  prediction `Δlat = Δalong·tan(yaw)` (known along-track). On GT this model
  predicts lateral to 0.02–0.05 m/frame vs 0.16–0.63 m for a constant model.
  NIS gate χ²₂(99.9 %), cold re-acquisition after 3 rejections.

## 3. Running

```bash
PYTHONPATH=. python scripts/infer_yolo_xp12.py --every 1 --stride 2   # YOLO masks (~0.1 s/frame CPU)
PYTHONPATH=. python scripts/eval_xp12.py --suite solver_cold --workers 6
PYTHONPATH=. python scripts/eval_xp12.py --suite tracking   --workers 6
PYTHONPATH=. python scripts/eval_xp12.py --suite robustness --workers 6
PYTHONPATH=. python scripts/report_xp12.py                            # REPORT.md + figures
PYTHONPATH=. python scripts/render_xp12_video.py --seq 000018 --method maskfit_analytic_lm --source yolo --kf
```

Suites: `solver_cold`, `solver_cold_yolo`, `tracking`, `tracking_orig`, `robustness`.

## 4. Metric definitions

- **success**: |yaw err| < 0.5° and |lat err| < 5 m (frames without output count as failures)
- **gross**: |yaw err| > 2° or |lat err| > 20 m
- **meas.**: fraction of frames with a vision measurement (YOLO misses the runway at long range)
- **95 % cov.**: fraction of frames whose error lies inside the reported 95 % ellipse
  (NEES < 5.99) — 95 % means a calibrated covariance
