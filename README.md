# e2e_runway_dof_estimation

Runway yaw and cross-track from a forward-looking camera, end to end, on X-Plane 12 approaches.

```
camera frame ──► DeepLabV3+ ──► runway mask ──► mask-fit estimator ──► crab-angle EKF ──► yaw, cross-track
                 (segment.py)                    (Samuel Moyal's, capstone-dof-estimation submodule)
```

The along-track distance, height, pitch and roll are taken as known (from the INS); the runway's
width and length come from the runway database. The estimator fits the runway's projected
outline to each mask, and the EKF tracks yaw, yaw rate, cross-track and the crab angle, the
angle between the nose and the ground track that a crosswind sets up.

## Results

On the 88 test flights at airports the models never saw in training (20,832 frames,
150–1500 m before the threshold, easy and weather conditions):

| Range to threshold | Yaw MAE | Cross-track MAE |
|---|---|---|
| 150–300 m | 0.059° | 0.28 m |
| 300–600 m | 0.041° | 0.33 m |
| 600–1000 m | 0.045° | 0.51 m |
| 1000–1500 m | 0.048° | 0.78 m |
| **All** | **0.047°** | **0.53 m** (95th percentile 1.70 m) |

DeepLabV3+'s runway masks average 0.891 IoU against the labels over the 49,792-frame test
split. Two design choices are measured in the `exp/` branches:
- **Crab-angle EKF.** The original 3-state filter averaged 2.17 m of cross-track error, because
  it drifts sideways in a crosswind (`exp/crab-ekf`).
- **Runway geometry from the database.** Using the database instead of fitting each flight's
  runway to its labels costs 0.03 m of cross-track.

## Quick start

```bash
git clone --recurse-submodules git@github.com:linok-bc/valnet.git
cd valnet
conda env create -f environment.yml && conda activate e2e-runway-dof
pip install -e . --no-deps
hf auth login        # the assets repo is private
```

Run one flight from the command line; this writes `replay.mp4`, `frames.csv` and `summary.json`:

```bash
python -m e2e_runway_dof_estimation.demo --flight weather_000185
```

Or open the web app at http://127.0.0.1:7860:

```bash
python -m e2e_runway_dof_estimation.app
```

To share the app, `--share` adds a public `gradio.live` link (valid about a week) that anyone
with the URL can open. Add a login with `--auth user:password`. While shared, the "run a flight
folder" box is hidden, because it would let visitors run the pipeline on any path on this
machine; `--allow-folder-runs` brings it back.

```bash
python -m e2e_runway_dof_estimation.app --share --auth demo:<password>
```

The app opens on a **preloaded run**: a saved result with its replay video. Switching between
the eight preloaded runs computes nothing, so it works without a GPU; the first view of each
downloads its frames (40–150 MB). Scrub through the frames with the slider. To process a new
flight folder, enter its path and press Run (needs a GPU).

The preloaded runs are a showcase: for each condition, the test flight with the lowest
cross-track error. Over all 88 test flights the average is 0.047° yaw and 0.53 m cross-track
(see Results).

| Preloaded run | Flight | Frames | Yaw MAE | Cross-track MAE |
|---|---|---|---|---|
| `weather_000185` | ESIB 19, clear, 7.3 m/s crosswind | 237 | 0.029° | 0.19 m |
| `weather_000292` | UWKJ 16, light rain | 230 | 0.040° | 0.14 m |
| `weather_000331` | UMBB 29, heavy rain, 5.9 m/s crosswind | 233 | 0.023° | 0.30 m |
| `weather_000414` | ESOK 21, heavy rain | 195 | 0.014° | 0.19 m |
| `weather_000270` | EETU 26, haze | 232 | 0.012° | 0.20 m |
| `weather_000410` | ESTA 14, fog, 1.5 km visibility | 108 | 0.024° | 0.23 m |
| `weather_000494` | EDLN 13, mist | 255 | 0.024° | 0.47 m |
| `easy_000126` | USNN 21, clear | 245 | 0.011° | 0.17 m |

Fog and heavy-rain approaches start 600–1350 m out, where the runway comes into view. Mist is
the hardest condition: even its best flight is off by 0.7 m beyond 1 km.

From Python:

```python
from e2e_runway_dof_estimation import pipeline
result = pipeline.run("weather_000410")    # a sample name, or a flight folder
print(result.summary())
saved = pipeline.load_preloaded("weather_000185")   # a preloaded run, no computation
```

`--flight` takes one of the five sample names above, downloaded on first use, or any flight
folder from the dataset.

Check an install with `python tests/test_smoke.py`. It loads every preloaded run and checks
its saved summary, then runs two samples through the full pipeline against reference numbers
(about 30 s on a GPU).

## Data and assets

- **Dataset:** [`linok-bc/xp12_flights`](https://huggingface.co/datasets/linok-bc/xp12_flights),
  145 easy and 468 weather approaches in seven conditions (68 GB), with airport-grouped splits.
- **Assets:** the private model repo `linok-bc/e2e-runway-dof-estimation` holds the DeepLabV3+
  weights, the runway database (`runways.dat`), five sample flights and their preloaded runs
  (rebuild with `python -m tools.precompute_runs`). They are cached
  under `~/.cache/huggingface` and `~/.cache/e2e_runway_dof_estimation`.

A flight folder holds `images/`, `pose/` (yaw pitch roll), `position/` (along height right,
along < 0 before the threshold), `labels/` (runway polygons, used for IoU and by the
estimator's loader) and `meta.txt`. By default the demo runs on frames 150–1500 m before the
threshold, and skips frames beyond half the visibility, where the camera can barely see the
runway.

## Repository layout

| Path | What |
|---|---|
| `e2e_runway_dof_estimation/` | The package: `data` (flights, runway database), `segment` (DeepLabV3+), `estimate` (wrapper around the submodule), `pipeline`, `viz`, `demo` (command line), `app` (Gradio), `assets` |
| `capstone-dof-estimation/` | Submodule: fork of [samuelmoyal/Restart](https://github.com/samuelmoyal/Restart), branch `crab-ekf`, which adds the crab-angle EKF state |
| `tools/` | Reproduce the model: `filter_visibility.py`, `subsample_split.py`, then `python -m tools.deeplab.train` and `python -m tools.deeplab.eval`; `precompute_runs.py` rebuilds the preloaded runs |
| `tests/` | The smoke test |

## Experiments

Each finished experiment is a frozen branch that holds its code and a README with its results.
These branches are not kept in sync with `main`.

| Branch | Experiment |
|---|---|
| `exp/seg-comparison` | YOLOv8-seg vs VALNet vs DeepLabV3+ on the easy + weather data, and the 960-px box-cap fix |
| `exp/crab-ekf` | Samuel's estimator on the test split; the crosswind (crab-angle) EKF fix |
| `exp/joint-demo` | Samuel's heading with our edge-line cross-track, and their fusion |
| `exp/edge-lines` | Cross-track from the runway's edge lines; the logged-axis rotation fix |
| `exp/bev-ryu` | Bird's-eye-view rectification and Ryu et al.'s lateral estimator |
| `exp/valnet-vs-yolo` | VALNet vs YOLOv8-seg on the original dataset |
| `exp/legacy-datasets` | Early data tools: RLD, BARS, YouTube footage |

Tag `pre-restructure` is the repository as it was before this split.

## Limitations

- Processing a flight needs a CUDA GPU (preloaded runs do not). DeepLabV3+'s network takes about 16 ms per 1280×720 frame on an
  RTX 3090 Ti; `--device cpu` works but is slow.
- The known quantities (along-track, height, pitch, roll) come from the simulator's logs,
  standing in for an INS.
- The estimator's loader reads the label files, so a flight folder needs `labels/`.
- Yaw is smoothed more than it needs to be: the EKF assumes 0.15° of yaw measurement noise
  where the per-frame estimate is good to about 0.04°.
- The runway model is flat, so a sloped runway (for example ZMUB 14) fits less well.

## Credits

The yaw/cross-track estimator is Samuel Moyal's
([samuelmoyal/Restart](https://github.com/samuelmoyal/Restart)); the crab-angle EKF state is
on this project's fork. DeepLabV3+: Chen et al., ECCV 2018, on torchvision's COCO weights.
