# Edge-line cross-track: lane keeping, applied to a runway

The BEV experiments ([../bev_yolo/README.md](../bev_yolo/README.md)) read
cross-track from where the runway *sits*, and that position depends on heading:
a heading error δψ enters as `range · sin δψ`, which set the camera-only result
at 6.21 m.

Car lane keeping reads the offset from how the lane lines *lean* instead. The
same works for a runway. After removing the known roll, a ground line parallel
to the runway at lateral offset `s` from a camera at height `h` appears with a
lean of `s / h` about its vanishing point. Heading slides the line sideways in
the image but does not change its lean, so

    cross-track = −(W/2) · (s_L + s_R) / (s_R − s_L)     (runway width W, no height)
    cross-track = −(s_L + s_R) / 2                       (height, no width)

Only roll and pitch are used, as a change of coordinates on the fitted lines;
the image is never warped. Heading and along-track distance are not used at all.
In a synthetic check, a 1° heading error costs ~0.05 m here against ~17 m at 1 km
for a position-based estimator.

References: Dickmanns & Mysliwetz, *Recursive 3-D road and relative ego-state
recognition*, IEEE TPAMI 14(2), 1992 (the classical geometry); Loganathan et
al., *ML-based lane detection and lateral offset estimation*, SAE 2025-01-8020
(YOLOv8 segmentation → lane offset); CLRerNet, WACV 2024 (the row-wise lane
detector to borrow if masks stay the bottleneck).

## Pipeline

1. Segment the runway (any mask).
2. Row-scan the mask in roll-free pixel coordinates: the leftmost and rightmost
   mask pixel of each row are the two edges. Rows touching the image border and
   the first/last few rows (near and far threshold edges) are dropped.
3. Robust line fit per edge.
4. Each line plus the camera centre is a plane; its tilt about the runway
   direction — where the two edge planes meet, i.e. the *measured* vanishing
   point — gives `s / h`.
5. Optional forward Kalman filter along distance flown, with an innovation gate.

## Results

Stock YOLOv8s-seg (`baseline/runs/stock`), scored on its own held-out test
split: 13 flights, 3,147 frames, 150–1500 m. Mean absolute cross-track error
in metres, `free_width` (no heading, no height).

| | exact label lines | ground-truth mask | YOLO @ 640 | YOLO @ 960 | YOLO @ 1280 |
|---|---|---|---|---|---|
| all ranges | 0.25 | 0.40 | 1.54 | 1.00 | 0.98 |
| 150–300 m | 0.25 | 0.26 | **0.35** | 0.64 | 0.77 |
| 1000–1500 m | 0.25 | 0.57 | 3.35 | 1.68 | **1.54** |
| + Kalman | 0.24 | 0.29 | 1.51 | 0.94 | 0.87 |
| + gated Kalman (3σ) | | | | | **0.85** |

- **The method works:** 0.40 m on a perfect mask, with no heading.
- **The segmenter is the limit at range.** At 640 px, YOLO's masks beyond 1 km
  are ~4 m too narrow: the runway is ~38 rows tall there and the mask
  prototypes are coarse. Inference at 1280 px halves the far-range error.
- **The model was trained at 640.** At 1280, close-in frames get occasional
  wild misses (up to 40 m) and 17 detection failures. Retraining at 1280 is the
  follow-up experiment.
- **Gating fixes isolated misses, not sustained ones.** Worst error at
  300–600 m drops from 15.6 to 2.4 m. What remains is two flights (000144,
  000148) where YOLO is wrong for 7–14 consecutive frames — a mask problem, not
  a filtering one. Gate settings (3σ, re-acquire after 3 misses) were fixed in
  advance, not tuned on test.
- The **exact-lines floor is a constant −0.25 m** at every range, unexplained;
  candidates are the principal point or the threshold position.

## Data fixes found along the way

**Logged yaw and cross-track were measured against a rotated runway axis.**
Against the recorded polygon, the logged cross-track erred by `angle × range`
with a fixed angle per flight (−0.46° to +0.52°, median |0.22°|; up to 13 m at
1.5 km; >2 m on 60% of frames). The logged yaw was off by exactly the same angle.
The polygon, not the logs, matches the rendered image: painted edges measured
from pixels alone agree with the polygon (r = 0.975 on 53 clean-edge flights).
Both runways at one airport share the angle (r = 0.987), and the polygon and the
collector's runway bearing come from the same runways.dat endpoints, so this is a
frame mismatch in the collector's `_monitor_loop`, not airport data. The collector
is being fixed separately.

The dataset has been **corrected in place** by `fix_rotation.py`: each flight's
angle is measured from its polygons, along/right are rotated about the threshold
and the angle is subtracted from yaw. The residual angle afterwards is ≤ 0.02°.
Originals: `datasets/xp12_original_logs_backup/`; per-flight angles:
`datasets/xp12_dataset/rotation_fix_manifest.json`.

**Ten crashed approaches were removed** (000036 053 074 076 078 091 092 118 121
129). They sank into terrain or water 600–1300 m short of the runway while the
collector kept recording and labelling. The dataset is now 108 flights. The split
files were *filtered* rather than regenerated, because `make_splits.py` assigns by
sorted position and rerunning it would move flights between train/val/test. The
test split is unchanged.

## Files

| file | what |
|---|---|
| `geometry.py` | row scan, robust line fit, plane-tilt cross-track |
| `eval.py` | exact lines vs ground-truth mask vs YOLO mask, by range, with Kalman / gated Kalman |
| `check_rotation.py` | three flights: polygon vs logs vs painted edges in the image |
| `verify_rotation.py` | per-flight rotation angle, intercept and residual over the whole dataset |
| `fix_rotation.py` | writes the corrected pose/position to a buffer, verifies, before/after figures |
| `../bev/ryu.py` | `kalman_track` gained an optional innovation gate (off by default) |
| `../tests/test_edge_lines.py` | synthetic round trip, heading independence, clipping, gating |

```bash
python -m experiments.tests.test_edge_lines
python -m experiments.edge_lines.eval --truth logged                 # 640, as trained
python -m experiments.edge_lines.eval --truth logged --imgsz 1280
python -m experiments.edge_lines.verify_rotation                     # all angles ~0 after the fix
```

The dataset is corrected, so use `--truth logged`. The default `--truth
corrected` refits the rotation from the polygons, which is now a no-op.
