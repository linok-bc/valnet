# BEV toy experiment

Does rectifying the image to a bird's-eye view give a better-fitting runway
target than segmenting the perspective image — and once rectified, is an
**oriented box** enough, with no pixel mask at all?

After Ryu et al., *An estimation method for vision-based autonomous landing
system for fixed wing aircraft*, IJRR 44(6), 2024 (§3.1.2.4), who warp to a BEV
using roll and pitch and then measure the runway's front edge in that frame.

## The assumption

**Four of the six degrees of freedom are known: altitude, roll, pitch, and
along-track distance to the threshold. Only yaw and cross-track are unknown.**

This is what makes the experiment well posed, and it is what the warp exploits.
The homography that flattens the ground consumes exactly the known quantities.
Yaw is deliberately *not* used — so whatever heading error the aircraft has
survives the warp as a rotation of the runway rectangle, and the cross-track
survives as its lateral offset. Both unknowns are then two numbers read off one
oriented box:

    yaw         = -(the box's angle from BEV "up")
    cross-track = -(near-edge midpoint) . (the box's right-normal)

## The arms

An arm is one cell: a **view**, a **target geometry**, and for BEV arms a
**ground raster**. They are defined in `config.yaml` under `arms`, read through
`arms.py` by the builder, trainer, evaluator and visualiser, so adding one is a
config edit plus a rebuild.

| arm | view | target | grid |
|---|---|---|---|
| `persp_seg` | perspective | polygon | — |
| `persp_obb` | perspective | oriented box | — |
| `bev_seg` | bev | polygon | `threshold` |
| `bev_obb` | bev | oriented box | `threshold` |
| `bev_seg_alt` | bev | polygon | `altitude` |
| `bev_obb_alt` | bev | oriented box | `altitude` |

`persp_seg` is the control — what VALNet and the YOLOv8-seg baseline already do.
`persp_obb` is the other control: it shows the BEV step is doing the work rather
than the box.

**Every arm sees exactly the same frames.** A frame is kept only if it survives
in the perspective view *and* under every grid policy, so a difference between
arms cannot come from one of them having an easier subset. This is why the grids
are built together rather than as separate datasets: a per-policy build keeps a
per-policy frame set, and comparing then needs an after-the-fact intersection
that is easy to forget.

## The grids

| | `threshold` | `altitude` |
|---|---|---|
| canvas placed by | known along-track range, fixed scale | height above the runway |
| runway width in canvas | 60 px at every range | 56 px → 20 px |
| runway inside the canvas | 617 m at every range | 670 m → 1212 m |
| along-range stretch | 2.5× → **31×** | **2.1× flat** |

`threshold` pins the threshold to the same canvas row and the runway to the same
pixel size in every image, so the detector never has to learn scale. Its cost is
that the along-range direction — the one the runway's long axis runs along — is
upsampled 31× at 1.4 km, and the angle is then fitted to interpolation.

`altitude` ties the canvas to height instead. Because the range/height ratio on
these approaches is roughly constant (median 13, p5–p95 6.9–21.9), a 2h..30h
window spans a near-constant number of source rows, so the stretch stays at 2.1×
everywhere and the fitting baseline doubles at range. Its cost is that the
runway narrows to 20 px and the threshold no longer lands on a fixed row.

## Scoring: metres, not mAP

mAP in each arm's own pixels is not comparable across arms. A BEV pixel is a
fixed patch of ground, a perspective pixel is not, so an IoU threshold means a
different physical tolerance in each. Worse, mAP rewards the wrong thing — a
mask can score well while its edges wander by the few metres that are the whole
signal.

So every arm goes through **one shared pipeline**: prediction → polygon → ground
plane in metres (using the known altitude and attitude) → minimum-area rectangle
→ heading and cross-track, scored against the logs and binned by range.

`eval.py` prints an **ORACLE** row above every result: the ground-truth label
through the same pipeline. It is the floor — no detector on that arm can beat
it. Read it first.

## Run it

```bash
conda activate ValNet

# 0. does the camera model agree with the logs? do this on any regenerated data
python -m experiments.bev_yolo.calibrate --data datasets/xp12_dataset --all-signs

# 1. look at the warps before spending a GPU-hour on them
python -m experiments.bev_yolo.preview --data datasets/xp12_dataset -n 8

# 2. how many frames survive, at what resolution? writes nothing
python -m experiments.bev_yolo.build_dataset --dry-run

# 3. build every arm from one frame set
python -m experiments.bev_yolo.build_dataset --clean

# 4. the floor, before any training
python -m experiments.bev_yolo.eval --oracle-only --split val

# 5. train and score
python -m experiments.bev_yolo.train --arm all
python -m experiments.bev_yolo.eval --arm all

# 6. look at what it actually predicted
python -m experiments.bev_yolo.visualize --n 6
python -m experiments.bev_yolo.whole_runway_bev --range 800 --full-fov
```

Synthetic tests, no data needed:

```bash
python -m experiments.tests.test_bev
python -m experiments.tests.test_estimator
```

## Results

118 episodes, stride 5, 4735 frames, airport-grouped splits. Test split n=567,
median absolute cross-track error in metres. `imgsz=1024`, 30 epochs,
`yolov8s`, no tuning.

| arm | oracle | predicted | 150–300 m | 1000–1500 m | median heading err @1.4 km |
|---|---|---|---|---|---|
| `persp_seg` | 2.12 | **2.83** | 1.51 | **5.03** | **0.185°** |
| `bev_seg` | 2.14 | 3.72 | **1.07** | 9.01 | 0.359° |
| `bev_obb` | 2.12 | 3.73 | **1.08** | 13.31 | 0.577° |
| `bev_seg_alt` | 2.30 | 3.64 | 1.83 | 5.60 | 0.257° |
| `bev_obb_alt` | 2.25 | 3.78 | 1.68 | 9.63 | 0.347° |
| `persp_obb` | 9.84 | 20.81 | 4.70 | 33.67 | 1.088° |

**Perspective segmentation wins overall**, and everywhere beyond 300 m. BEV wins
inside 300 m by ~30%. The crossover is around 400 m.

**`bev_seg` and `bev_obb` are a dead tie** (3.72 vs 3.73), and their oracles
agree to two decimals: once rectified, the polygon *is* its own minimum-area
rectangle, so the mask carries no information the four corners do not. That is
the direct answer to "could this just be a box?" — yes, at this level. The one
exception is long range, where fitting a rectangle to a predicted mask recovers
the angle better than the OBB head regresses it (9.01 vs 13.31 m).

**`persp_obb` fails as designed**, with a fitted width 137 m too large at close
range. A box around a perspective trapezoid is not a description of a runway.

**The error is almost entirely heading error, levered by range.** Cross-track is
read off the threshold midpoint projected onto the runway normal, and that
normal comes from the fitted box angle, so an angular error δψ enters as
`range · sin δψ`. Removing that lever collapses the oracle error and flattens it:

| range | as measured | lever removed |
|---|---|---|
| 150–300 m | 1.29 m | **0.34 m** |
| 600–1000 m | 1.28 m | **0.33 m** |
| 1000–1500 m | 3.19 m | **0.27 m** |

Correlation −0.982, fitted slope 0.92. Threshold *position* is recovered to
0.3 m flat with range; what drifts is a steady 0.14–0.19° of angle.

**Bounding the stretch fixes a large part of that.** `bev_seg_alt` cuts the
1000–1500 m angular error 28% against `bev_seg` (0.359° → 0.257°) and the
cross-track 38% (9.01 → 5.60 m), nearly closing the gap to perspective. It pays
for it in the near field, where losing the fixed threshold row costs it
1.07 → 1.83 m. The two policies each fix what the other breaks.

## Why the canvas cannot just hold everything

The camera's footprint on the ground is an unbounded wedge — from ~100 m out to
the horizon, widening as `±0.637 × range`. A BEV canvas is a bounded rectangle
of square metric pixels. Every BEV must truncate; the only question is where.

`whole_runway_bev.py` renders the alternative. For a 4406 m runway at 799 m, a
canvas holding all of it at the working 0.75 m/px is **9104 × 6884 px = 63 Mpx**,
about half of it ground the camera never saw, and the runway's far threshold is
**8.0 px wide in the source** — so the far two-thirds is radial smear, each
streak one source pixel stretched over hundreds of metres.

The underlying reason: a perspective camera allocates ground pixels as `1/X²`,
which is how resolvable detail actually falls off, so it is already the
information-optimal sampling of the ground plane. A metric BEV allocates them
uniformly in metres — optimal for *shape*, which is the entire point, and
pessimal for information density. You can have metric rectangularity or
information-matched sampling, not both; a canvas uniform in `1/X` *is* the
perspective image.

This is the same fact behind every result above, and behind Ryu et al. warping a
chosen ROI trapezoid rather than rasterizing the ground.

## Design decisions worth knowing about

**Range is capped at 1500 m.** Beyond that the aircraft is still *capturing* the
localizer, not flying an approach: past 2.2 km, |yaw| runs to 14° and median
cross-track to 54 m, which throws the threshold hundreds of metres off the
heading axis.

**Frames where the runway runs off the *side* of the canvas are dropped.**
Far-end clipping is fine — the estimator only uses the near edge. Side clipping
shaves the runway lengthwise, so the fitted box comes out narrower and its
centre is pulled inboard, biasing cross-track by metres. The tolerable yaw
scales as 1/range (`test_canvas_yaw_limit`).

**The perspective arm is clipped to the same ground window as the BEV arms.**
Without it the comparison is rigged: a perspective mask runs to the horizon,
where one pixel of error is tens of metres of ground. Left unclipped it costs
the perspective oracle 4× (1.09 → 4.46 m, with a −25 m bias close in);
`--no-ground-window` shows it.

## Known limits

- **The ~2.1 m oracle floor is angular, and unexplained.** It is 0.14–0.19° of
  heading error, not label noise and not the old elevation bug — these labels
  come from the collector that terrain-probes both runway ends. Remaining
  candidates: the camera-model residual (1.37 px median), runway gradient, and
  `runways.dat` geometry disagreeing with the painted runway. Until it is
  understood, nothing below ~3 m at 1.2 km is measurable here.
- **The range lever is inherent.** Inferring a lateral offset from a distant
  object's bearing always pays `range × angle`, so the angular term dominates at
  range however good the detector gets. That is the gap Ryu et al. close with a
  Kalman filter.
- **Perspective's better angle is not isolated.** It may be the crisper edges,
  or the 4× longer fitting baseline (it sees the whole runway; a BEV canvas holds
  617–1212 m). Testable by masking the far half out of the perspective arm's
  labels and seeing whether its angular error degrades toward BEV's.
- **The weights in `runs/` predate the unified build.** Each arm was trained on
  its own policy's build, which was a superset of the current shared frame set;
  the test split here is a subset of both, so the numbers stand, but a clean
  `train --arm all` on the unified build would remove the caveat.
- **`detect` (axis-aligned) is not an arm**, because it cannot represent yaw: at
  ±2° of lean the upright box is 67 m wide around a 45 m runway
  (`test_estimator.py`). It is implemented in `labels.py` if you want to
  quantify that.
