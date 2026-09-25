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

---

# Second experiment: Ryu et al.'s estimator on the same BEV segmentation

The experiment above asked whether a BEV gives a better-fitting *target*, and
answered no. It left a specific culprit: the error was almost entirely **heading
error levered by range**, because cross-track was read off the threshold midpoint
projected onto a normal that came from a fitted box angle, so `δψ` entered as
`range · sin δψ`.

Ryu et al. do not have that lever, and the reason is worth stating plainly: they
never estimate heading. This experiment ports their estimator onto the segmentation
we already have and asks whether that is what the BEV was missing.

## What the paper's estimator actually is

Their equations 4–21 look like a long chain — project the principal ray to the
ground, convert camera → body → heading frame, take pixel errors from the
principal point, scale them, add them back. Every step is affine about one point,
so the whole chain collapses to a **radial rescaling of the threshold midpoint
about the camera's principal ground point**:

    p_H = c_H + k_s · (t − c_H)          then     p_R = R_z(ψ_r − ψ) · p_H

That leaves exactly **three** differences from `eval.estimate`, and
`experiments/bev/ryu.py` makes each an independent flag:

| | `eval.estimate` | Ryu et al. |
|---|---|---|
| `edge` | whole polygon → `minAreaRect` | **two points**: the front edge (§3.1.1) |
| `scale` | known `mpp` from the metric grid | **measured**: `k_s = W_R / w` (eq 18) |
| `yaw` | **fitted** from the box's long axis | **known**, from the INS (eq 22) |

plus a first-order Kalman filter on the output (§3.1.2.5).

## The result

Test split, n=567, same `bev_seg` weights as above. Columns are the paper's
Table 3 — MAE, RMSE, MAX — so they can be read against its 1.779 / 2.380 / 6.618 m.

| variant | swaps | oracle MAE | **pred MAE** | pred RMSE | pred MAX |
|---|---|---|---|---|---|
| `rect` | (the estimator above) | 3.14 | 6.36 | 8.53 | 39.96 |
| `front` | + two points only | 2.68 | 6.27 | 8.22 | 36.96 |
| `fitted_yaw` | + measured scale | 2.69 | 6.21 | 8.13 | 33.31 |
| `paper` | + INS heading | **0.25** | **1.87** | 2.91 | 18.69 |
| `paper` + Kalman | | 0.27 | **1.66** | **2.20** | 10.59 |

**This reproduces the paper.** 1.66 m MAE / 2.20 m RMSE against their reported
1.78 / 2.38 over eight real flight tests.

**The 2.1 m oracle floor the first experiment could not explain was the yaw
lever, and it is now gone.** Handing the estimator the true heading drops the
oracle from 3.14 m to 0.25 m and — the real tell — *flattens it with range*:

| range | `rect` oracle | `paper` oracle | `rect` pred | `paper` pred |
|---|---|---|---|---|
| 150–300 m | 1.75 | 0.25 | 3.27 | 2.50 |
| 300–600 m | 2.39 | 0.24 | 3.43 | 2.04 |
| 600–1000 m | 3.14 | 0.24 | 5.35 | 1.69 |
| 1000–1500 m | 4.31 | 0.28 | 11.37 | 1.69 |

The predicted error stops growing with range entirely. That is the `range · sin δψ`
term being deleted, and it is the whole result.

**The ablation says it is all the yaw.** Two points instead of the polygon buys
15% on the oracle and nothing on the prediction. The measured scale `k_s` costs
nothing at all (2.679 → 2.687 m) — as predicted, because `k_s` multiplies the
offset from the pivot, which is hundreds of metres along-track and a few metres
laterally, so a 10% width error is 120 m of range error and 0.8 m of cross-track
(`test_scale_error_is_asymmetric`). The paper's own numbers show the same
asymmetry: 1.78 m lateral against ~30 m along-track RMSE.

**And with this estimator, BEV finally beats perspective** — reversing the first
experiment's conclusion:

| arm | pred MAE | RMSE | MAX | median |
|---|---|---|---|---|
| `bev_seg` | **1.87** | 2.91 | 18.7 | 1.32 |
| `bev_obb` | 2.09 | 4.71 | 59.9 | 1.25 |
| `bev_seg_alt` | 3.07 | 4.08 | 19.1 | 2.36 |
| `persp_seg` | 4.83 | 8.09 | 32.0 | **0.62** |

Look at `persp_seg` carefully: its **median is the best of the four** and its mean
is the worst. The paper's estimator reads two points off the front edge, and in a
perspective frame those two points sit where one pixel is metres of ground, so
the distribution is heavy-tailed — brilliant when it works, catastrophic when it
does not. In BEV the same two points are measured in a metric frame. That is the
first thing the BEV has been shown to actually buy.

## The caveat, and it is the whole interpretation

**`yaw="known"` is solving an easier problem.** The first experiment's premise is
that yaw and cross-track are the two unknowns; the paper hands the estimator one
of them from an INS. That is entirely legitimate for Ryu et al. — they are not
claiming to estimate heading from the camera — but `paper` and `rect` are not
answering the same question. **`fitted_yaw` is the like-for-like row, and it is
6.21 m.** Do not quote 1.66 m against a camera-only baseline without saying it
used the INS.

What the result does establish, camera-only claims aside: if heading is available
to ~0.05° (an INS, or a magnetometer plus the runway's published bearing), then
BEV segmentation plus this estimator gives single-GNSS-grade cross-track, and the
detector is no longer the limiting factor.

## The paper's pivot point does not exist on this dataset

Eq 12 needs the optical axis to hit the ground. Ryu et al. mount their camera
tilted **down** for exactly that reason (§3.1.2, "it looks slightly down to better
see the runway"). The X-Plane camera here looks straight along the body x axis —
the fit gives tilt **+0.04°** — and the approach is flown nose-up at **3.4° median**,
so the depression is negative and **the axis points at the sky on 98.8% of frames**.
On the other 1.2% it lands at 3.3 km against a threshold at 1.4 km, so the lever
`|t − c_H|` is ~1.9 km *pointing backwards*.

`--pivot` therefore selects the point the rescaling turns about. Measured on the
`bev_seg` oracle:

| `--pivot` | detection rate | MAE |
|---|---|---|
| `principal` — eq 12 as written | **0.01** | 1.34 |
| `nadir` — the ground point under the aircraft | 1.00 | **0.241** |
| `auto` — eq 12 where it exists (default) | 1.00 | 0.254 |

Conditioning is what matters, not availability — the scale error costs
`(k_s − 1) · |t − pivot|`, so a far pivot is worse than no pivot. `auto` is
*worse* than `nadir` for exactly that reason: on the 1% of frames where eq 12
does resolve, it resolves 1.9 km on the wrong side of the threshold, and those
frames alone move the mean. **Tilting the camera down ~8° would put
`c_H` near the threshold and make the method better conditioned than either
option here.** That is a claim about mounting, and it is the most actionable thing
this port has to say about the data collection.

## The new floor is 0.25 m, and it is the labels

The oracle bias is a near-constant −0.25 m, flat with range and almost noise-free.
It is not the estimator: run on the runway quad rebuilt from `runways.dat` and the
pose logs, with no image, no warp and no label file, the estimator is exact to
1e-4 m. Comparing the **stored label's** threshold against the logged one puts the
disagreement at 2.5 m, mostly along-track, of which ~0.25 m leaks laterally — so
this is the README's own standing candidate, *the collector's polygon disagreeing
with `runways.dat`*, now measured. `BIRK` is 39 m out on 3 frames and should be
dropped or re-probed.

## Run it

```bash
# the floor, no GPU
python -m experiments.bev_yolo.ryu_eval --arm bev_seg --oracle-only

# the full ablation ladder against the trained weights
python -m experiments.bev_yolo.ryu_eval --arm bev_seg

# how it lands on the other arms
python -m experiments.bev_yolo.ryu_eval --arm all --variant rect,paper

# the paper's eq 12 as written -- watch the detection rate collapse
python -m experiments.bev_yolo.ryu_eval --arm bev_seg --oracle-only \
    --variant paper --pivot principal

# animate one approach twice, known vs fitted heading, into media/
python -m experiments.bev_yolo.ryu_video --episode 000131 --fps 60 --gif
```

`ryu_video.py` replays a single descent beside the source camera and the BEV
canvas the detector was given. Four things about it are deliberate:

- **Every captured frame, not the built subset.** It reads the collector episode
  directly and warps on the fly, so `EFJO/28` is 431 frames from 2749 m -- 238
  inside the window -- rather than the 48 the stride-5 build kept. The frames outside the
  experiment's range window are shown greyed and labelled — watching the
  estimator have nothing to work with at 2.5 km, then lock on as the runway
  resolves, is most of the value of animating this.
- **The filter is causal.** `kalman_track` is a forward pass, so the blue line at
  any frame uses only measurements up to that frame, and it is blank until the
  first detection — before that the filter has only its prior, and drawing the
  prior would put a confident line through a stretch where nothing was observed.
- **Shared axes across the pair**, because two plots that each rescale to their
  own error are two plots that hide the difference. Three rules make that
  readable, all forced by measurements on this episode:
  - *Limits are asymmetric, not `±max`.* In-window cross-track runs −17.1 … +10.2,
    so a symmetric axis leaves the whole upper band empty and puts the signal
    off-centre.
  - *Estimates enter the limits at the 1st/99th percentile*, the truth whole. The
    truth is the line everything is read against; one blown frame should not set
    the scale.
  - *Lines break where they leave the axes* rather than being clipped. Outside
    the window the estimates reach −365 m and +711 m; clipped to an 18 m axis
    they draw as solid vertical bars that dominate the plot and say nothing.
    In-window there are no such points at all — 0/238 frames beyond ±30 m for
    either variant — so every one of those stripes came from the dead zone.
  - *The x-axis leads in 30% of the window*, not all the way to 2749 m, which
    was 48% of the plot width for almost no content.
- **The gif is coarser than the mp4** (20 fps, 1000 px). A 60 fps gif of 431
  frames is tens of megabytes and plays badly.

On `EFJO/28`, over the 238 in-window frames: known heading **1.92 m MAE /
2.09 m RMSE**, fitted **8.33 / 9.32** — and the two converge over the last 200 m
as the `range · sin δψ` lever collapses.

Synthetic tests, no data needed:

```bash
python -m experiments.tests.test_ryu
```

## The shape leg: stop reading two points

**The paper's error is worst where it should be easiest** — 2.50 m MAE inside
300 m against 1.69 m at 1000–1500 m, the opposite of what the range lever
predicts. The cause is not the mask. Measured on the test split, the predicted
mask's area is within 0.3% of the label's at every range and its fitted width is
good to 0.64 m inside 300 m. What degrades is the **tail of the threshold
midpoint**:

| range | threshold lateral error, median | **p90** | cross-track MAE |
|---|---|---|---|
| 150–300 m | −1.11 m | **6.21** | 2.50 |
| 300–600 m | −1.36 m | 3.87 | 2.04 |
| 600–1000 m | −1.41 m | **2.25** | 1.69 |

The worst frames are 13–23 m out — about half a runway width (45/2 = 22.5 m).
That is the signature of the two-point rule latching onto a **partial** front
edge: read two points, and a clipped edge or a rounded corner moves their
midpoint by up to half a width. `test_ryu.py` reproduces it — eating half a front
edge costs the two-point rule 16.5 m and the centroid 5.0 m.

So replace the two points with something fitted to the whole mask. Two candidates,
both of which land on the runway's centre line:

| `edge` | what it reads |
|---|---|
| `rect` | near edge of the **minimum-area rectangle** |
| `axis` | the polygon's **area centroid** |

**Not a rhombus.** In a metric BEV with square pixels — which `grids.py`
guarantees deliberately — a runway is *exactly* a rectangle. A rhombus has four
equal sides; a 45 × 2500 m runway is not remotely that. A general parallelogram
would only be right under anisotropic BEV scaling, which this does not have.

### Result

Test split, n=567, `bev_seg`, predicted masks:

| variant | MAE | RMSE | **MAX** | 150–300 | 300–600 | 600–1000 | 1000–1500 |
|---|---|---|---|---|---|---|---|
| `paper` (two points) | 1.873 | 2.909 | 18.69 | 2.497 | 2.035 | 1.693 | 1.694 |
| `paper_rect` (enclosing rect) | 1.869 | 2.438 | 12.55 | 2.082 | 1.535 | 1.646 | 2.343 |
| **`paper_axis`** (centroid) | **1.224** | **1.478** | **4.77** | **1.530** | **1.324** | **1.133** | **1.120** |
| `paper_axis` + Kalman | 1.224 | 1.452 | **3.52** | 1.529 | 1.311 | 1.099 | 1.172 |

**The centroid wins by 35% and kills the tail** — MAX falls 18.69 → 4.77 m. And
it finally behaves the way the geometry says it should: error *decreasing*
monotonically as the aircraft closes in (1.53 → 1.12 m), instead of getting worse.
Bias drops from +0.92 to −0.16 m.

**The enclosing rectangle is the right shape but the wrong estimator.** It fixes
the near field (2.50 → 2.08 m) and halves the worst case, but *degrades* at long
range (1.69 → 2.34 m) — because it is a **max** statistic, set by the outermost
pixels, and at range the far end of the mask is smeared across the canvas. The
centroid is an average and does not care.

**The Kalman filter has almost nothing left to do** (1.224 → 1.224 MAE). Most of
what it was buying was absorbing exactly the outliers the shape fit now prevents.

### The oracle says the trade explicitly

| variant | oracle | predicted | gap |
|---|---|---|---|
| `paper` | **0.254** | 1.873 | 1.62 — detector-limited |
| `paper_axis` | 1.118 | **1.224** | **0.11** — geometry-limited |

The centroid gives up accuracy on a *perfect* mask to gain robustness on a *real*
one, and ends up within 0.11 m of its own floor while the two-point rule sits 7×
above its.

**That floor is a label defect, not a method defect**, and it is the README's own
"unexplained angular floor" showing up a second time. The label quads are rotated
**0.235°** from the true runway axis; the centroid sits **309 m** past the
threshold, where that rotation has a lever. Predicted offset `lever · sin(angle)`
tracks the measured offset with **correlation +0.999 and slope 0.99**, and the
residual once the lever is removed is **0.247 m** — the same floor the front edge
has. On correct geometry the centroid is exact to 0.007 m
(`test_axis_mode_recovers_crosstrack`). **Fixing the label rotation would take
`paper_axis` from 1.22 m toward 0.3 m**; nothing else on this list has that much
headroom.

### It rescues the perspective arm too

Running the same swap on `persp_seg`, whose failure was a heavy tail (best median
of any arm at 0.62 m, worst mean at 4.83 m):

| `persp_seg` | MAE | RMSE | 150–300 | 1000–1500 |
|---|---|---|---|---|
| `paper` | 4.833 | 8.087 | 1.238 | 4.869 |
| `paper_axis` | **1.100** | 2.051 | **0.942** | **1.409** |

A 4.4× improvement, and it puts perspective (1.100) back ahead of BEV (1.224).
**This weakens the "BEV finally beats perspective" claim above**: that result was
substantially about which estimator tolerates a heavy-tailed mask, not about the
view. With a tail-resistant estimator the two are close, and perspective's better
angular resolution reasserts itself. Its MAX is still 36 m against BEV's 4.8 m,
so BEV remains the safer of the two — but the honest summary is now *"BEV is more
robust, perspective is more accurate"*, not *"BEV wins"*.

```bash
python -m experiments.bev_yolo.ryu_eval --arm bev_seg \
    --variant paper,paper_rect,paper_axis
python -m experiments.bev_yolo.ryu_video --episode 000131 --variant paper,paper_axis
```

## Where this sits in the field

[LITERATURE.md](LITERATURE.md) surveys the alternatives — keypoints+PnP, line and
vanishing-point methods, BEV-free 3D anchors, differentiable PnP, metric depth,
vision-inertial fusion, conformal/integrity bounds — and scores each against what
this dataset actually holds. Three things from it bear directly on the results
above:

- **The BEV negative result is corroborated externally.** Monocular 3D lane
  detection moved from IPM-BEV (3D-LaneNet, PersFormer) to BEV-free (Anchor3DLane,
  LATR) for the same reason: the flat-ground assumption IPM requires.
- **Four-corner PnP is weakly conditioned for cross-track at range.** An
  integrity-bound paper puts the cross-track protection level at 24–52 m at 1 km
  with four corners — against 1.69 m MAE here — which is Ryu et al.'s thesis
  arriving independently.
- **Generic heading sources are not accurate enough to close the `fitted_yaw`
  gap.** Vanishing-point detectors run 0.5–1.5°, horizon attitude ~1.7° RMS; the
  fitted box here is already 0.36°, and 1 m of cross-track at 1.4 km costs 0.041°.

## What this leaves open

- **The warp is still ours, not theirs.** This ports the estimator onto the
  existing metric BEV raster. The paper's own BEV is an attitude-only quad warp
  picked from vanishing-point geometry (their Figs 8–10) with no metric meaning —
  which is *why* it needs `k_s`. Since `k_s` costs nothing here, that warp is
  probably not worth building, but it is untested: it would go in as a `vp` grid
  mode in `grids.py` plus an arm, and needs a rebuild and a retrain.
- **The filter is not gated.** It helps (MAX 18.7 → 10.6 m) but a chi-square gate
  on the innovation would help `persp_seg` far more, since its failure mode is
  exactly the heavy tail a gate removes.
- **No PnP comparison.** The paper's headline is beating PnP (3.14 → 1.78 m MAE).
  That is step 4 of the capstone plan and would slot into `ryu_eval.py` as another
  variant reading four corners instead of two.
- **`yaw` is binary here.** The interesting sweep is injecting graded heading
  noise — 0.01°, 0.05°, 0.2° — to find how good a heading source has to be before
  the estimator degrades to `fitted_yaw`. `test_heading_error_is_levered_by_range`
  has the mechanism; the sweep is a loop over `yaw_deg + δ`.
