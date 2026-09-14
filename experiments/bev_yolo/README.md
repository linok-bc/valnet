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
survives as its lateral offset. Both unknowns end up as two numbers you can read
off one oriented box:

    yaw        = -(the box's angle from BEV "up")
    cross-track = -(near-edge midpoint) . (the box's right-normal)

## The four arms

|             | `segment` (polygon) | `obb` (oriented box) |
|-------------|---------------------|----------------------|
| perspective | `persp_seg` — the control: what VALNet and the YOLOv8-seg baseline already do | `persp_obb` — a box around a trapezoid; expected to fail, and it is here to show *how* it fails |
| bev         | `bev_seg` | `bev_obb` — **the proposal** |

Every arm sees **exactly the same frames**. A frame is kept only if it is usable
in both views, so a difference between arms cannot come from one of them having
been handed an easier subset.

## Scoring: metres, not mAP

mAP in each arm's own pixels is not comparable across arms. A BEV pixel is a
fixed patch of ground; a perspective pixel is not; an IoU threshold therefore
means a different physical tolerance in each. Worse, mAP rewards the wrong
thing — a mask can score well while its edges wander by the few metres that are
the whole signal.

So every arm goes through **one shared pipeline**: prediction → polygon →
ground plane in metres (using the known altitude and attitude) → minimum-area
rectangle → heading and cross-track, scored against the logs and binned by
range. Identical post-processing everywhere, so only the representation differs.

`eval.py` prints an **ORACLE** row above every result: the ground-truth label
through the same pipeline. It is the floor — no detector on that arm can beat
it. Read it first. If the oracle is already several metres out, the arm is
limited by its geometry or its labels, and training harder will not help.

## Run it

```bash
# 0. does the camera model agree with the logs? do this on any regenerated data
python -m experiments.bev_yolo.calibrate --data ~/Documents/xp12_dataset --all-signs

# 1. look at the warps before spending a GPU-hour on them
python -m experiments.bev_yolo.preview --data ~/Documents/xp12_dataset -n 8

# 2. how many frames survive, at what resolution? writes nothing
python -m experiments.bev_yolo.build_dataset --dry-run

# 3. build all four arms (~10 min for 250 episodes at stride 5)
python -m experiments.bev_yolo.build_dataset --clean

# 4. the floor, before any training
python -m experiments.bev_yolo.eval --oracle-only --split val

# 5. train and score
python -m experiments.bev_yolo.train --arm all
python -m experiments.bev_yolo.eval --arm all
```

Everything is driven by `config.yaml`; point `source.root` at the regenerated
dataset. The synthetic tests need no data at all:

```bash
python -m experiments.tests.test_bev
python -m experiments.tests.test_estimator
```

## What is already known, before training anything

Measured on the *old* `datasets/xp12_dataset` (40 episodes, stride 25), so the
absolute numbers will move on the regenerated data — but the structure will not.

**The premise holds.** In BEV the runway warps to a rectangle to within
floating-point error, under arbitrary roll and pitch, with the heading error
preserved exactly (`test_bev.py`). Cross-track and yaw come back through the
full camera → warp → box-fit chain with zero error (`test_estimator.py`).

**`bev_seg` and `bev_obb` have the *same* oracle, to three decimal places.**
Once rectified, the polygon *is* its own minimum-area rectangle. The mask
carries no information the four corners do not — which is the strongest
available argument for dropping segmentation in favour of a box.

**`persp_obb`'s oracle is ~127 m of cross-track error** against ~1 m for the
others. A box around a perspective trapezoid is not a description of a runway.
That arm is the control that shows the BEV step is doing the work, not the box.

**The perspective and BEV oracles are otherwise identical** (1.09 m median,
which is old-label noise), because the back-projection geometry is the same.
The experiment therefore isolates exactly one question: does a *network* find
the target more accurately in one representation than the other?

## Design decisions worth knowing about

**The canvas is anchored on the known along-track range, at a fixed scale**
(`bev.mode: threshold`). A 45 m runway is ~60 px across in *every* image, at
1.4 km and at 200 m alike, and the threshold always lands on the same canvas
row. The detector never has to learn scale — only the two unknowns. The
alternatives (`altitude`, `horizon`, `fixed`) do not use along-track and are
there for the case where that assumption is dropped.

**Range is capped at 1500 m.** Beyond that the aircraft in this dataset is still
*capturing* the localizer, not flying an approach: measured over 60 episodes,
|yaw| runs to 14° and cross-track to 54 m median past 2.2 km, which throws the
threshold hundreds of metres off the heading axis. Inside 1500 m the p99 lateral
offset is 63 m and the canvas holds it comfortably.

**Frames where the runway runs off the *side* of the canvas are dropped.**
Far-end clipping is fine — the estimator only uses the near edge. Side clipping
shaves the runway lengthwise, so the fitted box comes out narrower than the
runway and its centre is pulled inboard, biasing cross-track by metres. The
tolerable yaw scales as 1/range (`test_canvas_yaw_limit`); the builder reports
how many frames this costs.

**The perspective arm is clipped to the same ground window as the BEV arm.**
Without this the comparison is rigged: the BEV canvas stops a few hundred metres
past the threshold, while a perspective mask runs to the horizon, where one
pixel of error is tens of metres of ground. That is a real property of
perspective masks, but of the *window*, not the representation. Left unclipped
it costs the perspective oracle 4× (1.09 m → 4.46 m median, and a −25 m bias
close in); `--no-ground-window` shows it.

**Warping cannot add detail.** The builder prints the ratio of BEV pixels to
source pixels across the runway width; at the current settings it is ~1.0
median. Far above 1 means the canvas is interpolating; far below means it is
throwing information away.

## Known limits

- **The far field is out of scope.** A fixed-scale canvas cannot hold a runway
  that is 500 m off the heading axis. The fix, if it is ever needed, is Ryu's own
  two-step approach: a coarse detection first, then a tight warp around it.
- **The camera model is fitted, not measured.** `calibrate.py` recovers
  fx ≈ 1003, fy ≈ 1015, tilt ≈ +0.03° from the logs, at a 1.4 px median corner
  residual on the old data. Re-run it on the regenerated dataset; a residual
  above ~10 px means something in the chain disagrees and nothing downstream is
  trustworthy.
- **The old labels are noisy** (near-plane clipping, runway corners projected at
  the airport reference elevation). That noise is the 1.09 m oracle floor. The
  regenerated dataset should lower it — the oracle row is how you will know.
- **`detect` (axis-aligned) is not one of the four arms**, because it cannot
  represent yaw at all: at ±2° of lean the upright box is 67 m wide around a
  45 m runway (`test_estimator.py`). It is implemented in `labels.py` if you
  want to quantify that.
