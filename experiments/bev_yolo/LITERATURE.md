# Cross-track estimation: what else exists, and what this dataset can support

A survey of methods for recovering lateral offset from a forward-facing camera —
in the runway domain and in three adjacent ones — scored against what
`datasets/xp12_dataset` actually contains.

Reviewed 2026-09-14. Companion to [README.md](README.md); the numbers quoted for
"ours" come from its two experiments.

## What the dataset constrains

Measured, not assumed (118 episodes, 50 820 frames):

| available | not available |
|---|---|
| 1280×720 RGB, X-Plane 12 | **timestamps** |
| 4-corner runway quad, unclipped | **velocities, angular rates** |
| yaw / pitch / roll per frame | real imagery (synthetic only) |
| along / height / right per frame | weather (0 rain, all daytime, wind ≤3 m/s) |
| `runways.dat`: width, length, heading, endpoints, elevation | a clean frame — the X-Plane HUD is baked in |
| fitted intrinsics (fx 1004, tilt +0.04°, 1.4 px residual) | camera tilted down (see README) |

**The missing clock is the binding constraint.** Sim speed varied 1.3–5× between
episodes, so frame index is not proportional to time. Every method whose value
comes from temporal dynamics — EKF/UKF vision-inertial fusion, optical flow,
tracking-by-filtering — is either blocked or must be re-expressed against a
different independent variable, as `kalman_track` does by filtering along
distance flown rather than seconds.

## The result everything below has to beat

From README.md, `bev_seg` test split, n=567:

- **1.65 m MAE / 2.19 m RMSE** with heading from the INS (Ryu et al. + Kalman)
- **6.21 m MAE** with heading fitted from the mask — the camera-only figure
- the gap is *entirely* the heading lever: cross-track error ≈ `range · sin δψ`

So the question for every method below is not "does it estimate cross-track"
but **"does it estimate heading better than 0.36°, or avoid needing it?"** At
1.4 km, 1 m of cross-track costs **0.041°** of heading. That is the bar.

---

## A. Keypoints + PnP

The mainstream. Detect the four runway corners, solve PnP against the known 3D
rectangle. [LARD](https://arxiv.org/pdf/2304.09938) (Airbus/ONERA/IRT
Saint-Exupéry) is the standard dataset and annotates exactly those four corners;
[Robust Validation to Geometric Perturbations](https://arxiv.org/html/2608.21066)
runs YOLOv8-Pose → iterative PnP (and BPnP, which it verifies matches to a 2.2%
median relative translation error) as the reference pipeline.

**The damning number.** [Protection Levels for Vision-Based Pose
Estimation](https://arxiv.org/html/2608.10023) (2026) computes RAIM-style
integrity bounds for this exact pipeline and reports, at 1 km with four corners:
**cross-track protection level 24–52 m**, along-track ~140 m position-only, and
**yaw/roll orientation bounds >4°**. Those are bounds rather than errors, but
they say four corners at 1 km constrain cross-track weakly — against the 1.69 m
MAE the width-scaled front-edge estimator gets in the 1000–1500 m bin. The
robustness paper's ~70 m median 3D translation error beyond 1 km points the
same way.

**Feasibility: immediate, and it is roadmap step 4.** The labels *are* four
corners, `runways.dat` gives the 3D model, intrinsics are fitted. Drops into
`ryu_eval.py` as another variant. Worth doing precisely because the literature
predicts it loses — Ryu et al.'s own headline is beating PnP 3.14 → 1.78 m MAE,
and reproducing that on your data is a strong result.

## B. Lines, vanishing points, and visual servoing

The oldest and, for this problem, the best-motivated family: the runway's long
edges carry the lateral signal over a far longer baseline than its corners do.
Bourquardez & Chaumette ([IROS/ICRA
2007](https://www.irisa.fr/lagadic/pdf/2007_iros_bourquardez.pdf)) servo an
airplane onto a runway from the two edge lines in bi-normalised Plücker
coordinates, never forming a metric pose at all. Modern learned versions predict
lines alongside the mask: **ERFE** (Chen et al., IET Image Processing 2024)
emits a segmentation map *and* a line-probability map from a MobileNetV3
backbone, and [joint segmentation + line detection via multi-task
learning](https://www.sciencedirect.com/science/article/abs/pii/S1047320325002032)
(2025) does the same with task-specific decoders.

**The trap.** Generic vanishing-point detectors are not accurate enough here.
[NeurVPS](https://arxiv.org/pdf/1910.06316) reports ~0.5° median for the up-VP
and ~1.5° over all VPs; horizon-based attitude estimation runs
[RMS < 1.7°](https://journals.sagepub.com/doi/full/10.1177/1756829318804761).
Both are **worse than the 0.36° your fitted box already achieves**, and 1.5° at
1.4 km is 37 m of cross-track. A VP detector is not the fix for the heading
lever; a *runway-specific, long-baseline edge fit with an uncertainty model*
might be.

**Feasibility: high, and it needs no new labels.** The 4-corner quad gives both
edge lines for free. This is roadmap step 2. The one thing to measure first is
whether fitting two long edges beats fitting a rectangle to a mask — that is a
one-evening experiment against the existing oracle.

## C. BEV / IPM — and why the driving literature agrees with your negative result

Your first experiment found BEV segmentation does not beat perspective
segmentation. Monocular 3D lane detection has converged on the same conclusion.
The [2024 survey](https://arxiv.org/html/2404.06860v1) splits the field into
BEV-based (3D-LaneNet, Gen-LaneNet, PersFormer, BEV-LaneDet — all IPM) and
**BEV-free** (Anchor3DLane, CurveFormer++, LATR), and the BEV-free methods now
lead: LATR reaches the lowest chamfer error on ONCE-3DLane working directly from
perspective features with 3D anchors and no view transform. The named culprit is
the **flat-ground assumption** that IPM requires — the same assumption your
`ground_to_image` homography makes.

This is worth citing in the writeup: your result is not an artefact of your
setup, it is the same finding the driving community reached independently, for
the same reason.

**Feasibility: done.** The interesting remaining variant is BEV-*free* — 3D
anchors in the perspective image, LATR-style — which would be a genuinely new
arm rather than a re-run.

## D. Known-width metric scaling, outside aviation

Ryu et al.'s `k_s = W_R / w` trick — recover metric scale from a structure of
known size rather than from altitude — is not aviation-specific. The railway
community uses it with track gauge: [Near-incident detection in railroad
environments](https://openaccess.thecvf.com/content/CVPR2025W/USM3D/papers/Wang_Near-incident_detection_in_railroad_environments_lateral_distance_estimation_from_train-mounted_CVPRW_2025_paper.pdf)
(CVPR 2025 workshop) calibrates monocular depth against the constant rail gauge
and measures lateral distance from the track, with a two-stage near-field
(sleeper-based) / far-field (geometric) calibration.

That two-stage idea transfers directly: near the threshold the runway *width* is
well resolved, far out the runway *length/edges* are the better-conditioned
reference. Your `scale=width` uses one reference at all ranges.

**Feasibility: implemented; the two-stage extension is cheap.** Note the measured
result that the width scale costs essentially nothing for cross-track
(2.679 → 2.687 m oracle) — so this buys along-track, not lateral.

## E. Differentiable PnP and learned correspondences

[EPro-PnP](https://arxiv.org/html/2303.12787) (CVPR 2022, TPAMI 2024) makes the
PnP layer differentiable by predicting a *distribution* over pose on SE(3),
learning 2D-3D correspondences and per-point weights end-to-end from a pose
loss. BPnP is the simpler backpropagatable alternative, already validated
against iterative PnP in the LARD robustness paper.

**Feasibility: possible but data-thin.** 118 episodes over ~178 airports, all
synthetic, no weather variation. End-to-end correspondence learning is exactly
the regime where that becomes the limit — and your memory already records
Sattler et al.'s warning that pose regressors degenerate toward image retrieval.
Keep it as roadmap step 6, after the geometric baselines are pinned.

## F. Multi-task segmentation + depth + slope

[Monocular visual estimation for autonomous aircraft landing guidance in unknown
structured scenes](https://www.sciopen.com/article/10.1016/j.cja.2025.103479)
(Chinese J. Aeronautics, 2025) trains one network for segmentation, depth and
slope jointly, targeting runways whose geometry is *not* known in advance — the
case where `runways.dat` would not exist.

**Feasibility: moderate, and partly synthesisable.** You have no depth ground
truth, but you can generate it: the flat-ground homography plus `runways.dat`
gives exact depth for every ground pixel. Whether that is worth it depends on
whether you care about the unknown-runway case; for a capstone scoped to known
runways, probably not.

## G. Metric depth foundation models

Metric3D v2 ([arXiv 2404.15506](https://arxiv.org/abs/2404.15506), TPAMI 2024)
recovers metric depth zero-shot via a canonical camera-space transform.
Tempting, because it would remove the need for a width or altitude reference
entirely.

**Feasibility: poor, and the evidence is against it.** [AerialMetric](https://arxiv.org/pdf/2606.29716)
(2026) benchmarks these models on UAV imagery and finds metric accuracy **drops
significantly as altitude increases** — the regime you live in. You also already
know altitude exactly, so the ambiguity these models resolve is not one you have.

## H. Filtering and vision-inertial fusion

Standard in the field: EKF fusing the runway stripe for height and lateral
deviation, [CFC-CKF](https://www.researchgate.net/publication/383131627_Real-time_vision-inertial_landing_navigation_for_fixed-wing_aircraft_with_CFC-CKF)
tightly integrating vision and IMU for fixed-wing landing, and [optical
navigation for runway-relative positioning](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8004248/)
fusing camera, low-cost IMU and barometer. Typical rates are IMU ~100 Hz against
camera ~10 Hz.

**Feasibility: BLOCKED by the dataset, and this is the most important finding
here.** There are no timestamps, no velocities and no angular rates, and sim
speed varied 1.3–5×, so no honest EKF can be built on this data as collected.
`kalman_track` works around it by using along-track distance as the independent
variable, which is defensible for a smoother but is not vision-inertial fusion.

Given that heading is the entire error budget and temporal accumulation is the
standard way to sharpen heading, **adding a clock and body rates to the collector
is likely the single highest-leverage change available** — higher than any
method on this list.

## I. Uncertainty and integrity — the cheapest real win

Three recent papers, all aimed at certification rather than accuracy:

- [Conformal prediction for runway detection](https://arxiv.org/pdf/2505.16740)
  — distribution-free coverage guarantees on the detection, with a "conformal
  mAP" metric.
- [Protection levels](https://arxiv.org/html/2608.10023) — RAIM-style bounds
  that hold under undetected keypoint faults; needs only per-keypoint
  covariances and the linearised projection.
- [Predictive uncertainty for runtime assurance](https://arxiv.org/pdf/2508.09732)
  of a real-time vision-based landing system.

**Feasibility: high, low effort, and it is already roadmap step 3 (the σ head).**
Conformal prediction needs nothing but a held-out calibration split, and you
already have airport-grouped splits. Protection levels need a covariance per
keypoint, which the same σ head would give you. For a capstone, "my estimator
reports a bound it actually meets" is a stronger claim than another 0.2 m of MAE.

---

## Where this leaves the roadmap

Ranked by (value × feasibility) on the data as it exists today:

| | direction | why | cost |
|---|---|---|---|
| 1 | **Add a clock + body rates to the collector** | unblocks the entire filtering literature; heading is the error budget and time is how you beat it down | collector change + recollect |
| 2 | **Long-baseline edge/line fit with σ** (roadmap 2) | targets the actual bottleneck; no new labels; beware that generic VP detectors are *worse* than what you have | days |
| 3 | **4-corner PnP variant** (roadmap 4) | the comparison Ryu et al. claim to win; literature predicts 24–52 m cross-track PL at 1 km, so this should lose — which is the result | ~1 day |
| 4 | **Conformal / protection levels** (roadmap 3) | cheapest credible contribution; certification framing is where this field is going | ~2 days |
| 5 | **BEV-free 3D-anchor arm** (LATR-style) | the driving field's answer to the same negative result you found | weeks |
| 6 | EPro-PnP (roadmap 6) | data-thin at 118 episodes; park it | weeks |
| — | metric depth foundation models | evidence says they degrade with altitude, and you already know altitude | skip |

The two claims worth carrying into the writeup:

1. **Your BEV negative result is corroborated** by monocular 3D lane detection,
   which moved from IPM-BEV to BEV-free for the same flat-ground reason.
2. **Four-corner PnP is weakly conditioned for cross-track at range**, and the
   width-scaled front-edge estimator is the better-conditioned use of the same
   mask — which is Ryu et al.'s thesis, now with an independent integrity-bound
   paper agreeing.

## Sources

- Ryu et al., *An estimation method for vision-based autonomous landing system for fixed wing aircraft*, IJRR 44(6) 2025 — https://journals.sagepub.com/doi/abs/10.1177/02783649241287229
- Ducoffe et al., *LARD — Landing Approach Runway Detection* — https://arxiv.org/pdf/2304.09938
- *Robust Validation to Geometric Perturbations for Autonomous Pose Estimation* — https://arxiv.org/html/2608.21066
- *Protection Levels for Vision-Based Pose Estimation* — https://arxiv.org/html/2608.10023
- *Robust Vision-Based Runway Detection through Conformal Prediction* — https://arxiv.org/pdf/2505.16740
- *Predictive Uncertainty for Runtime Assurance* — https://arxiv.org/pdf/2508.09732
- *Monocular 3D Lane Detection for Autonomous Driving: survey* — https://arxiv.org/html/2404.06860v1
- LATR — https://arxiv.org/pdf/2308.04583 · PersFormer — https://arxiv.org/abs/2203.11089 · Anchor3DLane++ — https://arxiv.org/pdf/2412.16889
- Wang et al., *Near-incident detection in railroad environments* (CVPR 2025 W) — https://openaccess.thecvf.com/content/CVPR2025W/USM3D/papers/Wang_Near-incident_detection_in_railroad_environments_lateral_distance_estimation_from_train-mounted_CVPRW_2025_paper.pdf
- Bourquardez & Chaumette, *Visual servoing of an airplane for auto-landing* — https://www.irisa.fr/lagadic/pdf/2007_iros_bourquardez.pdf
- Chen et al., ERFE, IET Image Processing 2024 — https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ipr2.13087
- *Joint airport runway segmentation and line detection* — https://www.sciencedirect.com/science/article/abs/pii/S1047320325002032
- YOLO-RWY, Drones 8(10) 2024 — https://www.mdpi.com/2504-446X/8/10/571
- NeurVPS — https://arxiv.org/pdf/1910.06316
- Carrio et al., *Attitude estimation using horizon detection* — https://journals.sagepub.com/doi/full/10.1177/1756829318804761
- Chen et al., EPro-PnP — https://arxiv.org/html/2303.12787
- *Monocular visual estimation … unknown structured scenes*, CJA 2025 — https://www.sciopen.com/article/10.1016/j.cja.2025.103479
- Metric3D v2 — https://arxiv.org/abs/2404.15506 · AerialMetric — https://arxiv.org/pdf/2606.29716
- *Real-time vision-inertial landing navigation with CFC-CKF* — https://www.researchgate.net/publication/383131627_Real-time_vision-inertial_landing_navigation_for_fixed-wing_aircraft_with_CFC-CKF
- *Optical Navigation Sensor for Runway Relative Positioning* — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8004248/
- Ferreira et al., *Vision-Based Landing Guidance through Tracking and Orientation Estimation*, WACV 2025 — https://openaccess.thecvf.com/content/WACV2025/papers/Ferreira_Vision-Based_Landing_Guidance_through_Tracking_and_Orientation_Estimation_WACV_2025_paper.pdf
- Cheng et al., *Vision-Based Runway Detection and Localization … GNSS-Denied*, J. Field Robotics 2026 — https://onlinelibrary.wiley.com/doi/10.1002/rob.70147
