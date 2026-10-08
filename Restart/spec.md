# Monocular Runway Relative Pose Estimation — Design Spec

## 1. Problem Statement

Given a monocular RGB cockpit camera view during landing, estimate the two
degrees of freedom that cannot be recovered from onboard instruments:

- **yaw** — aircraft heading relative to runway heading (a.k.a. Heading Error, HE)
- **lateral offset** — cross-track distance from the runway centerline (a.k.a.
  Cross-Track Error, CTE)

The other 4 DOF (altitude / height AGL, pitch, roll, along-track distance —
exact set TBD by the team) are assumed known from other sensors (radalt,
AHRS/IMU), each with **known or estimable noise characteristics**, and are
**optional** inputs to the system (must degrade gracefully when missing or
untrusted).

The runway is modeled as a **known planar rectangle** (known or optionally
known width/length prior).

## 2. Prior Art (must read before implementation / publication)

The general direction (segmentation/keypoints + geometric pose solve for
runway-relative pose during landing, with learned uncertainty) is an **active
research area**, not novel by itself. Closest known work:

- **arXiv:2304.09938 — LARD** (Ducoffe et al., Airbus/ONERA/IRT Saint
  Exupéry/Scalian): public dataset, >17k synthetic images + ~1800 annotated
  real landing-footage images, includes a synthetic generator (Google Earth
  Studio based) and full pose metadata (along-track distance, lateral/vertical
  path angle, pitch/roll/yaw). GitHub: `deel-ai/LARD`. **Primary dataset
  candidate.**
- **arXiv:2508.09732 — "Predictive Uncertainty for Runtime Assurance of a
  Real-Time Computer Vision-Based Landing System"**: soft-argmax keypoint
  regression + calibrated predictive uncertainty + RAIM-inspired integrity
  monitoring adapted to nonlinear PnP. **Closest prior work — must be cited
  and differentiated against explicitly.**
- **arXiv:2608.10023 — "Protection Levels for Vision-Based Pose Estimation"**
  (accepted AIAA/IEEE DASC 2026): follow-up deriving probabilistic protection
  levels (integrity bounds) for the above pipeline, handling undetected
  keypoint faults.
- **US Patent 12148183**: PnP with n=2 points assuming rectangular/planar
  runway, reducing the problem to 2 unknown angles (lateral/vertical angular
  deviation) given known runway length/width — same DOF-reduction idea used
  here.
- Older patent (US8284997): classical vanishing-point method — fit
  centerline slope/intercept, compute vanishing point position, derive yaw
  and lateral offset from known focal length + pitch. Useful as a reference
  baseline / sanity-check implementation.
- VALNet (MDPI Remote Sensing 2024), CJA 2025 multi-task network (seg +
  depth + slope + sparse keypoints), MDPI 2026 line-feature width-estimation
  paper: general evidence the segmentation→pose pipeline space is active.

**Differentiation angle for this project**: (a) working purely in
mask-probability space (not RGB) to reduce sim2real gap, (b) treating known
DOF as *fully optional, noisy, learnable-confidence* side inputs rather than
hard-coded PnP constants, (c) analysis-by-synthesis (render-and-compare)
directly against the soft mask instead of extracting intermediate
keypoints/lines, (d) redundant metric-scale anchoring (runway width prior vs.
altitude) with explicit cross-consistency as an anomaly signal, (e) temporal
tracking via EKF/IEKF with NIS-based reacquisition instead of frame-independent
regression.

## 3. Discarded / Superseded Approaches (kept for reference)

### 3.1 Approach A — segmentation → centerline extraction → closed-form geometry
Classical vanishing-point method. Interpretable, zero training data needed,
but fragile: unstable near yaw≈0 (vanishing point → infinity), sensitive to
noisy/incomplete mask contours, assumes perfect rectangle. Useful as a
reference baseline for benchmarking, not as the primary method.

### 3.2 Approach B — black-masked image + optional side-inputs → direct CNN regression
Robust to segmentation noise but loses interpretability, requires large
representative training data, direct point-estimate regression struggles
with multimodal/ambiguous cases. Superseded by Approach D below, but the
input-conditioning design (§5.4) is retained.

### 3.3 Approach C — hybrid: learned keypoint heatmaps + differentiable PnP solver
Soft-argmax keypoint heatmaps (4 corners or edge-line params) + visibility
confidence per point, fed into a differentiable geometric solver
(BPnP-style), trained end-to-end with a keypoint loss + task loss + a
reprojection-consistency auxiliary loss. Viable and closely matches the
"Predictive Uncertainty..." prior-art pipeline (§2). **Retained as a fast
initializer / fallback**, superseded as primary estimator by Approach D.

## 4. Selected Approach (D) — Analysis-by-Synthesis / Render-and-Compare

### 4.1 Core idea
Let `g_dof: R^2 -> {H x W mask}` be the **deterministic, analytic** forward
render of the known rectangular runway model as a function of
`(yaw, lateral_offset)` only, given the other (fixed) DOF and runway
dimension prior. Given the observed soft segmentation mask `Y`, solve:

```
(yaw*, offset*) = argmin_{yaw, offset} distance(Y, g_dof(yaw, offset))
```

`g_dof` is available in closed form (standard pinhole projection of a
rectangle's edges/corners as a function of the 2 unknowns), including its
**analytic Jacobian** w.r.t. `(yaw, offset)`.

### 4.2 Why this approach
- No fragile intermediate feature extraction (no line/corner fitting from a
  noisy mask) — the comparison happens directly in mask space.
- With only 2 unknowns, dense/coarse grid search and local refinement are
  computationally trivial (unlike standard 6DOF render-and-compare).
- The residual/distance landscape over `(yaw, offset)` **is** the likelihood
  — gives calibrated, "free" uncertainty (including multimodality) without a
  separate learned uncertainty head.
- `g_dof` requires zero training data — purely geometric.

### 4.3 Distance metric
Candidates (all computed on an ROI around expected+observed runway location,
never on the full image — background dominance otherwise swamps the metric):
- **Chamfer distance** on soft-mask contours, **truncated/robustified** (cap
  max per-point distance) to limit sensitivity to isolated false-positive
  blobs.
- **Dice / soft-IoU** on the soft mask vs. rendered rectangle indicator —
  robust to small FPs, less precise for fine edge alignment.
- **Weighted L2 / cross-entropy** between soft mask probability and rendered
  rectangle indicator.

**Action item**: empirically benchmark these against the procedurally
corrupted synthetic masks (§6.1) before committing.

### 4.4 Optimization pipeline (per frame)
1. **Predict** (temporal warm-start): `x_pred = x_{t-1} + v_t * dt` using
   known aircraft velocity (or richer motion model if turn-rate/accel
   available).
2. **Coarse search** (only on reacquisition — see §4.6): downsampled mask
   (e.g. 64x64-128x128), grid or coarse-to-fine search over
   `(yaw, offset)` to find candidate basin(s); cheap because dimensionality
   is 2.
3. **Local refinement**: Gauss-Newton / Levenberg-Marquardt using the
   analytic Jacobian of `g_dof`, evaluated against the full-resolution soft
   mask. In steady tracking, **a single GN/LM step per frame** is sufficient
   given small inter-frame motion — do not force full reconvergence every
   frame.
4. Update state — see EKF formalization below.

### 4.5 Temporal tracking = EKF / IEKF, not "velocity-scaled learning rate"
Formalize as an Extended (Iterated) Kalman Filter rather than an ad hoc
gradient step scaled by aircraft speed (speed must **not** directly scale the
correction step — that amplifies measurement noise exactly when the aircraft
is moving fast and the "constant velocity between frames" model is least
reliable).

- **Predict**: `x_t^- = x_{t-1} + v_t * dt`,
  `P_t^- = P_{t-1} + Q` (Q = process noise; increase Q when velocity/dynamics
  are uncertain or maneuvering).
- **Update**: innovation `= Y - g_dof(x_t^-)` (via the chosen distance's
  gradient/residual), analytic Jacobian `H` of `g_dof` at `x_t^-`,
  Kalman gain `K = P_t^- H^T (H P_t^- H^T + R)^-1` (R = measurement noise,
  derived from current-frame mask quality/confidence).
  Single-iteration update: `x_t = x_t^- + K * innovation`,
  `P_t = (I - K H) P_t^-`.
- Velocity's correct role: informs the **predict** step (state shift) and,
  through its own uncertainty, inflates **Q** — not the gain/step size
  directly. The gain is what should legitimately vary with confidence.

### 4.6 Reacquisition / integrity monitoring
Compute **NIS (Normalized Innovation Squared)** each frame. If NIS exceeds a
statistical threshold given `P_t^-` and `R`, treat local tracking as invalid
(trajectory continuity broken) and fall back to the coarse multimodal search
(§4.4 step 2) instead of forcing a local correction into a wrong basin. This
mirrors the RAIM-inspired fault detection in the closest prior work (§2).

### 4.7 Multimodality handling
Point-estimate regression risks silently averaging two valid geometric
hypotheses (dangerous in safety-critical use). Preserve multimodality when
present:
- Coarse search should report **all significant local minima**, not just the
  global one.
- Consider a **particle filter / histogram filter** instead of a plain
  Gaussian EKF for the temporal fusion layer if multimodality is expected to
  persist across frames (EKF assumes unimodal Gaussian belief).
- Periodically re-run a low-frequency coarse search in parallel even without
  an NIS trigger, to catch silently-emerging second modes.

## 5. Handling Known/Optional DOF and Priors

### 5.1 Do not hard-fix known DOF in the solver
Runway dimension prior and the 4 "known" DOF are themselves noisy (sensor
error) and occasionally **grossly wrong** (e.g. wrong runway identified —
width prior wrong by 20-50%, a real historical failure mode in aviation, not
just Gaussian noise).

### 5.2 Joint MAP formulation
Treat the "known" DOF as soft-constrained, not fixed: extend the optimization
to include them as free variables regularized around their measured value
weighted by their own inverse-variance (quadratic penalty), while
`(yaw, offset)` keep a flat/uninformative prior. This is a small
weighted least-squares / bundle-adjustment-like problem (state dim = 4+2)
instead of a pure 2D grid — still tractable.

### 5.3 Redundant metric-scale anchoring
Lateral offset is a **metric** quantity — requires at least one scale anchor
(runway width prior OR altitude). When both are available, compute offset via
both independently, fuse by inverse-variance weighting (Kalman-gain style),
and use their **disagreement** as an explicit anomaly/diagnostic signal
(exposed as an output, not silently absorbed) — analogous to the RAIM
approach in prior art but applied to scale-anchor sources rather than
redundant keypoints.

### 5.4 Optional-input mechanics (if a learned component is kept, e.g. the
fallback keypoint initializer of §3.3)
- Represent each optional scalar as `(value, presence_flag)`, or better
  `(value, sigma)` if sensor noise characteristics are known.
- Inject via FiLM conditioning rather than plain concatenation.
- Train with random input dropout (classifier-free-guidance style) so the
  model doesn't collapse when a subset of inputs is missing.
- Learned components should expose per-source confidence/gating outputs
  (inspectable, auditable) rather than opaque fused scalars — relevant for
  certification/safety review.

## 6. Data Strategy

### 6.1 Synthetic soft-mask generation pipeline
For each sampled ground-truth 6DOF pose + camera intrinsics:
1. Project the known 3D rectangle → perfect quadrilateral in pixel space.
2. `cv2.fillPoly` → hard binary reference mask.
3. Apply procedural corruption (randomized per sample):
   - vertex jitter / elastic boundary deformation
   - random local erosion/dilation
   - local holes (markings, shadows, threshold bars breaking continuity)
   - far-end fade/truncation (probability increasing with distance)
   - occasional unrelated false-positive blobs
4. Convert to soft mask: Gaussian blur on the boundary (randomized band
   width) + interior value drawn from a Beta distribution (e.g. centered
   ~0.9) instead of a hard 1.0, to mimic imperfect calibration.

**Critical**: calibrate corruption parameters against the **actual**
error/IoU/false-positive statistics of the team's real Stage-1 segmentation
model on a held-out validation set — not hand-picked hyperparameters.

**Better long-term approach**: run the real Stage-1 segmentation model on
synthetic RGB renders from a simulator and use its real (soft, unthresholded)
output as training/validation masks, paired with the simulator's exact 6DOF
ground truth. Re-refresh this periodically as Stage-1 segmentation improves,
to avoid training/testing against a stale noise distribution.

### 6.2 Datasets / simulators
- **LARD** (primary): `github.com/deel-ai/LARD` — synthetic + real annotated
  images with full pose metadata; real split (~1800 images) is the main
  source for sim2real validation given its limited size.
- **X-Plane / FlightGear / MSFS**: scriptable flight simulators with
  telemetry (UDP/property tree) for generating effectively unlimited labeled
  synthetic data across airports/approach trajectories; needed to
  supplement LARD's synthetic volume and to run Stage-1 on rendered RGB for
  §6.1's "better" pipeline.
- No confirmed fully-public TaxiNet-style dataset found — worth checking
  independently if a taxiing-domain analog pipeline reference is desired.

## 7. Open Questions / To Decide

- [ ] Exact set of the "4 known DOF" (confirm which of altitude/pitch/
      roll/along-track, and their sensor noise models).
- [ ] Final choice of distance metric (§4.3) — needs empirical ablation.
- [ ] Coarse-search grid resolution / adaptive refinement strategy near
      ill-conditioned regions (yaw≈0, long range).
- [ ] Particle filter vs. EKF for the temporal layer — depends on how often
      multimodality is empirically observed.
- [ ] Whether to keep the learned keypoint initializer (§3.3) in production,
      or whether coarse-grid + warm-start alone (§4.4) is sufficient/faster.
- [ ] Target publication venue if pursued: AIAA/IEEE DASC (best fit given
      closest prior work), IEEE Aerospace Conference, JAIS/IEEE TAES
      (journal route), or ION GNSS+/ITM if emphasizing the integrity angle.

## 8. Suggested Implementation Phases

1. Implement analytic `g_dof` (rectangle projection) + its Jacobian; unit
   test against known closed-form vanishing-point baseline (Approach A /
   patent US8284997) for sanity checking.
2. Implement synthetic soft-mask generator with procedural corruption
   (§6.1), uncalibrated first pass.
3. Implement distance metrics (§4.3) and benchmark them on synthetic data
   with known ground truth vs. corruption severity.
4. Implement coarse-search + GN/LM local refinement (§4.4), no temporal
   filtering yet — validate per-frame accuracy on synthetic + LARD real
   split.
5. Add EKF/IEKF temporal layer (§4.5) with NIS-based reacquisition (§4.6);
   validate on synthetic trajectories with simulated sensor noise on the
   "known" DOF.
6. Add joint MAP handling of noisy/optional known DOF (§5.2) and redundant
   scale-anchor fusion + disagreement signal (§5.3).
7. Calibrate synthetic mask corruption against the real Stage-1 segmentation
   model's actual error statistics (§6.1, "better approach").
8. Evaluate end-to-end on LARD real split; write up ablations for potential
   publication (§7 last item).
