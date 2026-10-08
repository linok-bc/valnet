# Precise Covariance Computation for (yaw, offset) Estimates

## Context / Motivation

Empirical cost-landscape visualization (L2 and 1-Dice heatmaps over
`(yaw, offset)` for a real ROI) revealed an **elongated, near-flat diagonal
valley** — convex, but ill-conditioned. `grid min`, `Est` (LM refinement) and
`GT` all sit inside the low-cost valley but at different points along its
main axis, because the cost is nearly insensitive to movement along that
direction (classic effect of near-degenerate geometry, expected close to
yaw≈0 / at long range, as flagged earlier in the design).

**Conclusion**: this is not an optimizer bug (LM/Gauss-Newton is still the
right choice over gradient descent for this kind of elongated-but-convex
valley — it uses curvature information via `J^T J` to auto-rescale steps per
direction, which plain GD cannot do). It is a genuine **partial
unobservability of `(yaw, offset)` from a single frame** in this geometric
regime. The correct fix is not "a better optimizer" but (a) fusing the
temporal prior (this valley is exactly what the IEKF prior term is for) and
(b) exposing the *directional* uncertainty explicitly rather than reporting
a single point estimate. A scalar variance cannot represent "confident
perpendicular to the valley, unconfident along it" — a full 2×2 covariance
with its eigenstructure is required.

## 1. Fisher information from the current frame's data alone

At the converged estimate `x* = (yaw*, offset*)`, with the measurement
Jacobian `H` (the `dd_dx` from `refine`, shape `(N,2)`) and diagonal
measurement noise `R` (`(N,N)`):

```
I_data = H^T · R^-1 · H          # (2,2) Fisher information matrix
```

If `I_data` has a near-zero eigenvalue along some direction, the data alone
barely constrains that direction — this is exactly the flat valley observed
empirically. Without a prior, `I_data` can be near-singular in that
direction (variance → infinity), which is why fusing the temporal prior is
not optional in this regime.

## 2. Fuse the temporal (IEKF) prior — required for a well-posed result

```
I_total = I_data + P_pred^-1
P_t     = I_total^-1
```

This is algebraically equivalent to the already-defined
`P_t = (I - K·H) · P_pred` update, just written in information form — useful
here because it makes explicit *why* the prior term is what prevents the
covariance from blowing up along the ill-conditioned direction. **If a
refine/LM run is done standalone without `P_pred` (data-only), it is
structurally expected to misbehave in this kind of valley — this is a
missing-prior issue, not an LM defect.**

## 3. Eigendecomposition — read out the uncertainty direction

```python
eigvals, eigvecs = np.linalg.eigh(P_t)     # P_t is symmetric
# numpy convention: eigvals ascending
uncertain_direction = eigvecs[:, 1]         # eigenvector of the larger eigenvalue
sigma_max = np.sqrt(eigvals[1])             # stddev along the ill-conditioned direction
sigma_min = np.sqrt(eigvals[0])             # stddev perpendicular to it (should be much smaller)
```

**Sanity check**: `uncertain_direction` should be near-collinear with the
diagonal axis of the observed cost valley. Verify this on real
frames/heatmaps before trusting the covariance pipeline.

## 4. Confidence ellipse (diagnostic / visualization)

```
ellipse: x* + R(θ) · diag(sqrt(χ² · eigvals)) · [cos t, sin t]^T,   t in [0, 2π]
```
with `χ² = 5.991` for a 95% confidence region in 2D
(`scipy.stats.chi2.ppf(0.95, df=2)`). Overlay this directly on the cost
heatmap as the primary diagnostic — it should hug the valley shape if the
computation is correct.

## 5. Independent numerical check (Laplace approximation)

Cross-check the analytical `H`/`I_data` against a numerical Hessian of the
scalar cost function, to catch Jacobian derivation errors (sign errors are
the most common bug in this kind of estimator):

```python
def numerical_hessian(cost_fn, x, eps=1e-4):
    Hn = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            ei, ej = np.eye(2)[i] * eps, np.eye(2)[j] * eps
            Hn[i, j] = (cost_fn(x+ei+ej) - cost_fn(x+ei-ej)
                        - cost_fn(x-ei+ej) + cost_fn(x-ei-ej)) / (4 * eps**2)
    return Hn

P_check = np.linalg.inv(0.5 * numerical_hessian(cost_fn, x_star))
# factor 0.5: cost is a sum of squared residuals, Hessian of the squared
# term differs from the Fisher/Gauss-Newton approximation by this factor.
```
Compare `P_check` against `P_t` from step 2 (eigenvalues and eigenvector
directions) — large discrepancy flags a bug in the analytical Jacobian
rather than a real modeling issue.

## 6. Integration into the pipeline / API contract

Every frame's output must include, in addition to `(yaw, offset)`:
- Full `P_t` (2×2), or equivalently `(eigvals, eigvecs)`.
- NIS (already defined in main spec.md §4.6/6).

**These two diagnostics are complementary, not redundant**:
- **NIS** flags a broken consistency between prediction and measurement
  ("something doesn't add up" — triggers reacquisition/coarse search).
- **`P_t` eigenstructure** flags *why* the current estimate might be
  imprecise even when everything is otherwise consistent — e.g. a large
  isolated eigenvalue means "roughly right, but not well pinned down along
  this specific direction", a known and expected condition at long range /
  near yaw≈0, not an anomaly.

Downstream consumers (integrity monitor, Kalman fusion with other sensors)
should use the eigenstructure to decide how much to trust the estimate along
each direction, rather than treating the point estimate as equally reliable
in all directions.

## Suggested follow-up validation

Plot the same cost-landscape heatmap across frames at decreasing range to
confirm the valley narrows as the aircraft approaches — this validates that
the ill-conditioning is a geometry-driven, range-dependent effect (expected
to resolve via temporal fusion as range decreases) rather than a pipeline
defect, and calibrates how many frames of temporal fusion are typically
needed to resolve the ambiguity at a given range.
