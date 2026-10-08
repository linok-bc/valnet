# Edge-Based Joint Fit for (yaw, offset) — Tier 1 Initializer

## Context

This is a fast, training-free method to get an initial estimate of
`(yaw, offset)` from the runway mask's contour points, to be used as the
warm-start `x0` fed into the full dense-mask `refine` (IEKF/Gauss-Newton on
the soft mask, see main spec.md). It sits between the OBB head (Tier 0,
learned, near-free) and the coarse grid search (Tier 2, last resort) in the
reacquisition hierarchy.

**Important**: this operates on the runway edges in the **original
(non-warped) perspective image**, not the IPM-rectified image — vanishing
point convergence only exists in the unrectified view. In the warped/BEV
view the two edges are parallel by construction, so this method does not
apply there (use the rigid-fit-on-warped-mask method instead in that space).

## The problem with the naive pipeline

The straightforward approach — fit the left edge as an independent line, fit
the right edge as an independent line, intersect them for yaw, then compute
offset separately from the apparent width — chains three sequential steps
with no cross-constraint between them, so estimation error compounds at each
step (and is numerically unstable near yaw ≈ 0, where the vanishing point
goes to infinity).

## The fix: joint constrained nonlinear fit

The two edges are not free/independent lines — they are the projections of
two real-world parallel lines separated by the known runway width `W`, fully
parametrized by the same 2 unknowns `(yaw, offset)` via the same projection
model used for `g_dof`. Fit both simultaneously against all edge points,
directly in `(yaw, offset)` space — no intersection step needed.

### 1. Edge line model — `edge_lines_from_dof(yaw, offset, known_dof)`

For each side (left: `x=-W/2`, right: `x=+W/2` in the runway frame), project
the near and far corners into the image using the same pinhole + rotation
chain as `g_dof`:

```
C_near = (0, ±W/2, 0)      C_far = (L, ±W/2, 0)
p_near(yaw, offset) = π( R(yaw)^T · (C_near - P_ac(offset)) )
p_far(yaw, offset)  = π( R(yaw)^T · (C_far  - P_ac(offset)) )
```

Normalized line form `a·u + b·v + c = 0` (with `a²+b²=1`, so point-to-line
distance is a plain dot product, no division):

```
d = p_far - p_near
(a, b) = normalize(-d.v, d.u)
c = -(a, b) · p_near
```

Both `p_near`/`p_far` (and hence `a, b, c`) are differentiable functions of
`(yaw, offset)` — same chain as `g_dof`, just evaluated at 2 points per edge
instead of a full sampled contour.

### 2. Residual — `point_to_line_dist`

For an observed contour point `p = (u, v)`:
```
dist(p, line) = a·u + b·v + c
```

### 3. Joint cost, with a robust loss (not raw squared error)

```python
def cost(yaw, offset):
    line_L, line_R = edge_lines_from_dof(yaw, offset, known_dof)
    r_L = [huber(point_to_line_dist(p, line_L)) for p in left_pts]
    r_R = [huber(point_to_line_dist(p, line_R)) for p in right_pts]
    return sum(r_L) + sum(r_R)
```

**Huber (or truncated-square) loss is required**, not plain sum-of-squares —
the joint formulation fixes the geometric coupling between the two edges,
but does nothing on its own about outlier robustness (isolated false-positive
mask blobs). Prefer Huber-in-cost over RANSAC-wrapped LM here: same
robustness benefit, one LM solve instead of many, much cheaper — important
given the <200ms/frame budget.

### 4. Point assignment (left_pts / right_pts)

Do **not** cluster contour points blindly. Preferred: reuse the temporal
warm-start already in the pipeline — project the previous frame's estimated
edges into the current frame and assign each contour point to its nearest
expected edge. Falls back to directional filtering (drop near-horizontal
contour tangents — these belong to the near/far short edges, not the
laterals — then 2-way clustering by orientation) only when no prior state is
available (post-reacquisition).

### 5. Solve — Levenberg-Marquardt

Stack all residuals into `r(x) ∈ R^N`, `x = (yaw, offset)`:

1. Linearize: `r(x+Δx) ≈ r(x) + J·Δx`, `J` = `(N×2)` Jacobian.
2. Damped normal equations: `(J^T J + λI) Δx = -J^T r(x)`. The `λI` term
   prevents an overly aggressive step when `J` is ill-conditioned — the
   known failure mode near yaw≈0.
3. Update `x ← x+Δx`; if cost decreased, shrink `λ` (trust Gauss-Newton);
   else grow `λ` (fall back toward gradient descent) and retry.
4. Repeat to convergence.

**Get `J` via autodiff** (`jax.jacfwd` / `torch.autograd`), not by hand —
the full chain (pinhole projection, rotation, line construction, point-line
distance) is composed of elementary differentiable ops; manual derivation is
error-prone (sign errors are the most common bug in this kind of estimator —
verify numerically against finite differences regardless of the derivation
method used).

## Output and role in the pipeline

`(yaw0, offset0)` from this solve is a **fast initializer only** — feed it as
`x0` into the full dense-mask `refine` (main spec.md §4.4/4.5), not as a
final answer. It inherits the same limitations as any contour-based method:
sensitive to the quality of left/right point assignment, and only uses
boundary information rather than the full soft mask's confidence. Its value
is speed (no training required, cheap relative to a full grid search) and a
different failure mode than the learned OBB head (Tier 0) and the dense-mask
refine — useful as a cross-check / fallback layer, not a replacement for
either.
