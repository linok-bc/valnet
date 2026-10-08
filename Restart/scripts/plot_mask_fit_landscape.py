#!/usr/bin/env python3
"""
Plot 2-D cost landscapes dist(g_dof(yaw, lat), mask_yolo) with covariance ellipses.

Per frame (sequential):
  1. Warm grid around init t−1
  2. One LM refinement step from grid winner
  3. Fisher covariance (data + temporal prior) at each point
  4. Overlay 95% confidence ellipses on the L2 heatmap
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pose_estimation.cost_landscape import cost_at_state, evaluate_cost_grid, make_eval_context
from pose_estimation.covariance import (
    PoseCovariance,
    compute_pose_covariance,
    confidence_ellipse_xy,
    default_prior_covariance,
    process_noise,
)
from pose_estimation.g_dof import GDofParams, GDofState
from pose_estimation.optimize import (
    OptimizerConfig,
    WARM_START_COARSE_LEVELS,
    estimate_pose_mask_space,
)
from scripts.validate_mask_fit_sequence import collect_sequence_with_masks


@dataclass(frozen=True)
class AnnotatedPoint:
    state: GDofState
    P: np.ndarray
    label: str
    color: str
    fill: bool = True


def _plot_ellipse(
    ax,
    point: AnnotatedPoint,
    *,
    linestyle: str = "-",
    linewidth: float = 1.6,
    alpha_fill: float = 0.18,
) -> None:
    xl, yw = confidence_ellipse_xy(point.state, point.P)
    ax.plot(
        xl,
        yw,
        color=point.color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=point.label,
        zorder=6,
    )
    if point.fill:
        ax.fill(xl, yw, color=point.color, alpha=alpha_fill, zorder=5)


def plot_landscape_figure(
    sample,
    landscape,
    *,
    state_gt: GDofState,
    points: list[AnnotatedPoint],
    ctx,
    lm_min_gain: float = 0.03,
) -> plt.Figure:
    extent = [
        landscape.lateral_m[0],
        landscape.lateral_m[-1],
        landscape.yaw_deg[0],
        landscape.yaw_deg[-1],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    panels = [
        (axes[0], landscape.l2_masked, "L2  dist(g_dof, mask_yolo)", "L2"),
        (axes[1], landscape.dice_loss_masked, "1−Dice  dist(g_dof, mask_yolo)", "1−Dice"),
    ]

    for ax, grid, title, key in panels:
        vmin = np.nanpercentile(grid, 5)
        vmax = np.nanpercentile(grid, 95)
        im = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.contour(
            landscape.lateral_m,
            landscape.yaw_deg,
            grid,
            levels=12,
            colors="white",
            alpha=0.35,
            linewidths=0.8,
        )
        if key == "L2":
            for pt in points:
                _plot_ellipse(ax, pt)
        for pt in points:
            l2_c, d_c, _ = cost_at_state(ctx, pt.state)
            z = l2_c if key == "L2" else d_c
            ax.scatter(
                pt.state.lateral_offset_m,
                pt.state.yaw_cam_deg,
                c=pt.color,
                s=55,
                edgecolors="black",
                linewidths=0.5,
                zorder=7,
            )
            if key == "L2":
                ax.annotate(
                    pt.label.split()[0],
                    (pt.state.lateral_offset_m, pt.state.yaw_cam_deg),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=7,
                    color=pt.color,
                    zorder=8,
                )
        # GT marker (no ellipse by default)
        l2_gt, d_gt, _ = cost_at_state(ctx, state_gt)
        z_gt = l2_gt if key == "L2" else d_gt
        ax.scatter(
            state_gt.lateral_offset_m,
            state_gt.yaw_cam_deg,
            c="lime",
            marker="*",
            s=140,
            edgecolors="black",
            linewidths=0.6,
            label=f"GT ({key}={z_gt:.4f})",
            zorder=7,
        )
        ax.set_xlabel("Lateral offset (m)")
        ax.set_ylabel("yaw_cam (°)")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper right", fontsize=6.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{sample.airport} rwy {sample.runway}  row {sample.row_index}  "
        f"— YOLO mask + 95% ellipses (grid → LM until ΔL2<{lm_min_gain})",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def _cov_summary(name: str, cov: PoseCovariance) -> str:
    ratio = cov.sigma_max / max(cov.sigma_min, 1e-9)
    return (
        f"{name}: σ_min={cov.sigma_min:.3f} σ_max={cov.sigma_max:.1f} "
        f"ratio={ratio:.0f}×"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/LARD"))
    parser.add_argument("--mask-dir", type=Path, default=Path("data/LARD/masks/yolo_runway"))
    parser.add_argument("--airport", default="CYEG")
    parser.add_argument("--runway", default="20")
    parser.add_argument("--min-row", type=int, default=21)
    parser.add_argument("--yaw-half-range", type=float, default=3.0)
    parser.add_argument("--lat-half-range", type=float, default=30.0)
    parser.add_argument("--grid", type=int, default=51)
    parser.add_argument("--measurement-sigma", type=float, default=0.15)
    parser.add_argument(
        "--lm-min-gain",
        type=float,
        default=0.03,
        help="Stop LM when one step reduces mean L2 cost by less than this (default 0.03)",
    )
    parser.add_argument(
        "--max-lm-iters",
        type=int,
        default=128,
        help="Safety cap on LM iterations per frame",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/LARD/results/mask_fit_landscape_CYEG_20"),
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    frames = collect_sequence_with_masks(
        data_root,
        args.mask_dir.resolve(),
        args.airport,
        args.runway,
    )
    if args.min_row is not None:
        frames = [f for f in frames if f[0].row_index >= args.min_row]
    if not frames:
        raise SystemExit("No frames.")

    opt_config = OptimizerConfig(
        refine_method="lm",
        max_refine_iters=args.max_lm_iters,
        lm_min_cost_drop=args.lm_min_gain,
        search_mode="coarse_to_fine",
        coarse_levels=WARM_START_COARSE_LEVELS,
        measurement_sigma=args.measurement_sigma,
    )
    Q = process_noise()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"Landscape + covariance for {len(frames)} frames (observed = YOLO)…")
    P_pred = default_prior_covariance()
    state_est_prev: GDofState | None = None

    for i, (sample, scene, pose_gt, observed) in enumerate(frames):
        params = GDofParams.from_pose(pose_gt, scene)
        state_gt = GDofState(pose_gt.yaw_cam_deg, pose_gt.lateral_offset_m)
        state_init_raw = state_gt if state_est_prev is None else state_est_prev

        est = estimate_pose_mask_space(
            scene,
            params,
            observed,
            init=state_init_raw,
            search_center=state_init_raw,
            config=opt_config,
            use_coarse=True,
            P_pred=P_pred,
            compute_covariance=True,
            relocate_invalid_seed=True,
        )
        state_init = (
            est.seed_search.seed
            if est.seed_search is not None and est.seed_search.relocated
            else state_init_raw
        )
        ctx = make_eval_context(scene, params, observed, init=state_init, config=opt_config)

        state_grid = est.state_coarse or state_init
        cov_grid = compute_pose_covariance(
            ctx,
            state_grid,
            config=opt_config,
            data_only=True,
        )

        cov_lm = est.covariance
        assert cov_lm is not None

        points = [
            AnnotatedPoint(state_init_raw, P_pred, "prior t−1", "gold", fill=True),
        ]
        if est.seed_search is not None and est.seed_search.relocated:
            points.append(
                AnnotatedPoint(
                    state_init,
                    P_pred,
                    f"seed ring {est.seed_search.ring}",
                    "orange",
                    fill=False,
                )
            )
        points.extend(
            [
                AnnotatedPoint(state_grid, cov_grid.P_posterior, "grid min (data)", "cyan", fill=False),
                AnnotatedPoint(est.state, cov_lm.P_posterior, f"LM×{est.n_refine_iters}", "red", fill=True),
            ]
        )

        landscape = evaluate_cost_grid(
            scene,
            params,
            observed,
            center=state_gt,
            yaw_half_range_deg=args.yaw_half_range,
            lateral_half_range_m=args.lat_half_range,
            n_yaw=args.grid,
            n_lat=args.grid,
            init_for_roi=state_init,
            config=opt_config,
        )

        fig = plot_landscape_figure(
            sample,
            landscape,
            state_gt=state_gt,
            points=points,
            ctx=ctx,
            lm_min_gain=args.lm_min_gain,
        )
        path = out / f"landscape_row{sample.row_index:05d}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)

        P_pred = cov_lm.P_posterior + Q
        state_est_prev = est.state

        dy = est.state.yaw_cam_deg - state_grid.yaw_cam_deg
        dl = est.state.lateral_offset_m - state_grid.lateral_offset_m
        reloc = ""
        if est.seed_search is not None and est.seed_search.relocated:
            reloc = (
                f" seed_reloc ring={est.seed_search.ring} "
                f"dice_loss={est.seed_search.dice_loss:.3f}"
            )
        print(
            f"  row {sample.row_index:05d}: LM×{est.n_refine_iters} "
            f"Δgrid→LM yaw={dy:+.3f}° lat={dl:+.2f}m | "
            f"{_cov_summary('fused', cov_lm)}{reloc}"
        )
        print(f"    saved {path.name}")

    print(f"\nDone — {len(frames)} landscapes in {out}")


if __name__ == "__main__":
    main()
