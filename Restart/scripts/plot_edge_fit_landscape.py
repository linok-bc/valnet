#!/usr/bin/env python3
"""
Plot 2-D edge cost landscapes vs mask L2 for the same frames.

Compares whether the yaw–lateral valley persists when scoring only contour
points against jointly constrained lateral edge lines (spec_edges.md).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pose_estimation.cost_landscape import cost_at_state, evaluate_cost_grid, make_eval_context
from pose_estimation.edge_cost_landscape import edge_cost_at_state, evaluate_edge_cost_grid
from pose_estimation.edge_fit import prepare_edge_observation
from pose_estimation.g_dof import GDofState
from scripts.validate_mask_fit_sequence import collect_sequence_with_masks, track_sequence_viz


def _mark_minima(ax, landscape, grid: np.ndarray, color: str, label: str) -> None:
    flat = grid.copy()
    flat[~landscape.valid] = np.nan
    if not np.any(np.isfinite(flat)):
        return
    idx = np.nanargmin(flat)
    iy, ix = np.unravel_index(idx, grid.shape)
    ax.scatter(
        landscape.lateral_m[ix],
        landscape.yaw_deg[iy],
        c=color,
        marker="v",
        s=80,
        edgecolors="white",
        linewidths=0.8,
        label=label,
        zorder=4,
    )


def _ridge_slope(landscape, grid: np.ndarray) -> float | None:
    flat = grid.copy()
    flat[~landscape.valid] = np.nan
    ridge = []
    for iy in range(len(landscape.yaw_deg)):
        row = flat[iy, :]
        if not np.any(np.isfinite(row)):
            continue
        ix = int(np.nanargmin(row))
        ridge.append((landscape.yaw_deg[iy], landscape.lateral_m[ix]))
    if len(ridge) < 3:
        return None
    ys = np.array([r[0] for r in ridge])
    xs = np.array([r[1] for r in ridge])
    return float(np.polyfit(ys, xs, 1)[0])


def plot_compare_figure(
    frame,
    mask_landscape,
    edge_landscape,
    *,
    state_gt: GDofState,
    state_init: GDofState,
    state_est: GDofState,
    ctx,
    left_pts,
    right_pts,
    huber_delta_px: float,
) -> plt.Figure:
    extent = [
        mask_landscape.lateral_m[0],
        mask_landscape.lateral_m[-1],
        mask_landscape.yaw_deg[0],
        mask_landscape.yaw_deg[-1],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    panels = [
        (axes[0], mask_landscape.l2_masked, "Mask L2 (dense 48×48)", "mask"),
        (axes[1], edge_landscape.edge_cost_masked, "Edge Huber (contour → lines)", "edge"),
    ]

    markers = [
        (state_gt, "GT", "lime", "*", 14),
        (state_init, "Init t−1", "gold", "s", 9),
        (state_est, "Est", "red", "x", 12),
    ]

    ridge_slopes: dict[str, float | None] = {}
    for ax, grid, title, key in panels:
        ridge_slopes[key] = _ridge_slope(
            edge_landscape if key == "edge" else mask_landscape,
            grid,
        )
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
            mask_landscape.lateral_m,
            mask_landscape.yaw_deg,
            grid,
            levels=12,
            colors="white",
            alpha=0.35,
            linewidths=0.8,
        )
        for state, label, color, mk, ms in markers:
            if key == "mask":
                z, _, _ = cost_at_state(ctx, state)
            else:
                z = edge_cost_at_state(
                    frame.scene,
                    frame.params,
                    state,
                    left_pts,
                    right_pts,
                    huber_delta_px=huber_delta_px,
                )
            ax.scatter(
                state.lateral_offset_m,
                state.yaw_cam_deg,
                c=color,
                marker=mk,
                s=ms,
                edgecolors="black",
                linewidths=0.6,
                label=f"{label} ({key}={z:.3f})",
                zorder=5,
            )
        ls = edge_landscape if key == "edge" else mask_landscape
        _mark_minima(ax, ls, grid, "cyan", "grid min")
        ax.set_xlabel("Lateral offset (m)")
        ax.set_ylabel("yaw_cam (°)")
        slope_txt = ridge_slopes[key]
        slope_str = f"{slope_txt:.1f} m/°" if slope_txt is not None else "n/a"
        ax.set_title(f"{title}\nvalley slope ≈ {slope_str}", fontsize=10)
        ax.legend(loc="upper right", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    sample = frame.sample
    n_left, n_right = left_pts.shape[0], right_pts.shape[0]
    fig.suptitle(
        f"{sample.airport} rwy {sample.runway}  row {sample.row_index}  "
        f"— mask vs edge cost  (contour L={n_left} R={n_right}, Huber δ={huber_delta_px}px)",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


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
    parser.add_argument("--huber-delta", type=float, default=3.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/LARD/results/edge_fit_landscape_CYEG_20"),
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

    print(f"Tracking + edge/mask landscapes for {len(frames)} frames…")
    viz_frames = track_sequence_viz(frames, mode="sequential")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    summary: list[str] = []
    for vf in viz_frames:
        mask_landscape = evaluate_cost_grid(
            vf.scene,
            vf.params,
            vf.observed,
            center=vf.state_gt,
            yaw_half_range_deg=args.yaw_half_range,
            lateral_half_range_m=args.lat_half_range,
            n_yaw=args.grid,
            n_lat=args.grid,
            init_for_roi=vf.state_init,
        )
        edge_landscape = evaluate_edge_cost_grid(
            vf.scene,
            vf.params,
            vf.observed,
            center=vf.state_gt,
            assign_state=vf.state_gt,
            yaw_half_range_deg=args.yaw_half_range,
            lateral_half_range_m=args.lat_half_range,
            n_yaw=args.grid,
            n_lat=args.grid,
            huber_delta_px=args.huber_delta,
        )
        left_pts, right_pts = prepare_edge_observation(
            vf.observed,
            vf.scene,
            vf.params,
            vf.state_gt,
        )
        ctx = make_eval_context(vf.scene, vf.params, vf.observed, init=vf.state_init)

        fig = plot_compare_figure(
            vf,
            mask_landscape,
            edge_landscape,
            state_gt=vf.state_gt,
            state_init=vf.state_init,
            state_est=vf.state_est,
            ctx=ctx,
            left_pts=left_pts,
            right_pts=right_pts,
            huber_delta_px=args.huber_delta,
        )
        path = out / f"landscape_row{vf.sample.row_index:05d}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)

        m_slope = _ridge_slope(mask_landscape, mask_landscape.l2_masked)
        e_slope = _ridge_slope(edge_landscape, edge_landscape.edge_cost_masked)
        line = (
            f"row {vf.sample.row_index:05d}: "
            f"mask valley {m_slope:.1f} m/°  edge valley {e_slope:.1f} m/°  "
            f"pts L={left_pts.shape[0]} R={right_pts.shape[0]}"
        )
        summary.append(line)
        print(f"  saved {path.name} — {line}")

    (out / "ridge_slopes.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"\nDone — {len(viz_frames)} landscapes in {out}")


if __name__ == "__main__":
    main()
