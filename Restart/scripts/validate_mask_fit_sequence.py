#!/usr/bin/env python3
"""
Sequential g_dof mask-fit: initialize frame t from estimate at t-1.

Compares tracking modes on YOLO masks along one runway sequence:
  - sequential : init frame 0 = GT, then t-1 estimate (warm narrow search)
  - gt         : per-frame GT init (oracle upper bound)
  - cold       : per-frame GT + fixed offset, full coarse search
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pose_estimation.benchmark import (
    annotated_runway_bbox_area_px,
    height_agl_meters,
    load_saved_mask,
    mask_path_for_image,
)
from pose_estimation.g_dof import GDofParams, GDofState, g_dof, mask_iou
from pose_estimation.covariance import default_prior_covariance, process_noise
from pose_estimation.optimize import (
    OptimizerConfig,
    WARM_START_COARSE_LEVELS,
    bounds_around,
    estimate_pose_mask_space,
)
from pose_estimation.runway_model import CameraPoseLocal, RunwayScene
from runway_detection.lard.loader import iter_samples, load_runways_database
from runway_detection.yolo.inference import mask_has_foreground


def collect_sequence_with_masks(
    data_root: Path,
    mask_dir: Path,
    airport: str,
    runway: str,
    *,
    source: str = "flsim",
    split: str = "train",
    require_yolo: bool = True,
) -> list[tuple[object, RunwayScene, CameraPoseLocal, np.ndarray]]:
    runways_db = load_runways_database(data_root, source)
    out = []
    for sample in iter_samples(data_root, source=source, split=split, require_image=True):
        if sample.airport != airport or sample.runway != runway:
            continue
        path = mask_path_for_image(mask_dir, data_root, sample.image_path)
        if not path.exists():
            if require_yolo:
                continue
            observed = None
        else:
            observed = load_saved_mask(path)
            if not mask_has_foreground(observed):
                if require_yolo:
                    continue
                observed = None
        if require_yolo and observed is None:
            continue
        scene = RunwayScene.from_lard_sample(sample, runways_db)
        pose = CameraPoseLocal.from_lard_sample(sample, scene)
        out.append((sample, scene, pose, observed))
    out.sort(key=lambda x: x[0].row_index)
    return out


def _optimizer_config(
    *,
    mode: str,
    refine_method: str,
    max_refine_iters: int,
) -> OptimizerConfig:
    if mode == "sequential":
        return OptimizerConfig(
            search_mode="coarse_to_fine",
            coarse_levels=WARM_START_COARSE_LEVELS,
            beam_top_k=3,
            refine_method=refine_method,
            max_refine_iters=max_refine_iters,
        )
    return OptimizerConfig(
        search_mode="coarse_to_fine",
        refine_method=refine_method,
        max_refine_iters=max_refine_iters,
    )


def _fit_frame(
    scene: RunwayScene,
    params: GDofParams,
    observed: np.ndarray,
    state_gt: GDofState,
    init: GDofState,
    *,
    mode: str,
    cold_yaw_offset: float,
    cold_lat_offset: float,
    refine_method: str = "lm",
    max_refine_iters: int = 12,
    use_coarse: bool = True,
    P_pred: np.ndarray | None = None,
) -> tuple[GDofState, object, np.ndarray | None]:
    if mode == "gt":
        search_init = state_gt
    elif mode == "cold":
        search_init = GDofState(
            state_gt.yaw_cam_deg + cold_yaw_offset,
            state_gt.lateral_offset_m + cold_lat_offset,
        )
    elif mode == "sequential":
        search_init = init
    else:
        raise ValueError(mode)

    config = _optimizer_config(
        mode=mode,
        refine_method=refine_method,
        max_refine_iters=max_refine_iters,
    )

    bounds = bounds_around(search_init, yaw_half_range_deg=2.0, lateral_half_range_m=25.0)
    est = estimate_pose_mask_space(
        scene,
        params,
        observed,
        init=search_init,
        bounds=bounds,
        config=config,
        use_coarse=use_coarse,
        search_center=search_init,
        P_pred=P_pred,
        compute_covariance=True,
    )
    return est.state, est, est.covariance.P_posterior if est.covariance is not None else None


def track_sequence(
    frames: list[tuple[object, RunwayScene, CameraPoseLocal, np.ndarray]],
    *,
    mode: str,
    cold_yaw_offset: float = 3.0,
    cold_lat_offset: float = 30.0,
    refine_method: str = "lm",
    max_refine_iters: int = 12,
    use_coarse: bool = True,
) -> list[dict]:
    rows: list[dict] = []
    state_est: GDofState | None = None
    P_pred = default_prior_covariance()
    Q = process_noise()

    for i, (sample, scene, pose_gt, observed) in enumerate(frames):
        params = GDofParams.from_pose(pose_gt, scene)
        state_gt = GDofState(pose_gt.yaw_cam_deg, pose_gt.lateral_offset_m)

        if mode == "sequential":
            if state_est is None:
                init = state_gt
            else:
                init = state_est
        else:
            init = state_gt

        state_out, est, P_post = _fit_frame(
            scene,
            params,
            observed,
            state_gt,
            init,
            mode=mode,
            cold_yaw_offset=cold_yaw_offset,
            cold_lat_offset=cold_lat_offset,
            refine_method=refine_method,
            max_refine_iters=max_refine_iters,
            use_coarse=use_coarse,
            P_pred=P_pred if mode == "sequential" else default_prior_covariance(),
        )
        state_est = state_out
        if mode == "sequential" and P_post is not None:
            P_pred = P_post + Q

        gt_mask = g_dof(scene, params, state_gt)
        yaw_err = state_out.yaw_cam_deg - state_gt.yaw_cam_deg
        lat_err = state_out.lateral_offset_m - state_gt.lateral_offset_m
        cov = est.covariance
        sigma_min = cov.sigma_min if cov else float("nan")
        sigma_max = cov.sigma_max if cov else float("nan")

        rows.append(
            {
                "seq_idx": i,
                "row_index": sample.row_index,
                "mode": mode,
                "along_track_m": pose_gt.along_track_m,
                "height_agl_m": height_agl_meters(sample),
                "runway_area_px": annotated_runway_bbox_area_px(sample),
                "mask_iou_vs_gt": mask_iou(observed > 0.25, gt_mask),
                "yaw_gt": state_gt.yaw_cam_deg,
                "lat_gt": state_gt.lateral_offset_m,
                "yaw_est": state_out.yaw_cam_deg,
                "lat_est": state_out.lateral_offset_m,
                "yaw_err_deg": yaw_err,
                "lat_err_m": lat_err,
                "dice": est.dice,
                "n_coarse": est.n_coarse_evals,
                "n_refine": est.n_refine_iters,
                "refine_method": est.refine_method,
                "sigma_yaw_lat_min": sigma_min,
                "sigma_yaw_lat_max": sigma_max,
                "init_yaw": init.yaw_cam_deg,
                "init_lat": init.lateral_offset_m,
            }
        )

    return rows


def track_sequence_viz(
    frames: list[tuple[object, RunwayScene, CameraPoseLocal, np.ndarray]],
    *,
    mode: str = "sequential",
    cold_yaw_offset: float = 3.0,
    cold_lat_offset: float = 30.0,
    refine_method: str = "lm",
    max_refine_iters: int = 12,
    use_coarse: bool = True,
) -> list:
    """Run tracking and return :class:`SequenceFrameResult` list for the viewer."""
    from pose_estimation.sequence_viz import SequenceFrameResult

    out: list[SequenceFrameResult] = []
    state_est: GDofState | None = None

    for i, (sample, scene, pose_gt, observed) in enumerate(frames):
        params = GDofParams.from_pose(pose_gt, scene)
        state_gt = GDofState(pose_gt.yaw_cam_deg, pose_gt.lateral_offset_m)

        if mode == "sequential":
            init = state_gt if state_est is None else state_est
        else:
            init = state_gt

        state_out, est, _ = _fit_frame(
            scene,
            params,
            observed,
            state_gt,
            init,
            mode=mode,
            cold_yaw_offset=cold_yaw_offset,
            cold_lat_offset=cold_lat_offset,
            refine_method=refine_method,
            max_refine_iters=max_refine_iters,
            use_coarse=use_coarse,
        )
        state_est = state_out
        gt_mask = g_dof(scene, params, state_gt)

        out.append(
            SequenceFrameResult(
                sample=sample,
                scene=scene,
                params=params,
                observed=observed,
                state_gt=state_gt,
                state_init=init,
                state_est=state_out,
                dice=est.dice,
                mask_iou_yolo_gt=mask_iou(observed > 0.25, gt_mask),
                yaw_err_deg=state_out.yaw_cam_deg - state_gt.yaw_cam_deg,
                lat_err_m=state_out.lateral_offset_m - state_gt.lateral_offset_m,
                seq_idx=i,
            )
        )

    return out


def summarize(label: str, rows: list[dict]) -> None:
    if not rows:
        print(f"{label}: no frames")
        return
    yaw = np.array([r["yaw_err_deg"] for r in rows])
    lat = np.array([r["lat_err_m"] for r in rows])
    print(f"\n=== {label} ({len(rows)} frames) ===")
    print(
        f"  |Δyaw| mean={np.mean(np.abs(yaw)):.3f}°  final={yaw[-1]:+.3f}°  max={np.max(np.abs(yaw)):.3f}°"
    )
    print(
        f"  |Δlat| mean={np.mean(np.abs(lat)):.1f}m  final={lat[-1]:+.1f}m  max={np.max(np.abs(lat)):.1f}m"
    )
    if len(rows) > 1:
        print(
            f"  drift (final−first err): yaw={yaw[-1]-yaw[0]:+.3f}°  lat={lat[-1]-lat[0]:+.1f}m"
        )
    print(f"  dice mean={np.mean([r['dice'] for r in rows]):.3f}")


def plot_trajectories(
    rows_seq: list[dict],
    rows_gt: list[dict] | None,
    rows_cold: list[dict] | None,
    *,
    title: str,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    def _plot(ax, key_gt, key_est, ylabel, rows_list, labels):
        for rows, label, style in rows_list:
            if not rows:
                continue
            x = [r["seq_idx"] for r in rows]
            ax.plot(x, [r[key_gt] for r in rows], "k-", alpha=0.25, linewidth=1)
            ax.plot(x, [r[key_est] for r in rows], style, label=label, linewidth=1.5)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    series = [
        (rows_seq, "sequential t−1", "C0-"),
        (rows_gt, "GT init", "C2--"),
        (rows_cold, "cold (+3°/+30m)", "C3:"),
    ]
    _plot(axes[0, 0], "yaw_gt", "yaw_est", "Yaw (°)", series, None)
    _plot(axes[0, 1], "lat_gt", "lat_est", "Lateral (m)", series, None)

    if rows_seq:
        x = [r["seq_idx"] for r in rows_seq]
        axes[1, 0].plot(x, [r["yaw_err_deg"] for r in rows_seq], "C0-o", markersize=3, label="sequential")
        if rows_cold:
            axes[1, 0].plot(
                [r["seq_idx"] for r in rows_cold],
                [r["yaw_err_deg"] for r in rows_cold],
                "C3-o",
                markersize=3,
                alpha=0.7,
                label="cold",
            )
        axes[1, 0].axhline(0, color="k", linewidth=0.8)
        axes[1, 0].set_ylabel("Yaw error (°)")
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(x, [r["lat_err_m"] for r in rows_seq], "C0-o", markersize=3, label="sequential")
        if rows_cold:
            axes[1, 1].plot(
                [r["seq_idx"] for r in rows_cold],
                [r["lat_err_m"] for r in rows_cold],
                "C3-o",
                markersize=3,
                alpha=0.7,
                label="cold",
            )
        axes[1, 1].axhline(0, color="k", linewidth=0.8)
        axes[1, 1].set_ylabel("Lat error (m)")
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)

    for ax in axes[1, :]:
        ax.set_xlabel("Sequence index (YOLO frames)")

    fig.suptitle(title)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/LARD"))
    parser.add_argument("--mask-dir", type=Path, default=Path("data/LARD/masks/yolo_runway"))
    parser.add_argument("--source", default="flsim")
    parser.add_argument("--split", default="train")
    parser.add_argument("--airport", default="CYEG")
    parser.add_argument("--runway", default="20")
    parser.add_argument("--cold-yaw-offset", type=float, default=3.0)
    parser.add_argument("--cold-lat-offset", type=float, default=30.0)
    parser.add_argument(
        "--refine",
        choices=("lm", "gd"),
        default="lm",
        help="Refinement after coarse search: Levenberg-Marquardt or gradient descent",
    )
    parser.add_argument(
        "--max-refine-iters",
        type=int,
        default=None,
        help="Refinement iterations (default: 12 for lm, 64 for gd)",
    )
    parser.add_argument(
        "--no-coarse",
        action="store_true",
        help="Skip coarse grid; refine only from the frame init (for GD experiments)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/LARD/results"),
    )
    args = parser.parse_args()

    max_refine_iters = args.max_refine_iters
    if max_refine_iters is None:
        max_refine_iters = 64 if args.refine == "gd" else 12
    use_coarse = not args.no_coarse

    data_root = args.data_root.resolve()
    mask_dir = args.mask_dir.resolve()
    tag = f"{args.airport}_{args.runway}"

    frames = collect_sequence_with_masks(
        data_root,
        mask_dir,
        args.airport,
        args.runway,
        source=args.source,
        split=args.split,
    )
    if len(frames) < 2:
        raise SystemExit(f"Need ≥2 YOLO frames for {tag}, got {len(frames)}")

    print(f"g_dof sequential tracking — {tag}, {len(frames)} YOLO frames")
    print(f"  refine={args.refine}  max_refine_iters={max_refine_iters}  coarse={use_coarse}")

    refine_kw = dict(
        refine_method=args.refine,
        max_refine_iters=max_refine_iters,
        use_coarse=use_coarse,
    )
    rows_seq = track_sequence(frames, mode="sequential", **refine_kw)
    rows_gt = track_sequence(frames, mode="gt", **refine_kw)
    rows_cold = track_sequence(
        frames,
        mode="cold",
        cold_yaw_offset=args.cold_yaw_offset,
        cold_lat_offset=args.cold_lat_offset,
        **refine_kw,
    )

    summarize("Sequential (t−1 init, warm search)", rows_seq)
    summarize("Oracle (GT init / frame)", rows_gt)
    summarize(f"Cold (GT+{args.cold_yaw_offset}°/+{args.cold_lat_offset}m init)", rows_cold)

    out_dir = args.output_dir.resolve()
    tag_suffix = args.refine + ("_nocoarse" if args.no_coarse else "")
    for name, rows in [("sequential", rows_seq), ("gt", rows_gt), ("cold", rows_cold)]:
        path = out_dir / f"mask_fit_seq_{tag}_{name}_{tag_suffix}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    plot_path = out_dir / f"mask_fit_seq_{tag}_{tag_suffix}.png"
    plot_trajectories(
        rows_seq,
        rows_gt,
        rows_cold,
        title=(
            f"{tag} g_dof on YOLO masks "
            f"(refine={args.refine}, iters={max_refine_iters}, coarse={use_coarse})"
        ),
        save_path=plot_path,
    )
    print(f"\nSaved CSVs and {plot_path}")


if __name__ == "__main__":
    main()
