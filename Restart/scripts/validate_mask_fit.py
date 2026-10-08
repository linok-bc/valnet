#!/usr/bin/env python3
"""
Benchmark mask-space pose estimation.

Modes:
  - synthetic corruption of GT masks (default)
  - precomputed YOLO masks (--mask-dir)
  - perfect GT masks (--perfect-mask)

With --stratify, reports errors binned by height AGL and annotated runway area.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pose_estimation.benchmark import (
    annotated_runway_bbox_area_px,
    assign_tertile_bin,
    height_agl_meters,
    load_saved_mask,
    mask_path_for_image,
    summarize_by_bin,
)
from pose_estimation.g_dof import GDofParams, GDofState, g_dof, mask_iou
from pose_estimation.mask_corrupt import CorruptionConfig, corrupt_mask
from pose_estimation.optimize import (
    OptimizerConfig,
    bounds_around,
    estimate_pose_mask_space,
)
from pose_estimation.runway_model import CameraPoseLocal, RunwayScene
from runway_detection.lard.loader import iter_samples, load_runways_database
from runway_detection.yolo.inference import mask_has_foreground


def _resolve_observed_mask(
    sample,
    *,
    perfect_mask: bool,
    corrupt_cfg: CorruptionConfig,
    rng: np.random.Generator,
    scene: RunwayScene,
    params: GDofParams,
    state_gt: GDofState,
    mask_dir: Path | None,
    data_root: Path,
) -> tuple[np.ndarray | None, str]:
    if mask_dir is not None:
        path = mask_path_for_image(mask_dir, data_root, sample.image_path)
        if not path.exists():
            return None, "missing_mask"
        observed = load_saved_mask(path)
        if not mask_has_foreground(observed):
            return None, "empty_mask"
        return observed, "yolo"

    gt_mask = g_dof(scene, params, state_gt).astype(np.float32)
    if perfect_mask:
        return gt_mask, "perfect"
    return corrupt_mask(gt_mask, config=corrupt_cfg, rng=rng), "corrupt"


def run_benchmark(
    data_root: Path,
    *,
    source: str = "flsim",
    split: str = "train",
    max_samples: int | None = 10,
    seed: int = 0,
    yaw_init_offset_deg: float = 1.0,
    lateral_init_offset_m: float = 10.0,
    use_coarse: bool = True,
    perfect_mask: bool = False,
    mask_dir: Path | None = None,
    init_from_gt: bool = False,
    stratify: bool = False,
    search_mode: str = "coarse_to_fine",
    beam_top_k: int = 3,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    runways_db = load_runways_database(data_root, source)
    config = OptimizerConfig(search_mode=search_mode, beam_top_k=beam_top_k)
    corrupt_cfg = CorruptionConfig()
    results: list[dict] = []
    skipped: dict[str, int] = {}

    for sample in iter_samples(data_root, source=source, split=split, require_image=True):
        scene = RunwayScene.from_lard_sample(sample, runways_db)
        pose_gt = CameraPoseLocal.from_lard_sample(sample, scene)
        params = GDofParams.from_pose(pose_gt, scene)
        state_gt = GDofState(pose_gt.yaw_cam_deg, pose_gt.lateral_offset_m)

        observed, mask_source = _resolve_observed_mask(
            sample,
            perfect_mask=perfect_mask,
            corrupt_cfg=corrupt_cfg,
            rng=rng,
            scene=scene,
            params=params,
            state_gt=state_gt,
            mask_dir=mask_dir,
            data_root=data_root,
        )
        if observed is None:
            skipped[mask_source] = skipped.get(mask_source, 0) + 1
            continue

        if init_from_gt:
            init = state_gt
        else:
            init = GDofState(
                state_gt.yaw_cam_deg + yaw_init_offset_deg,
                state_gt.lateral_offset_m + lateral_init_offset_m,
            )
        bounds = bounds_around(init, yaw_half_range_deg=2.0, lateral_half_range_m=25.0)

        est = estimate_pose_mask_space(
            scene,
            params,
            observed,
            init=init,
            bounds=bounds,
            config=config,
            use_coarse=use_coarse,
            search_center=init,
        )

        gt_mask = g_dof(scene, params, state_gt)
        mask_iou_gt = mask_iou(observed > 0.25, gt_mask)

        h_agl = height_agl_meters(sample)
        area_px = annotated_runway_bbox_area_px(sample)
        yaw_err = est.state.yaw_cam_deg - state_gt.yaw_cam_deg
        lat_err = est.state.lateral_offset_m - state_gt.lateral_offset_m

        row = {
            "row_index": sample.row_index,
            "image": str(sample.image_path.relative_to(data_root)),
            "airport": sample.airport,
            "runway": sample.runway,
            "mask_source": mask_source,
            "height_agl_m": h_agl,
            "runway_area_px": area_px,
            "mask_iou_vs_gt": mask_iou_gt,
            "yaw_gt": state_gt.yaw_cam_deg,
            "lat_gt": state_gt.lateral_offset_m,
            "yaw_est": est.state.yaw_cam_deg,
            "lat_est": est.state.lateral_offset_m,
            "yaw_err_deg": yaw_err,
            "lat_err_m": lat_err,
            "dice": est.dice,
            "cost": est.cost,
            "converged": est.converged,
            "n_coarse": est.n_coarse_evals,
            "n_refine": est.n_refine_iters,
            "search_mode": est.search_mode,
        }
        results.append(row)
        h_str = f"{h_agl:.0f}m" if h_agl is not None else "n/a"
        print(
            f"[{len(results):03d}] {sample.airport} {sample.runway} "
            f"h={h_str} area={area_px:.0f}px² "
            f"maskIoU={mask_iou_gt:.2f} "
            f"Δyaw={yaw_err:+.2f}° Δlat={lat_err:+.1f}m dice={est.dice:.3f}"
        )

        if max_samples is not None and len(results) >= max_samples:
            break

    if skipped:
        print(f"\nSkipped: {skipped}")

    if results:
        yaw_errs = [abs(r["yaw_err_deg"]) for r in results]
        lat_errs = [abs(r["lat_err_m"]) for r in results]
        print(
            f"\n{len(results)} samples | "
            f"|Δyaw| mean={np.mean(yaw_errs):.3f}° max={max(yaw_errs):.3f}° | "
            f"|Δlat| mean={np.mean(lat_errs):.2f}m max={max(lat_errs):.2f}m | "
            f"mask IoU vs GT mean={np.mean([r['mask_iou_vs_gt'] for r in results]):.3f}"
        )
        if stratify and results:
            heights = [r["height_agl_m"] for r in results if r["height_agl_m"] is not None]
            areas = [r["runway_area_px"] for r in results]
            for r in results:
                r["height_bin"] = (
                    assign_tertile_bin(heights, r["height_agl_m"])
                    if r["height_agl_m"] is not None
                    else "unknown"
                )
                r["area_bin"] = assign_tertile_bin(areas, r["runway_area_px"])
            summarize_by_bin(results, "height_bin")
            summarize_by_bin(results, "area_bin")

    return results


def save_results(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    else:
        with path.open("w", newline="", encoding="utf-8") as handle:
            if results:
                writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
                writer.writeheader()
                writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/LARD"))
    parser.add_argument("--source", default="flsim")
    parser.add_argument("--split", default="train")
    parser.add_argument("-n", "--max-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--yaw-init-offset", type=float, default=1.0)
    parser.add_argument("--lat-init-offset", type=float, default=10.0)
    parser.add_argument("--no-coarse", action="store_true")
    parser.add_argument("--perfect-mask", action="store_true")
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Directory of precomputed .npy masks (from infer_yolo_lard.py)",
    )
    parser.add_argument(
        "--init-gt",
        action="store_true",
        help="Center search at GT pose (cheat — isolates mask error only)",
    )
    parser.add_argument(
        "--search-mode",
        choices=("coarse_to_fine", "single"),
        default="coarse_to_fine",
        help="coarse_to_fine = hierarchical beam search (default)",
    )
    parser.add_argument("--beam-top-k", type=int, default=3)
    parser.add_argument("--stratify", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save results CSV/JSON (e.g. data/LARD/results/mask_fit_yolo.csv)",
    )
    args = parser.parse_args()

    mask_dir = args.mask_dir.resolve() if args.mask_dir else None
    results = run_benchmark(
        args.data_root.resolve(),
        source=args.source,
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed,
        yaw_init_offset_deg=args.yaw_init_offset,
        lateral_init_offset_m=args.lat_init_offset,
        use_coarse=not args.no_coarse,
        perfect_mask=args.perfect_mask,
        mask_dir=mask_dir,
        init_from_gt=args.init_gt,
        stratify=args.stratify,
        search_mode=args.search_mode,
        beam_top_k=args.beam_top_k,
    )
    if args.output and results:
        save_results(results, args.output.resolve())


if __name__ == "__main__":
    main()
