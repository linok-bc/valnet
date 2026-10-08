#!/usr/bin/env python3
"""Validate g_dof mask rendering at ground-truth yaw / lateral offset on LARD."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pose_estimation.g_dof import (
    GDofParams,
    GDofState,
    corner_pixel_error,
    corners_dict_to_polygon,
    g_dof,
    g_dof_jacobian_corners,
    mask_iou,
    render_runway_mask,
)
from pose_estimation.runway_model import CameraPoseLocal, RunwayScene
from pose_estimation.visualize import plot_g_dof_overlay
from runway_detection.lard.loader import iter_samples, load_runways_database


def validate_g_dof(
    data_root: Path,
    *,
    source: str = "flsim",
    split: str = "train",
    max_samples: int = 5,
    output_dir: Path | None = None,
    show: bool = False,
) -> list[dict]:
    runways_db = load_runways_database(data_root, source)
    results: list[dict] = []

    for sample in iter_samples(
        data_root, source=source, split=split, require_image=True
    ):
        scene = RunwayScene.from_lard_sample(sample, runways_db)
        pose = CameraPoseLocal.from_lard_sample(sample, scene)
        params = GDofParams.from_pose(pose, scene)
        state = GDofState(yaw_cam_deg=pose.yaw_cam_deg, lateral_offset_m=pose.lateral_offset_m)

        mask, corners = g_dof(scene, params, state, return_corners=True)
        ann_mask = render_runway_mask(
            corners_dict_to_polygon(sample.annotated_corners),
            scene.intrinsics.width,
            scene.intrinsics.height,
        )
        iou = mask_iou(mask, ann_mask)
        err = corner_pixel_error(corners, sample.annotated_corners)
        jac = g_dof_jacobian_corners(scene, params, state)

        result = {
            "row_index": sample.row_index,
            "airport": sample.airport,
            "runway": sample.runway,
            "yaw_cam_deg": pose.yaw_cam_deg,
            "lateral_offset_m": pose.lateral_offset_m,
            "along_track_m": pose.along_track_m,
            "height_m": pose.height_m,
            "corner_errors_px": err.tolist(),
            "mean_corner_error_px": float(np.mean(err)),
            "mask_iou": iou,
            "jacobian_norm": float(np.linalg.norm(jac)),
        }
        results.append(result)
        print(
            f"[{len(results):02d}] {sample.airport} {sample.runway} "
            f"IoU={iou:.4f} corner_err={np.mean(err):.3f}px "
            f"lat={pose.lateral_offset_m:.1f}m yaw={pose.yaw_cam_deg:.1f}°"
        )

        if output_dir is not None:
            out = output_dir / f"{sample.airport}_{sample.runway}_row{sample.row_index:05d}.png"
            plot_g_dof_overlay(sample, mask, corners, iou=iou, save_path=out, show=show)

        if len(results) >= max_samples:
            break

    if results:
        ious = [r["mask_iou"] for r in results]
        errs = [r["mean_corner_error_px"] for r in results]
        print(
            f"\nValidated {len(results)} sample(s): "
            f"mask IoU mean={np.mean(ious):.4f} (min={min(ious):.4f}), "
            f"corner err mean={np.mean(errs):.3f}px"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/LARD"))
    parser.add_argument("--source", default="flsim")
    parser.add_argument("--split", default="train")
    parser.add_argument("-n", "--max-samples", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/LARD/g_dof_overlays"),
    )
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    validate_g_dof(
        args.data_root.resolve(),
        source=args.source,
        split=args.split,
        max_samples=args.max_samples,
        output_dir=None if args.no_save else args.output_dir.resolve(),
        show=args.show,
    )


if __name__ == "__main__":
    main()
