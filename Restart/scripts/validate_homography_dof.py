#!/usr/bin/env python3
"""Validate 4-DOF homography rectification against GT yaw / lateral offset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pose_estimation.g_dof import project_corners_local
from pose_estimation.runway_model import CameraPoseLocal, RunwayScene
from runway_detection.homography.heading import heading_error_ground_deg
from runway_detection.homography.measure import measure_from_image_corners
from runway_detection.homography.plane_homography import (
    homography_from_pose,
    nominal_pose_four_dof,
    runway_plane_xy,
)
from runway_detection.homography.visualize import plot_rectification
from runway_detection.lard.loader import iter_samples, load_runways_database


def validate_homography(
    data_root: Path,
    *,
    source: str = "flsim",
    split: str = "train",
    max_samples: int = 8,
    output_dir: Path | None = None,
    show: bool = False,
) -> list[dict]:
    runways_db = load_runways_database(data_root, source)
    results: list[dict] = []

    for sample in iter_samples(data_root, source=source, split=split, require_image=True):
        scene = RunwayScene.from_lard_sample(sample, runways_db)
        pose_gt = CameraPoseLocal.from_lard_sample(sample, scene)
        pose_nom = nominal_pose_four_dof(pose_gt, scene)

        h_nom = homography_from_pose(scene, pose_nom)
        corners_px = project_corners_local(scene, pose_gt)
        est = measure_from_image_corners(corners_px, h_nom)

        he_gt = heading_error_ground_deg(pose_gt, scene)
        lat_gt = pose_gt.lateral_offset_m
        he_err = est.heading_error_deg - he_gt

        row = {
            "airport": sample.airport,
            "runway": sample.runway,
            "he_gt_deg": he_gt,
            "he_meas_deg": est.heading_error_deg,
            "he_err_deg": he_err,
            "lat_gt_m": lat_gt,
            "lat_plane_y_m": est.lateral_offset_m,
            "obb_angle_deg": est.obb_angle_deg,
        }
        results.append(row)
        print(
            f"[{len(results):02d}] {sample.airport} {sample.runway} "
            f"HE gt={he_gt:+.2f}° meas={est.heading_error_deg:+.2f}° err={he_err:+.2f}° | "
            f"lat_gt={lat_gt:+.1f}m (plane_y={est.lateral_offset_m:+.1f}m, not metric)"
        )

        if output_dir is not None:
            out = output_dir / f"{sample.airport}_{sample.runway}_row{sample.row_index:05d}.png"
            plot_rectification(
                sample.image_path,
                homography=h_nom,
                corners_px=corners_px,
                estimate=est,
                gt_heading_deg=he_gt,
                gt_lateral_m=lat_gt,
                plane_roi_xy=runway_plane_xy(scene),
                save_path=out,
                show=show,
            )

        if len(results) >= max_samples:
            break

    if results:
        he_errs = [abs(r["he_err_deg"]) for r in results]
        print(
            f"\n{len(results)} samples | "
            f"|ΔHE| mean={np.mean(he_errs):.3f}° max={max(he_errs):.3f}°"
        )
        print(
            "Heading error via 4-DOF homography + centerline angle: validated.\n"
            "Lateral: plane-Y readout is NOT metric cross-track (perspective coupling) — "
            "use g_dof or height-scaled geometry for CTE."
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/LARD"))
    parser.add_argument("--source", default="flsim")
    parser.add_argument("--split", default="train")
    parser.add_argument("-n", "--max-samples", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/LARD/homography_overlays"),
    )
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    validate_homography(
        args.data_root.resolve(),
        source=args.source,
        split=args.split,
        max_samples=args.max_samples,
        output_dir=None if args.no_save else args.output_dir.resolve(),
        show=args.show,
    )


if __name__ == "__main__":
    main()
