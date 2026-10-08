#!/usr/bin/env python3
"""
Fit runway width on one runway (train frames), evaluate lateral on hold-out.

Uses 4-DOF homography (along-track, height, pitch, roll from each frame).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pose_estimation.g_dof import project_corners_local
from pose_estimation.runway_model import CameraPoseLocal, RunwayScene
from runway_detection.homography.heading import heading_error_ground_deg
from runway_detection.homography.measure import measure_from_image_corners
from runway_detection.homography.plane_homography import homography_from_pose, nominal_pose_four_dof, runway_plane_xy
from runway_detection.homography.scene_width import scene_with_runway_width
from runway_detection.homography.visualize import plot_rectification
from runway_detection.homography.width_fit import evaluate_width, fit_runway_width
from runway_detection.lard.loader import iter_samples, load_runways_database


def collect_runway_samples(
    data_root: Path,
    airport: str,
    runway: str,
    *,
    source: str = "flsim",
    split: str = "train",
    require_image: bool = True,
) -> list[tuple[object, RunwayScene, CameraPoseLocal, int]]:
    runways_db = load_runways_database(data_root, source)
    out = []
    for sample in iter_samples(data_root, source=source, split=split, require_image=require_image):
        if sample.airport != airport or sample.runway != runway:
            continue
        scene = RunwayScene.from_lard_sample(sample, runways_db)
        pose = CameraPoseLocal.from_lard_sample(sample, scene)
        out.append((sample, scene, pose, sample.row_index))
    out.sort(key=lambda x: x[3])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/LARD"))
    parser.add_argument("--airport", default="CYEG")
    parser.add_argument("--runway", default="20")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--method", choices=("plane_y", "center_shift", "similarity_ty"), default="similarity_ty")
    parser.add_argument("--output-dir", type=Path, default=Path("data/LARD/homography_overlays"))
    parser.add_argument("--save-viz", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    all_samples = collect_runway_samples(data_root, args.airport, args.runway)
    if len(all_samples) < 4:
        raise SystemExit(f"Need more frames for {args.airport} {args.runway}, got {len(all_samples)}")

    n_train = max(2, int(len(all_samples) * args.train_frac))
    train = [(s, p) for _, s, p, _ in all_samples[:n_train]]
    test = [(s, p) for _, s, p, _ in all_samples[n_train:]]

    fit = fit_runway_width(train, method=args.method)
    eval_train = evaluate_width(train, fit["fitted_width_m"], method=args.method)
    eval_test_db = evaluate_width(test, fit["db_width_m"], method=args.method)
    eval_test_fit = evaluate_width(test, fit["fitted_width_m"], method=args.method)

    print(f"Runway {args.airport} {args.runway} — {len(train)} train / {len(test)} test frames")
    print(f"4-DOF per frame: along_track={train[0][1].along_track_m:.0f}m (varies), "
          f"height, pitch, roll from CSV")
    print(f"\nDB width:     {fit['db_width_m']:.1f} m")
    print(f"Fitted width: {fit['fitted_width_m']:.1f} m  (scale {fit['width_scale']:.3f})")
    odd_ok = 0.5 <= fit["width_scale"] <= 2.0
    print(f"ODD sanity (0.5×–2× DB): {'OK' if odd_ok else 'FAIL — fitted width not plausible'}")
    print(f"Train MAE lateral: DB={fit['train_mae_db_width_m']:.1f}m → fitted={fit['train_mae_m']:.1f}m")
    print(f"Test  MAE lateral: DB={eval_test_db['mae_m']:.1f}m → fitted={eval_test_fit['mae_m']:.1f}m")
    print(f"Test RMSE: {eval_test_fit['rmse_m']:.1f}m")

    # Heading error still works with fitted width (sample on test)
    if test:
        scene0, pose0 = test[0]
        scene_w = scene_with_runway_width(scene0, fit["fitted_width_m"])
        h_nom = homography_from_pose(scene_w, nominal_pose_four_dof(pose0, scene_w))
        corners = project_corners_local(scene0, pose0)
        he_meas = measure_from_image_corners(corners, h_nom).heading_error_deg
        he_gt = heading_error_ground_deg(pose0, scene0)
        print(f"\nHE check (1 test frame): gt={he_gt:+.2f}° meas={he_meas:+.2f}° (sign convention)")

    result = {
        "airport": args.airport,
        "runway": args.runway,
        "n_train": len(train),
        "n_test": len(test),
        "fit": fit,
        "eval_train": eval_train,
        "eval_test_db": eval_test_db,
        "eval_test_fit": eval_test_fit,
    }
    out_json = data_root / "results" / f"width_fit_{args.airport}_{args.runway}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved {out_json}")

    if args.save_viz and test:
        sample, scene, pose, row_idx = all_samples[n_train]
        scene_w = scene_with_runway_width(scene, fit["fitted_width_m"])
        h_nom = homography_from_pose(scene_w, nominal_pose_four_dof(pose, scene_w))
        corners = project_corners_local(scene, pose)
        est = measure_from_image_corners(corners, h_nom)
        out_png = args.output_dir / f"{args.airport}_{args.runway}_test_row{row_idx:05d}.png"
        plot_rectification(
            sample.image_path,
            homography=h_nom,
            corners_px=corners,
            estimate=est,
            gt_heading_deg=heading_error_ground_deg(pose, scene),
            gt_lateral_m=pose.lateral_offset_m,
            plane_roi_xy=runway_plane_xy(scene),
            save_path=out_png,
        )
        print(f"Overlay: {out_png}")


if __name__ == "__main__":
    main()
