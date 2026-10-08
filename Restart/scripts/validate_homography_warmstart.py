#!/usr/bin/env python3
"""
Warm-start homography experiment (simulated t-1 prior).

Build H from a pose close to GT (prior = GT + small delta on HE / lateral),
then measure GT runway corners in the prior rectified plane. If the 2D
readout is locally linear, measured values should track the injected deltas.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pose_estimation.g_dof import project_corners_local
from pose_estimation.runway_model import CameraPoseLocal, RunwayScene
from runway_detection.homography.heading import heading_error_ground_deg
from runway_detection.homography.measure import measure_from_image_corners
from runway_detection.homography.plane_homography import (
    homography_from_pose,
    nominal_pose_four_dof,
    nominal_pose_with_prior,
)
from runway_detection.homography.readout import correct_plane_residuals
from runway_detection.lard.loader import iter_samples, load_runways_database


def _parse_deltas(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def evaluate_warmstart(
    data_root: Path,
    *,
    source: str = "flsim",
    split: str = "train",
    airport: str | None = None,
    runway: str | None = None,
    max_samples: int = 30,
    delta_he_deg: list[float] | None = None,
    delta_lat_m: list[float] | None = None,
    output_csv: Path | None = None,
) -> list[dict]:
    if delta_he_deg is None:
        delta_he_deg = [-2.0, -1.0, 0.0, 1.0, 2.0]
    if delta_lat_m is None:
        delta_lat_m = [-20.0, -10.0, 0.0, 10.0, 20.0]

    runways_db = load_runways_database(data_root, source)
    rows: list[dict] = []

    n_frames = 0
    for sample in iter_samples(data_root, source=source, split=split, require_image=True):
        if airport is not None and sample.airport != airport:
            continue
        if runway is not None and sample.runway != runway:
            continue

        scene = RunwayScene.from_lard_sample(sample, runways_db)
        pose_gt = CameraPoseLocal.from_lard_sample(sample, scene)
        corners_px = project_corners_local(scene, pose_gt)
        he_gt = heading_error_ground_deg(pose_gt, scene)
        lat_gt = pose_gt.lateral_offset_m

        for d_he in delta_he_deg:
            for d_lat in delta_lat_m:
                pose_prior = nominal_pose_with_prior(
                    pose_gt,
                    scene,
                    delta_he_deg=d_he,
                    delta_lat_m=d_lat,
                )
                he_prior = heading_error_ground_deg(pose_prior, scene)
                lat_prior = pose_prior.lateral_offset_m

                h_prior = homography_from_pose(scene, pose_prior)
                est = measure_from_image_corners(corners_px, h_prior)
                he_corr, lat_corr = correct_plane_residuals(
                    est,
                    along_track_m=pose_gt.along_track_m,
                )

                # Zero-prior baseline: meas ≈ -gt; residual vs injected delta.
                h_zero = homography_from_pose(scene, nominal_pose_four_dof(pose_gt, scene))
                est_zero = measure_from_image_corners(corners_px, h_zero)

                row = {
                    "airport": sample.airport,
                    "runway": sample.runway,
                    "row_index": sample.row_index,
                    "along_track_m": pose_gt.along_track_m,
                    "height_m": pose_gt.height_m,
                    "he_gt_deg": he_gt,
                    "lat_gt_m": lat_gt,
                    "delta_he_deg": d_he,
                    "delta_lat_m": d_lat,
                    "he_prior_deg": he_prior,
                    "lat_prior_m": lat_prior,
                    "he_inj_deg": he_prior - he_gt,
                    "lat_inj_m": lat_prior - lat_gt,
                    "he_meas_deg": est.heading_error_deg,
                    "lat_meas_m": est.lateral_offset_m,
                    "he_corr_deg": he_corr,
                    "lat_corr_m": lat_corr,
                    "he_zero_deg": est_zero.heading_error_deg,
                    "lat_zero_m": est_zero.lateral_offset_m,
                    "he_meas_minus_zero_deg": est.heading_error_deg - est_zero.heading_error_deg,
                    "lat_meas_minus_zero_m": est.lateral_offset_m - est_zero.lateral_offset_m,
                }
                rows.append(row)

        n_frames += 1
        if n_frames >= max_samples:
            break

    if not rows:
        return rows

    he_inj = np.array([r["he_inj_deg"] for r in rows])
    lat_inj = np.array([r["lat_inj_m"] for r in rows])
    he_meas = np.array([r["he_meas_deg"] for r in rows])
    lat_meas = np.array([r["lat_meas_m"] for r in rows])

    lat_corr = np.array([r["lat_corr_m"] for r in rows])
    he_corr = np.array([r["he_corr_deg"] for r in rows])

    def _fit_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        if np.std(x) < 1e-9:
            return float("nan"), float(np.mean(y))
        k, b = np.polyfit(x, y, 1)
        return float(k), float(b)

    he_k, he_b = _fit_slope(he_inj, he_meas)
    lat_k, lat_b = _fit_slope(lat_inj, lat_meas)

    he_only = [r for r in rows if abs(r["delta_lat_m"]) < 1e-9]
    lat_only = [r for r in rows if abs(r["delta_he_deg"]) < 1e-9]
    he_inj_1d = np.array([r["he_inj_deg"] for r in he_only])
    he_meas_1d = np.array([r["he_meas_deg"] for r in he_only])
    lat_inj_1d = np.array([r["lat_inj_m"] for r in lat_only])
    lat_meas_1d = np.array([r["lat_meas_m"] for r in lat_only])

    lat_coupled_mask = np.array(
        [
            abs(r["delta_he_deg"]) > 1e-9 and abs(r["delta_lat_m"]) > 1e-9
            for r in rows
        ]
    )

    print(f"Warm-start homography — {n_frames} frames, {len(rows)} (frame × delta) samples")
    print(f"Deltas HE (°): {delta_he_deg}")
    print(f"Deltas lat (m): {delta_lat_m}")
    print()
    print("Absolute readout in prior frame (meas should equal injected delta):")
    print(
        f"  HE:  slope={he_k:+.3f}  intercept={he_b:+.4f}°  "
        f"corr={np.corrcoef(he_inj, he_meas)[0, 1]:+.4f}  "
        f"MAE={np.mean(np.abs(he_meas - he_inj)):.4f}°"
    )
    print(
        f"  lat: slope={lat_k:+.3f}  intercept={lat_b:+.2f}m  "
        f"corr={np.corrcoef(lat_inj, lat_meas)[0, 1]:+.4f}  "
        f"MAE={np.mean(np.abs(lat_meas - lat_inj)):.2f}m"
    )
    print()
    print("Decoupled (one DOF perturbed at a time):")
    print(
        f"  HE only (Δlat=0): corr={np.corrcoef(he_inj_1d, he_meas_1d)[0, 1]:+.4f}  "
        f"MAE={np.mean(np.abs(he_meas_1d - he_inj_1d)):.5f}°"
    )
    print(
        f"  lat only (ΔHE=0): corr={np.corrcoef(lat_inj_1d, lat_meas_1d)[0, 1]:+.4f}  "
        f"MAE={np.mean(np.abs(lat_meas_1d - lat_inj_1d)):.4f}m"
    )
    if np.any(lat_coupled_mask):
        lat_coupled_err = lat_meas[lat_coupled_mask] - lat_inj[lat_coupled_mask]
        print()
        print(
            f"Coupled (both ΔHE≠0 and Δlat≠0): lateral MAE={np.mean(np.abs(lat_coupled_err)):.1f}m "
            f"(HE/lat cross-talk in plane readout)"
        )
    print()
    print("After deterministic correction  lat_corr = lat_meas − tan(HE)·along_track:")
    coupled_lat_mae = (
        float(np.mean(np.abs(lat_corr[lat_coupled_mask] - lat_inj[lat_coupled_mask])))
        if np.any(lat_coupled_mask)
        else float("nan")
    )
    print(
        f"  HE MAE={np.mean(np.abs(he_corr - he_inj)):.4f}°  "
        f"lat MAE={np.mean(np.abs(lat_corr - lat_inj)):.3f}m  "
        f"(coupled lat MAE={coupled_lat_mae:.3f}m)"
    )

    zero_rows = [r for r in rows if r["delta_he_deg"] == 0 and r["delta_lat_m"] == 0]
    if zero_rows:
        print()
        print(
            "Sanity (prior = GT, delta=0): "
            f"|HE| mean={np.mean([abs(r['he_meas_deg']) for r in zero_rows]):.2e}°  "
            f"|lat| mean={np.mean([abs(r['lat_meas_m']) for r in zero_rows]):.2e}m"
        )

    for d_he, d_lat in [(1.0, 0.0), (0.0, 10.0), (1.0, 10.0)]:
        sub = [
            r
            for r in rows
            if abs(r["delta_he_deg"] - d_he) < 1e-9 and abs(r["delta_lat_m"] - d_lat) < 1e-9
        ]
        if sub:
            r0 = sub[0]
            print(
                f"\nExample inj ΔHE={d_he:+.0f}° Δlat={d_lat:+.0f}m → "
                f"meas HE={float(r0['he_meas_deg']):+.3f}° lat={float(r0['lat_meas_m']):+.1f}m | "
                f"corr lat={float(r0['lat_corr_m']):+.1f}m"
            )

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {output_csv}")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/LARD"))
    parser.add_argument("--source", default="flsim")
    parser.add_argument("--split", default="train")
    parser.add_argument("--airport", default="CYEG")
    parser.add_argument("--runway", default="20")
    parser.add_argument("-n", "--max-samples", type=int, default=30)
    parser.add_argument(
        "--delta-he",
        default="-2,-1,0,1,2",
        help="Injected HE deltas (degrees), comma-separated",
    )
    parser.add_argument(
        "--delta-lat",
        default="-20,-10,0,10,20",
        help="Injected lateral deltas (meters), comma-separated",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/LARD/results/homography_warmstart.csv"),
    )
    args = parser.parse_args()

    evaluate_warmstart(
        args.data_root.resolve(),
        source=args.source,
        split=args.split,
        airport=args.airport,
        runway=args.runway,
        max_samples=args.max_samples,
        delta_he_deg=_parse_deltas(args.delta_he),
        delta_lat_m=_parse_deltas(args.delta_lat),
        output_csv=args.output.resolve(),
    )


if __name__ == "__main__":
    main()
