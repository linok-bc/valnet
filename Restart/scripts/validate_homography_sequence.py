#!/usr/bin/env python3
"""
Sequential homography tracking: prior at frame t = estimate at t-1.

Uses GT runway corners (oracle detection) and known 4-DOF per frame.
Reports drift of HE / lateral vs ground truth over a runway sequence.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pose_estimation.g_dof import project_corners_local
from pose_estimation.runway_model import CameraPoseLocal, RunwayScene
from runway_detection.homography.heading import heading_error_ground_deg
from runway_detection.homography.measure import measure_from_image_corners
from runway_detection.homography.plane_homography import (
    homography_from_pose,
    nominal_pose_four_dof,
    pose_with_he_lat,
)
from runway_detection.homography.readout import apply_prior_residual, correct_plane_residuals
from runway_detection.lard.loader import iter_samples, load_runways_database


def collect_runway_sequence(
    data_root: Path,
    airport: str,
    runway: str,
    *,
    source: str = "flsim",
    split: str = "train",
) -> list[tuple[object, RunwayScene, CameraPoseLocal]]:
    runways_db = load_runways_database(data_root, source)
    out: list[tuple[object, RunwayScene, CameraPoseLocal]] = []
    for sample in iter_samples(data_root, source=source, split=split, require_image=True):
        if sample.airport != airport or sample.runway != runway:
            continue
        scene = RunwayScene.from_lard_sample(sample, runways_db)
        pose = CameraPoseLocal.from_lard_sample(sample, scene)
        out.append((sample, scene, pose))
    out.sort(key=lambda x: x[0].row_index)
    return out


def _step_update(
    scene: RunwayScene,
    pose_gt: CameraPoseLocal,
    corners_px: dict[str, tuple[float, float]],
    pose_prior: CameraPoseLocal,
    he_prior: float,
    lat_prior: float,
    *,
    use_correction: bool,
) -> tuple[float, float, float, float, float, float]:
    """Measure plane residual and apply prior update. Returns d_he, d_lat, he_est, lat_est."""
    h_prior = homography_from_pose(scene, pose_prior)
    est = measure_from_image_corners(corners_px, h_prior)
    he_raw = est.heading_error_deg
    lat_raw = est.lateral_offset_m
    if use_correction:
        he_est, lat_est = apply_prior_residual(
            he_prior,
            lat_prior,
            est,
            along_track_m=pose_gt.along_track_m,
        )
        d_he, d_lat = correct_plane_residuals(est, along_track_m=pose_gt.along_track_m)
    else:
        d_he, d_lat = he_raw, lat_raw
        he_est, lat_est = he_prior - d_he, lat_prior - d_lat
    return d_he, d_lat, he_raw, lat_raw, he_est, lat_est


def track_sequence(
    sequence: list[tuple[object, RunwayScene, CameraPoseLocal]],
    *,
    init_mode: str = "gt",
    use_correction: bool = True,
) -> list[dict]:
    if not sequence:
        return []

    rows: list[dict] = []
    he_est = lat_est = 0.0

    for i, (sample, scene, pose_gt) in enumerate(sequence):
        he_gt = heading_error_ground_deg(pose_gt, scene)
        lat_gt = pose_gt.lateral_offset_m
        corners_px = project_corners_local(scene, pose_gt)

        if i == 0:
            if init_mode == "gt":
                he_est, lat_est = he_gt, lat_gt
            elif init_mode == "zero":
                he_est, lat_est = 0.0, 0.0
            else:
                raise ValueError(f"Unknown init_mode: {init_mode}")

            rows.append(
                _row(
                    sample=sample,
                    frame_idx=i,
                    pose_gt=pose_gt,
                    he_gt=he_gt,
                    lat_gt=lat_gt,
                    he_est=he_est,
                    lat_est=lat_est,
                    d_he=0.0,
                    d_lat=0.0,
                    d_he_raw=0.0,
                    d_lat_raw=0.0,
                    init=True,
                )
            )
            continue

        pose_prior = pose_with_he_lat(
            pose_gt,
            scene,
            heading_error_deg=he_est,
            lateral_offset_m=lat_est,
            yaw_reference=pose_gt,
        )
        d_he, d_lat, d_he_raw, d_lat_raw, he_est, lat_est = _step_update(
            scene,
            pose_gt,
            corners_px,
            pose_prior,
            he_est,
            lat_est,
            use_correction=use_correction,
        )

        rows.append(
            _row(
                sample=sample,
                frame_idx=i,
                pose_gt=pose_gt,
                he_gt=he_gt,
                lat_gt=lat_gt,
                he_est=he_est,
                lat_est=lat_est,
                d_he=d_he,
                d_lat=d_lat,
                d_he_raw=d_he_raw,
                d_lat_raw=d_lat_raw,
                init=False,
            )
        )

    return rows


def _row(
    *,
    sample,
    frame_idx: int,
    pose_gt: CameraPoseLocal,
    he_gt: float,
    lat_gt: float,
    he_est: float,
    lat_est: float,
    d_he: float,
    d_lat: float,
    d_he_raw: float,
    d_lat_raw: float,
    init: bool,
) -> dict:
    return {
        "frame_idx": frame_idx,
        "row_index": sample.row_index,
        "along_track_m": pose_gt.along_track_m,
        "height_m": pose_gt.height_m,
        "he_gt_deg": he_gt,
        "lat_gt_m": lat_gt,
        "he_est_deg": he_est,
        "lat_est_m": lat_est,
        "he_err_deg": he_est - he_gt,
        "lat_err_m": lat_est - lat_gt,
        "d_he_deg": d_he,
        "d_lat_m": d_lat,
        "d_he_raw_deg": d_he_raw,
        "d_lat_raw_m": d_lat_raw,
        "init": init,
    }


def track_oracle_gt_prior(
    sequence: list[tuple[object, RunwayScene, CameraPoseLocal]],
    *,
    use_correction: bool = True,
) -> list[dict]:
    """Upper bound: prior at t = GT at t-1 (no accumulated error)."""
    rows: list[dict] = []
    for i, (sample, scene, pose_gt) in enumerate(sequence):
        he_gt = heading_error_ground_deg(pose_gt, scene)
        lat_gt = pose_gt.lateral_offset_m
        if i == 0:
            he_est, lat_est = he_gt, lat_gt
            d_he = d_lat = d_he_raw = d_lat_raw = 0.0
        else:
            pose_prev_gt = sequence[i - 1][2]
            he_prior = heading_error_ground_deg(pose_prev_gt, scene)
            lat_prior = pose_prev_gt.lateral_offset_m
            pose_prior = pose_with_he_lat(
                pose_gt,
                scene,
                heading_error_deg=he_prior,
                lateral_offset_m=lat_prior,
                yaw_reference=pose_gt,
            )
            corners_px = project_corners_local(scene, pose_gt)
            d_he, d_lat, d_he_raw, d_lat_raw, he_est, lat_est = _step_update(
                scene,
                pose_gt,
                corners_px,
                pose_prior,
                he_prior,
                lat_prior,
                use_correction=use_correction,
            )

        rows.append(
            _row(
                sample=sample,
                frame_idx=i,
                pose_gt=pose_gt,
                he_gt=he_gt,
                lat_gt=lat_gt,
                he_est=he_est,
                lat_est=lat_est,
                d_he=d_he if i > 0 else 0.0,
                d_lat=d_lat if i > 0 else 0.0,
                d_he_raw=d_he_raw if i > 0 else 0.0,
                d_lat_raw=d_lat_raw if i > 0 else 0.0,
                init=i == 0,
            )
        )
    return rows


def summarize(label: str, rows: list[dict]) -> None:
    he_err = np.array([r["he_err_deg"] for r in rows])
    lat_err = np.array([r["lat_err_m"] for r in rows])
    print(f"\n=== {label} ===")
    print(f"  frames: {len(rows)}")
    print(f"  |ΔHE|  mean={np.mean(np.abs(he_err)):.4f}°  final={he_err[-1]:+.4f}°  max={np.max(np.abs(he_err)):.4f}°")
    print(f"  |Δlat| mean={np.mean(np.abs(lat_err)):.1f}m  final={lat_err[-1]:+.1f}m  max={np.max(np.abs(lat_err)):.1f}m")
    if len(rows) > 1:
        he_drift = he_err[-1] - he_err[0]
        lat_drift = lat_err[-1] - lat_err[0]
        print(f"  drift (final−initial err): HE={he_drift:+.4f}°  lat={lat_drift:+.1f}m")


def plot_trajectory(
    rows_est: list[dict],
    rows_oracle: list[dict] | None,
    *,
    title: str,
    save_path: Path,
) -> None:
    frames = [r["frame_idx"] for r in rows_est]
    he_gt = [r["he_gt_deg"] for r in rows_est]
    lat_gt = [r["lat_gt_m"] for r in rows_est]
    he_est = [r["he_est_deg"] for r in rows_est]
    lat_est = [r["lat_est_m"] for r in rows_est]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(frames, he_gt, "k-", label="GT", linewidth=2)
    axes[0, 0].plot(frames, he_est, "C0-", label="sequential est")
    if rows_oracle:
        axes[0, 0].plot(
            [r["frame_idx"] for r in rows_oracle],
            [r["he_est_deg"] for r in rows_oracle],
            "C2--",
            label="oracle GT prior",
            alpha=0.8,
        )
    axes[0, 0].set_ylabel("HE (°)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(frames, lat_gt, "k-", label="GT", linewidth=2)
    axes[0, 1].plot(frames, lat_est, "C0-", label="sequential est")
    if rows_oracle:
        axes[0, 1].plot(
            [r["frame_idx"] for r in rows_oracle],
            [r["lat_est_m"] for r in rows_oracle],
            "C2--",
            label="oracle GT prior",
            alpha=0.8,
        )
    axes[0, 1].set_ylabel("Lateral (m)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    he_err = [r["he_err_deg"] for r in rows_est]
    lat_err = [r["lat_err_m"] for r in rows_est]
    axes[1, 0].plot(frames, he_err, "C3-o", markersize=3)
    axes[1, 0].axhline(0, color="k", linewidth=0.8)
    axes[1, 0].set_xlabel("Frame")
    axes[1, 0].set_ylabel("HE error (°)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(frames, lat_err, "C3-o", markersize=3)
    axes[1, 1].axhline(0, color="k", linewidth=0.8)
    axes[1, 1].set_xlabel("Frame")
    axes[1, 1].set_ylabel("Lateral error (m)")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/LARD"))
    parser.add_argument("--source", default="flsim")
    parser.add_argument("--split", default="train")
    parser.add_argument("--airport", default="CYEG")
    parser.add_argument("--runway", default="20")
    parser.add_argument(
        "--init",
        choices=("gt", "zero"),
        default="gt",
        help="Frame-0 initialization",
    )
    parser.add_argument("--no-correction", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/LARD/results"),
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    sequence = collect_runway_sequence(
        data_root,
        args.airport,
        args.runway,
        source=args.source,
        split=args.split,
    )
    if len(sequence) < 2:
        raise SystemExit(f"Need ≥2 frames for {args.airport} {args.runway}, got {len(sequence)}")

    use_corr = not args.no_correction
    tag = f"{args.airport}_{args.runway}"
    rows = track_sequence(sequence, init_mode=args.init, use_correction=use_corr)
    rows_oracle = track_oracle_gt_prior(sequence, use_correction=use_corr)

    corr_label = "corrected" if use_corr else "raw"
    print(f"Sequential tracking {tag} — {len(sequence)} frames, init={args.init}, {corr_label}")

    summarize("Sequential (est t-1 → prior t)", rows)
    summarize("Oracle (GT t-1 → prior t)", rows_oracle)

    out_dir = args.output_dir.resolve()
    csv_path = out_dir / f"homography_seq_{tag}_{args.init}_{corr_label}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_path = out_dir / f"homography_seq_{tag}_{args.init}_{corr_label}.png"
    plot_trajectory(
        rows,
        rows_oracle,
        title=f"{tag} sequential homography ({args.init} init, {corr_label})",
        save_path=plot_path,
    )
    print(f"\nSaved {csv_path}")
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
