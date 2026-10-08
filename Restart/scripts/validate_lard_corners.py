#!/usr/bin/env python3
"""Validate LARD corner reprojection on local metadata + images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from runway_detection.lard.loader import iter_samples, load_runways_database
from runway_detection.lard.projection import (
    corner_reprojection_error,
    project_runway_corners,
    runway_corners_wgs84,
)
from runway_detection.lard.visualize import plot_corner_overlay


def validate_samples(
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
        airport = runways_db.get(sample.airport)
        if airport is None:
            continue
        runway_entry = airport.get(sample.runway)
        if runway_entry is None:
            continue

        corners_latlon = runway_corners_wgs84(runway_entry)
        projected = project_runway_corners(
            runway_corners_latlon=corners_latlon,
            lat_cam=sample.lat,
            lon_cam=sample.lon,
            alt_cam=sample.alt,
            yaw_cam_deg=sample.yaw,
            pitch_cam_deg=sample.pitch,
            roll_cam_deg=sample.roll,
            intrinsics=sample.intrinsics,
        )
        errors = corner_reprojection_error(projected, sample.annotated_corners)
        result = {
            "row_index": sample.row_index,
            "image": str(sample.image_path.relative_to(data_root)),
            "airport": sample.airport,
            "runway": sample.runway,
            "errors_px": errors.tolist(),
            "mean_error_px": float(np.mean(errors)),
            "max_error_px": float(np.max(errors)),
        }
        results.append(result)

        print(
            f"[{len(results):02d}] {sample.airport} {sample.runway} "
            f"mean={result['mean_error_px']:.3f}px "
            f"errors={np.round(errors, 2).tolist()}"
        )

        if output_dir is not None:
            out = output_dir / f"{sample.airport}_{sample.runway}_row{sample.row_index:05d}.png"
            plot_corner_overlay(
                sample,
                projected,
                errors=errors,
                save_path=out,
                show=show,
            )

        if len(results) >= max_samples:
            break

    if not results:
        print("No samples validated (check data_root, images, runways DB).")
    else:
        means = [r["mean_error_px"] for r in results]
        print(
            f"\nValidated {len(results)} sample(s). "
            f"Mean error across samples: {np.mean(means):.3f}px "
            f"(min={min(means):.3f}, max={max(means):.3f})"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/LARD"),
        help="LARD data directory (metadata + images)",
    )
    parser.add_argument("--source", default="flsim")
    parser.add_argument("--split", default="train")
    parser.add_argument("-n", "--max-samples", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/LARD/validation_overlays"),
        help="Where to save overlay PNGs (omit with --no-save)",
    )
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    output_dir = None if args.no_save else args.output_dir.resolve()
    validate_samples(
        data_root,
        source=args.source,
        split=args.split,
        max_samples=args.max_samples,
        output_dir=output_dir,
        show=args.show,
    )


if __name__ == "__main__":
    main()
