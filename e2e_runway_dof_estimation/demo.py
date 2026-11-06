"""Command line: runway masks, yaw and cross-track for one flight; writes a video and a CSV.

    python -m e2e_runway_dof_estimation.demo --flight weather_000185
    python -m e2e_runway_dof_estimation.demo --flight /path/to/xp12_dataset_weather/000414 --out runs/000414

--flight is a sample name (assets.SAMPLES, e.g. weather_000185: 7.3 m/s crosswind; weather_000270:
haze), downloaded on first use, or a flight folder from the XP12 dataset.
"""

import argparse
import csv
import json
import time
from pathlib import Path

from e2e_runway_dof_estimation import assets, pipeline
from e2e_runway_dof_estimation.estimate import Estimator


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flight", required=True, help=f"sample ({', '.join(assets.SAMPLES)}) or flight folder")
    ap.add_argument("--out", default=None, help="output folder (default: demo_out/<flight>)")
    ap.add_argument("--min-range", type=float, default=150.0, help="m before the threshold")
    ap.add_argument("--max-range", type=float, default=1500.0)
    ap.add_argument("--no-visibility-filter", action="store_true",
                    help="also run on frames beyond half the visibility")
    ap.add_argument("--no-crab", action="store_true", help="the original 3-state EKF (drifts in crosswind)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from e2e_runway_dof_estimation.segment import Segmenter
    t0 = time.time()
    estimator = Estimator(crab=not args.no_crab)
    result = pipeline.run(args.flight, segmenter=Segmenter(device=args.device), estimator=estimator,
                          min_range=args.min_range, max_range=args.max_range,
                          visibility_filter=not args.no_visibility_filter,
                          progress=lambda d, n: print(f"\rsegmenting {d}/{n}", end="", flush=True))
    print()
    out = Path(args.out or Path("demo_out") / result.flight.name.replace("/", "_"))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "frames.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(result.records[0]))
        w.writeheader()
        w.writerows(result.records)
    s = result.summary()
    (out / "summary.json").write_text(json.dumps(s, indent=1))

    print(f"{s['flight']}  {s['airport']}  {s['weather']}  {s['frames']} frames  ({time.time() - t0:.0f} s)")
    print(f"  {'':14s} {'yaw MAE':>9s} {'cross-track MAE':>16s} {'mask IoU':>9s}")
    print(f"  {'all':14s} {s['yaw_mae_deg']:8.3f}° {s['xtrack_mae_m']:14.2f} m {s['mask_iou']:9.3f}")
    for k, v in s["by_range"].items():
        print(f"  {k:14s} {v['yaw_mae_deg']:8.3f}° {v['xtrack_mae_m']:14.2f} m {v['mask_iou']:9.3f}")
    if not args.no_video:
        from e2e_runway_dof_estimation.viz import write_video
        write_video(result, estimator, out / "replay.mp4")
    print(f"wrote {out}/" + ("" if args.no_video else "replay.mp4, ") + "frames.csv, summary.json")


if __name__ == "__main__":
    main()
