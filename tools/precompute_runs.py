#!/usr/bin/env python
"""Build the app's preloaded runs (assets.SAMPLES) and, with --upload, publish them.

For each run: the pipeline on the local flight folder, Result.save -> runs/<name>/
{records.json, masks.npz}, the replay video -> runs/<name>/replay.mp4, and the flight's
frames -> samples/<name>.tar.zst. --upload sends the new files to the asset repo.

    python -m tools.precompute_runs --out build/assets
    python -m tools.precompute_runs --out build/assets --upload
"""

import argparse
import subprocess
from pathlib import Path

from e2e_runway_dof_estimation import assets, pipeline
from e2e_runway_dof_estimation.estimate import Estimator
from e2e_runway_dof_estimation.segment import Segmenter
from e2e_runway_dof_estimation.viz import write_video

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="build/assets")
    ap.add_argument("--datasets", default=str(REPO / "datasets"), help="local XP12 dataset root")
    ap.add_argument("--runs", nargs="*", default=list(assets.SAMPLES))
    ap.add_argument("--upload", action="store_true", help=f"upload runs/ and samples/ to {assets.REPO}")
    args = ap.parse_args()

    out = Path(args.out)
    segmenter, estimator = Segmenter(), Estimator()
    for name in args.runs:
        dataset, episode = name.split("_")
        flight = Path(args.datasets) / f"xp12_dataset_{dataset}" / episode
        result = pipeline.run(flight, segmenter=segmenter, estimator=estimator)
        result.save(out / "runs" / name)
        write_video(result, estimator, out / "runs" / name / "replay.mp4")
        archive = out / "samples" / f"{name}.tar.zst"
        if not archive.exists():
            archive.parent.mkdir(parents=True, exist_ok=True)
            tar = subprocess.Popen(["tar", "-C", str(flight.parents[1]), "-cf", "-",
                                    f"{flight.parent.name}/{flight.name}"], stdout=subprocess.PIPE)
            subprocess.run(["zstd", "-q", "-19", "-T8", "-o", str(archive)], stdin=tar.stdout, check=True)
            tar.wait()
        s = result.summary()
        print(f"{name:16s} {s['frames']:4d} frames  yaw {s['yaw_mae_deg']:.3f} deg  "
              f"cross-track {s['xtrack_mae_m']:.2f} m  IoU {s['mask_iou']:.3f}", flush=True)

    if args.upload:
        from huggingface_hub import HfApi
        HfApi().upload_folder(repo_id=assets.REPO, folder_path=str(out), allow_patterns=["runs/**", "samples/**"],
                              commit_message=f"Preloaded runs: {', '.join(args.runs)}")
        print(f"uploaded to {assets.REPO}")


if __name__ == "__main__":
    main()
