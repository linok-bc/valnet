#!/usr/bin/env python
"""Samuel's estimator on the same flights fed by different segmenters' masks.

Same frames, calibration, method and filter as run.py; only the mask changes:

    yolo     cache/masks/          soft YOLOv8 masks (run.py's masks stage)
    deeplab  cache/masks_deeplab/  DeepLabV3+ runway probability, x255 (built here)

Writes cache/rows_<source>_<filter>/<ds>/<ep>.csv and prints yaw and
cross-track error per source, overall and by range.

    python -m experiments.restart_test.compare_masks --flights xp12_dataset_weather/000414
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "capstone-dof-estimation"))

from experiments.restart_test.run import BINS, CACHE, stats

MASKS = {"yolo": "masks", "deeplab": "masks_deeplab"}


def deeplab_masks(flights, weights):
    from deeplab.data import MEAN, STD
    from deeplab.model import DeepLabV3Plus

    model = DeepLabV3Plus(pretrained=False).cuda().to(memory_format=torch.channels_last).eval()
    model.load_state_dict(torch.load(weights, map_location="cuda"))
    for ds, ep in flights:
        dst = CACHE / "masks_deeplab" / ds / f"{ep}.npz"
        if dst.exists():
            continue
        with np.load(CACHE / "masks" / ds / f"{ep}.npz") as z:  # the same frames as the YOLO cache
            idx = z["indices"]
        masks = []
        for k in range(0, len(idx), 4):
            ims = [
                (
                    cv2.cvtColor(
                        cv2.imread(str(ROOT / "datasets" / ds / ep / "images" / f"{i:06d}.jpg")), cv2.COLOR_BGR2RGB
                    ).astype(np.float32)
                    / 255
                    - MEAN
                )
                / STD
                for i in idx[k : k + 4]
            ]
            x = torch.from_numpy(np.stack([im.transpose(2, 0, 1) for im in ims])).cuda()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                p = torch.sigmoid(model(x.to(memory_format=torch.channels_last))[:, 0].float())
            masks += list(np.round(p.cpu().numpy() * 255).astype(np.uint8))
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, indices=idx, masks=np.stack(masks))
        print(f"deeplab masks {ds}/{ep}: {len(idx)} frames", flush=True)


def run(ds, ep, source, filt):
    from evaluation.configs import make_method
    from evaluation.observations import YoloMaskCache
    from evaluation.runner import RunConfig, run_sequence
    from runway_detection.xp12.calibration import load_calibrations
    from runway_detection.xp12.loader import load_sequence

    out = CACHE / f"rows_{source}_{filt}" / ds / f"{ep}.csv"
    if not out.exists():
        calib = load_calibrations(CACHE / f"calibration_{ds}.json")[ep]
        seq = load_sequence(ROOT / "datasets" / ds, ep)
        cache = YoloMaskCache(CACHE / MASKS[source] / ds)
        cfg = RunConfig(source="yolo", mode="track", kf=filt != "none", kf_crab=filt == "crab")
        rows = run_sequence(
            seq, calib, make_method("maskfit_analytic_lm"), cfg, yolo=cache, frame_indices=cache.available_indices(ep)
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return list(csv.DictReader(open(out)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flights", nargs="+", required=True, help="<dataset>/<episode> ...")
    ap.add_argument("--weights", default=str(ROOT / "deeplab/runs/xp12_combined/best.pt"))
    ap.add_argument("--filters", nargs="+", default=["crab", "none"])
    args = ap.parse_args()
    flights = [tuple(f.split("/")) for f in args.flights]
    deeplab_masks(flights, args.weights)

    for filt in args.filters:
        print(f"\nfilter: {filt}")
        print(f"  {'':22s} {'n':>5s} | {'yaw MAE':>8s} {'p95':>6s} (deg) | {'xt MAE':>7s} {'p95':>6s} {'max':>6s} (m)")
        for source in MASKS:
            rows = [r for ds, ep in flights for r in run(ds, ep, source, filt)]
            groups = {"all": rows}
            for lo, hi in zip(BINS[:-1], BINS[1:]):
                groups[f"{lo}-{hi} m"] = [r for r in rows if lo <= abs(float(r["along_m"])) <= hi]
            for name, g in groups.items():
                if not g:
                    continue
                y, x = stats([float(r["yaw_err"]) for r in g]), stats([float(r["lat_err"]) for r in g])
                print(
                    f"  {source + ' ' + name:22s} {y['n']:5d} | {y['mae']:8.3f} {y['p95']:6.3f}       | "
                    f"{x['mae']:7.2f} {x['p95']:6.2f} {x['max']:6.2f}"
                )


if __name__ == "__main__":
    main()
