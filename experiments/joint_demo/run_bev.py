#!/usr/bin/env python
"""BEV cross-track (Ryu et al.'s estimator on bev_seg) for the joint demo's frames.

Reuses experiments.bev_yolo.ryu_video unchanged: warp each source frame to the
bev_seg canvas, run the bev_seg YOLO, back-project, and read cross-track with the
`paper` variant (front edge, width-measured scale, yaw from the INS log) plus
the paper's causal Kalman filter. The only differences from ryu_video: frames
are restricted to 150-1500 m, so the filter starts cold at 1500 m like the
partner's EKF, and everything is written to disk for render_compare.py.

Only flights 000145 / 000146 are unseen by BOTH the stock YOLO (tilt, Samuel)
and bev_seg: bev_seg trained on most of split_test, and the stock model trained
on most of the bev_toy test split.

Writes cache/bev/<ep>/<stem>.jpg (BEV canvas) and cache/bev/<ep>.json (per frame).

    python -m experiments.joint_demo.run_bev --episodes 000145 000146
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from experiments.bev_yolo.arms import load_config
from experiments.bev_yolo.ryu_eval import VARIANTS
from experiments.bev_yolo.ryu_video import collect, estimate_track, to_bev_px
from experiments.joint_demo import data


def _lst(a):
    return None if a is None else np.asarray(a, float).round(2).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", default=["000145", "000146"])
    ap.add_argument("--config", default="experiments/bev_yolo/config.yaml")
    ap.add_argument("--arm", default="bev_seg")
    ap.add_argument("--variant", default="paper")
    ap.add_argument("--pivot", default="auto")
    ap.add_argument("--dp-frac", type=float, default=0.01)
    ap.add_argument("--kf-q", type=float, default=1e-5)
    ap.add_argument("--kf-r", type=float, default=4.0)
    args = ap.parse_args()

    cfg, arms, _ = load_config(args.config)
    arm = arms[args.arm]
    weights = str(Path(cfg["train"]["project"]).resolve() / arm.name / "weights" / "best.pt")
    for ep in args.episodes:
        cache = data.HERE / "cache/bev" / ep
        cache.mkdir(parents=True, exist_ok=True)
        cam, frames = collect(arm, cfg, weights, ep, 1, None, args.dp_frac, cache)
        frames = [f for f in frames if f["range_m"] <= data.HEADING_FROM_M]
        raw, kf, marks = estimate_track(frames, cam, VARIANTS[args.variant], args.pivot,
                                        args.kf_q, args.kf_r)
        rows = []
        for f, r, k, m in zip(frames, raw, kf, marks):
            bev = cv2.imread(str(f["bev_path"]))
            cv2.imwrite(str(cache / f"{f['stem']}.jpg"), bev, [cv2.IMWRITE_JPEG_QUALITY, 92])
            f["bev_path"].unlink()
            edge = pt = None
            if m is not None:
                if m["edge"] is not None:
                    edge = to_bev_px(np.array(m["edge"]), f["grid"])
                pt = to_bev_px(np.asarray(m["point"]).reshape(1, 2), f["grid"])[0]
            rows.append({"frame": int(f["stem"]), "range_m": f["range_m"],
                         "bev_raw": float(r), "bev_kf": float(k),
                         "poly_src": _lst(f["poly_src"]), "poly_bev": _lst(f["poly_px"]),
                         "edge_bev": _lst(edge), "point_bev": _lst(pt)})
        for p in cache.glob("bev_*.png"):
            p.unlink()
        (data.HERE / "cache/bev" / f"{ep}.json").write_text(json.dumps(rows))
        print(f"{ep}: {len(rows)} frames, {np.isfinite(raw).mean():.0%} with a measurement",
              flush=True)


if __name__ == "__main__":
    main()
