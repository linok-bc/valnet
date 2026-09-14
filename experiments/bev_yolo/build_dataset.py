#!/usr/bin/env python
"""Build every arm of the BEV experiment from one frame set.

The matrix is views x targets x ground rasters, defined in the config's `arms`
block. `persp_seg` is the control -- what VALNet and the YOLOv8-seg baseline
already do. The bev arms are the proposal: warp to a bird's-eye view where the
runway is a true rectangle, and regress that rectangle instead of painting
pixels.

EVERY ARM SEES EXACTLY THE SAME FRAMES. A frame is kept only if it survives in
the perspective view AND under every grid policy any arm uses, so a difference
between arms cannot come from one of them having been handed an easier subset.
This is why the grids are built together rather than as separate datasets: a
per-policy build keeps a per-policy frame set, and the comparison then needs an
after-the-fact intersection that is easy to forget.

The kept set, the per-frame ground truth and the per-frame grid for each policy
all land in meta/<split>.jsonl, which is what eval.py reads to score every arm
in metres rather than in each arm's own pixels.

Splits are grouped BY AIRPORT, not by episode: two approaches to the same runway
are near-duplicates, and letting them straddle a split would inflate everything.

    python -m experiments.bev_yolo.build_dataset --clean
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from experiments.bev.camera import CameraModel
from experiments.bev.episodes import find_episodes, load_runways
from experiments.bev.grids import resolution_report
from experiments.bev.labels import FORMATTERS, clip_to_canvas, polygon_area, warp_points
from experiments.bev.transform import bev_from_image, warp_image
from experiments.bev_yolo.arms import load_config


def split_of(airport: str, fractions) -> str:
    """Deterministic airport-grouped split. Stable across dataset regenerations."""
    h = int(hashlib.sha1(airport.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    train, val = fractions["train"], fractions["val"]
    return "train" if h < train else ("val" if h < train + val else "test")


def load_camera(cfg):
    """Camera model, preferring the fitted camera.json over the config guess."""
    path = Path(cfg["camera"]["fit"]) if cfg["camera"].get("fit") else None
    if path and path.exists():
        fitted = json.loads(path.read_text())
        print(f"camera: {path} (fitted, {fitted['residual_px']['median_px']:.2f} px "
              f"median residual on {fitted['n_frames']} frames)")
        if Path(fitted.get("data", "")).resolve() != Path(cfg["source"]["root"]).resolve():
            print(f"  WARNING: fitted on {fitted.get('data')}, building from "
                  f"{cfg['source']['root']}.\n  Re-run calibrate.py unless you are "
                  f"sure the camera did not change.")
        return CameraModel(**fitted["camera"])
    c = cfg["camera"]
    print("camera: config fallback -- run calibrate.py, do not trust this blindly")
    return CameraModel.from_fov(c["width"], c["height"], c["fov_x_deg"],
                                c["fov_y_deg"], c.get("tilt_deg", 0.0))


class FramePlan:
    """One kept frame: its source polygon, and its geometry under every grid."""

    def __init__(self, frame, runway, poly_src, per_grid, split, name):
        self.frame, self.runway = frame, runway
        self.poly_src = poly_src
        self.per_grid = per_grid           # {grid name: (BEVGrid, polygon in it)}
        self.split, self.name = split, name

    def record(self):
        fr = self.frame
        return {
            "name": self.name,
            "episode": fr.episode.path.name,
            "stem": fr.stem,
            "airport": fr.episode.airport,
            "runway": fr.episode.runway_name,
            "runway_width_m": self.runway.width_m,
            "runway_length_m": self.runway.length_m,
            "range_m": fr.range_m,
            "height_m": fr.height_m,
            "along_m": fr.along_m,
            "right_m": fr.right_m,
            "yaw_deg": fr.yaw_deg,
            "pitch_deg": fr.pitch_deg,
            "roll_deg": fr.roll_deg,
            "grids": {name: {"width": g.width, "height": g.height,
                             "mpp": g.mpp, "x_near": g.x_near}
                      for name, (g, _) in self.per_grid.items()},
        }


def plan_frames(cfg, cam, grids, runways):
    """Decide which frames make the cut, under every grid, before writing anything."""
    root = Path(cfg["source"]["root"])
    sel = cfg["select"]
    episodes = find_episodes(root)
    if sel.get("max_episodes"):
        episodes = episodes[: sel["max_episodes"]]
    if not episodes:
        raise SystemExit(f"no episodes under {root}")

    plans = []
    dropped = {"unlabelled": 0, "range": 0, "no_runway": 0, "behind_camera": 0,
               "off_canvas": 0, "side_clipped": 0, "too_small": 0}
    src_size = (cam.width, cam.height)

    for ep in tqdm(episodes, desc="planning"):
        rw = runways.get((ep.airport, ep.runway_name))
        if rw is None:
            dropped["no_runway"] += 1
            continue
        split = split_of(ep.airport, cfg["split_fractions"])
        for fr in ep.frames(stride=sel["stride"]):
            if not fr.labelled:
                dropped["unlabelled"] += 1
                continue
            if not (sel["min_range_m"] <= fr.range_m <= sel["max_range_m"]):
                dropped["range"] += 1
                continue

            poly_src = fr.polygon_px(*src_size)
            per_grid, reason = {}, None
            for name, policy in grids.items():
                grid = policy.for_frame(cam, fr.roll_deg, fr.pitch_deg, fr.height_m,
                                        range_m=fr.range_m)
                m_bev = bev_from_image(cam, grid, fr.roll_deg, fr.pitch_deg,
                                       fr.height_m)
                pts, valid = warp_points(poly_src, m_bev)
                if not valid.all():
                    reason = "behind_camera"      # a corner at or above the horizon
                    break
                # Clipping at the FAR end is expected and harmless: the canvas
                # stops before the far threshold, and the near edge -- the only
                # edge the estimator uses -- is untouched. Clipping at the SIDES
                # shaves the runway lengthwise, so the fitted box comes out
                # narrower than the runway and its centre is pulled inboard,
                # biasing cross-track by metres. Those frames are dropped.
                if sel.get("drop_side_clipped", True) and (
                        pts[:, 0].min() < 0 or pts[:, 0].max() > grid.width - 1):
                    reason = "side_clipped"
                    break
                poly_bev = clip_to_canvas(pts, grid.width, grid.height)
                if len(poly_bev) < 3:
                    reason = "off_canvas"
                    break
                if polygon_area(poly_bev) < sel["min_area_px"]:
                    reason = "too_small"
                    break
                per_grid[name] = (grid, poly_bev)

            if reason:
                dropped[reason] += 1
                continue
            plans.append(FramePlan(fr, rw, poly_src, per_grid, split,
                                   f"{ep.path.name}_{fr.stem}"))

    return plans, dropped


def write_arm_trees(out: Path, plans, arms, cam):
    """images/ (symlinks) + labels/ per arm, so Ultralytics finds its pairs."""
    for arm in arms.values():
        fmt = FORMATTERS[arm.target]
        for split in ("train", "val", "test"):
            (out / "arms" / arm.name / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "arms" / arm.name / "labels" / split).mkdir(parents=True, exist_ok=True)

        for p in plans:
            if arm.is_bev:
                grid, poly = p.per_grid[arm.grid]
                img = out / "bev" / arm.grid / p.split / f"{p.name}.jpg"
                w, h = grid.width, grid.height
            else:
                img, poly = p.frame.image_path, p.poly_src
                w, h = cam.width, cam.height

            link = out / "arms" / arm.name / "images" / p.split / f"{p.name}{img.suffix}"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(img.resolve())
            (out / "arms" / arm.name / "labels" / p.split / f"{p.name}.txt").write_text(
                fmt(poly, w, h) + "\n")

        (out / "arms" / arm.name / "data.yaml").write_text(yaml.safe_dump({
            "path": str((out / "arms" / arm.name).resolve()),
            "train": "images/train", "val": "images/val", "test": "images/test",
            "names": {0: "runway"},
        }, sort_keys=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="experiments/bev_yolo/config.yaml")
    ap.add_argument("--out", default=None, help="override dataset.out")
    ap.add_argument("--dry-run", action="store_true", help="plan and report, write nothing")
    ap.add_argument("--clean", action="store_true", help="wipe the output first")
    args = ap.parse_args()

    cfg, arms, grids = load_config(args.config)
    out = Path(args.out or cfg["dataset"]["out"])
    cam = load_camera(cfg)
    runways = load_runways(cfg["source"].get("runways_dat") or None)
    print(f"arms:  {', '.join(arms)}")
    print(f"grids: {', '.join(f'{k} ({v.mode})' for k, v in grids.items())}")

    plans, dropped = plan_frames(cfg, cam, grids, runways)
    if not plans:
        raise SystemExit(f"every frame was dropped: {dropped}")

    counts = {s: sum(p.split == s for p in plans) for s in ("train", "val", "test")}
    print(f"\nkept {len(plans)} frames  {counts}")
    print(f"dropped: {dropped}")

    ranges = np.array([p.frame.range_m for p in plans])
    print(f"range m: {ranges.min():.0f} .. {ranges.max():.0f} "
          f"(median {np.median(ranges):.0f})")
    for name in grids:
        ratios = np.array([
            resolution_report(cam, p.per_grid[name][0], p.frame.roll_deg,
                              p.frame.pitch_deg, p.frame.height_m, p.frame.range_m,
                              p.runway.width_m)["ratio"] for p in plans])
        print(f"  grid {name:10s} BEV/source resolution on the runway width: "
              f"median {np.median(ratios):.2f}  p10 {np.percentile(ratios, 10):.2f}"
              f"  p90 {np.percentile(ratios, 90):.2f}")
    print("  (<1 means the canvas samples the runway more coarsely than the source "
          "did;\n   warping cannot add detail, so this is the price of rectification)")
    if args.dry_run:
        return

    if args.clean and out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    for name in grids:
        for split in ("train", "val", "test"):
            (out / "bev" / name / split).mkdir(parents=True, exist_ok=True)

    quality = [cv2.IMWRITE_JPEG_QUALITY, cfg["dataset"].get("jpeg_quality", 92)]
    for p in tqdm(plans, desc="warping"):
        img = cv2.imread(str(p.frame.image_path))
        if img is None:
            raise SystemExit(f"unreadable frame {p.frame.image_path}")
        for name, (grid, _) in p.per_grid.items():
            m_bev = bev_from_image(cam, grid, p.frame.roll_deg, p.frame.pitch_deg,
                                   p.frame.height_m)
            cv2.imwrite(str(out / "bev" / name / p.split / f"{p.name}.jpg"),
                        warp_image(img, m_bev, grid), quality)

    write_arm_trees(out, plans, arms, cam)

    for split in ("train", "val", "test"):
        with open(out / "meta" / f"{split}.jsonl", "w") as fp:
            for p in plans:
                if p.split == split:
                    fp.write(json.dumps(p.record()) + "\n")
    (out / "meta" / "camera.json").write_text(json.dumps({
        "camera": dict(width=cam.width, height=cam.height, fx=cam.fx, fy=cam.fy,
                       cx=cam.cx, cy=cam.cy, tilt_deg=cam.tilt_deg),
        "arms": cfg["arms"], "grids": cfg["grids"], "select": cfg["select"],
        "source": cfg["source"]["root"],
    }, indent=2) + "\n")

    print(f"\nwrote {out}")
    for name in arms:
        print(f"  {name:12s} {out / 'arms' / name / 'data.yaml'}")


if __name__ == "__main__":
    main()
