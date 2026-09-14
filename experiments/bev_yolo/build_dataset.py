#!/usr/bin/env python
"""Build the four YOLO datasets the toy experiment compares.

The matrix is 2 views x 2 targets:

                 segment (polygon)     obb (oriented box)
    perspective   persp_seg             persp_obb
    bev           bev_seg               bev_obb

`persp_seg` is what VALNet and the YOLOv8-seg baseline already do, so it is the
control. `bev_obb` is the proposal: warp to a bird's-eye view where the runway
is a true rectangle, and regress that rectangle directly instead of painting
pixels.

EVERY ARM SEES EXACTLY THE SAME FRAMES. A frame is kept only if it is usable in
both views, so a difference between arms cannot come from one of them having
been handed an easier subset. The kept set, the per-frame ground truth and the
per-frame BEV grid all land in meta/<split>.jsonl, which is what eval.py reads
to score every arm in metres rather than in each arm's own pixels.

Splits are grouped BY AIRPORT, not by episode: two approaches to the same runway
are near-duplicates, and letting them straddle a split would inflate every
number here.

    python -m experiments.bev_yolo.build_dataset --config experiments/bev_yolo/config.yaml
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
from experiments.bev.episodes import find_episodes, load_runways, runway_to_ground
from experiments.bev.grids import GridPolicy, resolution_report
from experiments.bev.labels import (
    FORMATTERS, clip_to_canvas, polygon_area, warp_points,
)
from experiments.bev.transform import bev_from_image, warp_image

ARMS = {
    "persp_seg": ("perspective", "segment"),
    "persp_obb": ("perspective", "obb"),
    "bev_seg": ("bev", "segment"),
    "bev_obb": ("bev", "obb"),
}


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
    """One kept frame: where it came from, and its geometry in both views."""

    def __init__(self, frame, runway, poly_src, poly_bev, grid, split, name):
        self.frame, self.runway = frame, runway
        self.poly_src, self.poly_bev = poly_src, poly_bev
        self.grid, self.split, self.name = grid, split, name

    def record(self):
        fr, g = self.frame, self.grid
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
            "grid": {"width": g.width, "height": g.height,
                     "mpp": g.mpp, "x_near": g.x_near},
        }


def plan_frames(cfg, cam, policy, runways):
    """Decide which frames make the cut, in both views, before writing anything."""
    root = Path(cfg["source"]["root"])
    sel = cfg["select"]
    episodes = find_episodes(root)
    if sel.get("max_episodes"):
        episodes = episodes[: sel["max_episodes"]]
    if not episodes:
        raise SystemExit(f"no episodes under {root}")

    plans, dropped = [], {"unlabelled": 0, "range": 0, "no_runway": 0,
                          "behind_camera": 0, "off_canvas": 0, "side_clipped": 0,
                          "too_small": 0}
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
            grid = policy.for_frame(cam, fr.roll_deg, fr.pitch_deg, fr.height_m,
                                    range_m=fr.range_m)
            m_bev = bev_from_image(cam, grid, fr.roll_deg, fr.pitch_deg, fr.height_m)
            pts, valid = warp_points(poly_src, m_bev)
            if not valid.all():
                dropped["behind_camera"] += 1      # a corner at or above the horizon
                continue
            # Clipping at the FAR end is expected and harmless: the canvas simply
            # stops before the far threshold, and the near edge -- the only edge
            # the estimator uses -- is untouched. Clipping at the SIDES is not: it
            # shaves the runway lengthwise, so the fitted box is narrower than the
            # runway and its centre is pulled inboard, biasing cross-track by
            # metres. That happens when range * sin(yaw) + half the runway width
            # exceeds the canvas half-width, i.e. early in the approach with a
            # large heading error. Those frames are dropped rather than learned.
            if sel.get("drop_side_clipped", True) and (
                    pts[:, 0].min() < 0 or pts[:, 0].max() > grid.width - 1):
                dropped["side_clipped"] += 1
                continue
            poly_bev = clip_to_canvas(pts, grid.width, grid.height)
            if len(poly_bev) < 3:
                dropped["off_canvas"] += 1
                continue
            if polygon_area(poly_bev) < sel["min_area_px"]:
                dropped["too_small"] += 1
                continue

            name = f"{ep.path.name}_{fr.stem}"
            plans.append(FramePlan(fr, rw, poly_src, poly_bev, grid, split, name))

    return plans, dropped


def write_arm_trees(out: Path, plans, cfg):
    """images/ (symlinks) + labels/ per arm, so Ultralytics finds its pairs."""
    for arm, (view, target) in ARMS.items():
        fmt = FORMATTERS[target]
        for split in ("train", "val", "test"):
            (out / "arms" / arm / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "arms" / arm / "labels" / split).mkdir(parents=True, exist_ok=True)

        for p in plans:
            if view == "bev":
                img = out / "bev" / p.split / f"{p.name}.jpg"
                poly, w, h = p.poly_bev, p.grid.width, p.grid.height
            else:
                img = p.frame.image_path
                poly = p.poly_src
                w, h = cfg["camera"]["width"], cfg["camera"]["height"]

            link = out / "arms" / arm / "images" / p.split / f"{p.name}{img.suffix}"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(img.resolve())
            label = out / "arms" / arm / "labels" / p.split / f"{p.name}.txt"
            label.write_text(fmt(poly, w, h) + "\n")

        yaml_path = out / "arms" / arm / "data.yaml"
        yaml_path.write_text(yaml.safe_dump({
            "path": str((out / "arms" / arm).resolve()),
            "train": "images/train", "val": "images/val", "test": "images/test",
            "names": {0: "runway"},
        }, sort_keys=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="experiments/bev_yolo/config.yaml")
    ap.add_argument("--out", default=None, help="override dataset.out")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan and report, write nothing")
    ap.add_argument("--clean", action="store_true", help="wipe the output first")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out or cfg["dataset"]["out"])
    cam = load_camera(cfg)
    policy = GridPolicy(**cfg["bev"])
    runways = load_runways(cfg["source"].get("runways_dat") or None)

    plans, dropped = plan_frames(cfg, cam, policy, runways)
    if not plans:
        raise SystemExit(f"every frame was dropped: {dropped}")

    counts = {s: sum(p.split == s for p in plans) for s in ("train", "val", "test")}
    print(f"\nkept {len(plans)} frames  {counts}")
    print(f"dropped: {dropped}")

    ranges = np.array([p.frame.range_m for p in plans])
    ratios = np.array([
        resolution_report(cam, p.grid, p.frame.roll_deg, p.frame.pitch_deg,
                          p.frame.height_m, p.frame.range_m, p.runway.width_m)["ratio"]
        for p in plans])
    print(f"range m: {ranges.min():.0f} .. {ranges.max():.0f} "
          f"(median {np.median(ranges):.0f})")
    print(f"BEV/source resolution ratio on the runway width: "
          f"median {np.median(ratios):.2f}, p10 {np.percentile(ratios, 10):.2f}, "
          f"p90 {np.percentile(ratios, 90):.2f}")
    print("  (<1 means the BEV canvas samples the runway more coarsely than the "
          "source image did;\n   warping cannot add detail, so this is the price "
          "of the rectification)")
    if args.dry_run:
        return

    if args.clean and out.exists():
        shutil.rmtree(out)
    for split in ("train", "val", "test"):
        (out / "bev" / split).mkdir(parents=True, exist_ok=True)
        (out / "meta").mkdir(parents=True, exist_ok=True)

    for p in tqdm(plans, desc="warping"):
        img = cv2.imread(str(p.frame.image_path))
        if img is None:
            raise SystemExit(f"unreadable frame {p.frame.image_path}")
        m_bev = bev_from_image(cam, p.grid, p.frame.roll_deg, p.frame.pitch_deg,
                               p.frame.height_m)
        bev = warp_image(img, m_bev, p.grid)
        cv2.imwrite(str(out / "bev" / p.split / f"{p.name}.jpg"), bev,
                    [cv2.IMWRITE_JPEG_QUALITY, cfg["dataset"].get("jpeg_quality", 92)])

    write_arm_trees(out, plans, cfg)

    for split in ("train", "val", "test"):
        with open(out / "meta" / f"{split}.jsonl", "w") as fp:
            for p in plans:
                if p.split == split:
                    fp.write(json.dumps(p.record()) + "\n")
    (out / "meta" / "camera.json").write_text(json.dumps({
        "camera": dict(width=cam.width, height=cam.height, fx=cam.fx, fy=cam.fy,
                       cx=cam.cx, cy=cam.cy, tilt_deg=cam.tilt_deg),
        "grid_policy": cfg["bev"], "select": cfg["select"],
        "source": cfg["source"]["root"],
    }, indent=2) + "\n")

    print(f"\nwrote {out}")
    for arm in ARMS:
        print(f"  {arm:10s} {out / 'arms' / arm / 'data.yaml'}")


if __name__ == "__main__":
    main()
