#!/usr/bin/env python
"""Score every arm in metres, on the two quantities the system actually needs.

THE ASSUMPTION: four of the aircraft's six degrees of freedom are known --
altitude, roll, pitch, and along-track distance to the threshold. Only YAW
(heading error against the runway) and CROSS-TRACK (metres right of the
centreline) are unknown. That is what makes this experiment well posed: the BEV
warp consumes exactly the known quantities and leaves exactly the unknown ones
visible in the image, as the rotation and the lateral offset of a rectangle.

WHY NOT mAP. mAP in each arm's own pixels is not comparable across arms: a BEV
pixel is metres of ground whose scale changes with altitude, a perspective pixel
is not, and an IoU threshold therefore means a different physical tolerance in
each. Worse, it rewards the wrong thing -- a mask can score well while its edges
wander by the few metres that are the entire signal. So every arm is put through
one shared pipeline:

    prediction -> polygon in image pixels
               -> ground plane, metres, using the KNOWN altitude and attitude
               -> minimum-area rectangle
               -> heading from the long axis, cross-track from the near edge

and scored against the logged truth, binned by range. Identical post-processing
for all four arms, so only the representation differs.

THE ORACLE ROW IS THE ONE TO READ FIRST. It runs the ground-truth label through
the same pipeline. It is the floor: no detector on this arm can beat it. If the
oracle is already several metres out, the arm is limited by its geometry, not by
its network, and training harder will not help.

    python -m experiments.bev_yolo.eval --arm all
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from experiments.bev.camera import CameraModel
from experiments.bev.labels import clip_to_canvas
from experiments.bev.transform import BEVGrid, ground_to_image
from experiments.bev_yolo.build_dataset import ARMS


# ----------------------------------------------------------------- geometry --

def ground_from_bev(pts_px: np.ndarray, grid: BEVGrid) -> np.ndarray:
    """BEV pixels -> ground metres. Exact: the grid is a metric raster."""
    p = np.column_stack([pts_px, np.ones(len(pts_px))]) @ grid.A.T
    return p[:, :2] / p[:, 2:3]


def ground_from_image(pts_px: np.ndarray, cam: CameraModel, roll_deg: float,
                      pitch_deg: float, height_m: float):
    """Perspective pixels -> ground metres, dropping anything at or above the horizon.

    Returns (points, valid). Points beyond the vanishing line have no ground
    intersection in front of the aircraft, which is why a perspective mask that
    leaks upward a few pixels can throw its back-projection kilometres away.
    """
    H = np.linalg.inv(ground_to_image(cam, roll_deg, pitch_deg, height_m))
    p = np.column_stack([pts_px, np.ones(len(pts_px))]) @ H.T
    valid = p[:, 2] > 1e-9
    out = np.full((len(pts_px), 2), np.nan)
    out[valid] = p[valid, :2] / p[valid, 2:3]
    return out, valid


def estimate(poly_ground: np.ndarray):
    """Heading error and cross-track from a runway polygon in ground metres.

    The near edge is the threshold: it is the closest thing to the aircraft, it
    stays in frame longest, and it is the only edge Ryu et al. trust for the
    same reason. The long axis gives the heading.
    """
    if len(poly_ground) < 3:
        return None
    (cx, cy), (w, h), ang = cv2.minAreaRect(poly_ground.astype(np.float32))
    box = cv2.boxPoints(((cx, cy), (w, h), ang)).astype(np.float64)

    order = np.argsort(box[:, 0])              # X = forward, so the two smallest
    near, far = box[order[:2]], box[order[2:]]  # are the near edge
    t = near.mean(axis=0)                       # threshold midpoint estimate
    u = far.mean(axis=0) - t
    n_u = np.linalg.norm(u)
    if n_u < 1e-6:
        return None
    u = u / n_u

    yaw = -math.degrees(math.atan2(u[1], u[0]))
    n = np.array([math.sin(math.radians(yaw)), math.cos(math.radians(yaw))])
    return {
        "yaw_deg": yaw,
        "right_m": float(-t @ n),               # aircraft metres right of centreline
        "along_m": float(-t @ u),               # diagnostic: known in this setup
        "width_m": float(min(w, h)),
        "threshold_m": t,
    }


def polygon_to_ground(poly_px, rec, cam, view, ground_window=True):
    """Prediction in image pixels -> runway polygon on the ground, in metres.

    `ground_window` clips the perspective arm to the same patch of ground the BEV
    canvas covers. Without it the comparison is rigged: the BEV canvas ends a few
    hundred metres past the threshold, so the BEV arm never sees the far end of
    the runway, while the perspective arm's polygon runs all the way to the
    horizon -- where one pixel of label error is tens of metres of ground and the
    fitted rectangle skews badly. That is a real property of perspective masks,
    but it is a property of the WINDOW, not of the representation, so it is held
    equal here and called out separately.
    """
    g = rec["grid"]
    grid = BEVGrid(g["width"], g["height"], g["mpp"], g["x_near"])
    if view == "bev":
        return ground_from_bev(poly_px, grid)
    pts, valid = ground_from_image(poly_px, cam, rec["roll_deg"], rec["pitch_deg"],
                                   rec["height_m"])
    pts = pts[valid]
    if not ground_window or len(pts) < 3:
        return pts
    inv = np.linalg.inv(grid.A)                       # ground -> BEV pixels
    q = np.column_stack([pts, np.ones(len(pts))]) @ inv.T
    q = q[:, :2] / q[:, 2:3]
    return ground_from_bev(clip_to_canvas(q, grid.width, grid.height), grid)


# ------------------------------------------------------------- predictions --

def predicted_polygon(result, target):
    """Highest-confidence instance as a polygon in image pixels, or None."""
    if target == "obb":
        obb = getattr(result, "obb", None)
        if obb is None or len(obb) == 0:
            return None
        i = int(obb.conf.argmax())
        return obb.xyxyxyxy[i].cpu().numpy().reshape(4, 2).astype(np.float64)
    masks = getattr(result, "masks", None)
    if masks is None or len(masks) == 0:
        return None
    i = int(result.boxes.conf.argmax())
    return np.asarray(masks.xy[i], dtype=np.float64)


def label_polygon(path: Path, width: int, height: int):
    vals = Path(path).read_text().split()
    if len(vals) < 7:
        return None
    p = np.array([float(v) for v in vals[1:]], dtype=np.float64).reshape(-1, 2)
    return p * np.array([width, height])


# ------------------------------------------------------------------ report --

def summarise(rows, bins, label):
    """Per-range-bin cross-track and heading error. Medians, because a handful of
    blown-up back-projections would otherwise dominate every mean."""
    print(f"\n  {label}")
    print(f"    {'range (m)':>14}  {'n':>5}  {'det':>5}  "
          f"{'|xtrack| med':>12}  {'p90':>7}  {'|yaw| med':>9}  {'bias':>7}  "
          f"{'width err':>9}")
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = [r for r in rows if lo <= r["range_m"] < hi]
        if not sel:
            continue
        hit = [r for r in sel if r["ok"]]
        if not hit:
            print(f"    {lo:6.0f}-{hi:6.0f}  {len(sel):5d}  {0.0:5.2f}"
                  f"  {'-':>12}  {'-':>7}  {'-':>9}  {'-':>7}")
            continue
        ex = np.array([r["xtrack_err"] for r in hit])
        ey = np.array([r["yaw_err"] for r in hit])
        ew = np.array([r.get("width_err", np.nan) for r in hit], dtype=float)
        print(f"    {lo:6.0f}-{hi:6.0f}  {len(sel):5d}  {len(hit)/len(sel):5.2f}"
              f"  {np.median(np.abs(ex)):12.2f}  {np.percentile(np.abs(ex), 90):7.2f}"
              f"  {np.median(np.abs(ey)):9.3f}  {np.mean(ex):7.2f}"
              f"  {np.nanmedian(ew):9.2f}")
    hit = [r for r in rows if r["ok"]]
    if hit:
        ex = np.abs([r["xtrack_err"] for r in hit])
        ey = np.abs([r["yaw_err"] for r in hit])
        print(f"    {'ALL':>13}  {len(rows):5d}  {len(hit)/len(rows):5.2f}"
              f"  {np.median(ex):12.2f}  {np.percentile(ex, 90):7.2f}"
              f"  {np.median(ey):9.3f}")


def evaluate_arm(arm, dataset_root, weights, split, conf, bins, device,
                 oracle_only=False, ground_window=True):
    view, target = ARMS[arm]
    root = Path(dataset_root)
    cam = CameraModel(**json.loads((root / "meta" / "camera.json").read_text())["camera"])
    records = {json.loads(l)["name"]: json.loads(l)
               for l in (root / "meta" / f"{split}.jsonl").read_text().splitlines()}

    img_dir = root / "arms" / arm / "images" / split
    lbl_dir = root / "arms" / arm / "labels" / split
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    if not images:
        raise SystemExit(f"no images in {img_dir}")

    def size_of(rec):
        g = rec["grid"]
        return (g["width"], g["height"]) if view == "bev" else (cam.width, cam.height)

    def score(poly_px, rec):
        row = {"range_m": rec["range_m"], "ok": False,
               "xtrack_err": np.nan, "yaw_err": np.nan, "width_err": np.nan}
        if poly_px is None:
            return row
        est = estimate(polygon_to_ground(poly_px, rec, cam, view, ground_window))
        if est is not None:
            row.update(ok=True,
                       xtrack_err=est["right_m"] - rec["right_m"],
                       yaw_err=est["yaw_deg"] - rec["yaw_deg"],
                       width_err=est["width_m"] - rec["runway_width_m"])
        return row

    oracle_rows = [score(label_polygon(lbl_dir / f"{p.stem}.txt", *size_of(records[p.stem])),
                         records[p.stem]) for p in images]

    pred_rows = []
    if not oracle_only:
        from ultralytics import YOLO
        model = YOLO(weights)
        for chunk in range(0, len(images), 64):
            batch = images[chunk:chunk + 64]
            results = model.predict(batch, conf=conf, verbose=False, device=device,
                                    retina_masks=(target == "segment"))
            for path, res in zip(batch, results):
                pred_rows.append(score(predicted_polygon(res, target), records[path.stem]))

    print(f"\n=== {arm}  ({view} image, {target} target)  split={split} "
          f"n={len(images)} ===")
    summarise(oracle_rows, bins, "ORACLE (ground-truth label through the same pipeline)")
    if pred_rows:
        summarise(pred_rows, bins, f"PREDICTED ({Path(weights).name})")
    return {"arm": arm, "oracle": oracle_rows, "pred": pred_rows}


def main():
    import yaml

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="experiments/bev_yolo/config.yaml")
    ap.add_argument("--arm", default="all", choices=["all", *ARMS])
    ap.add_argument("--weights", default=None,
                    help="default: <project>/<arm>/weights/best.pt")
    ap.add_argument("--split", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-ground-window", action="store_true",
                    help="do not clip the perspective arm to the BEV's ground "
                         "window -- shows what the far field costs it")
    ap.add_argument("--oracle-only", action="store_true",
                    help="skip the network: how good could this arm possibly be?")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    split = args.split or cfg["eval"]["split"]
    bins = cfg["eval"]["range_bins_m"]
    root = cfg["dataset"]["out"]
    project = Path(cfg["train"]["project"]).resolve()

    out = []
    for arm in (list(ARMS) if args.arm == "all" else [args.arm]):
        weights = args.weights or str(project / arm / "weights" / "best.pt")
        if not args.oracle_only and not Path(weights).exists():
            print(f"\n=== {arm}: no weights at {weights}, skipping ===")
            continue
        out.append(evaluate_arm(arm, root, weights, split, cfg["eval"]["conf"],
                                bins, args.device, oracle_only=args.oracle_only,
                                ground_window=not args.no_ground_window))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            [{k: (v if k == "arm" else [dict(r) for r in v]) for k, v in o.items()}
             for o in out], indent=2, default=float))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
