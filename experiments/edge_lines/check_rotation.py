#!/usr/bin/env python
"""Is the LOGGED cross-track rotated, or are the LABELS? Let the image decide.

eval.py found that, against the label corners, the logged `right` is off by
range * (a per-episode angle of -0.21 .. +0.31 deg). That says the two
disagree; it does not say which is wrong. The rendered image is the arbiter,
because both claim to describe it.

For each chosen frame this draws, over the actual picture:

    green   the label quad (the collector projects lat/lon corners on screen)
    red     the runway placed from the LOGS: runways.dat rectangle, logged
            along / right / height / yaw / pitch / roll, fitted camera

and measures where the painted runway edges actually are, row by row, from
the image gradient alone -- no label, no log. Each outline's lateral offset
from those measured edges is printed in pixels.

If the logs are rotated, red should sit beside the runway far out and converge
onto it near the threshold, and green should sit on it everywhere. If the
labels are wrong, it is the other way round.

    python -m experiments.edge_lines.check_rotation
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.bev.camera import CameraModel
from experiments.bev.episodes import Episode, load_runways, runway_to_ground
from experiments.bev.transform import ground_to_image
from experiments.edge_lines.geometry import crosstrack, fit_line, line_through

# (episode, [target ranges]) -- the two worst angles, of opposite sign, and a
# near-zero control. Angles are eval.py's fit on split_test.
EPISODES = [("000135", "+0.31"), ("000148", "-0.21"), ("000139", "+0.01")]
RANGES = (1400, 800, 300)


def log_quad(f, rw, cam):
    """Runway corners (NL, FL, FR, NR) in pixels, from the logs alone."""
    g = runway_to_ground(rw.corners_runway_frame(), f.along_m, f.right_m, f.yaw_deg)
    H = ground_to_image(cam, f.roll_deg, f.pitch_deg, f.height_m)
    p = np.column_stack([g, np.ones(4)]) @ H.T
    return p[:, :2] / p[:, 2:3]


def u_on(p, q, v):
    """Column of line pq at row v."""
    t = (v - p[1]) / (q[1] - p[1])
    return p[0] + t * (q[0] - p[0])


def measure_edges(gray, quad_a, quad_b, pad=5, min_width_px=6):
    """Painted-edge columns along the WHOLE visible runway, from the gradient alone.

    The two outlines only say where to look: in each row the search window spans
    both of their edge positions plus `pad` px, so it favours neither. Inside it
    the strongest horizontal intensity step wins, with a parabolic sub-pixel
    peak. Rows where the runway is narrower than `min_width_px` are skipped --
    near the horizon the two edges blur into one. Returns [(v, u_left, u_right)].
    """
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 0.8)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    v_near = int(np.floor(min(quad_a[0][1], quad_a[3][1], quad_b[0][1], quad_b[3][1]))) - 2
    out = []
    for v in range(v_near, 0, -1):
        row = []
        for a_idx, b_idx in ((0, 1), (3, 2)):          # left edge NL-FL, right NR-FR
            us = [u_on(q[a_idx], q[b_idx], v) for q in (quad_a, quad_b)]
            a, b = int(np.floor(min(us) - pad)), int(np.ceil(max(us) + pad))
            if a < 1 or b >= gray.shape[1] - 1:
                break
            seg = np.abs(gx[v, a:b + 1])
            k = int(seg.argmax())
            if 0 < k < len(seg) - 1:
                y0, y1, y2 = seg[k - 1], seg[k], seg[k + 1]
                d = y0 - 2 * y1 + y2
                k = k + (0.5 * (y0 - y2) / d if abs(d) > 1e-9 else 0.0)
            row.append(a + k)
        if len(row) < 2 or row[1] - row[0] < min_width_px:
            break
        out.append((v, row[0], row[1]))
    return out


def image_crosstrack(edges, f, cam, width_m):
    """Robust line through each measured edge, then the edge-line estimator."""
    if len(edges) < 6:
        return None
    ev = np.array(edges, dtype=np.float64)
    lines = []
    for col in (1, 2):
        fit = fit_line(np.column_stack([ev[:, col], ev[:, 0]]))
        if fit is None:
            return None
        a, b = fit[0], fit[1]
        lines.append(np.array([1.0, -b, -a]))          # u = a + b v, original pixels
    out = crosstrack(lines[0], lines[1], cam, f.roll_deg, f.pitch_deg, f.height_m,
                     width_m, f.yaw_deg)
    return out["free_width"]


def label_crosstrack(f, cam, width_m):
    q = f.polygon_px(cam.width, cam.height)
    return crosstrack(line_through(q[0], q[1]), line_through(q[3], q[2]), cam,
                      f.roll_deg, f.pitch_deg, f.height_m, width_m, f.yaw_deg)["free_width"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="datasets/xp12_dataset")
    ap.add_argument("--camera", default="experiments/bev_yolo/camera.json")
    ap.add_argument("--out", default="experiments/edge_lines/media/rotation_check.png")
    ap.add_argument("--range", type=float, default=1400)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    cam = CameraModel(**json.loads(Path(args.camera).read_text())["camera"])
    runways = load_runways()

    fig, axes = plt.subplots(2, len(EPISODES), figsize=(5.6 * len(EPISODES), 8.4),
                             gridspec_kw={"height_ratios": [1.0, 1.0]})
    print(f"{'episode':>8} {'airport':>9}  n    image-vs-label   image-vs-logs"
          f"   (m, cross-track, 150-1500 m, MAE / mean)")
    for i, (e, ang) in enumerate(EPISODES):
        ep = Episode(Path(args.root) / e)
        rw = runways[(ep.airport, ep.runway_name)]
        frames = [ep.frame(s) for s in ep.stems]

        # --- top: one far frame, whole runway, both outlines over the image
        f = min(frames, key=lambda f: abs(f.range_m - args.range))
        img = cv2.imread(str(f.image_path))
        lab, log = f.polygon_px(cam.width, cam.height), log_quad(f, rw, cam)
        edges = measure_edges(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), lab, log)
        pts = np.vstack([lab, log])
        v_top = edges[-1][0] - 4 if edges else pts[:, 1].min()
        x0, x1 = int(pts[:, 0].min() - 25), int(pts[:, 0].max() + 25)
        y0, y1 = int(v_top - 6), int(max(lab[0][1], lab[3][1]) + 12)
        ax = axes[0, i]
        crop = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        ax.imshow(crop, extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5),
                  interpolation="nearest")
        for q, col, name in ((lab, "#22dd44", "label"), (log, "#ff3030", "logs")):
            ax.plot([q[0][0], q[1][0]], [q[0][1], q[1][1]], color=col, lw=1.3, label=name)
            ax.plot([q[3][0], q[2][0]], [q[3][1], q[2][1]], color=col, lw=1.3)
            ax.plot([q[0][0], q[3][0]], [q[0][1], q[3][1]], color=col, lw=1.3)
        if edges:
            ev = np.array(edges)
            ax.plot(ev[:, 1], ev[:, 0], "o", ms=2.4, color="#ffd400", label="edge found in image")
            ax.plot(ev[:, 2], ev[:, 0], "o", ms=2.4, color="#ffd400")
        ax.set_xlim(x0 - 0.5, x1 - 0.5), ax.set_ylim(y1 - 0.5, y0 - 0.5)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"{e}  {ep.airport} {ep.runway_name}  (eval fit {ang} deg)\n"
                     f"frame {f.stem}, {f.range_m:.0f} m out, zoomed", fontsize=9)
        if i == 0:
            ax.legend(loc="upper left", fontsize=7.5, framealpha=0.85)

        # --- bottom: the whole approach, three cross-track sources
        r, logged, labelled, imaged = [], [], [], []
        for f in frames[::args.stride]:
            if not f.labelled or not 150 <= f.range_m <= 1500:
                continue
            img = cv2.imread(str(f.image_path), cv2.IMREAD_GRAYSCALE)
            lab, log = f.polygon_px(cam.width, cam.height), log_quad(f, rw, cam)
            est = image_crosstrack(measure_edges(img, lab, log), f, cam, rw.width_m)
            r.append(f.range_m), logged.append(f.right_m)
            labelled.append(label_crosstrack(f, cam, rw.width_m))
            imaged.append(np.nan if est is None else est)
        r, logged, labelled, imaged = map(np.array, (r, logged, labelled, imaged))
        ok = np.isfinite(imaged)
        d_lab, d_log = imaged[ok] - labelled[ok], imaged[ok] - logged[ok]
        print(f"{e:>8} {ep.airport + ' ' + ep.runway_name:>9} {ok.sum():4d}"
              f"   {np.abs(d_lab).mean():5.2f} / {d_lab.mean():+5.2f}"
              f"    {np.abs(d_log).mean():5.2f} / {d_log.mean():+5.2f}")
        ax = axes[1, i]
        ax.plot(r, logged, color="#ff3030", lw=1.8, label="logged `right`")
        ax.plot(r, labelled, color="#22aa44", lw=1.8, label="from label corners")
        ax.plot(r[ok], imaged[ok], ".", ms=3.5, color="#333333",
                label="from edges found in image")
        ax.invert_xaxis()
        ax.set_xlabel("range to threshold (m)")
        ax.set_ylabel("cross-track, m right of centreline")
        ax.set_title(f"image vs label: {np.abs(d_lab).mean():.2f} m MAE   "
                     f"image vs logs: {np.abs(d_log).mean():.2f} m MAE", fontsize=9)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7.5)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
