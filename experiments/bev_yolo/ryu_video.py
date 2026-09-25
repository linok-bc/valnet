#!/usr/bin/env python
"""Animate one approach: estimated cross-track against the truth, frame by frame.

Renders the same descent twice, changing exactly one thing -- where the heading
comes from -- so the pair isolates the finding in README.md that the cross-track
error is heading error levered by range:

    paper       yaw from the INS (eq 22), as Ryu et al. do
    fitted_yaw  yaw fitted from the mask's minimum-area rectangle

Both run the paper's estimator and its Kalman filter (section 3.1.2.5), so the
two videos differ in the heading source and nothing else.

EVERY SOURCE FRAME, NOT THE BUILT SUBSET. The dataset under `datasets/bev_toy`
holds one frame in five, inside a 150-1500 m range window (see config.yaml,
`select`). This reads the collector episode directly instead and warps each
frame to BEV on the fly, so the video is the whole descent at capture rate --
including the long-range frames the experiment deliberately excludes, which are
drawn greyed out and marked. Watching the estimator fail out there, then lock on
as the runway resolves, is most of the value of animating this at all.

The filter is causal -- `kalman_track` is a forward pass, so the blue line at any
frame uses only measurements up to that frame. The animation is therefore an
honest replay of what the system would have output in flight, not a smoothing of
the finished track.

    python -m experiments.bev_yolo.ryu_video --episode 000131
    python -m experiments.bev_yolo.ryu_video --episode 000131 --fps 60 --gif
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from experiments.bev.camera import CameraModel
from experiments.bev.episodes import Episode, load_runways
from experiments.bev.ryu import RyuEstimator, front_edge, kalman_track, simplify
from experiments.bev.transform import bev_from_image, warp_image
from experiments.bev_yolo.arms import load_config, load_grids
from experiments.bev_yolo.eval import ground_from_bev, predicted_polygon
from experiments.bev_yolo.ryu_eval import VARIANTS

TRUTH, RAW, KF, MASK = "#111111", "#d1495b", "#1f6feb", "#2ea043"
OUTSIDE = "#b9b9b9"          # frames outside the experiment's range window


def _m(v):
    """Metres, or an em-dash before the filter has anything to say."""
    return f"{v:+7.2f} m" if np.isfinite(v) else "      — "


def _limits(reference, series, margin=0.08):
    """Axis limits that fit the data instead of straddling zero.

    Symmetric limits waste half the plot here: in-window cross-track runs
    -17 .. +10, so a +-18 axis leaves the whole upper band empty and puts the
    signal off-centre. `reference` (the truth) is included whole because it is
    the line everything is read against; `series` (the estimates) enter at the
    1st/99th percentile so one blown frame cannot set the scale.
    """
    ref = np.concatenate([np.asarray(a, float).ravel() for a in reference])
    ref = ref[np.isfinite(ref)]
    est = np.concatenate([np.asarray(a, float).ravel() for a in series])
    est = est[np.isfinite(est)]
    lo = min(ref.min(), np.percentile(est, 1)) if len(est) else ref.min()
    hi = max(ref.max(), np.percentile(est, 99)) if len(est) else ref.max()
    pad = max((hi - lo) * margin, 0.5)
    return lo - pad, hi + pad


def _break(y, lo, hi):
    """NaN out whatever leaves the axes, so the line BREAKS rather than drawing a
    full-height stripe. Out-of-window estimates reach -365 m and +711 m on this
    episode; clipped to an 18 m axis they render as solid vertical bars that
    dominate the plot and say nothing."""
    y = np.asarray(y, float).copy()
    y[(y < lo) | (y > hi)] = np.nan
    return y


def collect(arm, cfg, weights, episode, stride, device, dp_frac, cache):
    """Warp every frame of one collector episode to BEV, predict, back-project.

    Returns (cam, frames). Each frame carries its source image path, the warped
    BEV canvas (cached to `cache` so the two renders share one warp), the mask in
    both frames, and whether the built dataset kept it.
    """
    root = Path(cfg["dataset"]["out"])
    cam = CameraModel(**json.loads((root / "meta" / "camera.json").read_text())["camera"])

    # which frames the experiment actually trained and scored on
    kept = set()
    for split in ("train", "val", "test"):
        f = root / "meta" / f"{split}.jsonl"
        if f.exists():
            kept |= {json.loads(l)["name"] for l in f.read_text().splitlines()}

    ep_dir = Path(cfg["source"]["root"]) / episode
    if not ep_dir.exists():
        raise SystemExit(f"no episode at {ep_dir}")
    ep = Episode(ep_dir)
    runways = load_runways(cfg["source"].get("runways_dat"))
    rw = runways.get((ep.airport, ep.runway_name))
    if rw is None:
        raise SystemExit(f"{ep.airport}/{ep.runway_name} is not in runways.dat")

    policy = load_grids(cfg)[arm.grid]

    sel = cfg["select"]
    frames = []
    for fr in ep.frames(stride=stride):
        if fr.range_m <= 0:
            continue
        grid = policy.for_frame(cam, fr.roll_deg, fr.pitch_deg, fr.height_m,
                                range_m=fr.range_m)
        m_bev = bev_from_image(cam, grid, fr.roll_deg, fr.pitch_deg, fr.height_m)
        src = cv2.imread(str(fr.image_path))
        bev = warp_image(src, m_bev, grid)
        bev_path = cache / f"bev_{fr.stem}.png"
        cv2.imwrite(str(bev_path), bev)
        frames.append({
            "stem": fr.stem, "src_path": fr.image_path, "bev_path": bev_path,
            "grid": grid, "m_bev": m_bev, "frame": fr,
            "range_m": fr.range_m, "height_m": fr.height_m, "right_m": fr.right_m,
            "yaw_deg": fr.yaw_deg, "pitch_deg": fr.pitch_deg, "roll_deg": fr.roll_deg,
            "runway_width_m": rw.width_m,
            "label_src": fr.polygon_px(cam.width, cam.height) if fr.labelled else None,
            # the experiment's RANGE window -- what the tables are computed over.
            # NOT membership of the stride-5 build: at stride 1 that would be one
            # frame in five, and the on-screen tag would flicker every frame.
            "in_window": sel["min_range_m"] <= fr.range_m <= sel["max_range_m"],
            "built": f"{episode}_{fr.stem}" in kept,
        })

    # ---- one YOLO pass over the warped canvases, shared by both variants ----
    from ultralytics import YOLO
    model = YOLO(weights)
    for i in range(0, len(frames), 32):
        batch = frames[i:i + 32]
        res = model.predict([cv2.imread(str(f["bev_path"])) for f in batch],
                            conf=cfg["eval"]["conf"], verbose=False, device=device,
                            retina_masks=(arm.target == "segment"))
        for f, r in zip(batch, res):
            poly_px = predicted_polygon(r, arm.target)
            f["poly_px"] = poly_px
            f["poly_ground"] = None
            f["poly_src"] = None
            if poly_px is None:
                continue
            simple = simplify(poly_px, dp_frac)
            f["poly_px"] = simple
            f["poly_ground"] = ground_from_bev(simple, f["grid"])
            # back into the source image, so the mask can be drawn where it came from
            inv = np.linalg.inv(f["m_bev"])
            q = np.column_stack([simple, np.ones(len(simple))]) @ inv.T
            ok = np.abs(q[:, 2]) > 1e-9
            f["poly_src"] = (q[ok, :2] / q[ok, 2:3]) if ok.any() else None
    return cam, frames


def estimate_track(frames, cam, spec, pivot, q, r):
    """Run one variant over the episode, then the causal Kalman filter.

    The `marks` it returns are what the estimator READS on each frame -- the
    front edge and its midpoint for `edge="front"`/`"rect"`, the area centroid
    for `edge="axis"`. Drawing the front edge under an `axis` run would show the
    wrong thing entirely.
    """
    est = RyuEstimator(pivot=pivot, **spec)
    raw, marks = [], []
    for f in frames:
        out, mark = None, None
        pg = f["poly_ground"]
        if pg is not None and len(pg) >= 3:
            out = est(pg, cam, f["roll_deg"], f["pitch_deg"], f["height_m"],
                      f["runway_width_m"], yaw_deg=f["yaw_deg"])
            if out is not None:
                mark = {"point": out["threshold_m"],
                        "edge": front_edge(pg) if spec["edge"] != "axis" else None}
        raw.append(out["right_m"] if out else np.nan)
        marks.append(mark)
    s = np.array([-f["range_m"] for f in frames])     # along-track, rising
    return np.array(raw), kalman_track(np.array(raw), s, q=q, r=r), marks


def to_bev_px(pts_ground, grid):
    """Ground metres -> BEV pixels, for drawing on the canvas."""
    p = np.asarray(pts_ground, dtype=np.float64).reshape(-1, 2)
    q = np.column_stack([p, np.ones(len(p))]) @ np.linalg.inv(grid.A).T
    return q[:, :2] / q[:, 2:3]


def render(frames, truth, raw, kf, marks, title, subtitle, out_dir, ylim, elim,
           window_m, xlim, reads):
    """One PNG per frame: the source image, the BEV canvas, and the track.

    `ylim` and `elim` are passed in rather than fitted per variant so that the two
    videos share axes: the comparison is the point, and a pair of plots that each
    rescale to their own error is a pair that hides it.
    """
    rng = np.array([f["range_m"] for f in frames])
    e_raw, e_kf = raw - truth, kf - truth
    t_b, raw_b, kf_b = (_break(truth, *ylim), _break(raw, *ylim), _break(kf, *ylim))
    er_b, ek_b = _break(e_raw, *elim), _break(e_kf, *elim)

    for i in range(len(frames)):
        f = frames[i]
        inw = f["in_window"]
        fig = plt.figure(figsize=(16.0, 6.0), dpi=100)
        gs = GridSpec(2, 3, width_ratios=[2.35, 0.62, 3.6],
                      height_ratios=[3.0, 1.05], wspace=0.14, hspace=0.30)

        # ---- the source frame the camera actually saw --------------------
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(cv2.cvtColor(cv2.imread(str(f["src_path"])), cv2.COLOR_BGR2RGB))
        if f["label_src"] is not None:
            lp = np.vstack([f["label_src"], f["label_src"][:1]])
            ax.plot(lp[:, 0], lp[:, 1], color="white", lw=1.1, ls="--", alpha=0.85)
        if f.get("poly_src") is not None and len(f["poly_src"]) >= 3:
            pp = np.vstack([f["poly_src"], f["poly_src"][:1]])
            ax.plot(pp[:, 0], pp[:, 1], color=MASK, lw=1.5)
        ax.set_xlim(0, 1280); ax.set_ylim(720, 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("source camera  (white = truth, green = YOLO mask)",
                     fontsize=8.5, pad=4)

        # ---- the flight state, and whether this frame is in the experiment --
        ax = fig.add_subplot(gs[1, 0]); ax.axis("off")
        tag = ("in experiment window" if inw else
               f"OUTSIDE window (>{window_m:.0f} m: localizer capture, not approach)")
        ax.text(0.0, 0.92,
                f"frame {f['stem']}   range {f['range_m']:7.0f} m   "
                f"height {f['height_m']:5.0f} m\n"
                f"roll {f['roll_deg']:+5.1f}°   pitch {f['pitch_deg']:+5.1f}°   "
                f"yaw {f['yaw_deg']:+5.2f}°",
                transform=ax.transAxes, va="top", fontsize=9, family="monospace")
        ax.text(0.0, 0.16, tag, transform=ax.transAxes, va="center", fontsize=8.5,
                color=("#2ea043" if inw else "#a15c00"),
                bbox=dict(fc=("#eaf6ec" if inw else "#fdf3e3"),
                          ec=("#2ea043" if inw else "#d9a441"), alpha=0.9, pad=3))

        # ---- the BEV canvas the detector was given ------------------------
        ax = fig.add_subplot(gs[:, 1])
        bev = cv2.cvtColor(cv2.imread(str(f["bev_path"])), cv2.COLOR_BGR2RGB)
        ax.imshow(bev)
        if f["poly_px"] is not None and len(f["poly_px"]) >= 3:
            pp = np.vstack([f["poly_px"], f["poly_px"][:1]])
            ax.plot(pp[:, 0], pp[:, 1], color=MASK, lw=1.4)
        if marks[i] is not None:
            if marks[i]["edge"] is not None:
                e = to_bev_px(np.array(marks[i]["edge"]), f["grid"])
                ax.plot(e[:, 0], e[:, 1], color=RAW, lw=2.8, solid_capstyle="round")
            pt = to_bev_px(np.asarray(marks[i]["point"]).reshape(1, 2), f["grid"])[0]
            ax.plot(pt[0], pt[1], "o", color=RAW, ms=7, zorder=5)
        ax.set_xlim(0, bev.shape[1]); ax.set_ylim(bev.shape[0], 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"BEV\n(red = {reads})", fontsize=8.5, pad=4)

        # ---- the cross-track track ----------------------------------------
        ax = fig.add_subplot(gs[0, 2])
        ax.axvspan(xlim[0], window_m, color=OUTSIDE, alpha=0.22, lw=0)
        ax.axvline(window_m, color="#888", lw=0.9, ls=":")
        ax.text(window_m, ylim[1] - 0.035 * (ylim[1] - ylim[0]),
                f" {window_m:.0f} m ", fontsize=7, color="#666",
                ha="left", va="top")
        ax.axhline(0, color="#999", lw=0.8)
        ax.fill_between(rng, truth - 3, truth + 3, color="#bbbbbb", alpha=0.35,
                        lw=0, label="±3 m")
        ax.fill_between(rng, truth - 1, truth + 1, color="#f0a202", alpha=0.45,
                        lw=0, label="±1 m")
        ax.plot(rng, t_b, color=TRUTH, lw=1.8, label="truth (log)")
        ax.plot(rng[:i + 1], raw_b[:i + 1], color=RAW, lw=0.8, alpha=0.7,
                label="estimate, per frame")
        ax.plot(rng[:i + 1], kf_b[:i + 1], color=KF, lw=2.3, label="+ Kalman")
        ax.plot(rng[i], t_b[i], "o", color=TRUTH, ms=6, zorder=5)
        if np.isfinite(kf_b[i]):
            ax.plot(rng[i], kf_b[i], "o", color=KF, ms=7, zorder=6)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_ylabel("cross-track (m)", fontsize=9)
        ax.tick_params(labelsize=8, labelbottom=False)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5, loc="upper right", ncol=2, framealpha=0.9)
        ax.text(0.0, 1.085, title, transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="bottom")
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=8.5,
                color="#555", va="bottom")

        # ---- the running error --------------------------------------------
        ax = fig.add_subplot(gs[1, 2])
        ax.axvspan(xlim[0], window_m, color=OUTSIDE, alpha=0.22, lw=0)
        ax.axhline(0, color="#999", lw=0.8)
        ax.fill_between(rng, -1, 1, color="#f0a202", alpha=0.35, lw=0)
        ax.plot(rng[:i + 1], er_b[:i + 1], color=RAW, lw=0.8, alpha=0.65)
        ax.plot(rng[:i + 1], ek_b[:i + 1], color=KF, lw=2.0)
        ax.set_xlim(*xlim); ax.set_ylim(*elim)
        ax.set_xlabel("range to threshold (m)", fontsize=9)
        ax.set_ylabel("error (m)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)

        # MAE is reported over the experiment's window only, so the number on
        # screen is the one the tables quote
        m = np.array([fr["in_window"] for fr in frames[:i + 1]])
        inside = e_kf[:i + 1][m] if m.any() else np.array([])
        mae = np.nanmean(np.abs(inside)) if np.isfinite(inside).any() else np.nan
        ax.text(0.995, 0.08,
                f"err {_m(e_kf[i])}    MAE in window {_m(mae)}",
                transform=ax.transAxes, ha="right", fontsize=9, family="monospace",
                bbox=dict(fc="white", ec="#ccc", alpha=0.92, pad=3))

        fig.subplots_adjust(top=0.87)
        fig.savefig(out_dir / f"{i:05d}.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)


def encode(png_dir, stem, fps, hold, gif, gif_fps):
    """PNGs -> mp4, and optionally a decimated gif.

    A 60 fps gif of a few hundred frames is tens of megabytes and plays badly, so
    the gif is deliberately coarser than the mp4 rather than a copy of it.
    """
    pngs = sorted(png_dir.glob("*.png"))
    for k in range(hold):
        shutil.copy(pngs[-1], png_dir / f"{len(pngs) + k:05d}.png")

    out = []
    mp4 = f"{stem}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", str(png_dir / "%05d.png"),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", mp4],
                   check=True)
    out.append(mp4)
    if gif:
        pal, g = png_dir / "pal.png", f"{stem}.gif"
        vf = f"fps={gif_fps},scale=1000:-1:flags=lanczos"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4,
                        "-vf", f"{vf},palettegen=max_colors=128", str(pal)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-i", str(pal),
                        "-lavfi", f"{vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
                        g], check=True)
        out.append(g)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="experiments/bev_yolo/config.yaml")
    ap.add_argument("--arm", default="bev_seg")
    ap.add_argument("--episode", default="000131")
    ap.add_argument("--variant", default="paper,fitted_yaw",
                    help="comma-separated. 'paper,fitted_yaw' contrasts the "
                         "heading source; 'paper,paper_axis' contrasts the point "
                         "read off the mask")
    ap.add_argument("--stride", type=int, default=1,
                    help="1 = every captured frame (the default); the built "
                         "dataset uses 5")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--pivot", default="auto")
    ap.add_argument("--dp-frac", type=float, default=0.01)
    ap.add_argument("--kf-q", type=float, default=1e-5)
    ap.add_argument("--kf-r", type=float, default=4.0)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--hold", type=int, default=90, help="frames held at the end")
    ap.add_argument("--gif", action="store_true",
                    help="also write a decimated gif beside the mp4")
    ap.add_argument("--gif-fps", type=int, default=20)
    ap.add_argument("--out", default="experiments/bev_yolo/media")
    args = ap.parse_args()

    cfg, arms, _ = load_config(args.config)
    arm = arms[args.arm]
    if not arm.is_bev:
        raise SystemExit(f"{arm.name} is a perspective arm; this renders a BEV warp")
    weights = args.weights or str(Path(cfg["train"]["project"]).resolve()
                                  / arm.name / "weights" / "best.pt")
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as cache:
        cam, frames = collect(arm, cfg, weights, args.episode, args.stride,
                              args.device, args.dp_frac, Path(cache))
        truth = np.array([f["right_m"] for f in frames])
        inw = np.array([f["in_window"] for f in frames])
        print(f"episode {args.episode}: {len(frames)} frames, "
              f"{frames[0]['range_m']:.0f} -> {frames[-1]['range_m']:.0f} m; "
              f"{inw.sum()} inside the {cfg['select']['max_range_m']:.0f} m window, "
              f"{sum(f['built'] for f in frames)} of which the stride-"
              f"{cfg['select']['stride']} build kept")

        # first pass: every variant, so both videos can share one pair of axes
        tracks = {v: estimate_track(frames, cam, VARIANTS[v], args.pivot,
                                    args.kf_q, args.kf_r)
                  for v in args.variant.split(",")}
        # limits come from the IN-WINDOW frames only; the long-range excursions
        # are real but would flatten the part of the plot that matters
        # Limits come from the IN-WINDOW frames only, and are shared by both
        # variants. Outside the window the estimates reach hundreds of metres --
        # real, but letting them set the scale flattens the part under test into
        # nothing; `_break` keeps them from drawing stripes across the axes.
        est = [a[inw] for rw, k, _ in tracks.values() for a in (rw, k)]
        ylim = _limits([truth[inw]], est)
        elim = _limits([np.zeros(1)],
                       [(a - truth)[inw] for rw, k, _ in tracks.values()
                        for a in (rw, k)])
        # the dead stretch before the window was 48% of the plot width; a lead-in
        # of 30% of the window keeps the approach into it without the emptiness
        xlim = (min(float(np.max([f["range_m"] for f in frames])),
                    float(cfg["select"]["max_range_m"]) * 1.3),
                float(np.min([f["range_m"] for f in frames])) * 0.75)

        made = []
        for vname, (raw, kf, marks) in tracks.items():
            spec = VARIANTS[vname]
            e = (kf - truth)[inw]
            head = ("Heading KNOWN (from INS)" if spec["yaw"] == "known"
                    else "Heading UNKNOWN (fitted from the mask)")
            shape = {"front": "two-point front edge (Ryu et al.)",
                     "rect": "min-area rectangle",
                     "axis": "polygon area centroid"}[spec["edge"]]
            title = f"{head} — {shape} + Kalman"
            sub = (f"{args.arm} · episode {args.episode} · every frame at "
                   f"{args.fps} fps · scale={spec['scale']} yaw={spec['yaw']} "
                   f"edge={spec['edge']} · in-window MAE {np.nanmean(np.abs(e)):.2f} m, "
                   f"RMSE {np.sqrt(np.nanmean(e ** 2)):.2f} m")
            with tempfile.TemporaryDirectory() as td:
                render(frames, truth, raw, kf, marks, title, sub, Path(td),
                       ylim, elim, float(cfg["select"]["max_range_m"]), xlim,
                       {"front": "front edge", "rect": "rect near edge",
                        "axis": "area centroid"}[spec["edge"]])
                made += encode(Path(td),
                               str(out_root / f"xtrack_{args.episode}_{vname}"),
                               args.fps, args.hold, args.gif, args.gif_fps)
            print(f"  {vname:11s} in-window MAE {np.nanmean(np.abs(e)):5.2f} m   "
                  f"RMSE {np.sqrt(np.nanmean(e ** 2)):5.2f} m")

    print("\nwrote:")
    for m in made:
        print(f"  {m}  ({Path(m).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
