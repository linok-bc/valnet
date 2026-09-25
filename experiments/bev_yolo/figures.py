#!/usr/bin/env python
"""Report figures for the Ryu estimator + shape-leg experiments.

One YOLO pass over the test split produces everything, cached to an npz so the
plots can be restyled without touching the GPU:

    fig1_error_vs_range.png   the headline: where each estimator's error lives
    fig2_ablation.png         MAE and worst case for every variant
    fig3_oracle_gap.png       detector-limited vs geometry-limited
    fig4_why_two_points.png   the tail that breaks the front-edge rule
    fig5_label_lever.png      the centroid's oracle floor is a label defect

    python -m experiments.bev_yolo.figures            # plots from cache if present
    python -m experiments.bev_yolo.figures --recompute

Colours are slots 1-3 of the data-viz reference palette, which is documented as
validated all-pairs in both modes; nothing here re-steps them.
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
from experiments.bev.episodes import ground_to_runway
from experiments.bev.ryu import (RyuEstimator, front_edge, polygon_centroid,
                                 simplify)
from experiments.bev.transform import BEVGrid
from experiments.bev_yolo.arms import load_config
from experiments.bev_yolo.eval import (ground_from_bev, label_polygon,
                                       predicted_polygon)
from experiments.bev_yolo.ryu_eval import VARIANTS

# data-viz reference palette, light mode, slots 1-3 (validated all-pairs)
C = {"front": "#2a78d6", "rect": "#eb6834", "axis": "#1baf7a"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e3e2dd"
BINS = [150, 300, 600, 1000, 1500]
SHOW = [("paper", "front", "Ryu: two-point front edge"),
        ("paper_rect", "rect", "Min-area rectangle"),
        ("paper_axis", "axis", "Polygon area centroid")]
CACHE = Path("experiments/bev_yolo/media/figdata.npz")


def _style(ax, xlabel=None, ylabel=None, title=None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3, color=GRID)
    ax.grid(color=GRID, lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9.5, color=INK2)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9.5, color=INK2)
    if title: ax.set_title(title, fontsize=12.5, color=INK, fontweight="bold",
                           loc="left", pad=10)


def compute(cfg, arm_name="bev_seg", split="test"):
    """One pass: per-frame estimates for every variant, plus the two diagnostics."""
    _, arms, _ = cfg
    cfg = cfg[0]
    arm = arms[arm_name]
    root = Path(cfg["dataset"]["out"])
    cam = CameraModel(**json.loads((root / "meta" / "camera.json").read_text())["camera"])
    meta = {json.loads(l)["name"]: json.loads(l)
            for l in (root / "meta" / f"{split}.jsonl").read_text().splitlines()}
    img_dir = root / "arms" / arm_name / "images" / split
    lbl_dir = root / "arms" / arm_name / "labels" / split
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".png"))

    from ultralytics import YOLO
    model = YOLO(str(Path(cfg["train"]["project"]).resolve() / arm_name / "weights" / "best.pt"))

    out = {k: [] for k in ("rng", "thr_lat", "lever_off", "lever_pred")}
    for v in VARIANTS:
        out[f"p_{v}"] = []
        out[f"o_{v}"] = []

    for i in range(0, len(images), 64):
        batch = images[i:i + 64]
        res = model.predict(batch, conf=cfg["eval"]["conf"], verbose=False,
                            retina_masks=True)
        for path, r in zip(batch, res):
            rec = meta[path.stem]
            g = rec["grids"][arm.grid]
            grid = BEVGrid(g["width"], g["height"], g["mpp"], g["x_near"])
            pp = predicted_polygon(r, "segment")
            lp = label_polygon(lbl_dir / f"{path.stem}.txt", g["width"], g["height"])
            if lp is None:
                continue
            lg = ground_from_bev(lp, grid)
            pg = ground_from_bev(simplify(pp, 0.01), grid) if pp is not None else None
            out["rng"].append(rec["range_m"])

            kw = (cam, rec["roll_deg"], rec["pitch_deg"], rec["height_m"],
                  rec["runway_width_m"])
            for v, spec in VARIANTS.items():
                est = RyuEstimator(**spec)
                for tag, poly in (("o", lg), ("p", pg)):
                    e = np.nan
                    if poly is not None and len(poly) >= 3:
                        o = est(poly, *kw, yaw_deg=rec["yaw_deg"])
                        if o is not None:
                            e = o["right_m"] - rec["right_m"]
                    out[f"{tag}_{v}"].append(e)

            # diagnostic 1: lateral error of the predicted FRONT-EDGE midpoint
            d = np.nan
            if pg is not None and len(pg) >= 3:
                fp, fl = front_edge(pg), front_edge(lg)
                if fp is not None and fl is not None:
                    d = ((fp[0] + fp[1]) / 2)[1] - ((fl[0] + fl[1]) / 2)[1]
            out["thr_lat"].append(d)

            # diagnostic 2: the LABEL's rotation, levered by how far down the
            # runway the centroid sits
            rc = ground_to_runway(lg, rec["along_m"], rec["right_m"], rec["yaw_deg"])
            c = polygon_centroid(rc)
            o_ = np.argsort(rc[:, 0])
            near, far = rc[o_[:2]].mean(0), rc[o_[-2:]].mean(0)
            u = far - near
            out["lever_off"].append(c[1])
            out["lever_pred"].append((c[0] - near[0]) * np.sin(np.arctan2(u[1], u[0])))

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **{k: np.array(v, dtype=float) for k, v in out.items()})
    print(f"wrote {CACHE}  (n={len(out['rng'])})")


def _binned(rng, err, stat):
    xs, ys = [], []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (rng >= lo) & (rng < hi) & np.isfinite(err)
        if m.sum():
            xs.append((lo + hi) / 2)
            ys.append(stat(err[m]))
    return np.array(xs), np.array(ys)


def fig1(d, out):
    """The headline: the two-point rule degrades as the aircraft closes in."""
    rng = d["rng"]
    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=200)
    top = 0.0
    for v, c, label in SHOW:
        x, y = _binned(rng, d[f"p_{v}"], lambda e: np.abs(e).mean())
        top = max(top, y.max())
        ax.plot(x, y, "-o", color=C[c], lw=2.0, ms=8, mec="white", mew=1.5,
                zorder=3)
        # direct labels: no legend box, so nothing can collide with the data
        ax.annotate(f"  {label}\n  {y[0]:.2f} m", (x[0], y[0]),
                    textcoords="offset points", xytext=(12, 0), ha="left",
                    va="center", fontsize=9.5, color=C[c], fontweight="bold",
                    linespacing=1.5)
        ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9, color=INK2)
    _style(ax, "range to threshold (m)", "cross-track MAE (m)",
           "Where each estimator's error lives")
    ax.set_xticks([(a + b) / 2 for a, b in zip(BINS[:-1], BINS[1:])])
    ax.set_xticklabels([f"{a}–{b}" for a, b in zip(BINS[:-1], BINS[1:])])
    ax.set_xlim(BINS[-1], -520)          # room on the right for the direct labels
    ax.set_ylim(0, top * 1.18)
    ax.text(0, -0.21, "Reading two points gets WORSE close in; fitting the whole "
            "mask gets better.", transform=ax.transAxes, fontsize=9.5, color=MUTED)
    fig.savefig(out / "fig1_error_vs_range.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def fig2(d, out):
    """Ablation: mean and worst case, as two panels (never a second y-axis)."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), dpi=200)
    fig.subplots_adjust(wspace=0.12)
    names = [s[0] for s in SHOW][::-1]
    cols = [C[s[1]] for s in SHOW][::-1]
    labs = [s[2] for s in SHOW][::-1]
    for ax, stat, title, unit in (
            (axes[0], lambda e: np.abs(e).mean(), "Mean absolute error", "m"),
            (axes[1], lambda e: np.abs(e).max(), "Worst single frame", "m")):
        vals = [stat(d[f"p_{v}"][np.isfinite(d[f"p_{v}"])]) for v in names]
        y = np.arange(len(names))
        ax.barh(y, vals, color=cols, height=0.62)
        for yi, val in zip(y, vals):
            ax.text(val + max(vals) * 0.025, yi, f"{val:.2f} {unit}", va="center",
                    fontsize=9.5, color=INK2)
        ax.set_yticks(y)
        # the categories are shared, so label them once instead of letting the
        # right panel's ticks collide with the left panel's value labels
        ax.set_yticklabels(labs if ax is axes[0] else [] * len(labs),
                           fontsize=9.5, color=INK2)
        _style(ax, "metres", None, title)
        ax.xaxis.grid(True); ax.yaxis.grid(False)
        ax.set_xlim(0, max(vals) * 1.28)
    fig.savefig(out / "fig2_ablation.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig3(d, out):
    """Oracle vs predicted: which estimator is limited by what."""
    fig, ax = plt.subplots(figsize=(7.6, 3.2), dpi=200)
    for i, (v, c, label) in enumerate(SHOW):
        o = np.abs(d[f"o_{v}"][np.isfinite(d[f"o_{v}"])]).mean()
        p = np.abs(d[f"p_{v}"][np.isfinite(d[f"p_{v}"])]).mean()
        ax.plot([o, p], [i, i], "-", color=C[c], lw=2.5, alpha=0.45, zorder=2,
                solid_capstyle="round")
        ax.plot(o, i, "o", color="white", mec=C[c], mew=2.5, ms=10, zorder=3)
        ax.plot(p, i, "o", color=C[c], ms=10, zorder=3)
        ax.text(max(o, p) + 0.09, i, f"gap {abs(p - o):.2f} m", va="center",
                fontsize=9.5, color=INK2)
    ax.set_yticks(range(len(SHOW)))
    ax.set_yticklabels([s[2] for s in SHOW], fontsize=9.5, color=INK2)
    _style(ax, "cross-track MAE (m)", None,
           "Oracle (hollow) → predicted (filled)")
    ax.xaxis.grid(True); ax.yaxis.grid(False)
    ax.set_xlim(0, 2.45); ax.invert_yaxis()
    ax.text(0, -0.30, "The centroid lands within 0.11 m of its own floor; the "
            "front edge sits 7× above its.",
            transform=ax.transAxes, fontsize=9.5, color=MUTED)
    fig.savefig(out / "fig3_oracle_gap.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig4(d, out):
    """The mechanism: it is the TAIL of the front-edge midpoint that grows."""
    rng, lat = d["rng"], np.abs(d["thr_lat"])
    fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=200)
    series = [("worst frame", C["front"], lambda e: e.max()),
              ("p90", C["rect"], lambda e: np.percentile(e, 90)),
              ("median", MUTED, np.median)]
    x = _binned(rng, lat, np.median)[0]
    ax.axhline(22.5, color=MUTED, lw=1.2, ls=":", zorder=1)
    ax.text(x[-1], 21.7, "half a runway width (22.5 m)", fontsize=9,
            color=MUTED, ha="left", va="top")
    for name, col, stat in series:
        _, y = _binned(rng, lat, stat)
        ax.plot(x, y, "-o", color=col, lw=2.0, ms=8, mec="white", mew=1.5, zorder=3)
        ax.annotate(f"  {name}\n  {y[0]:.1f} m", (x[0], y[0]),
                    textcoords="offset points", xytext=(12, 0), ha="left",
                    va="center", fontsize=9.5, color=col, fontweight="bold",
                    linespacing=1.5)
    _style(ax, "range to threshold (m)",
           "front-edge midpoint lateral error (m)",
           "The median is flat — the tail is what grows")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}–{b}" for a, b in zip(BINS[:-1], BINS[1:])])
    ax.set_xlim(BINS[-1], -520); ax.set_ylim(0, 27)
    fig.savefig(out / "fig4_why_two_points.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def fig5(d, out):
    """The centroid's oracle floor is the LABEL's rotation, levered by distance."""
    a, b = d["lever_pred"], d["lever_off"]
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=200)
    lim = [min(a.min(), b.min()) - 0.3, max(a.max(), b.max()) + 0.3]
    ax.plot(lim, lim, color=MUTED, lw=1.2, ls="--", zorder=1)
    ax.scatter(a, b, s=13, color=C["axis"], alpha=0.5, lw=0, zorder=2)
    _style(ax, "predicted:  lever × sin(label rotation)  (m)",
           "measured centroid offset (m)", "The floor is a label defect")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    r = np.corrcoef(a, b)[0, 1]
    ax.text(0.04, 0.95, f"r = {r:+.3f}\nslope {np.polyfit(a, b, 1)[0]:.2f}\n"
            f"label rotation 0.235°\nlever 309 m",
            transform=ax.transAxes, va="top", fontsize=9.5, color=INK2,
            linespacing=1.6)
    fig.savefig(out / "fig5_label_lever.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="experiments/bev_yolo/config.yaml")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--out", default="experiments/bev_yolo/media/figs")
    args = ap.parse_args()

    if args.recompute or not CACHE.exists():
        compute(load_config(args.config))
    d = dict(np.load(CACHE))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for fn in (fig1, fig2, fig3, fig4, fig5):
        fn(d, out)
        print(f"  {fn.__name__}")
    print(f"\nwrote {len(list(out.glob('*.png')))} figures to {out}")


if __name__ == "__main__":
    main()
