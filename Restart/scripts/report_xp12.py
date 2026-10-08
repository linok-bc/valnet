#!/usr/bin/env python3
"""
Build tables + figures from ``data/xp12/results/<suite>/`` into ``REPORT.md``.

    PYTHONPATH=. python scripts/report_xp12.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

from evaluation.metrics import add_along_bin

# Validated categorical palette (dataviz reference instance), fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
})

TABLE_COLS = [
    ("run", "run", "{}"), ("n_frames", "frames", "{:.0f}"), ("meas_rate", "meas.", "{:.0%}"),
    ("yaw_mae", "yaw MAE °", "{:.3f}"), ("yaw_p95", "yaw p95 °", "{:.2f}"),
    ("lat_mae", "lat MAE m", "{:.2f}"), ("lat_p95", "lat p95 m", "{:.1f}"),
    ("success", "success", "{:.1%}"), ("gross", "gross", "{:.1%}"),
    ("cov95", "95% cov.", "{:.0%}"), ("runtime_ms", "ms/frame", "{:.0f}"),
]


def _log_axis(axis) -> None:
    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(ScalarFormatter())
    axis.set_minor_formatter(NullFormatter())


def md_table(df: pd.DataFrame) -> str:
    cols = [c for c in TABLE_COLS if c[0] in df]
    lines = ["| " + " | ".join(c[1] for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for key, _, fmt in cols:
            v = r[key]
            cells.append("–" if (isinstance(v, float) and np.isnan(v)) else fmt.format(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fig_errors(summary: pd.DataFrame, path: Path, title: str) -> None:
    s = summary.sort_values("lat_mae", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.32 * len(s) + 1.2), sharey=True)
    y = np.arange(len(s))
    for ax, key, unit in ((axes[0], "yaw", "deg"), (axes[1], "lat", "m")):
        ax.hlines(y, s[f"{key}_mae"], s[f"{key}_p95"], color=SERIES[0], alpha=0.35, lw=2)
        ax.plot(s[f"{key}_mae"], y, "o", color=SERIES[0], ms=7, label="MAE")
        ax.plot(s[f"{key}_p95"], y, "|", color=SERIES[0], ms=12, mew=2, label="p95")
        ax.set_xscale("log")
        _log_axis(ax.xaxis)
        ax.set_xlabel(f"|{key} error| ({unit}, log)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, s["run"])
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_along(frames: pd.DataFrame, runs: list[str], path: Path, title: str) -> None:
    df = add_along_bin(frames[frames["run"].isin(runs)])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for k, run in enumerate(runs):
        g = df[df["run"] == run].groupby("along_bin", observed=True)
        for ax, col in ((axes[0], "yaw_err"), (axes[1], "lat_err")):
            med = g[col].apply(lambda v: np.nanmedian(np.abs(v)))
            ax.plot(range(len(med)), med.to_numpy(), "-o", color=SERIES[k % len(SERIES)], lw=2, ms=5, label=run)
            ax.set_xticks(range(len(med)), med.index.astype(str), rotation=0)
    axes[0].set_ylabel("median |yaw error| (deg)")
    axes[1].set_ylabel("median |lateral error| (m)")
    for ax in axes:
        ax.set_xlabel("along-track distance to threshold (m)")
        ax.set_yscale("log")
        _log_axis(ax.yaxis)
    axes[1].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_speed(summary: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(summary["runtime_ms"], summary["lat_mae"], "o", color=SERIES[0], ms=8)
    for _, r in summary.iterrows():
        ax.annotate(r["run"], (r["runtime_ms"], r["lat_mae"]), fontsize=7, color=INK,
                    xytext=(5, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _log_axis(ax.xaxis)
    _log_axis(ax.yaxis)
    ax.set_xlabel("runtime per frame (ms, log)")
    ax.set_ylabel("lateral MAE (m, log)")
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


SUITE_TEXT = {
    "solver_cold": "Single frame, oracle masks (label quad), no temporal prior — isolates the pose solver.",
    "solver_cold_yolo": "Single frame, real YOLO masks, no temporal prior.",
    "tracking": "Sequential tracking (t−1 warm start) and kinematic EKF, GT vs YOLO masks.",
    "tracking_orig": "Original repo mask-fit (raster + warm hierarchical grid) on a subset.",
    "robustness": "Tracking + EKF under mask corruption, known-DOF noise and width-prior error.",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("data/xp12/results"))
    args = ap.parse_args()
    fig_dir = args.results / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    out = ["# XP12 evaluation report", ""]
    findings = args.results / "FINDINGS.md"
    if findings.exists():
        out += [findings.read_text(), ""]
    for suite in SUITE_TEXT:
        d = args.results / suite
        if not (d / "summary.csv").exists():
            continue
        summary = pd.read_csv(d / "summary.csv")
        frames = pd.read_csv(d / "frames.csv.gz", dtype={"seq": str})
        n_seq = frames["seq"].nunique()
        out += [f"## {suite}", "", f"{SUITE_TEXT[suite]} {n_seq} sequences.", ""]
        out += [md_table(summary.sort_values("lat_mae")), ""]
        fig_errors(summary, fig_dir / f"{suite}_errors.png", f"{suite}: MAE (dot) and p95 (bar end)")
        fig_speed(summary, fig_dir / f"{suite}_speed.png", f"{suite}: accuracy vs runtime")
        pool = summary[summary["run"].str.contains("yolo")]
        pool = pool if len(pool) else summary
        which = "YOLO runs" if pool is not summary else "runs"
        best = pool.sort_values("lat_mae")["run"].head(6).tolist()
        fig_along(frames, best, fig_dir / f"{suite}_along.png",
                  f"{suite}: error vs distance (6 best {which} by lateral MAE)")
        out += [f"![]({fig_dir.name}/{suite}_errors.png)", "", f"![]({fig_dir.name}/{suite}_along.png)", "",
                f"![]({fig_dir.name}/{suite}_speed.png)", ""]
    (args.results / "REPORT.md").write_text("\n".join(out))
    print(f"→ {args.results / 'REPORT.md'}")


if __name__ == "__main__":
    main()
