#!/usr/bin/env python
"""All-flights summary: heading error from 1500 m, cross-track error from 1000 m.

Writes media/summary.png (error vs distance, every flight + median) and
media/summary.csv (per-flight and per-distance-bin MAE / p95), and prints both.

    python -m experiments.joint_demo.summary
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.joint_demo import data  # noqa: E402
from experiments.joint_demo.render import HEX, RGB, _style  # noqa: E402

BINS = [150, 300, 600, 1000, 1500]


def stats(e):
    e = np.abs(e.dropna())
    return pd.Series({"n": len(e), "MAE": e.mean(), "p95": e.quantile(0.95), "max": e.max()})


def main():
    out = Path(data.HERE / "media")
    out.mkdir(exist_ok=True)
    dfs = []
    for ep in data.episodes():
        df, _, _ = data.load(ep)
        df["episode"] = ep
        df["airport"] = " ".join(data.meta(ep))
        dfs.append(df)
    d = pd.concat(dfs, ignore_index=True)
    hd, xt = d[d.heading_on], d[d.xtrack_on]

    per_flight = pd.concat({
        "heading_deg": hd.groupby(["episode", "airport"]).yaw_err.apply(stats).unstack(),
        "xtrack_m": xt.groupby(["episode", "airport"]).xt_err.apply(stats).unstack(),
    }, axis=1)
    d["bin"] = pd.cut(d.range_m, BINS)
    per_bin = pd.concat({
        "heading_deg": d[d.heading_on].groupby("bin", observed=True).yaw_err.apply(stats).unstack(),
        "xtrack_m": d[d.xtrack_on].groupby("bin", observed=True).xt_err.apply(stats).unstack(),
    }, axis=1)
    overall = pd.DataFrame({"heading_deg (150-1500 m)": stats(hd.yaw_err),
                            "xtrack_m (150-1000 m)": stats(xt.xt_err)}).T
    pd.set_option("display.width", 200)
    print(overall.round(3), "\n")
    print(per_bin.round(3), "\n")
    print(per_flight.round(3))
    with open(out / "summary.csv", "w") as f:
        for name, t in (("overall", overall), ("per_distance_bin", per_bin),
                        ("per_flight", per_flight)):
            f.write(f"# {name}\n")
            t.round(4).to_csv(f)
            f.write("\n")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), dpi=150)
    fig.patch.set_facecolor(HEX["surface"])
    grid = np.arange(150, 1501, 10.0)
    for ax, sub, key, col, lo, title, unit in (
            (axes[0], hd, "yaw_err", "heading", 1500, "Heading error (partner's joint fit + EKF)", "deg"),
            (axes[1], xt, "xt_err", "xtrack", 1000, "Cross-track error (our edge-line tilt)", "m")):
        curves = []
        for ep, g in sub.groupby("episode"):
            g = g.sort_values("range_m")
            a = np.interp(grid, g.range_m, g[key].abs(), left=np.nan, right=np.nan)
            a[(grid < g.range_m.min()) | (grid > g.range_m.max())] = np.nan
            curves.append(a)
            ax.plot(g.range_m, g[key].abs(), color="0.7", lw=0.7)
        c = np.array(curves)
        m = grid <= lo
        ax.fill_between(grid[m], np.nanpercentile(c[:, m], 25, 0), np.nanpercentile(c[:, m], 75, 0),
                        color=RGB[col], alpha=0.2, lw=0, label="middle 50% of flights")
        ax.plot(grid[m], np.nanmedian(c[:, m], 0), color=RGB[col], lw=2.2, label="median flight")
        ax.plot([], [], color="0.7", lw=0.7, label="each flight")
        mae = sub[key].abs().mean()
        ax.set_title(f"{title}\nMAE {mae:.2f} {unit} over {sub.episode.nunique()} held-out flights, "
                     f"{len(sub)} frames", loc="left", fontsize=11, color=HEX["ink"])
        ax.set_xlim(1520, 130)
        ax.set_ylim(0, np.nanpercentile(sub[key].abs(), 99) * 1.1)
        ax.set_xlabel("distance to threshold (m)", color=HEX["ink2"])
        ax.set_ylabel(f"|error| ({unit})", color=HEX["ink2"])
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        _style(ax)
    axes[1].axvspan(1520, 1000, color=RGB["zone"], lw=0)
    axes[1].text(1260, 0.5, "cross-track\nstarts at 1 km", transform=axes[1].get_xaxis_transform(),
                 ha="center", va="center", fontsize=9, color=HEX["ink2"])
    fig.tight_layout()
    fig.savefig(out / "summary.png", facecolor=fig.get_facecolor())
    print(f"\nwrote {out / 'summary.png'}, {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
