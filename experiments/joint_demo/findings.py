#!/usr/bin/env python
"""Every number the findings page quotes, in one JSON (media/findings.json).

Truth is the rotation-fixed logs (data.py). Needs cache/heading, cache/fusion,
cache/bev and the tilt eval in data.STOCK -- i.e. run_heading, fuse, run_bev
first.

    python -m experiments.joint_demo.findings
"""

import json

import numpy as np
import pandas as pd

from experiments.joint_demo import data

BINS = [150, 300, 600, 1000, 1500]
TWO = ("000145", "000146")


def boot_ci(diff, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    bs = [rng.choice(diff, len(diff)).mean() for _ in range(n)]
    return [float(np.mean(diff)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    s = data._stock()
    frames = []
    for ep in data.episodes():
        f = pd.read_csv(data.HERE / "cache/fusion" / f"{ep}.csv")
        m = f.merge(s[s.episode == ep][["frame", "range_m", "truth_m", "free_width",
                                        "err_kf_free_width"]], on="frame")
        m["ep"] = ep
        b = data.HERE / "cache/bev" / f"{ep}.json"
        m["bev"] = np.nan
        if b.exists():
            bb = pd.DataFrame(json.loads(b.read_text()))[["frame", "bev_kf"]]
            m = m.drop(columns="bev").merge(bb.rename(columns={"bev_kf": "bev"}), on="frame",
                                            how="left")
        frames.append(m)
    d = pd.concat(frames, ignore_index=True)
    t = d.truth_m
    err = pd.DataFrame({
        "ep": d.ep, "range_m": d.range_m,
        "samuel": -d.samuel_lat - t, "tilt": d.err_kf_free_width, "fused": -d.fused_lat - t,
        "tilt_raw": d.free_width - t, "bev": d.bev - t,
        "yaw_samuel": d.samuel_yaw - d.samuel_yaw_gt, "yaw_fused": d.fused_yaw - d.fused_yaw_gt,
    })
    main3 = ["samuel", "tilt", "fused"]
    err["bin"] = pd.cut(err.range_m, BINS)
    by_bin = {k: [float(v) for v in err.groupby("bin", observed=True)[k].apply(lambda e: e.abs().mean())]
              for k in main3 + ["tilt_raw"]}
    x = err[err.range_m <= data.XTRACK_FROM_M]
    overall = {k: {"mae": float(x[k].abs().mean()), "p95": float(x[k].abs().quantile(0.95)),
                   "max": float(x[k].abs().max())} for k in main3 + ["tilt_raw"]}
    pf = x.groupby("ep")[main3].apply(lambda g: g.abs().mean())
    per_flight = [{"ep": ep, "site": " ".join(data.meta(ep)),
                   **{k: float(pf.loc[ep, k]) for k in main3}} for ep in pf.index]
    ci = {f"{a}-{b}": boot_ci((pf[a] - pf[b]).values)
          for a, b in (("tilt", "samuel"), ("samuel", "fused"), ("tilt", "fused"))}
    wins = {"fused_vs_samuel": int((pf.fused < pf.samuel).sum()),
            "fused_vs_tilt": int((pf.fused < pf.tilt).sum()),
            "samuel_vs_tilt": int((pf.samuel < pf.tilt).sum()),
            "n": int(len(pf))}
    heading = {k: {"mae": float(err[f"yaw_{k}"].abs().mean()),
                   "p95": float(err[f"yaw_{k}"].abs().quantile(0.95))} for k in ("samuel", "fused")}

    two = x[x.ep.isin(TWO)]
    two_tab = [{"ep": ep, "site": " ".join(data.meta(ep)),
                **{k: float(g[k].abs().mean()) for k in ("samuel", "tilt_raw", "fused", "bev")},
                "bev_bias": float(g.bev.mean())}
               for ep, g in two.groupby("ep")]

    tilt_noise = json.loads((data.HERE / "cache/tilt_noise.json").read_text())
    out = {"bins": [f"{a}–{b}" for a, b in zip(BINS[:-1], BINS[1:])], "by_bin": by_bin,
           "overall_150_1000": overall, "per_flight": per_flight, "ci_per_flight_diff": ci,
           "wins": wins, "heading_150_1500": heading, "two_flights": two_tab,
           "tilt_noise": tilt_noise,
           "n_frames_13": int(len(err)), "n_frames_scored": int(len(x))}
    path = data.HERE / "media/findings.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    main()
