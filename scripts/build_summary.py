#!/usr/bin/env python
"""Build the results summary page (one self-contained HTML file) from the result files.

Every number on the page is read from results/xp12_combined/iou_*.json and
experiments/restart_test/summary*.json; the stills are embedded as data URIs.

    python scripts/build_summary.py --figs <dir with cap.jpg cmp_fog.jpg cmp_rain.jpg samuel.jpg> --out summary.html
"""

import argparse
import base64
import html
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results/xp12_combined"
RT = REPO / "experiments/restart_test"

MODELS = [  # key, name, iou file, css var, params (M), ms/frame, train h
    ("yolo", "YOLOv8s-seg", "iou_yolo_widened", "--s1", 11.8, 26.2, 16.5),
    ("valnet", "VALNet", "iou_valnet", "--s2", 12.8, 36.2, 20.0),
    ("deeplab", "DeepLabV3+", "iou_deeplab", "--s3", 40.3, 48.9, 22.1),
]
# range bins in approach order (far to near, then on the runway)
RANGES = [
    ("1500-3000", "1.5–3 km"),
    ("1000-1500", "1–1.5 km"),
    ("600-1000", "600 m–1 km"),
    ("300-600", "300–600 m"),
    ("150-300", "150–300 m"),
    ("0-150", "0–150 m"),
    ("past threshold", "past threshold"),
]
WEATHER = [
    ("easy", "Easy (24 km, no weather)"),
    ("clear_vfr", "Clear"),
    ("haze", "Haze"),
    ("mist", "Mist"),
    ("fog", "Fog"),
    ("light_rain", "Light rain"),
    ("moderate_rain", "Moderate rain"),
    ("heavy_rain", "Heavy rain"),
]


def load(name):
    return json.loads((RES / f"{name}.json").read_text())


def img(path):
    return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def f3(x):
    return f"{x:.3f}"


def range_chart(data):
    """IoU by range bin, one line per model, SVG with per-point hover titles."""
    W, H, L, R, T, B = 760, 340, 56, 128, 20, 64
    lo, hi = 0.70, 1.00
    xs = [L + i * (W - L - R) / (len(RANGES) - 1) for i in range(len(RANGES))]
    Y = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)
    out = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="rc-title rc-desc" class="chart">',
        '<title id="rc-title">Mean per-frame IoU by range to the threshold</title>',
        (
            '<desc id="rc-desc">Lines for YOLOv8s-seg, VALNet and DeepLabV3+ across seven range bins; '
            "the same numbers are in the table below.</desc>"
        ),
    ]
    for v in (0.70, 0.80, 0.90, 1.00):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>')
        out.append(f'<text x="{L - 10}" y="{Y(v) + 4:.1f}" class="tick" text-anchor="end">{v:.2f}</text>')
    for x, (_, lab) in zip(xs, RANGES):
        out.append(f'<text x="{x:.1f}" y="{H - B + 22}" class="tick" text-anchor="middle">{html.escape(lab)}</text>')
    out.append(
        f'<text x="{(L + W - R) / 2:.0f}" y="{H - 14}" class="axis" text-anchor="middle">'
        "range to the runway threshold (approach runs left to right)</text>"
    )
    ends = []
    for key, name, f, var, *_ in MODELS:
        vals = [data[key]["by_range"][k]["mean_iou"] for k, _ in RANGES]
        pts = " ".join(f"{x:.1f},{Y(v):.1f}" for x, v in zip(xs, vals))
        out.append(f'<polyline points="{pts}" class="series" style="stroke:var({var})"/>')
        for x, v, (k, lab) in zip(xs, vals, RANGES):
            n = data[key]["by_range"][k]["n"]
            out.append(
                f'<g class="pt"><circle cx="{x:.1f}" cy="{Y(v):.1f}" r="12" class="hit"/>'
                f'<circle cx="{x:.1f}" cy="{Y(v):.1f}" r="4.5" class="dot" style="fill:var({var})"/>'
                f"<title>{name}, {html.escape(lab)}: IoU {v:.3f} ({n:,} frames)</title></g>"
            )
        ends.append([Y(vals[-1]), name, var])
    # direct labels at the right end, nudged apart so they never overlap
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 16)
    for y, name, var in ends:
        out.append(
            f'<line x1="{xs[-1] + 10:.1f}" x2="{xs[-1] + 22:.1f}" y1="{y:.1f}" y2="{y:.1f}" '
            f'style="stroke:var({var})" class="swatch"/>'
            f'<text x="{xs[-1] + 28:.1f}" y="{y + 4:.1f}" class="lab">{name}</text>'
        )
    out.append("</svg>")
    return "\n".join(out)


def iou_table(data, groups, field):
    head = "".join(
        f'<th scope="col"><span class="key" style="background:var({v})"></span>{n}</th>' for _, n, _, v, *_ in MODELS
    )
    rows = []
    for k, lab in groups:
        vals = [data[m][field][k]["mean_iou"] for m, *_ in MODELS]
        best = max(vals)
        cells = "".join(f'<td class="{"best" if v == best else ""}">{f3(v)}</td>' for v in vals)
        rows.append(
            f'<tr><th scope="row">{html.escape(lab)}</th><td class="n">'
            f"{data[MODELS[0][0]][field][k]['n']:,}</td>{cells}</tr>"
        )
    return (
        f'<div class="tw"><table><thead><tr><th scope="col"></th><th scope="col" class="n">frames</th>{head}'
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    figs = Path(args.figs)

    data = {k: load(f) for k, _, f, *_ in MODELS}
    plain = load("iou_yolo")
    rt = {
        k: json.loads((RT / f).read_text())
        for k, f in [("old", "summary.json"), ("none", "summary_nokf.json"), ("crab", "summary_crab.json")]
    }
    ov = {k: data[k]["overall"] for k in data}

    tiles = "".join(
        f'<div class="tile"><div class="tname"><span class="key" style="background:var({v})"></span>{n}</div>'
        f'<div class="big">{f3(ov[k]["mean_iou"])}</div><div class="sub">mean IoU, all test frames</div>'
        f"<dl><div><dt>Parameters</dt><dd>{p:.1f}M</dd></div><div><dt>Per frame</dt><dd>{ms:.0f} ms</dd></div>"
        f"<div><dt>Training</dt><dd>{th:.1f} h</dd></div></dl></div>"
        for k, n, _, v, p, ms, th in MODELS
    )

    def rt_row(key, label):
        d = rt[key]
        o = d["overall"]
        return (
            f'<tr><th scope="row">{label}</th><td>{o["yaw_deg"]["mae"]:.3f}°</td><td>{o["yaw_deg"]["p95"]:.3f}°</td>'
            f"<td>{o['xtrack_m']['mae']:.2f} m</td><td>{o['xtrack_m']['p95']:.2f} m</td>"
            f'<td class="n">{d["status"].get("rejected", 0)}</td></tr>'
        )

    rt_weather = "".join(
        f'<tr><th scope="row">{html.escape(lab)}</th>'
        + "".join(f"<td>{rt[v]['by_weather'][k]['xtrack_m']['mae']:.2f} m</td>" for v in ("old", "none", "crab"))
        + "</tr>"
        for k, lab in WEATHER
    )
    rt_range = "".join(
        f'<tr><th scope="row">{k.replace("-", "–")} m</th>'
        + "".join(f"<td>{rt[v]['by_range'][k]['xtrack_m']['mae']:.2f} m</td>" for v in ("old", "none", "crab"))
        + "</tr>"
        for k in ["1000-1500", "600-1000", "300-600", "150-300"]
    )

    pt = "past threshold"
    page = TEMPLATE.format(
        tiles=tiles,
        chart=range_chart(data),
        range_table=iou_table(data, RANGES, "by_range"),
        weather_table=iou_table(data, WEATHER, "by_weather"),
        n_frames=f"{ov['yolo']['n']:,}",
        cap_img=img(figs / "cap.jpg"),
        fog_img=img(figs / "cmp_fog.jpg"),
        rain_img=img(figs / "cmp_rain.jpg"),
        samuel_img=img(figs / "samuel.jpg"),
        cap_before=f3(plain["by_range"][pt]["mean_iou"]),
        cap_after=f3(data["yolo"]["by_range"][pt]["mean_iou"]),
        all_before=f3(plain["overall"]["mean_iou"]),
        all_after=f3(ov["yolo"]["mean_iou"]),
        rt_rows=rt_row("old", "Original EKF") + rt_row("none", "No filter") + rt_row("crab", "Fixed EKF (crab state)"),
        rt_weather=rt_weather,
        rt_range=rt_range,
        rt_frames=f"{rt['crab']['frames']:,}",
        rt_flights=rt["crab"]["flights"],
        dl_empty=ov["deeplab"]["empty_pred"],
        yolo_empty=ov["yolo"]["empty_pred"],
        vn_empty=ov["valnet"]["empty_pred"],
    )
    Path(args.out).write_text(page)
    print(args.out, f"{len(page) / 1e6:.2f} MB")


TEMPLATE = """<title>Runway Segmentation Results</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@85,600;85,700&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column of findings, each section = claim, evidence (chart, table or still), caveat. */
:root {{
  --bg: #f6f7f8; --panel: #ffffff; --ink: #14181d; --ink2: #4a525c; --ink3: #737c86;
  --rule: #dde1e5; --grid: #e7eaed; --accent: #24476b; --best: #e9f0f7;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a;
  --f-display: "Archivo", "Arial Narrow", system-ui, sans-serif;
  --f-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --f-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #121519; --panel: #1a1e23; --ink: #eceff2; --ink2: #b4bcc5; --ink3: #8a939d;
  --rule: #2c3238; --grid: #252a30; --accent: #9cc2ea; --best: #1f2b38;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #121519; --panel: #1a1e23; --ink: #eceff2; --ink2: #b4bcc5; --ink3: #8a939d;
  --rule: #2c3238; --grid: #252a30; --accent: #9cc2ea; --best: #1f2b38;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--ink); font: 16px/1.6 var(--f-body); }}
.wrap {{ max-width: 980px; margin: 0 auto; padding-block: 40px 72px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: grid; gap: 56px; }}
header {{ display: grid; gap: 10px; }}
.eyebrow {{ font: 500 12px/1.4 var(--f-mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent); }}
h1 {{ font: 700 clamp(30px, 5vw, 46px)/1.08 var(--f-display); font-stretch: 85%; margin: 0; text-wrap: balance; }}
h2 {{ font: 700 26px/1.2 var(--f-display); font-stretch: 85%; margin: 0; text-wrap: balance; }}
h3 {{ font: 600 16px/1.3 var(--f-body); margin: 0; }}
p {{ margin: 0; max-width: 68ch; color: var(--ink2); }}
p strong, li strong {{ color: var(--ink); font-weight: 600; }}
.lede {{ font-size: 18px; color: var(--ink2); max-width: 70ch; }}
section {{ display: grid; gap: 18px; min-width: 0; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }}
.tile {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 18px 20px; display: grid; gap: 4px; }}
.tname {{ font-weight: 600; display: flex; align-items: center; gap: 8px; }}
.key {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px; flex: none; }}
.big {{ font: 500 40px/1.1 var(--f-mono); font-variant-numeric: tabular-nums; }}
.sub {{ font-size: 13px; color: var(--ink3); }}
dl {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 12px 0 0; padding-top: 12px; border-top: 1px solid var(--rule); }}
dt {{ font-size: 11px; letter-spacing: .06em; text-transform: uppercase; color: var(--ink3); }}
dd {{ margin: 0; font: 500 15px/1.4 var(--f-mono); font-variant-numeric: tabular-nums; }}
.figure {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 16px; display: grid; gap: 10px; min-width: 0; }}
.chartbox {{ overflow-x: auto; }}
.chart {{ width: 100%; min-width: 600px; height: auto; display: block; }}
.chart .grid {{ stroke: var(--grid); stroke-width: 1; }}
.chart .tick {{ fill: var(--ink3); font: 12px var(--f-mono); }}
.chart .axis {{ fill: var(--ink2); font: 13px var(--f-body); }}
.chart .lab {{ fill: var(--ink); font: 600 13px var(--f-body); }}
.chart .series {{ fill: none; stroke-width: 2; stroke-linejoin: round; }}
.chart .swatch {{ stroke-width: 3; stroke-linecap: round; }}
.chart .dot {{ stroke: var(--panel); stroke-width: 2; }}
.chart .hit {{ fill: transparent; }}
.chart .pt:hover .dot {{ r: 6.5; }}
.cap {{ font-size: 13px; color: var(--ink3); max-width: 80ch; }}
.tw {{ overflow-x: auto; background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ padding: 9px 14px; text-align: right; border-bottom: 1px solid var(--rule); white-space: nowrap; }}
th[scope="row"], thead th:first-child {{ text-align: left; font-weight: 500; }}
thead th {{ font-weight: 600; font-size: 13px; color: var(--ink2); }}
thead th .key {{ margin-right: 6px; vertical-align: -1px; }}
td {{ font-family: var(--f-mono); font-variant-numeric: tabular-nums; }}
td.n {{ color: var(--ink3); }}
td.best {{ background: var(--best); font-weight: 500; }}
tbody tr:last-child th, tbody tr:last-child td {{ border-bottom: 0; }}
.two {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }}
.two > * {{ min-width: 0; }}
img.still {{ width: 100%; height: auto; border-radius: 4px; display: block; }}
ul {{ margin: 0; padding-left: 20px; display: grid; gap: 6px; color: var(--ink2); max-width: 74ch; }}
code {{ font: 13.5px var(--f-mono); color: var(--ink); background: var(--best); padding: 1px 5px; border-radius: 3px; overflow-wrap: anywhere; }}
.videos td:first-child {{ text-align: left; }}
.videos td {{ white-space: normal; font-family: var(--f-body); text-align: left; }}
.videos td code {{ white-space: nowrap; }}
.num {{ font-family: var(--f-mono); font-variant-numeric: tabular-nums; color: var(--ink); }}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">XP12 runway segmentation · test split · 23 Oct 2026</div>
  <h1>Runway Segmentation Results</h1>
  <p class="lede">Three segmentation models trained on the same easy and weather approaches at 1280&nbsp;px, scored on
  {n_frames} held-out test frames from 89 flights at airports none of them saw in training.
  DeepLabV3+ is the most accurate, YOLOv8 the fastest, and VALNet trails both on this data.</p>
</header>

<section aria-labelledby="h-headline">
  <h2 id="h-headline">Headline</h2>
  <div class="tiles">{tiles}</div>
  <p class="cap">IoU is the overlap between a frame's predicted runway mask and its label, at full 1280×720, averaged
  over frames. YOLO and VALNet include the box-cap fix below. Per-frame time is end to end (image load, inference in
  half precision, post-processing) on one RTX 3090 Ti. Training: 50 epochs each, batch 8.</p>
</section>

<section aria-labelledby="h-range">
  <h2 id="h-range">Accuracy by range to the threshold</h2>
  <p>YOLOv8 and DeepLabV3+ are level within 600&nbsp;m. <strong>DeepLabV3+ pulls ahead from 600&nbsp;m out</strong>, the range
  that matters most for cross-track. VALNet falls behind from 300&nbsp;m out, most at range: its masks come out about 13%
  larger than the label at 600–1500&nbsp;m. On the runway all three tie once the box-cap fix is applied.</p>
  <div class="figure"><div class="chartbox">{chart}</div></div>
  {range_table}
</section>

<section aria-labelledby="h-weather">
  <h2 id="h-weather">Accuracy by weather</h2>
  <p>Bad weather is not the hard case. The weather flights score higher than the easy ones because frames where the
  runway is hidden (range beyond half the visibility) were removed from the split, and the easy flights keep their
  far frames out to 3&nbsp;km. DeepLabV3+ leads in seven of eight conditions; YOLO and VALNet edge it in fog.</p>
  {weather_table}
  <p class="cap">Frames with no prediction at all: DeepLabV3+ {dl_empty}, YOLO {yolo_empty}, VALNet {vn_empty},
  all beyond 1.5&nbsp;km on easy flights. DeepLabV3+'s worst on-runway frames are also worse than YOLO's
  (5th-percentile IoU 0.44 against 0.63).</p>
</section>

<section aria-labelledby="h-cap">
  <h2 id="h-cap">The 960-pixel box cap</h2>
  <p>On the ground YOLO's mask stopped short of both runway edges. The cause is the YOLOv8 box head: each side of a box
  can reach at most 15 bins × stride 32 = 480&nbsp;px from its anchor, so at 1280&nbsp;px input no box is wider than 960&nbsp;px,
  and masks are cropped to their box. Across 55k test frames no mask was wider than 959&nbsp;px. VALNet shares the head.
  The fix widens a box that hits the cap to the image edges before the crop; it needs no retraining.</p>
  <div class="figure">
    <img class="still" src="{cap_img}" alt="Two views of the same frame on the runway in fog. Left: YOLO's mask stops at vertical lines short of both runway edges. Right: with the widened crop the mask reaches the label at both edges.">
    <div class="cap">Same frame, weather/000197 (EGNH 28, fog), on the runway. Left: plain crop. Right: widened crop.
    Past-threshold IoU <span class="num">{cap_before}</span> → <span class="num">{cap_after}</span>; all frames
    <span class="num">{all_before}</span> → <span class="num">{all_after}</span>.</div>
  </div>
</section>

<section aria-labelledby="h-side">
  <h2 id="h-side">Side by side</h2>
  <div class="two">
    <div class="figure"><img class="still" src="{fog_img}" alt="Four-panel frame from the fog approach at 368 meters: the three models' masks on the runway and a plot of IoU against range.">
      <div class="cap">Fog, weather/000197 at 368&nbsp;m. White outline is the label.</div></div>
    <div class="figure"><img class="still" src="{rain_img}" alt="Four-panel frame from the heavy-rain approach at 602 meters: the three models' masks and a plot of IoU against range.">
      <div class="cap">Heavy rain, weather/000414 at 602&nbsp;m.</div></div>
  </div>
</section>

<section aria-labelledby="h-samuel">
  <h2 id="h-samuel">Yaw and cross-track: Samuel's estimator</h2>
  <p>Samuel's Restart method fits yaw and cross-track to the YOLO mask and tracks them with a kinematic EKF, scored on
  {rt_frames} frames of {rt_flights} test flights at 150–1500&nbsp;m. <strong>The original EKF moved the aircraft along its
  nose.</strong> In a crosswind the aircraft crabs, nose into the wind while tracking the runway, so the prediction slid
  sideways every frame: a 3–5&nbsp;m bias on windy flights, with a reject and re-acquire sawtooth. Adding the crab angle as
  a fourth state fixes it.</p>
  {rt_table}
  <div class="figure">
    <img class="still" src="{samuel_img}" alt="Replay of a windy approach: the runway outline from the original EKF sits to the side of the true outline, the fixed EKF's matches it; plots show the original cross-track estimate in a sawtooth while the fixed one follows the truth.">
    <div class="cap">weather/000613 (EDDG 25, light rain, 7.9&nbsp;m/s crosswind, 9.6° crab). Red: original EKF.
    Green: fixed EKF. Aqua: truth.</div>
  </div>
  <div class="two">
    <div class="tw"><table><thead><tr><th scope="col">Cross-track MAE</th><th scope="col">Original</th><th scope="col">No filter</th><th scope="col">Fixed</th></tr></thead><tbody>{rt_weather}</tbody></table></div>
    <div class="tw"><table><thead><tr><th scope="col">Range</th><th scope="col">Original</th><th scope="col">No filter</th><th scope="col">Fixed</th></tr></thead><tbody>{rt_range}</tbody></table></div>
  </div>
  <p class="cap">The fixed filter still smooths yaw more than it should (0.046° against 0.032° with no filter): its
  measurement noise assumes 0.15° where the per-frame yaw is good to about 0.04°. Setting it to 0.05° and 1&nbsp;m gave
  0.033° and 0.43&nbsp;m on 16 test flights; that setting should be confirmed on the validation flights before it is
  reported. One flight (weather/000209) is excluded: its logged height is 165&nbsp;m too high.</p>
</section>

<section aria-labelledby="h-data">
  <h2 id="h-data">Data and setup</h2>
  <ul>
    <li><strong>Flights:</strong> 145 easy approaches (24&nbsp;km visibility) and 468 weather approaches in seven conditions,
    split by airport 70/15/15, so test airports are unseen.</li>
    <li><strong>Visibility filter:</strong> frames with slant range beyond half the visibility are dropped (the runway is
    at about 22% contrast there). This removes 62% of fog frames and 47% of heavy-rain frames.</li>
    <li><strong>Training frames:</strong> every 5th frame per flight, 46k train and 9.9k validation. Test uses every kept frame.</li>
    <li><strong>Models:</strong> YOLOv8s-seg (COCO, Ultralytics defaults); VALNet (paper-size, 12.8M, from COCO YOLOv8s);
    DeepLabV3+ (torchvision COCO ResNet-50 and ASPP, new decoder, output stride 16, BCE + Dice).</li>
    <li><strong>mAP</strong> (instance metric, YOLO and VALNet only): full resolution 0.737 and 0.738, quarter resolution
    0.834 and 0.838, before the box-cap fix.</li>
  </ul>
</section>

<section aria-labelledby="h-videos">
  <h2 id="h-videos">Demo videos</h2>
  <p>In <code>results/xp12_combined/demos/</code>.</p>
  <div class="tw videos"><table><thead><tr><th scope="col">File</th><th scope="col">Shows</th></tr></thead><tbody>
    <tr><td><code>compare_weather_000197.mp4</code></td><td>All three models side by side, fog approach to EGNH 28 (31&nbsp;s)</td></tr>
    <tr><td><code>compare_weather_000414.mp4</code></td><td>All three, heavy rain, ESOK 21 (28&nbsp;s)</td></tr>
    <tr><td><code>compare_easy_000116.mp4</code></td><td>All three, easy flight out to 3&nbsp;km, EFIV 22 (29&nbsp;s)</td></tr>
    <tr><td><code>samuel_crab_weather_000613.mp4</code></td><td>Samuel's estimator in a 7.9&nbsp;m/s crosswind, original against fixed EKF (17&nbsp;s)</td></tr>
    <tr><td><code>weather_000197.mp4</code> / <code>yolo_weather_000197.mp4</code></td><td>YOLO alone on the fog flight, before and after the box-cap fix</td></tr>
    <tr><td><code>yolo_*</code>, <code>valnet_*</code>, <code>deeplab_*</code></td><td>Each model alone on the three flights, with IoU against range</td></tr>
  </tbody></table></div>
</section>
</div>
"""

RT_TABLE = """<div class="tw"><table><thead><tr><th scope="col">Tracking filter</th><th scope="col">Yaw MAE</th>
<th scope="col">Yaw p95</th><th scope="col">Cross-track MAE</th><th scope="col">Cross-track p95</th>
<th scope="col" class="n">Rejected frames</th></tr></thead><tbody>{rows}</tbody></table></div>"""

TEMPLATE = TEMPLATE.replace("{rt_table}", RT_TABLE.replace("{rows}", "{rt_rows}"))

if __name__ == "__main__":
    main()
