"""Web demo (Gradio): preloaded runs show instantly; new flights run the full pipeline.

    python -m e2e_runway_dof_estimation.app            # http://127.0.0.1:7860
    python -m e2e_runway_dof_estimation.app --share    # + a public gradio.live link (~1 week)

A preloaded run (assets.SAMPLES) is a saved summary plus its replay video: picking one
downloads two small files and computes nothing, so it needs no GPU. "Run a flight folder"
runs segmentation and estimation, then renders the replay, with the same numbers as the
command line.
"""

import argparse
import json
import tempfile
from pathlib import Path

import gradio as gr

from e2e_runway_dof_estimation import assets, pipeline

EXPLAINER = """
**The problem.** On final approach the aircraft needs two numbers about the runway ahead: its
**yaw**, the angle between the nose and the runway centreline, and its **cross-track**, the
sideways distance from the centreline (+ right). This demo estimates both from the forward
camera alone. DeepLabV3+ segments the runway in every frame, Samuel Moyal's estimator fits the
runway's outline to that mask, and a filter that allows for a crosswind tracks the result down
the approach. Distance to the threshold, height, pitch and roll are taken as known, as an INS
would give them.

**The table**, for the whole approach and by distance to the threshold:
- **Yaw MAE**: average yaw error, in degrees. *Lower is better.*
- **Cross-track MAE**: average cross-track error, in metres. *Lower is better.* For scale,
  these runways are 30–60 m wide.
- **Mask IoU**: overlap between the predicted and the true runway mask, from 0 (none) to 1
  (perfect). *Higher is better.*
- **Frames**: camera frames scored, 150–1500 m before the threshold, within half the visibility.

In the video, teal is the predicted runway mask, white the true runway, and amber the runway
drawn from the estimated yaw and cross-track. The plots on the right compare estimate (amber)
with truth (white).
"""

# Keep the results table on one line per row so it fits beside the video.
CSS = """
.summary table { font-size: 0.92em; }
.summary th, .summary td { white-space: nowrap; padding: 6px 9px !important; }
"""

_models = {}
_videos = {}             # folder -> (summary, rendered replay)


def _summary_md(s):
    rows = [f"| all | {s['frames']} | {s['yaw_mae_deg']:.3f}° | {s['xtrack_mae_m']:.2f} m | {s['mask_iou']:.3f} |"]
    rows += [f"| {k} | {v['frames']} | {v['yaw_mae_deg']:.3f}° | {v['xtrack_mae_m']:.2f} m | {v['mask_iou']:.3f} |"
             for k, v in s["by_range"].items()]
    return (f"**{s['flight']}** · {s['airport']} · {s['weather']}\n\n"
            "| Range | Frames | Yaw MAE ↓ | Cross-track MAE ↓ | Mask IoU ↑ |\n|---|---|---|---|---|\n"
            + "\n".join(rows))


def _label(name):
    return f"{name} · {assets.SAMPLES[name]}"


def load_run(label):
    """A preloaded run: its saved summary and replay video, nothing computed."""
    name = label.split(" · ")[0]
    summary = json.loads((assets.preloaded_run(name) / "records.json").read_text())["meta"]["summary"]
    return _summary_md(summary), str(assets.preloaded_video(name))


def run_flight(folder, progress=gr.Progress()):
    from e2e_runway_dof_estimation.estimate import Estimator
    from e2e_runway_dof_estimation.segment import Segmenter
    from e2e_runway_dof_estimation.viz import write_video
    folder = folder.strip()
    if not folder:
        raise gr.Error("Enter a flight folder, e.g. datasets/xp12_dataset_weather/000414.")
    key = str(Path(folder).resolve())
    if key not in _videos:
        if not _models:
            _models.update(segmenter=Segmenter(), estimator=Estimator())
        progress(0, desc="segmenting")
        result = pipeline.run(folder, segmenter=_models["segmenter"], estimator=_models["estimator"],
                              progress=lambda d, n: progress(d / n, desc=f"segmenting {d}/{n}"))
        progress(1, desc="rendering the replay")
        path = Path(tempfile.gettempdir()) / f"replay_{result.flight.name.replace('/', '_')}.mp4"
        write_video(result, _models["estimator"], path)
        _videos[key] = (_summary_md(result.summary()), str(path))
    return _videos[key]


def build(folder_runs=True):
    labels = [_label(n) for n in assets.SAMPLES]
    with gr.Blocks(title="Runway DOF estimation") as ui:
        gr.Markdown("## Runway yaw and cross-track from a camera")
        gr.Markdown(EXPLAINER)
        with gr.Row():
            run = gr.Dropdown(labels, value=labels[0], label="Preloaded run",
                              info="Saved results: shown instantly, no GPU needed")
            with gr.Column(visible=folder_runs):
                folder = gr.Textbox(label="…or run a flight folder (GPU)", lines=1, max_lines=1,
                                    placeholder="datasets/xp12_dataset_weather/000414")
                go = gr.Button("Run", variant="primary")
        with gr.Row(equal_height=False):
            with gr.Column(scale=2, min_width=440):
                summary = gr.Markdown(elem_classes="summary")
            with gr.Column(scale=3, min_width=480):
                video = gr.Video(label="Replay", autoplay=True, interactive=False)
        run.change(load_run, run, [summary, video])
        ui.load(load_run, run, [summary, video])
        if folder_runs:
            go.click(run_flight, folder, [summary, video])
    return ui


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true",
                    help="also serve a public gradio.live link that anyone with the URL can open")
    ap.add_argument("--auth", metavar="USER:PASSWORD", help="require this login (recommended with --share)")
    ap.add_argument("--allow-folder-runs", action="store_true",
                    help="with --share, keep 'run a flight folder': it lets visitors run the pipeline "
                         "on any path on this machine (off by default when shared)")
    args = ap.parse_args()
    auth = tuple(args.auth.split(":", 1)) if args.auth else None
    if args.auth and len(auth) != 2:
        ap.error("--auth takes USER:PASSWORD")
    folder_runs = not args.share or args.allow_folder_runs
    from huggingface_hub.constants import HF_HUB_CACHE
    # Preloaded replay videos are served straight from the Hugging Face cache.
    build(folder_runs).launch(server_name=args.host, server_port=args.port, share=args.share, auth=auth,
                              allowed_paths=[HF_HUB_CACHE], css=CSS)


if __name__ == "__main__":
    main()
