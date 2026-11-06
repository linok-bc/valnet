"""Web demo (Gradio): preloaded runs show instantly; new flights run the full pipeline.

    python -m e2e_runway_dof_estimation.app            # http://127.0.0.1:7860
    python -m e2e_runway_dof_estimation.app --share    # + a public gradio.live link (~1 week)

A preloaded run (assets.SAMPLES) is a saved result plus its replay video: picking one
downloads it and computes nothing, so it needs no GPU. "Run a flight folder" runs
segmentation and estimation, with the same numbers as the command line.
"""

import argparse
import tempfile
from pathlib import Path

import cv2
import gradio as gr

from e2e_runway_dof_estimation import assets, pipeline

_models = {}
_results = {}            # key -> (Result, Renderer)


def _estimator():
    if "estimator" not in _models:
        from e2e_runway_dof_estimation.estimate import Estimator
        _models["estimator"] = Estimator()
    return _models["estimator"]


def _segmenter():
    if "segmenter" not in _models:
        from e2e_runway_dof_estimation.segment import Segmenter
        _models["segmenter"] = Segmenter()
    return _models["segmenter"]


def _summary_md(s):
    rows = [f"| all | {s['frames']} | {s['yaw_mae_deg']:.3f}° | {s['xtrack_mae_m']:.2f} m | {s['mask_iou']:.3f} |"]
    rows += [f"| {k} | {v['frames']} | {v['yaw_mae_deg']:.3f}° | {v['xtrack_mae_m']:.2f} m | {v['mask_iou']:.3f} |"
             for k, v in s["by_range"].items()]
    return (f"**{s['flight']}** · {s['airport']} · {s['weather']}\n\n"
            "| Range | Frames | Yaw MAE | Cross-track MAE | Mask IoU |\n|---|---|---|---|---|\n" + "\n".join(rows))


def _show(key, video=None):
    """Outputs for a loaded result: state key, summary, slider, last frame, video."""
    result, _ = _results[key]
    n = len(result.records)
    slider = gr.Slider(minimum=0, maximum=n - 1, value=n - 1, step=1,
                       label=f"Frame (0 = {result.records[0]['range_m']:,.0f} m out)")
    return key, _summary_md(result.summary()), slider, show_frame(key, n - 1), video


def _label(name):
    return f"{name} · {assets.SAMPLES[name]}"


def load_run(label, progress=gr.Progress()):
    from e2e_runway_dof_estimation.viz import Renderer
    name = label.split(" · ")[0]
    key = f"run:{name}"
    if key not in _results:
        progress(0, desc="downloading")
        result = pipeline.load_preloaded(name)
        _results[key] = (result, Renderer(result, _estimator()))
    return _show(key, str(assets.preloaded_video(name)))


def run_flight(folder, progress=gr.Progress()):
    from e2e_runway_dof_estimation.viz import Renderer
    if not folder.strip():
        raise gr.Error("Enter a flight folder, e.g. datasets/xp12_dataset_weather/000414.")
    key = str(Path(folder.strip()).resolve())
    if key not in _results:
        progress(0, desc="segmenting")
        result = pipeline.run(folder.strip(), segmenter=_segmenter(), estimator=_estimator(),
                              progress=lambda d, n: progress(d / n, desc=f"segmenting {d}/{n}"))
        _results[key] = (result, Renderer(result, _estimator()))
    return _show(key)


def show_frame(key, pos):
    if not key:
        return None
    result, rend = _results[key]
    return cv2.cvtColor(rend.frame(result.records[int(pos)]["frame"]), cv2.COLOR_BGR2RGB)


def make_video(key):
    from e2e_runway_dof_estimation.viz import write_video
    if not key:
        raise gr.Error("Load a run first.")
    if key.startswith("run:"):
        return str(assets.preloaded_video(key[4:]))
    result, _ = _results[key]
    path = Path(tempfile.gettempdir()) / f"replay_{result.flight.name.replace('/', '_')}.mp4"
    write_video(result, _estimator(), path)
    return str(path)


def build(folder_runs=True):
    labels = [_label(n) for n in assets.SAMPLES]
    with gr.Blocks(title="Runway DOF estimation") as ui:
        gr.Markdown("## Runway yaw and cross-track from a camera\n"
                    "DeepLabV3+ segments the runway in each frame; Samuel Moyal's mask-fit estimator, "
                    "tracked by a crab-angle EKF, turns the masks into yaw and cross-track. "
                    "Frames 150–1500 m from the threshold, within half the visibility.")
        key = gr.State("")
        with gr.Row():
            run = gr.Dropdown(labels, value=labels[0], label="Preloaded run",
                              info="Saved results: shown instantly, no GPU needed")
            with gr.Column(visible=folder_runs):
                folder = gr.Textbox(label="…or run a flight folder (GPU)", lines=1, max_lines=1,
                                    placeholder="datasets/xp12_dataset_weather/000414")
                go = gr.Button("Run", variant="primary")
        summary = gr.Markdown()
        frame = gr.Image(label="Replay", type="numpy", interactive=False)
        slider = gr.Slider(minimum=0, maximum=1, value=0, step=1, label="Frame")
        with gr.Row():
            vbtn = gr.Button("Replay video")
            video = gr.Video(label="Replay video")
        outs = [key, summary, slider, frame, video]
        run.change(load_run, run, outs)
        ui.load(load_run, run, outs)
        if folder_runs:
            go.click(run_flight, folder, outs)
        slider.change(show_frame, [key, slider], frame)
        vbtn.click(make_video, key, video)
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
                              allowed_paths=[HF_HUB_CACHE])


if __name__ == "__main__":
    main()
