"""Web demo (Gradio): pick a flight, run it, scrub through the frames.

    python -m e2e_runway_dof_estimation.app            # http://127.0.0.1:7860

Same pipeline and numbers as the command line (e2e_runway_dof_estimation.demo).
"""

import argparse
import tempfile
from pathlib import Path

import cv2
import gradio as gr

from e2e_runway_dof_estimation import assets, pipeline

_models = {}
_results = {}            # flight folder -> (Result, Renderer)


def _get_models():
    if not _models:
        from e2e_runway_dof_estimation.estimate import Estimator
        from e2e_runway_dof_estimation.segment import Segmenter
        _models.update(segmenter=Segmenter(), estimator=Estimator())
    return _models


def _summary_md(s):
    rows = [f"| all | {s['frames']} | {s['yaw_mae_deg']:.3f}° | {s['xtrack_mae_m']:.2f} m | {s['mask_iou']:.3f} |"]
    rows += [f"| {k} | {v['frames']} | {v['yaw_mae_deg']:.3f}° | {v['xtrack_mae_m']:.2f} m | {v['mask_iou']:.3f} |"
             for k, v in s["by_range"].items()]
    return (f"**{s['flight']}** · {s['airport']} · {s['weather']}\n\n"
            "| Range | Frames | Yaw MAE | Cross-track MAE | Mask IoU |\n|---|---|---|---|---|\n" + "\n".join(rows))


def run_flight(choice, folder, progress=gr.Progress()):
    from e2e_runway_dof_estimation.viz import Renderer
    flight = folder.strip() or choice
    if not flight:
        raise gr.Error("Pick a sample flight or enter a flight folder.")
    key = str(pipeline.resolve(flight))
    if key not in _results:
        m = _get_models()
        progress(0, desc="segmenting")
        result = pipeline.run(flight, segmenter=m["segmenter"], estimator=m["estimator"],
                              progress=lambda d, n: progress(d / n, desc=f"segmenting {d}/{n}"))
        _results[key] = (result, Renderer(result, m["estimator"]))
    result, _ = _results[key]
    n = len(result.records)
    return (key, _summary_md(result.summary()), gr.Slider(minimum=0, maximum=n - 1, value=n - 1, step=1,
                                                         label=f"Frame (0 = {result.records[0]['range_m']:,.0f} m out)"),
            show_frame(key, n - 1))


def show_frame(key, pos):
    if not key:
        return None
    result, rend = _results[key]
    return cv2.cvtColor(rend.frame(result.records[int(pos)]["frame"]), cv2.COLOR_BGR2RGB)


def make_video(key):
    from e2e_runway_dof_estimation.viz import write_video
    if not key:
        raise gr.Error("Run a flight first.")
    result, _ = _results[key]
    path = Path(tempfile.gettempdir()) / f"replay_{result.flight.name.replace('/', '_')}.mp4"
    write_video(result, _get_models()["estimator"], path)
    return str(path)


def build():
    with gr.Blocks(title="Runway DOF estimation") as ui:
        gr.Markdown("## Runway yaw and cross-track from a camera\n"
                    "DeepLabV3+ segments the runway in each frame; Samuel Moyal's mask-fit estimator, "
                    "tracked by a crab-angle EKF, turns the masks into yaw and cross-track. "
                    "Frames 150–1500 m from the threshold, within half the visibility.")
        key = gr.State("")
        with gr.Row():
            choice = gr.Dropdown(list(assets.SAMPLES), value=assets.SAMPLES[1], label="Sample flight",
                                 info="weather_000197: fog · weather_000613: 7.9 m/s crosswind")
            folder = gr.Textbox(label="…or a flight folder", placeholder="datasets/xp12_dataset_weather/000414")
            run = gr.Button("Run", variant="primary")
        summary = gr.Markdown()
        frame = gr.Image(label="Replay", type="numpy", interactive=False)
        slider = gr.Slider(minimum=0, maximum=1, value=0, step=1, label="Frame")
        with gr.Row():
            vbtn = gr.Button("Render replay video")
            video = gr.Video(label="Replay video")
        run.click(run_flight, [choice, folder], [key, summary, slider, frame])
        slider.change(show_frame, [key, slider], frame)
        vbtn.click(make_video, key, video)
    return ui


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build().launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
