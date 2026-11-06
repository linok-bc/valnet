"""Smoke tests: the full pipeline on two of the sample flights (needs a GPU), and every preloaded
run loads and reproduces its saved summary (no GPU). Both need `hf auth login`.

    python -m pytest tests            # or: python tests/test_smoke.py

Reference numbers were produced on an RTX 3090 Ti with the published weights; the
tolerances leave room for GPU and library differences, not for a broken pipeline.
"""

import json
import math

from e2e_runway_dof_estimation import assets, pipeline

REFERENCE = {   # flight: (frames, yaw MAE deg, cross-track MAE m, mask IoU)
    "weather_000185": (237, 0.0294, 0.188, 0.946),     # 7.3 m/s crosswind, ESIB 19
    "weather_000270": (232, 0.0119, 0.202, 0.917),     # haze, EETU 26
}


def check(flight, reference):
    frames, yaw, xt, iou = reference
    s = pipeline.run(flight).summary()
    assert s["frames"] == frames, (flight, s["frames"])
    assert math.isclose(s["yaw_mae_deg"], yaw, abs_tol=0.005), (flight, s["yaw_mae_deg"])
    assert math.isclose(s["xtrack_mae_m"], xt, abs_tol=0.05), (flight, s["xtrack_mae_m"])
    assert math.isclose(s["mask_iou"], iou, abs_tol=0.01), (flight, s["mask_iou"])
    return s


def test_samples():
    for flight, ref in REFERENCE.items():
        check(flight, ref)


def check_preloaded(name):
    saved = json.loads((assets.preloaded_run(name) / "records.json").read_text())["meta"]["summary"]
    s = pipeline.load_preloaded(name).summary()
    assert json.dumps(s, sort_keys=True) == json.dumps(saved, sort_keys=True), name
    return s


def test_preloaded():
    for name in assets.SAMPLES:
        check_preloaded(name)


if __name__ == "__main__":
    for name in assets.SAMPLES:
        s = check_preloaded(name)
        print(f"ok  preloaded {name}: {s['frames']} frames, cross-track {s['xtrack_mae_m']:.2f} m")
    for flight, ref in REFERENCE.items():
        s = check(flight, ref)
        print(f"ok  {flight}: {s['frames']} frames, yaw {s['yaw_mae_deg']:.3f} deg, "
              f"cross-track {s['xtrack_mae_m']:.2f} m, IoU {s['mask_iou']:.3f}")
