"""End-to-end smoke test on the two sample flights (needs a GPU and `hf auth login`).

    python -m pytest tests            # or: python tests/test_smoke.py

Reference numbers were produced on an RTX 3090 Ti with the published weights; the
tolerances leave room for GPU and library differences, not for a broken pipeline.
"""

import math

from e2e_runway_dof_estimation import pipeline

REFERENCE = {   # flight: (frames, yaw MAE deg, cross-track MAE m, mask IoU)
    "weather_000197": (86, 0.0176, 0.237, 0.974),     # fog, EGNH 28
    "weather_000613": (260, 0.033, 0.29, 0.919),      # 7.9 m/s crosswind, EDDG 25
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


if __name__ == "__main__":
    for flight, ref in REFERENCE.items():
        s = check(flight, ref)
        print(f"ok  {flight}: {s['frames']} frames, yaw {s['yaw_mae_deg']:.3f} deg, "
              f"cross-track {s['xtrack_mae_m']:.2f} m, IoU {s['mask_iou']:.3f}")
