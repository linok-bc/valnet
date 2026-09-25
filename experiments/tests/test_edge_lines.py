"""Synthetic test of the edge-line (lane-keeping) cross-track estimator.

Same contract as test_ryu.py: put a runway at a KNOWN cross-track, project it
through the camera, and demand the number comes back. If this passes, an error
in edge_lines/eval.py belongs to the mask or the labels, not the geometry.

    python -m experiments.tests.test_edge_lines
"""

import math

import numpy as np

from experiments.bev.camera import CameraModel
from experiments.bev.episodes import runway_to_ground
from experiments.bev.transform import ground_to_image
from experiments.edge_lines.geometry import (crosstrack, estimate_from_mask,
                                             line_through, rasterize)

CAM = CameraModel.from_fov(1280, 720, 65.0, 39.43, tilt_deg=0.0)
W, L = 45.0, 3000.0


def project(range_m, right_m, yaw_deg, pitch_deg, roll_deg, height_m):
    """Runway quad (NL, FL, FR, NR) in image pixels."""
    rwy = np.array([[0, -W / 2], [L, -W / 2], [L, W / 2], [0, W / 2]])
    g = runway_to_ground(rwy, -range_m, right_m, yaw_deg)
    H = ground_to_image(CAM, roll_deg, pitch_deg, height_m)
    p = np.column_stack([g, np.ones(4)]) @ H.T
    return p[:, :2] / p[:, 2:3]


CASES = [  # range, right, yaw, pitch, roll
    (1400, 0.0, 0.0, 3.0, 0.0),
    (1400, 8.0, -1.5, 3.5, 1.0),
    (800, -12.0, 2.0, 2.0, -3.0),
    (300, 4.0, 0.5, 4.0, 2.0),
    (200, -3.0, -3.0, 3.0, -1.5),
]


def test_exact_lines_recover_crosstrack():
    """Lines through the true corners: every variant exact."""
    for rng, right, yaw, pitch, roll in CASES:
        h = rng * math.tan(math.radians(3.0))
        q = project(rng, right, yaw, pitch, roll, h)
        out = crosstrack(line_through(q[0], q[1]), line_through(q[3], q[2]),
                         CAM, roll, pitch, h, W, yaw)
        for k in ("free_height", "free_width", "known_height", "known_width"):
            assert abs(out[k] - right) < 1e-6, (k, rng, right, out[k])
        assert abs(out["yaw_fit_deg"] - yaw) < 1e-6


def test_heading_is_not_needed():
    """`free` never sees yaw, so a wrong yaw can only move `known`."""
    rng, right, yaw, pitch, roll = 1400, 8.0, -1.5, 3.5, 1.0
    h = rng * math.tan(math.radians(3.0))
    q = project(rng, right, yaw, pitch, roll, h)
    lL, lR = line_through(q[0], q[1]), line_through(q[3], q[2])
    good = crosstrack(lL, lR, CAM, roll, pitch, h, W, yaw)
    bad = crosstrack(lL, lR, CAM, roll, pitch, h, W, yaw + 1.0)
    assert good["free_width"] == bad["free_width"]
    # and a 1 deg heading error costs the `known` variant well under a metre,
    # against range * sin(1 deg) = 24 m for a position-based estimator
    assert abs(bad["known_width"] - right) < 1.0, bad["known_width"]


def test_width_variant_does_not_need_height():
    rng, right, yaw, pitch, roll = 800, -12.0, 2.0, 2.0, -3.0
    h = rng * math.tan(math.radians(3.0))
    q = project(rng, right, yaw, pitch, roll, h)
    out = crosstrack(line_through(q[0], q[1]), line_through(q[3], q[2]),
                     CAM, roll, pitch, h * 1.2, W, yaw)          # 20% height error
    assert abs(out["free_width"] - right) < 1e-6
    assert abs(out["free_height"] - right * 1.2) < 1e-6


def test_mask_pipeline_recovers_crosstrack():
    """Rasterise, row-scan, fit: within pixel quantisation of the truth."""
    for rng, right, yaw, pitch, roll in CASES:
        h = rng * math.tan(math.radians(3.0))
        q = project(rng, right, yaw, pitch, roll, h)
        mask = rasterize(q, CAM.width, CAM.height)
        out = estimate_from_mask(mask, CAM, roll, pitch, h, W, yaw)
        assert out is not None, (rng, right)
        err = out["free_width"] - right
        # a pixel of edge error is ~h / f of lateral offset per unit slope; allow
        # a generous metre, the point is that nothing is systematically wrong
        assert abs(err) < 1.0, (rng, right, out["free_width"])
        print(f"  range {rng:5.0f}  truth {right:+6.2f}  "
              f"free_width {out['free_width']:+7.3f}  free_height {out['free_height']:+7.3f}"
              f"  rows {out['rows_left']}/{out['rows_right']}")


def test_side_clipped_mask():
    """Close in with a big crab angle the runway's near end runs off the side.

    Rare on this dataset -- on a 3 deg path the runway sits within ~3 deg of the
    horizon, so it never reaches the bottom of the frame -- but a crosswind crab
    does it, and the row scan must drop the border rows rather than fit them.
    """
    rng, right, yaw, pitch, roll = 150, -10.0, -28.0, 3.0, -2.0
    h = rng * math.tan(math.radians(3.0))
    q = project(rng, right, yaw, pitch, roll, h)
    mask = rasterize(q, CAM.width, CAM.height)
    assert mask[:, 0].any() or mask[:, -1].any() or mask[-1].any(), "not clipped"
    out = estimate_from_mask(mask, CAM, roll, pitch, h, W, yaw)
    assert out is not None
    assert abs(out["free_width"] - right) < 1.0, out["free_width"]


def test_gated_kalman_rejects_outliers_and_reacquires():
    """The gate drops isolated wild frames but does not lock out a real change."""
    from experiments.bev.ryu import kalman_track
    rng = np.random.default_rng(0)
    s = np.arange(200) * 7.0                       # ~7 m flown per frame
    truth = np.full(200, 3.0)
    z = truth + rng.normal(0, 0.4, 200)
    z[[30, 31, 70, 150]] += np.array([35.0, -28.0, 40.0, -30.0])    # wild frames
    plain = kalman_track(z, s, r=4.0)
    gated = kalman_track(z, s, r=4.0, gate=9.0)
    bad = [30, 31, 70, 150]
    assert np.abs(gated[bad] - 3.0).max() < 1.0, gated[bad]
    assert np.abs(plain[bad] - 3.0).max() > 3.0      # the ungated filter is dragged

    # a genuine step it has never seen: after max_misses rejections it re-acquires
    z2 = np.r_[np.full(60, 0.0), np.full(60, 20.0)] + rng.normal(0, 0.3, 120)
    g2 = kalman_track(z2, np.arange(120) * 7.0, r=4.0, gate=9.0)
    assert abs(g2[-1] - 20.0) < 1.0, g2[-1]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
