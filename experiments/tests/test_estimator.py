"""End-to-end synthetic test of the estimator the whole experiment is scored with.

Build a runway at a KNOWN cross-track and yaw, project it through the camera,
warp to BEV, fit the oriented box, and demand the two unknowns come back exactly.
If this passes, a non-zero error in eval.py is the detector's, not the geometry's.

    python -m experiments.tests.test_estimator
"""

import math

import numpy as np

from experiments.bev.camera import CameraModel
from experiments.bev.episodes import Runway, ground_to_runway, runway_to_ground
from experiments.bev.grids import GridPolicy
from experiments.bev.labels import clip_to_canvas, warp_points
from experiments.bev.transform import bev_from_image, ground_to_image
from experiments.bev_yolo.eval import estimate, ground_from_bev

CAM = CameraModel.from_fov(1280, 720, 65.0, 39.43, tilt_deg=0.0)
POLICY = GridPolicy(mode="threshold", width_px=320, height_px=1024,
                    mpp_m=0.75, back_m=150.0)


def _runway(width=45.0, length=2500.0):
    """A synthetic runway; only width and length matter to the geometry."""
    rw = Runway("TEST", "09", 0.0, width, 0.0, 0.0, 0.0, 0.0, 90.0)
    object.__setattr__(rw, "_length", length)
    return rw, np.array([[0.0, -width / 2], [length, -width / 2],
                         [length, width / 2], [0.0, width / 2]])


def test_frame_round_trip():
    """runway frame -> ground frame -> runway frame is the identity."""
    pts = np.array([[0.0, -22.5], [2500.0, 22.5], [700.0, 0.0]])
    for along, right, yaw in [(-1200.0, 8.0, 3.0), (-300.0, -25.0, -7.5),
                              (-2000.0, 40.0, 11.0)]:
        g = runway_to_ground(pts, along, right, yaw)
        back = ground_to_runway(g, along, right, yaw)
        assert np.abs(back - pts).max() < 1e-9, np.abs(back - pts).max()
    # the sign that matters: aircraft right of the centreline sees the runway left
    g = runway_to_ground(np.array([[0.0, 0.0]]), -1000.0, 10.0, 0.0)[0]
    assert abs(g[0] - 1000.0) < 1e-9 and abs(g[1] + 10.0) < 1e-9, g
    print(f"  aircraft 10 m right at 1 km -> threshold at ground {tuple(round(v, 3) for v in g)}")


def test_recovers_crosstrack_and_yaw_through_bev():
    """The two unknowns survive the camera, the warp and the box fit."""
    _, corners = _runway()
    # Cross-track and yaw are held inside what the canvas can hold: the runway
    # must not run off the SIDE, which needs range*sin(yaw) + xtrack + width/2
    # under the canvas half-width. The dataset builder drops frames that violate
    # it; see test_canvas_yaw_limit.
    cases = [
        # range, cross-track, yaw, height, roll, pitch
        (1400.0, 12.0, 1.5, 75.0, 0.0, 3.0),
        (900.0, -18.0, -1.5, 48.0, 6.0, 2.5),
        (500.0, 4.0, 0.5, 27.0, -9.0, 3.5),
        (200.0, -1.0, -0.2, 11.0, 2.0, 1.0),
        (1200.0, 0.0, 0.0, 64.0, 0.0, 3.0),
    ]
    worst = 0.0
    for rng, right, yaw, h, roll, pitch in cases:
        along = -rng
        ground = runway_to_ground(corners, along, right, yaw)
        H = ground_to_image(CAM, roll, pitch, h)
        p = np.column_stack([ground, np.ones(4)]) @ H.T
        uv = p[:, :2] / p[:, 2:3]

        grid = POLICY.for_frame(CAM, roll, pitch, h, range_m=rng)
        m_bev = bev_from_image(CAM, grid, roll, pitch, h)
        pts, valid = warp_points(uv, m_bev)
        assert valid.all(), "a corner landed at or above the horizon"
        poly = clip_to_canvas(pts, grid.width, grid.height)
        est = estimate(ground_from_bev(poly, grid))

        d_right = est["right_m"] - right
        d_yaw = est["yaw_deg"] - yaw
        worst = max(worst, abs(d_right), abs(d_yaw))
        assert abs(d_right) < 0.05, f"cross-track off by {d_right} m"
        assert abs(d_yaw) < 0.05, f"yaw off by {d_yaw} deg"
        print(f"  range {rng:6.0f} m  xtrack {right:+6.1f} -> {est['right_m']:+7.3f} "
              f"({d_right:+.3f})   yaw {yaw:+5.2f} -> {est['yaw_deg']:+6.3f} "
              f"({d_yaw:+.3f})   width {est['width_m']:5.2f} m")
    print(f"  worst error across {len(cases)} cases: {worst:.4f}")


def test_axis_aligned_box_cannot_see_yaw():
    """Why the target geometry matters: an upright box has no angle to read.

    An axis-aligned box around the leaning runway is wider than the runway, by an
    amount that depends on the yaw it cannot report. Its centre is only unbiased
    at zero yaw. This is the failure mode a plain `detect` label would have.

    The yaw values here are realistic: measured over 60 episodes, |yaw| inside
    1500 m has a median of 0.9 deg and a p95 of 2.3 deg. Much beyond that and the
    runway leaves the narrow canvas entirely -- see canvas_yaw_limit_deg.
    """
    _, corners = _runway(length=2000.0)
    rng, h = 1000.0, 55.0
    widths = []
    for yaw in (-2.0, 0.0, 2.0):
        ground = runway_to_ground(corners, -rng, 0.0, yaw)
        grid = POLICY.for_frame(CAM, 0.0, 3.0, h, range_m=rng)
        H = ground_to_image(CAM, 0.0, 3.0, h)
        p = np.column_stack([ground, np.ones(4)]) @ H.T
        pts, _ = warp_points(p[:, :2] / p[:, 2:3], bev_from_image(CAM, grid, 0.0, 3.0, h))
        poly = clip_to_canvas(pts, grid.width, grid.height)
        aabb_w = (poly[:, 0].max() - poly[:, 0].min()) * grid.mpp
        widths.append(aabb_w)
        print(f"  yaw {yaw:+.1f} deg -> upright box is {aabb_w:6.1f} m wide "
              f"(runway is 45.0 m)")
    assert widths[1] < widths[0] and widths[1] < widths[2]
    assert widths[1] > 44.0, "at zero yaw the upright box should be the runway"


def test_canvas_yaw_limit():
    """How much yaw the narrow BEV canvas tolerates before the runway leaves it.

    A hard limit of the fixed-scale canvas, and the reason the dataset builder
    stops at 1500 m: the threshold sits |range * sin(yaw)| off the heading axis,
    so the tolerable yaw shrinks as 1/range.
    """
    half = (POLICY.width_px - 1) / 2 * POLICY.mpp_m
    margin = 22.5 + 20.0          # half a wide runway, plus typical cross-track
    print(f"  canvas half-width {half:.0f} m")
    for rng in (300.0, 600.0, 1000.0, 1500.0, 3000.0):
        limit = math.degrees(math.asin(min(1.0, max(half - margin, 0.0) / rng)))
        print(f"    range {rng:6.0f} m -> tolerates |yaw| up to {limit:5.2f} deg")
    assert math.degrees(math.asin((half - margin) / 1500.0)) > 2.3, \
        "the canvas must clear the measured p95 yaw at the builder's max range"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"{t.__name__}:")
        t()
    print(f"\nall {len(tests)} estimator tests passed")


if __name__ == "__main__":
    main()
