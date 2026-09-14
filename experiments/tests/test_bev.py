"""Synthetic correctness tests for the BEV transform.

These need no dataset: they build ground truth analytically. Run them after any
change to bev/transform.py, and before trusting a single warped frame.

    python -m experiments.tests.test_bev
"""

import math

import numpy as np

from experiments.bev import BEVGrid, CameraModel, bev_from_image, ground_to_image, horizon_v


CAM = CameraModel.from_fov(1280, 720, 65.0, 39.43, tilt_deg=0.0)


def _project(cam, roll, pitch, h, pts_ground):
    H = ground_to_image(cam, roll, pitch, h)
    p = np.column_stack([pts_ground, np.ones(len(pts_ground))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def test_level_horizon_at_principal_point():
    v = horizon_v(CAM, roll_deg=0.0, pitch_deg=0.0)
    assert abs(v - CAM.cy) < 1e-6, v
    print(f"  level camera: horizon v = {v:.3f}, cy = {CAM.cy:.3f}")


def test_looking_down_raises_horizon():
    cam = CameraModel.from_fov(1280, 720, 65.0, 39.43, tilt_deg=10.0)
    v = horizon_v(cam, roll_deg=0.0, pitch_deg=0.0)
    assert v < cam.cy, f"depression should push the horizon above centre, got {v}"
    # Ryu eq. 13: theta_cam = arctan((v_vp - cy) / fy)
    implied = math.degrees(math.atan((v - cam.cy) / cam.fy))
    assert abs(implied + 10.0) < 1e-6, implied
    print(f"  10 deg down-tilt: horizon v = {v:.2f} (above cy={cam.cy:.1f}); "
          f"Ryu eq.13 recovers {implied:.4f} deg")


def test_nose_up_pitch_cancels_tilt():
    cam = CameraModel.from_fov(1280, 720, 65.0, 39.43, tilt_deg=10.0)
    v = horizon_v(cam, roll_deg=0.0, pitch_deg=10.0)
    assert abs(v - cam.cy) < 1e-6, v
    print(f"  10 deg tilt + 10 deg nose-up: horizon back at centre ({v:.3f})")


def test_projection_matches_closed_form():
    """Level camera at height h: a point X ahead lands at v = cy + fy*h/X."""
    h, X = 100.0, 1500.0
    uv = _project(CAM, 0.0, 0.0, h, np.array([[X, 0.0]]))[0]
    assert abs(uv[0] - CAM.cx) < 1e-6
    assert abs(uv[1] - (CAM.cy + CAM.fy * h / X)) < 1e-6
    print(f"  ground point {X:.0f} m ahead at {h:.0f} m AGL -> "
          f"v = {uv[1]:.3f} (closed form {CAM.cy + CAM.fy*h/X:.3f})")


def test_ground_image_bev_roundtrip():
    """ground -> image -> BEV must land where the grid says it should."""
    grid = BEVGrid(width=512, height=512, mpp=0.5, x_near=50.0)
    roll, pitch, h = 7.0, -3.0, 120.0
    ground = np.array([[300.0, 0.0], [200.0, -20.0], [150.0, 25.0], [80.0, 5.0]])

    uv = _project(CAM, roll, pitch, h, ground)
    M = bev_from_image(CAM, grid, roll, pitch, h)
    p = np.column_stack([uv, np.ones(len(uv))]) @ M.T
    bev = p[:, :2] / p[:, 2:3]

    # what the grid definition predicts
    exp_u = ground[:, 1] / grid.mpp + (grid.width - 1) / 2.0
    exp_v = (grid.height - 1) - (ground[:, 0] - grid.x_near) / grid.mpp
    err = np.abs(bev - np.column_stack([exp_u, exp_v])).max()
    assert err < 1e-6, f"round-trip error {err}"
    print(f"  ground->image->BEV round trip: max error {err:.2e} px")


def test_runway_rectangle_stays_rectangular_and_keeps_heading():
    """The whole point: a runway rectangle comes back as a rectangle in BEV,
    rotated by exactly the heading error, under arbitrary roll and pitch."""
    grid = BEVGrid(width=512, height=768, mpp=0.4, x_near=20.0)
    W, L, heading = 45.0, 600.0, 6.0     # metres, metres, degrees of yaw error
    c, s = math.cos(math.radians(heading)), math.sin(math.radians(heading))
    rect = np.array([[0.0, -W/2], [0.0, W/2], [L, W/2], [L, -W/2]])
    rect = rect @ np.array([[c, s], [-s, c]])        # rotate about the threshold
    rect[:, 0] += 120.0                              # push it down-track

    for roll, pitch, h in [(0.0, 0.0, 150.0), (12.0, -4.0, 90.0), (-8.0, 2.5, 300.0)]:
        uv = _project(CAM, roll, pitch, h, rect)
        M = bev_from_image(CAM, grid, roll, pitch, h)
        p = np.column_stack([uv, np.ones(4)]) @ M.T
        q = p[:, :2] / p[:, 2:3]

        e = [q[(i + 1) % 4] - q[i] for i in range(4)]
        angles = [abs(math.degrees(math.acos(np.clip(
            np.dot(e[i], e[(i + 1) % 4]) / (np.linalg.norm(e[i]) * np.linalg.norm(e[(i + 1) % 4])),
            -1, 1)))) for i in range(4)]
        assert max(abs(a - 90.0) for a in angles) < 1e-3, angles

        width_px = np.linalg.norm(e[0])
        assert abs(width_px * grid.mpp - W) < 1e-3, width_px * grid.mpp
        # heading read off the long edge, measured from BEV "up"
        long_edge = q[2] - q[1]
        recovered = math.degrees(math.atan2(long_edge[0], -long_edge[1]))
        assert abs(recovered - heading) < 1e-3, recovered
        print(f"  roll={roll:+5.1f} pitch={pitch:+4.1f} h={h:5.0f}m -> "
              f"corners square (max dev {max(abs(a-90) for a in angles):.1e} deg), "
              f"width {width_px*grid.mpp:.3f} m, heading {recovered:+.4f} deg")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"{t.__name__}:")
        t()
    print(f"\nall {len(tests)} BEV tests passed")


if __name__ == "__main__":
    main()
