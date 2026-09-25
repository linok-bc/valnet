"""Synthetic test of Ryu et al.'s estimator, section 3.1.2 of the paper.

Same contract as test_estimator.py: build a runway at a KNOWN cross-track, push
it through the camera and the warp, and demand the number comes back. If this
passes, an error in ryu_eval.py is the detector's, not the geometry's.

It also pins down the two properties that decide whether the paper's method is
worth using here: that the measured scale factor k_s is nearly free for
cross-track and expensive for along-track, and that the Kalman filter earns its
place.

    python -m experiments.tests.test_ryu
"""

import math

import numpy as np

from experiments.bev.camera import CameraModel
from experiments.bev.episodes import runway_to_ground
from experiments.bev.grids import GridPolicy
from experiments.bev.labels import clip_to_canvas, warp_points
from experiments.bev.ryu import (RyuEstimator, front_edge, kalman_track,
                                 polygon_centroid, principal_ground_point,
                                 pivot_point)
from experiments.bev.transform import bev_from_image, ground_to_image
from experiments.bev_yolo.eval import ground_from_bev

CAM = CameraModel.from_fov(1280, 720, 65.0, 39.43, tilt_deg=0.0)
POLICY = GridPolicy(mode="threshold", width_px=320, height_px=1024,
                    mpp_m=0.75, back_m=150.0)
WIDTH_M, LENGTH_M = 45.0, 2500.0
CORNERS = np.array([[0.0, -WIDTH_M / 2], [LENGTH_M, -WIDTH_M / 2],
                    [LENGTH_M, WIDTH_M / 2], [0.0, WIDTH_M / 2]])

# range, cross-track, yaw, height, roll, pitch -- the same envelope
# test_estimator.py uses, so the two estimators are compared on like frames.
CASES = [
    (1400.0, 12.0, 1.5, 75.0, 0.0, 3.0),
    (900.0, -18.0, -1.5, 48.0, 6.0, 2.5),
    (500.0, 4.0, 0.5, 27.0, -9.0, 3.5),
    (200.0, -1.0, -0.2, 11.0, 2.0, 1.0),
    (1200.0, 0.0, 0.0, 64.0, 0.0, 3.0),
]


def _ground_polygon(rng, right, yaw, h, roll, pitch):
    """The runway as the BEV pipeline delivers it: ground metres, canvas-clipped."""
    ground = runway_to_ground(CORNERS, -rng, right, yaw)
    H = ground_to_image(CAM, roll, pitch, h)
    p = np.column_stack([ground, np.ones(4)]) @ H.T
    grid = POLICY.for_frame(CAM, roll, pitch, h, range_m=rng)
    pts, valid = warp_points(p[:, :2] / p[:, 2:3], bev_from_image(CAM, grid, roll, pitch, h))
    assert valid.all(), "a corner landed at or above the horizon"
    return ground_from_bev(clip_to_canvas(pts, grid.width, grid.height), grid)


def test_principal_ground_point():
    """Eq 12 against the flat-earth triangle it reduces to.

    Looking down by `d` from height `h` with no roll, the optical axis hits the
    ground straight ahead at h/tan(d). Roll swings it off the nose without
    changing that distance along the axis.
    """
    for h, pitch in [(100.0, 5.0), (60.0, 3.0), (300.0, 10.0)]:
        cam = CameraModel.from_fov(1280, 720, 65.0, 39.43, tilt_deg=pitch)
        c = principal_ground_point(cam, 0.0, 0.0, h)   # depression = tilt
        want = h / math.tan(math.radians(pitch))
        assert abs(c[0] - want) < 1e-6, (c, want)
        assert abs(c[1]) < 1e-9, c
        print(f"  h {h:5.0f} m, depression {pitch:4.1f} deg -> "
              f"principal point {c[0]:8.1f} m ahead (want {want:8.1f})")
    # looking level has no ground intersection ahead
    assert principal_ground_point(CAM, 0.0, 0.0, 100.0) is None


def test_pivot_is_undefined_for_a_forward_looking_camera():
    """The paper's eq 12 needs a camera tilted DOWN. Ours is not.

    Ryu et al. mount their camera looking slightly below the body x axis so the
    optical axis always meets the ground. The X-Plane camera here looks straight
    down the body axis (fitted tilt +0.04 deg), and the approach is flown nose-up
    at ~3.4 deg, so the axis points at the sky and there is no c_H to pivot about.

    That is why `pivot="auto"` exists. This test records the failure rather than
    working around it, because it is the finding.
    """
    misses = 0
    for rng, right, yaw, h, roll, pitch in CASES:
        if principal_ground_point(CAM, roll, pitch, h) is None:
            misses += 1
        assert pivot_point("auto", CAM, roll, pitch, h) is not None
        assert np.allclose(pivot_point("nadir", CAM, roll, pitch, h), 0.0)
    print(f"  optical axis misses the ground on {misses}/{len(CASES)} cases "
          f"(camera tilt {CAM.tilt_deg:+.2f} deg, nose-up approach)")
    assert misses >= 4, "expected a forward-looking camera to miss on climb-out attitudes"

    # where it DOES exist, on a properly tilted camera, it lands near the runway
    # and the lever is short -- which is the condition the paper's method wants
    tilted = CameraModel.from_fov(1280, 720, 65.0, 39.43, tilt_deg=8.0)
    c = principal_ground_point(tilted, 0.0, 3.0, 75.0)
    print(f"  with an 8 deg down-tilt instead: c_H at {c[0]:.0f} m ahead")
    assert 500.0 < c[0] < 2500.0


def test_front_edge_finds_the_threshold():
    """The near edge of the clipped runway is the threshold, at full width."""
    for rng, right, yaw, h, roll, pitch in CASES:
        poly = _ground_polygon(rng, right, yaw, h, roll, pitch)
        p1, p2 = front_edge(poly)
        w = math.hypot(*(p2 - p1))
        assert abs(w - WIDTH_M) < 0.5, f"front edge is {w:.2f} m, runway is {WIDTH_M}"
        # and it really is the near edge, not the far one
        assert (p1 + p2)[0] / 2 < poly[:, 0].mean(), "picked the far edge"
    print(f"  threshold edge recovered to <0.5 m on {len(CASES)} cases")


def test_recovers_crosstrack():
    """Both scale modes return the true cross-track on a perfect polygon.

    `scale="grid"` is exact by construction. `scale="width"` is exact too, but
    only because the warp is perfect here: its k_s = W_R / w_measured is 1 when
    the measured width is right. Every metre of error a real detector makes in
    that width becomes a scale error, which is what the next test prices.
    """
    for mode in ("grid", "width"):
        est = RyuEstimator(scale=mode, yaw="known", edge="front")
        worst = 0.0
        for rng, right, yaw, h, roll, pitch in CASES:
            poly = _ground_polygon(rng, right, yaw, h, roll, pitch)
            out = est(poly, CAM, roll, pitch, h, WIDTH_M, yaw_deg=yaw)
            d = out["right_m"] - right
            worst = max(worst, abs(d))
            assert abs(d) < 0.05, f"scale={mode} range {rng}: off by {d:.4f} m"
        print(f"  scale={mode:5s}: worst cross-track error {worst:.4f} m "
              f"over {len(CASES)} cases")


def _clip_half_plane(poly, x_max):
    """Sutherland-Hodgman against `x <= x_max`, i.e. the canvas's far edge."""
    out, prev = [], poly[-1]
    for cur in poly:
        ci, pi = cur[0] <= x_max, prev[0] <= x_max
        if ci != pi:
            t = (x_max - prev[0]) / (cur[0] - prev[0])
            out.append(prev + t * (cur - prev))
        if ci:
            out.append(cur)
        prev = cur
    return np.asarray(out, dtype=np.float64).reshape(-1, 2)


def test_centroid_is_robust_where_the_front_edge_is_not():
    """Why the `axis` mode exists, as the three failures it survives.

    Everything is scored as CROSS-TRACK METRES OUT OF THE ESTIMATOR, not as
    movement of the centroid. Those are different: the centroid slides a long way
    ALONG the centre line when the far end is clipped, and because the runway is
    tilted by the heading error that shows up as ground-frame lateral motion --
    154 m of slide at 1.5 deg of yaw is 4.0 m of ground y. It is still on the
    centre line, the estimator projects onto the runway normal, and the
    along-track component cancels. Only the estimate is the thing to measure.

    On the test split the front-edge rule's p90 threshold-midpoint lateral error
    grows from 2.25 m at 600-1000 m to 6.21 m inside 300 m, worst cases 13-23 m --
    about half a runway width. These are the mechanisms behind that.
    """
    rng, right, yaw, h, roll, pitch = 400.0, 6.0, 0.5, 21.0, 0.0, 3.0
    poly = _ground_polygon(rng, right, yaw, h, roll, pitch)
    kw = dict(scale="width", yaw="known")
    front = RyuEstimator(edge="front", **kw)
    axis = RyuEstimator(edge="axis", **kw)

    def err(e, q):
        o = e(q, CAM, roll, pitch, h, WIDTH_M, yaw_deg=yaw)
        return abs(o["right_m"] - right) if o else float("inf")

    # 1. HALF the front edge eaten -- the failure the eval traced. Two points
    #    slide; an average over the whole shape does not.
    near = np.argsort(poly[:, 0])[:2]
    eaten = poly.copy()
    eaten[near[0]] = (poly[near[0]] + poly[near[1]]) / 2.0
    e_f, e_a = err(front, eaten), err(axis, eaten)
    print(f"  half a front edge eaten:  front {e_f:6.2f} m   axis {e_a:6.2f} m")
    assert e_a < e_f, "the centroid should be the steadier of the two"

    # 2. far-end clipping, which EVERY frame here has: the canvas holds 617 m of
    #    a 2500 m runway. It shortens the rectangle without moving its centre
    #    line, so it must not change the estimate.
    for yaw_case in (0.0, 1.5):
        q = _ground_polygon(1000.0, 4.0, yaw_case, 53.0, 0.0, 3.0)
        e0 = RyuEstimator(edge="axis", **kw)(q, CAM, 0.0, 3.0, 53.0, WIDTH_M,
                                             yaw_deg=yaw_case)["right_m"]
        cut = _clip_half_plane(q, q[:, 0].min() + 0.5 * np.ptp(q[:, 0]))
        e1 = RyuEstimator(edge="axis", **kw)(cut, CAM, 0.0, 3.0, 53.0, WIDTH_M,
                                             yaw_deg=yaw_case)["right_m"]
        print(f"  far end cut in half at yaw {yaw_case:+.1f} deg: "
              f"estimate moves {abs(e1 - e0):.4f} m")
        assert abs(e1 - e0) < 0.05, "far-end clipping must not move cross-track"

    # 3. uneven vertex spacing, which Douglas-Peucker always leaves: the AREA
    #    centroid is immune, the vertex mean is not
    side = poly[np.argmin(poly[:, 1])]
    dense = np.vstack([poly[:1], [[side[0] + 1.0, side[1]], [side[0] + 2.0, side[1]]],
                       poly[1:]])
    d_area = np.linalg.norm(polygon_centroid(dense) - polygon_centroid(poly))
    d_mean = np.linalg.norm(dense.mean(0) - poly.mean(0))
    print(f"  two extra vertices on one edge:  area centroid moves {d_area:.3f} m, "
          f"vertex mean moves {d_mean:.2f} m")
    assert d_area < 0.5 < d_mean


def test_axis_mode_recovers_crosstrack():
    """`edge="axis"` is exact on a clean polygon, like the other modes."""
    est = RyuEstimator(scale="width", yaw="known", edge="axis")
    worst = 0.0
    for rng, right, yaw, h, roll, pitch in CASES:
        poly = _ground_polygon(rng, right, yaw, h, roll, pitch)
        out = est(poly, CAM, roll, pitch, h, WIDTH_M, yaw_deg=yaw)
        worst = max(worst, abs(out["right_m"] - right))
    print(f"  worst cross-track error {worst:.4f} m over {len(CASES)} cases")
    assert worst < 0.05


def test_scale_error_is_asymmetric():
    """The reason the paper can afford to measure its scale from the runway width.

    k_s multiplies the whole offset from the principal ground point. That offset
    is hundreds of metres along-track and metres laterally, so a relative width
    error costs proportionally in each -- which means it is ruinous for range and
    almost invisible for cross-track. This is the paper's own result (1.78 m
    lateral MAE against ~30 m along-track RMSE) reproduced in miniature.
    """
    rng, right, yaw, h, roll, pitch = 1200.0, 8.0, 0.0, 64.0, 0.0, 3.0
    poly = _ground_polygon(rng, right, yaw, h, roll, pitch)
    est = RyuEstimator(scale="width", yaw="known", edge="front")
    truth = est(poly, CAM, roll, pitch, h, WIDTH_M, yaw_deg=yaw)

    for frac in (0.05, 0.10):
        # a detector that thinks the runway is `frac` wider than it is
        out = est(poly, CAM, roll, pitch, h, WIDTH_M * (1 + frac), yaw_deg=yaw)
        d_lat = abs(out["right_m"] - truth["right_m"])
        d_along = abs(out["along_m"] - truth["along_m"])
        print(f"  {frac:.0%} width error at {rng:.0f} m -> "
              f"cross-track {d_lat:6.3f} m, along-track {d_along:7.2f} m "
              f"({d_along / max(d_lat, 1e-9):.0f}x)")
        assert d_along > 20 * d_lat, "expected the along-track channel to dominate"
        assert d_lat < 1.0, "a 10% width error should not cost a metre laterally"


def test_heading_error_is_levered_by_range():
    """Why this experiment is worth running at all.

    The README's finding is that cross-track error is angular error levered by
    range: dpsi of heading error moves the estimate by range * sin(dpsi). Here
    the polygon is left alone -- it is the true scene -- and only the HEADING THE
    ESTIMATOR IS TOLD is perturbed, which is exactly what a box fit gets wrong.

    The paper does not pay this, because eq 22 takes heading from the INS, whose
    error is an order of magnitude below what fitting a rectangle to a mask
    achieves (the README measures 0.14-0.19 deg for the latter). That is the
    single biggest reason to expect Ryu et al.'s estimator to win here.
    """
    est = RyuEstimator(scale="grid", yaw="known", edge="front")
    for rng in (1400.0, 600.0, 200.0):
        right, yaw, h, roll, pitch = 12.0, 1.5, rng / 19.0, 0.0, 3.0
        poly = _ground_polygon(rng, right, yaw, h, roll, pitch)
        exact = est(poly, CAM, roll, pitch, h, WIDTH_M, yaw_deg=yaw)["right_m"]
        for dpsi in (0.05, 0.2):
            off = est(poly, CAM, roll, pitch, h, WIDTH_M, yaw_deg=yaw + dpsi)["right_m"]
            lever = rng * math.sin(math.radians(dpsi))
            assert abs(abs(off - exact) - lever) < 0.15, (off, exact, lever)
        print(f"  range {rng:6.0f} m: 0.05 deg of heading error -> "
              f"{rng * math.sin(math.radians(0.05)):5.2f} m of cross-track, "
              f"0.2 deg -> {rng * math.sin(math.radians(0.2)):5.2f} m")
        assert abs(exact - right) < 0.05, "the unperturbed estimate should be exact"


def test_kalman_filter_helps():
    """Section 3.1.2.5, their Figure 20: the filter smooths the track.

    A true cross-track that drifts linearly down the approach, measured with
    noise and two dropped frames. The filter should beat the raw measurements on
    RMSE and fill the gaps.
    """
    rng = np.random.default_rng(0)
    s = np.linspace(-1500.0, -150.0, 120)          # along-track, rising to 0
    truth = 12.0 * (s / -1500.0)                   # drifts in toward the centreline
    meas = truth + rng.normal(0.0, 2.0, len(s))
    meas[40:45] = np.nan                            # the recogniser drops out

    filt = kalman_track(meas, s, q=1e-5, r=4.0)
    ok = np.isfinite(meas)
    raw_rmse = float(np.sqrt(((meas[ok] - truth[ok]) ** 2).mean()))
    kf_rmse = float(np.sqrt(((filt - truth) ** 2).mean()))
    print(f"  raw RMSE {raw_rmse:.3f} m -> filtered {kf_rmse:.3f} m "
          f"({100 * (1 - kf_rmse / raw_rmse):.0f}% better), gaps filled: "
          f"{int(np.isfinite(filt[40:45]).sum())}/5")
    assert kf_rmse < 0.75 * raw_rmse, "the filter should clearly reduce RMSE"
    assert np.isfinite(filt[40:45]).all(), "the filter should coast through dropouts"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"\n{name}")
            fn()
    print("\nall passed")
