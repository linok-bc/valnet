"""The lateral position estimator of Ryu et al. (IJRR 44(6), 2024), sections
3.1.2.2 - 3.1.2.5, as a drop-in alternative to `bev_yolo.eval.estimate`.

WHAT THE PAPER ACTUALLY DOES, once the coordinate algebra is unwound. Equations
4-21 look like a long chain -- project the principal ray to the ground, convert
camera to body to heading frame, take pixel errors from the principal point,
scale them, add them back -- but every step is affine about one point, so the
whole chain is a RADIAL RESCALING of the threshold midpoint about the camera's
principal ground point:

    p_H = c_H + k_s * (t - c_H)                             (eqs 12, 15-21)

where `t` is the runway front edge's midpoint on the ground, `c_H` is where the
optical axis pierces the ground (eqs 4-12), and `k_s` is a scale read off the
image (eq 18). Then eqs 22-23 rotate p_H into the runway frame by the known
heading error, and the lateral component is the answer.

So against the minimum-area-rectangle estimator this experiment already uses,
the paper differs in exactly THREE independent ways, and each is a flag here:

    `scale`     the paper measures the metre-per-pixel scale from the runway's
                apparent width against its known true width -- k_s = W_R / w --
                instead of taking it from the BEV grid, which it cannot do
                because its BEV is an attitude-only quad warp with no metric
                meaning. Our grid IS metric, so `scale="grid"` sets k_s = 1 and
                recovers the exact ground coordinate. That isolates what the
                measured scale costs. See `k_s` below for why it is nearly free
                for cross-track and expensive for along-track.

    `yaw`       the paper reads heading from the INS (eq 22) and never estimates
                it. The existing estimator fits it from the box's long axis, and
                the README's central finding is that THAT is where the error
                comes from, levered by range. `yaw="known"` removes the lever;
                `yaw="fitted"` keeps it, so the two are comparable.

    `edge`      the paper uses two points and no more: Douglas-Peucker the mask
                to a polygon, take the front edge (section 3.1.1, their Figure
                4). It says why -- the four-corner PnP methods it beats are
                limited by the accuracy of corner extraction. `edge="rect"` is
                the control that keeps the whole polygon.

WHY k_s IS ASYMMETRIC, and the prediction this experiment should test. k_s
multiplies the whole offset (t - c_H). That offset is hundreds of metres in the
along-track direction and a few metres laterally, so a relative width error eps
costs eps * X_E along-track -- tens of metres -- and only eps * Y_E laterally --
centimetres. The paper reports exactly this asymmetry: 1.78 m lateral MAE
against ~30 m along-track RMSE (their sections 5.2.1 and 5.3). The one leak is
the rotation: eq 23 mixes eps * X_E * sin(psi_err) into the lateral channel, so
the measured scale should start to hurt when the heading error is large, which
on these approaches means at long range.

THE PIVOT DOES NOT EXIST ON THIS DATASET, and that is a fourth flag. Eq 12
needs the optical axis to hit the ground. Ryu et al. mount their camera tilted
DOWN for exactly that reason (section 3.1.2: "the camera is mounted ... in such
a way that it looks slightly down to better see the runway"). Our X-Plane camera
looks straight along the body x axis -- the fit gives tilt = +0.04 deg -- and the
aircraft flies the approach nose-up at 3.4 deg median, so the depression is
NEGATIVE on 98.8% of frames and the axis points at the sky. On the remaining
1.2% the intersection lands at 3.3 km against a threshold at 1.4 km, so the lever
(t - c_H) is ~1.9 km pointing backwards and any error in k_s is amplified by it.

`pivot` therefore chooses the point the rescaling turns about:

    "principal"  eq 12 as written. Returns None where the axis misses the
                 ground, which here is almost everywhere. Keep it to reproduce
                 the paper on a camera that is actually tilted down.
    "nadir"      the ground point directly beneath the aircraft, which is the
                 heading-frame origin (0, 0) by construction and so is always
                 defined and exactly known. p_H = k_s * t.
    "auto"       principal where it exists, nadir elsewhere. The default.

Conditioning is the thing to watch, not availability: the scale error costs
(k_s - 1) * |t - pivot|, so a pivot far from the threshold is worse than no
pivot at all. Nadir puts that distance at the range itself; a correctly tilted
camera would put c_H near the threshold and make the paper's method much better
conditioned than either. That is a claim about MOUNTING, and it is the most
actionable thing this port has to say.

Frames follow `experiments.bev.transform`: ground/heading frame is X forward
along the aircraft nose, Y right, origin under the camera; `yaw_deg` from the
collector is the heading error psi - psi_r, positive nose-right of the runway.
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np

from experiments.bev.camera import CameraModel
from experiments.bev.transform import depression_deg, rotation_ground_to_camera


# ------------------------------------------------- eqs 4-12: the pivot point --

def principal_ground_point(cam: CameraModel, roll_deg: float, pitch_deg: float,
                           height_m: float):
    """Where the optical axis pierces the ground, in heading-frame metres.

    This is the paper's c_H (eq 12), reached in closed form rather than through
    its camera -> body -> heading chain: the optical axis is (0, 0, 1) in camera
    coordinates, so rotating it into the ground frame and scaling it until it
    drops `height_m` gives the same point. Returns None when the axis points at
    or above the horizon, where there is no intersection ahead of the aircraft.
    """
    R = rotation_ground_to_camera(roll_deg, depression_deg(pitch_deg, cam.tilt_deg))
    d = R.T @ np.array([0.0, 0.0, 1.0])     # optical axis, ground frame
    if d[2] <= 1e-9:                         # looking level or up
        return None
    t = height_m / d[2]
    return np.array([t * d[0], t * d[1]])


def pivot_point(mode: str, cam: CameraModel, roll_deg: float, pitch_deg: float,
                height_m: float):
    """The ground point the rescaling of eqs 19-21 turns about.

    See the module docstring for why this is a flag rather than just eq 12.
    Returns None only when `mode="principal"` and the axis misses the ground.
    """
    if mode == "nadir":
        return np.zeros(2)                # the heading frame's own origin
    c = principal_ground_point(cam, roll_deg, pitch_deg, height_m)
    if mode == "principal":
        return c
    if mode == "auto":
        return c if c is not None else np.zeros(2)
    raise ValueError(f"unknown pivot mode {mode!r}")


# ----------------------------------------- section 3.1.1: the two front points --

def front_edge(poly: np.ndarray, band: float = 0.25):
    """The runway's front edge as two endpoints, from a mask polygon.

    The paper's post-processing (section 3.1.1): simplify the mask to a polygon
    with Douglas-Peucker, then "extract the two farthest points in close
    proximity to the forward edge". Here `poly` is already in ground metres with
    X forward, so "forward edge" is the near one -- smallest X, the threshold --
    and the rule is: of the polygon's edges that lie wholly within the frontmost
    `band` of its along-track extent, take the longest.

    Taking the longest, rather than the frontmost, matters. A simplified mask
    often carries a short spur at the near corner; the frontmost edge is then
    that spur, a metre or two long, and `k_s = W_R / w` explodes. The runway's
    true front edge is a full width across, so length picks it out.

    Returns (p1, p2) or None.
    """
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if len(p) < 3:
        return None
    x0, x1 = p[:, 0].min(), p[:, 0].max()
    if x1 - x0 < 1e-6:
        return None
    cut = x0 + band * (x1 - x0)

    best, best_len = None, 0.0
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        if a[0] > cut or b[0] > cut:
            continue
        d = float(np.hypot(*(b - a)))
        if d > best_len:
            best, best_len = (a, b), d
    if best is None:
        # no edge lies wholly in the band -- happens when the mask is clipped so
        # hard that the threshold is a single vertex. Fall back to the two
        # frontmost vertices, which is the paper's phrasing read literally.
        order = np.argsort(p[:, 0])
        best = (p[order[0]], p[order[1]])
    return best


def simplify(poly_px: np.ndarray, frac: float = 0.01) -> np.ndarray:
    """Douglas-Peucker, with epsilon a fraction of the perimeter.

    Applied in the detector's own pixels before any back-projection, which is
    where the paper applies it (`cv2.approxPolyDP`, their section 3.1.1). Doing
    it after back-projection would set the tolerance in metres, and a metre
    means something different at 200 m and at 1.4 km.
    """
    p = np.asarray(poly_px, dtype=np.float32).reshape(-1, 1, 2)
    if len(p) < 4:
        return np.asarray(poly_px, dtype=np.float64).reshape(-1, 2)
    eps = frac * cv2.arcLength(p, True)
    out = cv2.approxPolyDP(p, eps, True).reshape(-1, 2).astype(np.float64)
    return out if len(out) >= 3 else np.asarray(poly_px, np.float64).reshape(-1, 2)


def polygon_centroid(poly: np.ndarray) -> np.ndarray:
    """AREA centroid of the polygon, not the mean of its vertices.

    For the runway's purposes this is the one robust point available: a runway
    in a metric BEV is a rectangle, and a rectangle's area centroid is the
    midpoint of its centre line. Two properties make it the right thing to
    measure cross-track from:

    1. It is an AVERAGE over the whole shape, not an extremum. The front-edge
       rule reads two points, so a near edge that is clipped or whose corner is
       rounded moves the midpoint by up to half a runway width -- measured on
       the test split, the p90 threshold-midpoint lateral error grows from
       2.25 m at 600-1000 m to 6.21 m inside 300 m, with worst cases at 13-23 m.
       The centroid cannot move like that.

    2. Far-end clipping, which happens in EVERY frame here because the canvas
       holds 617 m of a 2500 m runway, is laterally SYMMETRIC -- it shortens the
       rectangle without moving its centre line. So the failure that breaks
       "which edge is the front one" does not bias the centroid at all.

    The vertex mean would not do: Douglas-Peucker leaves vertices unevenly
    spaced, so it weights a corner with two nearby vertices twice.
    """
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if len(p) < 3:
        return p.mean(axis=0) if len(p) else np.array([np.nan, np.nan])
    x, y = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    a2 = cross.sum()
    if abs(a2) < 1e-12:                      # degenerate: collapse to the mean
        return p.mean(axis=0)
    return np.array([((x + xn) * cross).sum(), ((y + yn) * cross).sum()]) / (3.0 * a2)


# ----------------------------------------------- eqs 15-23: the estimate itself --

@dataclass(frozen=True)
class RyuEstimator:
    """Configured estimator. See the module docstring for what each flag swaps.

    Attributes:
        scale: "width" for the paper's k_s = W_R / w (eq 18), "grid" for k_s = 1,
            which trusts the metric BEV raster instead.
        yaw: "known" to take the heading error from the INS (eq 22, the paper),
            "fitted" to read it off the polygon's minimum-area rectangle.
        edge: which point on the runway's centre line to measure from.
            "front" is the paper's two points -- the front edge's midpoint.
            "rect" is the near edge of the minimum-area rectangle: the same
            quantity, but with the rectangle fitted to every vertex first, so a
            partial front edge cannot run away with it. "axis" drops the
            threshold entirely and uses the polygon's AREA CENTROID, which lies
            on the centre line too and is an average rather than an extremum.
            See `polygon_centroid` for why that is the robust choice near the
            threshold. Note "axis" gives no threshold RANGE -- its `along_m` is
            the centroid's, not the threshold's -- so it estimates cross-track
            only, which is what is under test.
        pivot: "principal" for eq 12, "nadir" for the point under the aircraft,
            "auto" for eq 12 where it exists. See the module docstring -- on
            this dataset eq 12 is undefined on 98.8% of frames.
        band: front-edge search band, see `front_edge`.
    """

    scale: str = "width"
    yaw: str = "known"
    edge: str = "front"
    pivot: str = "auto"
    band: float = 0.25

    def __call__(self, poly_ground: np.ndarray, cam: CameraModel, roll_deg: float,
                 pitch_deg: float, height_m: float, runway_width_m: float,
                 yaw_deg: float = None):
        """Estimate cross-track from a runway polygon in heading-frame metres.

        `yaw_deg` is the INS heading error, required when `yaw="known"`. Returns
        a dict with `right_m` (aircraft metres right of the centreline, the
        quantity under test), or None if the geometry degenerates.
        """
        p = np.asarray(poly_ground, dtype=np.float64).reshape(-1, 2)
        p = p[np.isfinite(p).all(axis=1)]
        if len(p) < 3:
            return None

        # --- the two points, and the width they imply (eq 17) ---------------
        box = cv2.boxPoints(cv2.minAreaRect(p.astype(np.float32))).astype(np.float64)
        if self.edge == "rect":
            order = np.argsort(box[:, 0])
            e = (box[order[0]], box[order[1]])
        elif self.edge == "front":
            e = front_edge(p, self.band)
        elif self.edge == "axis":
            # no edge at all: the centroid is already on the centre line
            e = None
        else:
            raise ValueError(f"unknown edge mode {self.edge!r}")

        if self.edge == "axis":
            t = polygon_centroid(p)
            # the scale still needs a width, and the rectangle's short side is
            # the least noisy estimate of it available from the whole polygon
            (_, _), (bw, bh), _ = cv2.minAreaRect(p.astype(np.float32))
            w_meas = float(min(bw, bh))
        else:
            if e is None:
                return None
            p1, p2 = e
            t = (p1 + p2) / 2.0                          # threshold midpoint
            w_meas = float(np.hypot(*(p2 - p1)))         # eq 17, in metres here
        if not np.isfinite(t).all() or w_meas < 1e-6:
            return None

        # --- the scale (eq 18) ---------------------------------------------
        # In BEV PIXELS the paper computes k_s = W_R / w_px and multiplies a
        # pixel offset. Our polygon is already in metres, so the nominal
        # metre-per-pixel cancels out of k_s * offset and what survives is the
        # dimensionless correction W_R / w_metres. Same number, no frame needed.
        if self.scale == "width":
            k_s = runway_width_m / w_meas
        elif self.scale == "grid":
            k_s = 1.0
        else:
            raise ValueError(f"unknown scale mode {self.scale!r}")

        # --- the rescaling about the pivot (eqs 12, 19-21) -------------------
        c_H = pivot_point(self.pivot, cam, roll_deg, pitch_deg, height_m)
        if c_H is None:
            return None
        p_H = c_H + k_s * (t - c_H)

        # --- the heading, and the rotation into the runway frame (eqs 22-23) -
        if self.yaw == "known":
            if yaw_deg is None:
                raise ValueError("yaw='known' needs the INS yaw_deg")
            psi = yaw_deg
        elif self.yaw == "fitted":
            psi = fitted_yaw_deg(p)
            if psi is None:
                return None
        else:
            raise ValueError(f"unknown yaw mode {self.yaw!r}")

        # eq 22 is R_z(psi_r - psi); the collector's yaw_deg is already the
        # heading error psi - psi_r, so the rotation angle is its negation.
        a = math.radians(-psi)
        ca, sa = math.cos(a), math.sin(a)
        p_R = np.array([ca * p_H[0] + sa * p_H[1],
                        -sa * p_H[0] + ca * p_H[1]])

        return {
            # p_R is the THRESHOLD relative to the aircraft in runway axes, so
            # the aircraft's own offset is its negation -- matching the sign
            # convention of `eval.estimate` and of the collector's `right_m`.
            "right_m": float(-p_R[1]),
            "along_m": float(-p_R[0]),
            "yaw_deg": float(psi),
            "width_m": w_meas,
            "k_s": k_s,
            # how far the scale error is levered: the estimate moves by
            # (k_s - 1) * lever_m. Small is good; see the module docstring.
            "lever_m": float(np.linalg.norm(t - c_H)),
            "threshold_m": t,
            "pivot_m": c_H,
        }


def fitted_yaw_deg(poly_ground: np.ndarray):
    """Heading error from the polygon's minimum-area rectangle, in degrees.

    The same quantity `eval.estimate` uses, extracted here so the `yaw="fitted"`
    ablation is running identical code and the comparison is about the flag.
    """
    p = np.asarray(poly_ground, dtype=np.float32).reshape(-1, 2)
    if len(p) < 3:
        return None
    box = cv2.boxPoints(cv2.minAreaRect(p)).astype(np.float64)
    order = np.argsort(box[:, 0])
    u = box[order[2:]].mean(axis=0) - box[order[:2]].mean(axis=0)
    n = np.linalg.norm(u)
    if n < 1e-6:
        return None
    return -math.degrees(math.atan2(u[1] / n, u[0] / n))


# ------------------------------------------------ section 3.1.2.5: the filter --

def kalman_track(measurements, s_m, q: float = 1e-5, r: float = 4.0,
                 p0: float = 1e3):
    """Constant-velocity Kalman filter over ONE approach, the paper's 3.1.2.5.

    The paper filters the lateral estimate with a first-order Kalman filter and
    shows the effect in its Figure 20: the same track, visibly smoother, with
    the spikes from bad frames pulled in.

    The state is [lateral position, its rate], and the independent variable is
    ALONG-TRACK DISTANCE FLOWN (`s_m`), not time. That is deliberate: we do not
    know the capture rate, the frames are strided, and cross-track drift is
    physically a function of distance down the approach rather than of seconds.
    It also makes the tuning transferable -- `q` is variance per metre flown.

    `measurements` may contain NaN for frames with no detection; those steps
    propagate without an update, which is what a real filter does when the
    recogniser drops out. Returns the filtered series.

    Output is NaN until the first measurement arrives. Before that the filter has
    no estimate -- only its prior -- and returning the prior would draw a
    confident line through a stretch where nothing has been observed. This
    matters on a full approach, where the recogniser has nothing to work with for
    the first kilometre.

    Args:
        measurements: per-frame lateral estimates, NaN where missing.
        s_m: along-track distance for each frame, monotonically increasing.
        q: process noise, m^2 per metre flown. The paper tunes its covariances
            against flight data (section 3.1.2.5); 1e-5 is the equivalent here,
            chosen so that over a ~15 m frame step the model admits ~0.1 m of
            unmodelled lateral drift -- loose enough to follow a real approach,
            tight enough to average four or five frames together. Raise it if
            the filtered track lags the truth, lower it if it stays noisy.
        r: measurement noise, m^2. 4.0 = a 2 m one-sigma frame.
        p0: initial covariance; large enough to let the first measurement in.
    """
    z = np.asarray(measurements, dtype=np.float64)
    s = np.asarray(s_m, dtype=np.float64)
    out = np.full(len(z), np.nan)
    if len(z) == 0:
        return out

    x = np.array([0.0, 0.0])
    P = np.diag([p0, p0])
    H = np.array([[1.0, 0.0]])
    started = False
    for i in range(len(z)):
        if started and i > 0:
            dt = max(float(s[i] - s[i - 1]), 1e-6)
            F = np.array([[1.0, dt], [0.0, 1.0]])
            # continuous white-noise-acceleration discretisation
            Q = q * np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0],
                              [dt ** 2 / 2.0, dt]])
            x = F @ x
            P = F @ P @ F.T + Q
        if np.isfinite(z[i]):
            y = z[i] - (H @ x)[0]
            S = (H @ P @ H.T)[0, 0] + r
            K = (P @ H.T / S).ravel()
            x = x + K * y
            P = (np.eye(2) - np.outer(K, H)) @ P
            started = True
        if started:
            out[i] = x[0]
    return out
