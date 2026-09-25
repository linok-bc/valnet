"""Cross-track from the TILT of the runway's edge lines, lane-keeping style.

THE IDEA. The estimators in `experiments.bev` read cross-track off where the
runway sits on the ground, and that position depends on heading: a heading
error dpsi enters as `range * sin dpsi`, which is ~9 m at 1.4 km for 0.36 deg.
Car lane-keeping reads the lateral offset off how the lane lines LEAN instead.
The same works for a runway:

    a ground line parallel to the runway, at lateral offset s from the camera,
    together with the camera centre spans a plane. That plane's tilt about the
    runway direction is atan(s / h). Heading slides the line's image sideways
    but does not change the tilt.

So each fitted edge line gives its lateral offset s (in units of height), and

    `height`   cross-track = -(s_L + s_R) / 2              needs h
    `width`    cross-track = -(W / 2) (s_L + s_R) / (s_R - s_L)   needs W, not h

Only roll and pitch are used to turn the image line into a plane. They are a
change of coordinates on the fitted line, never a resampling of the image: no
bird's-eye view, no flat-ground warp.

THE RUNWAY DIRECTION. The plane tilt is measured about the runway direction d,
which has to come from somewhere:

    `free`     d = n_L x n_R, where the two edge planes meet -- the measured
               vanishing point. Heading is not used at all.
    `known`    d from the logged yaw. A control, to show heading buys nothing.

EDGE EXTRACTION is a row scan, the representation lane detectors use. The mask
is rotated by the known roll -- its pixel COORDINATES, not its pixels -- so that
rows run parallel to the horizon, and in each row the leftmost and rightmost
mask pixels are the two edges. A row whose extreme pixel sits on the image
border is dropped for that side: the mask is clipped there, not ending. The
top and bottom few rows are dropped too, because there the extreme pixel lies
on the near or far threshold edge. A robust line fit takes care of the rest.

Frames follow `experiments.bev.transform`: ground X forward along the aircraft's
heading, Y right, Z down, camera at the origin, ground plane at Z = h.
"""

import math

import cv2
import numpy as np

from experiments.bev.camera import CameraModel
from experiments.bev.transform import depression_deg, rotation_ground_to_camera


# ------------------------------------------------------------ image lines --

def _R(cam: CameraModel, roll_deg: float, pitch_deg: float) -> np.ndarray:
    return rotation_ground_to_camera(roll_deg, depression_deg(pitch_deg, cam.tilt_deg))


def horizon_angle(cam: CameraModel, roll_deg: float, pitch_deg: float) -> float:
    """Angle of the horizon in the image, radians, measured from the +u axis.

    Taken from the vanishing line of the ground plane rather than from the roll
    sign convention, so a sign slip cannot silently tilt the row scan.
    """
    l = np.linalg.inv(cam.K).T @ _R(cam, roll_deg, pitch_deg) @ np.array([0.0, 0.0, 1.0])
    return math.atan2(-l[0], l[1])          # direction (l1, -l0) of l0 u + l1 v + l2 = 0


def _derotation(cam: CameraModel, angle: float) -> np.ndarray:
    """3x3 map from image pixels to roll-free pixels, rotating about (cx, cy)."""
    c, s = math.cos(-angle), math.sin(-angle)
    T = np.array([[1, 0, -cam.cx], [0, 1, -cam.cy], [0, 0, 1]], dtype=np.float64)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    return np.linalg.inv(T) @ Rz @ T


def edge_points(mask: np.ndarray, cam: CameraModel, roll_deg: float,
                pitch_deg: float, trim_px: int = 3, border_px: int = 1):
    """Row-scan a runway mask into left- and right-edge points.

    Returns (left, right, D): points in ROLL-FREE pixel coordinates, one per
    usable row and side, and the 3x3 derotation D that produced them. The half-
    pixel offset puts each point on the mask boundary rather than on the centre
    of the outermost pixel.
    """
    vs, us = np.nonzero(mask)
    if len(us) < 20:
        return None, None, None
    H, W = mask.shape[:2]
    D = _derotation(cam, horizon_angle(cam, roll_deg, pitch_deg))
    p = np.column_stack([us, vs, np.ones(len(us))]) @ D.T
    ur, vr = p[:, 0], p[:, 1]
    row = np.round(vr).astype(np.int64)
    on_border = ((us < border_px) | (us > W - 1 - border_px)
                 | (vs < border_px) | (vs > H - 1 - border_px))

    # the extreme pixel of every row, via a sort by (row, u')
    order = np.lexsort((ur, row))
    row_s = row[order]
    first = np.r_[0, np.flatnonzero(np.diff(row_s)) + 1]
    last = np.r_[first[1:] - 1, len(order) - 1]
    lo, hi = order[first], order[last]
    rows = row_s[first]

    keep = (rows >= rows.min() + trim_px) & (rows <= rows.max() - trim_px)
    left_ok = keep & ~on_border[lo]
    right_ok = keep & ~on_border[hi]
    left = np.column_stack([ur[lo[left_ok]] - 0.5, vr[lo[left_ok]]])
    right = np.column_stack([ur[hi[right_ok]] + 0.5, vr[hi[right_ok]]])
    return left, right, D


def fit_line(pts: np.ndarray, iters: int = 4, k: float = 3.0, floor_px: float = 0.5):
    """Robust fit of u = a + b v to roll-free edge points.

    Least squares, then repeatedly drop points beyond k robust sigmas and refit.
    Returns (a, b, rms_px, n_used) or None. Fitting u against v, not v against
    u, because runway edges are closer to vertical than horizontal.
    """
    if pts is None or len(pts) < 5:
        return None
    u, v = pts[:, 0], pts[:, 1]
    use = np.ones(len(u), dtype=bool)
    for _ in range(iters):
        if use.sum() < 5:
            return None
        b, a = np.polyfit(v[use], u[use], 1)
        r = u - (a + b * v)
        sigma = max(1.4826 * np.median(np.abs(r[use])), floor_px)
        new = np.abs(r) <= k * sigma
        if np.array_equal(new, use):
            break
        use = new
    rms = float(np.sqrt(np.mean(r[use] ** 2)))
    return float(a), float(b), rms, int(use.sum())


def line_from_fit(a: float, b: float, D: np.ndarray) -> np.ndarray:
    """Homogeneous image line (original pixels) for u' = a + b v' in roll-free pixels."""
    l_roll_free = np.array([1.0, -b, -a])
    return D.T @ l_roll_free


def line_through(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Homogeneous image line through two pixels."""
    return np.cross([p[0], p[1], 1.0], [q[0], q[1], 1.0])


# ------------------------------------------------------------- the ground --

def edge_plane(l_img: np.ndarray, cam: CameraModel, roll_deg: float,
               pitch_deg: float) -> np.ndarray:
    """Unit normal, in the ground frame, of the plane through the camera and an image line."""
    n = _R(cam, roll_deg, pitch_deg).T @ cam.K.T @ l_img
    return n / np.linalg.norm(n)


def runway_direction(n_left: np.ndarray, n_right: np.ndarray) -> np.ndarray:
    """Where the two edge planes meet: the runway direction, flattened to the ground."""
    d = np.cross(n_left, n_right)
    d = np.array([d[0], d[1], 0.0])
    d /= np.linalg.norm(d)
    return d if d[0] > 0 else -d


def lateral_offset(n: np.ndarray, d: np.ndarray, height_m: float) -> float:
    """Signed offset of an edge line from the aircraft, metres, positive to the RIGHT of d.

    The ground line is the plane's intersection with Z = h. Writing its points
    as s e + t d + h z with e = d rotated 90 deg right, the plane condition
    n . P = 0 gives s = -h n_z / (n . e) once n is made perpendicular to d. With
    `free` direction n is already perpendicular; with `known` direction this is
    the plane nearest the fitted one that is parallel to the runway.
    """
    n = n - (n @ d) * d
    e = np.array([-d[1], d[0], 0.0])
    ne = n @ e
    if abs(ne) < 1e-12:
        return float("nan")
    return float(-height_m * n[2] / ne)


def crosstrack(l_left: np.ndarray, l_right: np.ndarray, cam: CameraModel,
               roll_deg: float, pitch_deg: float, height_m: float,
               width_m: float, yaw_deg: float = None):
    """All four estimates from a pair of image edge lines.

    Returns a dict keyed "<direction>_<scale>" (free_height, free_width,
    known_height, known_width), each the aircraft's metres RIGHT of the
    centreline, plus diagnostics. `known_*` are omitted when yaw_deg is None.
    """
    nL = edge_plane(l_left, cam, roll_deg, pitch_deg)
    nR = edge_plane(l_right, cam, roll_deg, pitch_deg)
    dirs = {"free": runway_direction(nL, nR)}
    if yaw_deg is not None:
        psi = math.radians(yaw_deg)
        dirs["known"] = np.array([math.cos(psi), -math.sin(psi), 0.0])

    out = {}
    for name, d in dirs.items():
        sL = lateral_offset(nL, d, height_m)
        sR = lateral_offset(nR, d, height_m)
        out[f"{name}_height"] = -(sL + sR) / 2
        span = sR - sL
        out[f"{name}_width"] = (-(width_m / 2) * (sL + sR) / span
                                if span > 1e-6 else float("nan"))
        out[f"{name}_width_m"] = span
    d = dirs["free"]
    out["yaw_fit_deg"] = -math.degrees(math.atan2(d[1], d[0]))
    return out


def estimate_from_mask(mask: np.ndarray, cam: CameraModel, roll_deg: float,
                       pitch_deg: float, height_m: float, width_m: float,
                       yaw_deg: float = None, trim_px: int = 3, min_rows: int = 5):
    """Mask -> row scan -> two robust edge lines -> cross-track. None on failure."""
    left, right, D = edge_points(mask, cam, roll_deg, pitch_deg, trim_px=trim_px)
    if D is None or len(left) < min_rows or len(right) < min_rows:
        return None
    fl, fr = fit_line(left), fit_line(right)
    if fl is None or fr is None:
        return None
    out = crosstrack(line_from_fit(fl[0], fl[1], D), line_from_fit(fr[0], fr[1], D),
                     cam, roll_deg, pitch_deg, height_m, width_m, yaw_deg)
    out.update(rows_left=fl[3], rows_right=fr[3], rms_left=fl[2], rms_right=fr[2])
    return out


def rasterize(poly_px: np.ndarray, width: int, height: int) -> np.ndarray:
    """Polygon in pixels -> uint8 mask, clipped to the image as a detector's would be."""
    m = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(m, [np.round(poly_px * 16).astype(np.int32)], 1,
                 lineType=cv2.LINE_8, shift=4)
    return m
