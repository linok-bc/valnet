"""Warp runway polygons into the BEV frame and emit YOLO label formats.

Ultralytics label formats used here, all normalized to [0, 1]:

    segment : cls x1 y1 x2 y2 ... xn yn      (polygon)
    obb     : cls x1 y1 x2 y2 x3 y3 x4 y4    (4 corners, any rotation)
    detect  : cls xc yc w h                  (axis-aligned)

In BEV the runway is a rectangle rotated by the heading error, so `obb` fits it
exactly while `detect` cannot: an axis-aligned box around a rotated rectangle
is larger than the runway and its centre is only unbiased when the heading
error is zero. That difference is the experiment.
"""

import cv2
import numpy as np


def warp_points(pts_px: np.ndarray, m_bev: np.ndarray):
    """Map image pixels -> BEV pixels.

    Returns (points, valid). A point is invalid when it lies on or beyond the
    horizon, where the homography sends it to infinity or behind the camera --
    which is exactly what happens to runway corners in frames where the near
    plane clipping has already corrupted the label.
    """
    pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
    h = np.column_stack([pts, np.ones(len(pts))]) @ m_bev.T
    valid = h[:, 2] > 1e-9
    out = np.full_like(pts, np.nan)
    out[valid] = h[valid, :2] / h[valid, 2:3]
    return out, valid


def clip_to_canvas(poly: np.ndarray, width: int, height: int) -> np.ndarray:
    """Sutherland-Hodgman clip against the BEV canvas rectangle."""
    def inside(p, edge):
        if edge == 0: return p[0] >= 0
        if edge == 1: return p[0] <= width - 1
        if edge == 2: return p[1] >= 0
        return p[1] <= height - 1

    def intersect(a, b, edge):
        if edge in (0, 1):
            x = 0.0 if edge == 0 else width - 1.0
            t = (x - a[0]) / (b[0] - a[0])
            return np.array([x, a[1] + t * (b[1] - a[1])])
        y = 0.0 if edge == 2 else height - 1.0
        t = (y - a[1]) / (b[1] - a[1])
        return np.array([a[0] + t * (b[0] - a[0]), y])

    out = [np.asarray(p, dtype=np.float64) for p in poly]
    for edge in range(4):
        if not out:
            return np.zeros((0, 2))
        buf, prev = [], out[-1]
        for cur in out:
            if inside(cur, edge):
                if not inside(prev, edge):
                    buf.append(intersect(prev, cur, edge))
                buf.append(cur)
            elif inside(prev, edge):
                buf.append(intersect(prev, cur, edge))
            prev = cur
        out = buf
    return np.asarray(out, dtype=np.float64).reshape(-1, 2)


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def to_segment_label(poly: np.ndarray, width: int, height: int, cls: int = 0) -> str:
    n = poly / np.array([width, height])
    n = np.clip(n, 0.0, 1.0)
    return f"{cls} " + " ".join(f"{v:.6g}" for v in n.reshape(-1))


def to_obb_label(poly: np.ndarray, width: int, height: int, cls: int = 0) -> str:
    """Minimum-area rotated rectangle around the polygon, as 4 normalized corners."""
    box = cv2.boxPoints(cv2.minAreaRect(poly.astype(np.float32)))
    n = np.clip(box / np.array([width, height]), 0.0, 1.0)
    return f"{cls} " + " ".join(f"{v:.6g}" for v in n.reshape(-1))


def to_detect_label(poly: np.ndarray, width: int, height: int, cls: int = 0) -> str:
    x0, y0 = poly.min(axis=0)
    x1, y1 = poly.max(axis=0)
    xc, yc = (x0 + x1) / 2 / width, (y0 + y1) / 2 / height
    w, h = (x1 - x0) / width, (y1 - y0) / height
    return f"{cls} {xc:.6g} {yc:.6g} {w:.6g} {h:.6g}"


FORMATTERS = {
    "segment": to_segment_label,
    "obb": to_obb_label,
    "detect": to_detect_label,
}


def obb_geometry(poly: np.ndarray, mpp: float, grid_width: int, grid_height: int,
                 x_near: float):
    """Metric description of the fitted rotated box, for diagnostics.

    Returns dict with the runway's width and length in metres, its centre in
    ground coordinates (forward X, lateral Y), and the heading error in degrees
    measured from BEV "up". These are the quantities a downstream estimator
    wants, so having them here makes it cheap to build an oracle and check what
    a perfect detector would give you.
    """
    (cx, cy), (w, h), ang = cv2.minAreaRect(poly.astype(np.float32))
    short, long = (w, h) if w <= h else (h, w)
    X = x_near + (grid_height - 1 - cy) * mpp
    Y = (cx - (grid_width - 1) / 2.0) * mpp
    heading = ang if w <= h else ang + 90.0
    heading = (heading + 90.0) % 180.0 - 90.0
    return {
        "width_m": short * mpp,
        "length_m": long * mpp,
        "center_forward_m": X,
        "center_lateral_m": Y,
        "heading_deg": heading,
    }
