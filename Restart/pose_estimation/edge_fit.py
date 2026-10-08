"""
Edge-based joint fit for (yaw, lateral_offset) — spec_edges.md.

Projects the two lateral runway edges from g_dof geometry and scores observed
contour points by point-to-line distance with a Huber loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from pose_estimation.g_dof import GDofParams, GDofState, g_dof_corners
from pose_estimation.runway_model import RunwayScene


@dataclass(frozen=True)
class Line2D:
    """Normalized line a·u + b·v + c = 0 with a² + b² = 1."""

    a: float
    b: float
    c: float

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        return pts[:, 0] * self.a + pts[:, 1] * self.b + self.c


@dataclass(frozen=True)
class EdgeLines:
    left: Line2D
    right: Line2D


def line_from_endpoints(p_near: np.ndarray, p_far: np.ndarray) -> Line2D:
    """Build a normalized 2-D line through two image points (spec_edges §1)."""
    d = np.asarray(p_far, dtype=float) - np.asarray(p_near, dtype=float)
    norm = math.hypot(float(d[0]), float(d[1]))
    if norm < 1e-9:
        raise ValueError("Degenerate edge line: coincident endpoints")
    a = -d[1] / norm
    b = d[0] / norm
    p = np.asarray(p_near, dtype=float)
    c = -(a * p[0] + b * p[1])
    return Line2D(a, b, c)


def edge_lines_from_dof(
    scene: RunwayScene,
    params: GDofParams,
    state: GDofState,
) -> EdgeLines:
    """
    Left / right lateral edges from projected runway corners.

    Left edge (positive Y in runway frame): TL → BL.
    Right edge: TR → BR.
    """
    corners = g_dof_corners(scene, params, state)
    tl = np.array(corners["TL"], dtype=float)
    bl = np.array(corners["BL"], dtype=float)
    tr = np.array(corners["TR"], dtype=float)
    br = np.array(corners["BR"], dtype=float)
    return EdgeLines(
        left=line_from_endpoints(tl, bl),
        right=line_from_endpoints(tr, br),
    )


def huber(x: np.ndarray, delta: float) -> np.ndarray:
    """Element-wise Huber loss ρ(x)."""
    ax = np.abs(x)
    quad = 0.5 * x * x
    lin = delta * (ax - 0.5 * delta)
    return np.where(ax <= delta, quad, lin)


def extract_lateral_contour_points(
    mask: np.ndarray,
    *,
    min_tangent_ratio: float = 0.5,
    max_points: int = 400,
) -> np.ndarray:
    """
    Contour pixels likely belonging to lateral edges (not near/far short edges).

    Drops points whose local tangent is near-horizontal in image coordinates.
    """
    binary = (np.asarray(mask) > 0.25).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2), dtype=float)

    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    if contour.shape[0] < 5:
        return contour

    kept: list[np.ndarray] = []
    n = contour.shape[0]
    for i in range(n):
        prev_pt = contour[(i - 1) % n]
        next_pt = contour[(i + 1) % n]
        tangent = next_pt - prev_pt
        if abs(tangent[1]) < abs(tangent[0]) * min_tangent_ratio:
            continue
        kept.append(contour[i])

    if not kept:
        kept = [contour[i] for i in range(n)]

    pts = np.asarray(kept, dtype=float)
    if pts.shape[0] > max_points:
        idx = np.linspace(0, pts.shape[0] - 1, max_points, dtype=int)
        pts = pts[idx]
    return pts


def assign_contour_to_edges(
    points: np.ndarray,
    lines: EdgeLines,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each contour point to the nearer expected lateral edge."""
    if points.size == 0:
        z = np.zeros((0, 2), dtype=float)
        return z, z

    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    d_left = np.abs(lines.left.signed_distance(pts))
    d_right = np.abs(lines.right.signed_distance(pts))
    left_mask = d_left <= d_right
    return pts[left_mask], pts[~left_mask]


def prepare_edge_observation(
    mask: np.ndarray,
    scene: RunwayScene,
    params: GDofParams,
    assign_state: GDofState,
    *,
    min_tangent_ratio: float = 0.5,
    max_points: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract contour points and split left/right using projected edges at assign_state."""
    pts = extract_lateral_contour_points(
        mask,
        min_tangent_ratio=min_tangent_ratio,
        max_points=max_points,
    )
    lines = edge_lines_from_dof(scene, params, assign_state)
    return assign_contour_to_edges(pts, lines)


def edge_residuals(
    scene: RunwayScene,
    params: GDofParams,
    state: GDofState,
    left_pts: np.ndarray,
    right_pts: np.ndarray,
    *,
    huber_delta_px: float = 3.0,
) -> np.ndarray:
    """Stack signed point-to-line distances (Huber-transformed) for LM."""
    lines = edge_lines_from_dof(scene, params, state)
    parts: list[np.ndarray] = []
    for pts, line in ((left_pts, lines.left), (right_pts, lines.right)):
        if pts.size == 0:
            continue
        d = line.signed_distance(pts)
        parts.append(np.sqrt(2.0 * huber(d, huber_delta_px)))
    if not parts:
        return np.zeros(0, dtype=float)
    return np.concatenate(parts)


def edge_cost(
    scene: RunwayScene,
    params: GDofParams,
    state: GDofState,
    left_pts: np.ndarray,
    right_pts: np.ndarray,
    *,
    huber_delta_px: float = 3.0,
) -> float:
    """Mean Huber point-to-line cost over all assigned contour points."""
    res = edge_residuals(
        scene,
        params,
        state,
        left_pts,
        right_pts,
        huber_delta_px=huber_delta_px,
    )
    if res.size == 0:
        return float("inf")
    # residuals are sqrt(2*rho); square back for mean Huber loss.
    return float(np.mean(0.5 * res * res))
