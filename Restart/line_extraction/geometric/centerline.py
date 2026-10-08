"""
Geometric centerline: runway polygon → thin triangle for YOLO-seg labels / downstream use.

Migrated from CapstoneV2.ipynb / notebook_big_train.ipynb.
"""

from __future__ import annotations

import numpy as np

# Default used for training-oriented label prep (notebook_big_train centerline_triangle).
DEFAULT_HALF_WIDTH = 0.0005


def parse_polygon(coords_flat) -> np.ndarray:
    """Liste [x0,y0,x1,y1,...] → array (N,2)."""
    return np.array(coords_flat, dtype=float).reshape(-1, 2)


def top_mid_bottom_mid(pts: np.ndarray):
    """
    Retourne (top_mid, bot_mid) où :
      - top_mid = milieu des points ayant y minimal (côté haut)
      - bot_mid = milieu des points ayant y maximal (côté bas)
    On sélectionne les 2 points les plus proches du min/max de y.
    """
    ys = pts[:, 1]
    top_idx = np.argsort(ys)[:2]
    top_mid = pts[top_idx].mean(axis=0)
    bot_idx = np.argsort(ys)[-2:]
    bot_mid = pts[bot_idx].mean(axis=0)
    return top_mid, bot_mid


def centerline_triangle(pts: np.ndarray, half_width: float = DEFAULT_HALF_WIDTH):
    """
    Construit le triangle centerline :
      - sommet A : milieu du côté haut
      - sommet B : milieu côté bas décalé à gauche de half_width
      - sommet C : milieu côté bas décalé à droite de half_width
    """
    top_mid, bot_mid = top_mid_bottom_mid(pts)
    A = top_mid
    B = np.array([bot_mid[0] - half_width, bot_mid[1]])
    C = np.array([bot_mid[0] + half_width, bot_mid[1]])
    return A, B, C


def format_triangle_line(A, B, C, class_id: int = 0) -> str:
    coords = [A[0], A[1], B[0], B[1], C[0], C[1]]
    coords_str = " ".join(f"{v:.6f}" for v in coords)
    return f"{class_id} {coords_str}"


def is_centerline(coords) -> bool:
    """Un triangle = exactement 3 points = 6 coordonnées."""
    return len(coords) == 6
