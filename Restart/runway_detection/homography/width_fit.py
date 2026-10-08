"""Lateral offset readout with calibrated runway width."""

from __future__ import annotations

import cv2
import numpy as np

from pose_estimation.g_dof import project_corners_local
from pose_estimation.runway_model import CameraPoseLocal, RunwayScene
from runway_detection.homography.measure import measure_from_image_corners
from runway_detection.homography.plane_homography import (
    homography_from_pose,
    image_points_to_plane,
    nominal_pose_four_dof,
    runway_plane_xy,
)
from runway_detection.homography.scene_width import scene_with_runway_width
from runway_detection.lard.projection import CORNER_NAMES


def estimate_lateral_similarity_ty(
    scene_true: RunwayScene,
    pose_gt: CameraPoseLocal,
    *,
    runway_width_m: float,
) -> float:
    """
    Lateral readout: Y translation from a 2D similarity aligning the nominal
    runway rectangle (width ``runway_width_m``) to measured plane corners.

    Unlike ``plane_y`` (threshold midpoint Y), this **does** depend on assumed
    width — but still does not reliably recover metric cross-track offset.
    """
    scene_model = scene_with_runway_width(scene_true, runway_width_m)
    pose_nom = nominal_pose_four_dof(pose_gt, scene_model)
    h_nom = homography_from_pose(scene_model, pose_nom)
    corners_px = project_corners_local(scene_true, pose_gt)
    pts = np.array([corners_px[n] for n in CORNER_NAMES], dtype=np.float64)
    meas = image_points_to_plane(pts, h_nom)
    ref = runway_plane_xy(scene_model)
    m, _ = cv2.estimateAffinePartial2D(
        ref.astype(np.float32),
        meas.astype(np.float32),
    )
    if m is None:
        return float("nan")
    return float(m[1, 2])


def estimate_lateral_plane_y(
    scene_true: RunwayScene,
    pose_gt: CameraPoseLocal,
    *,
    runway_width_m: float,
) -> float:
    """
    Lateral readout: Y of threshold midpoint in nominal plane frame.

    Uses ``runway_width_m`` in the 4-DOF homography model; image corners come
    from the true runway geometry + GT pose.
    """
    scene_model = scene_with_runway_width(scene_true, runway_width_m)
    pose_nom = nominal_pose_four_dof(pose_gt, scene_model)
    h_nom = homography_from_pose(scene_model, pose_nom)
    corners_px = project_corners_local(scene_true, pose_gt)
    est = measure_from_image_corners(corners_px, h_nom)
    return est.lateral_offset_m


def estimate_lateral_center_shift(
    scene_true: RunwayScene,
    pose_gt: CameraPoseLocal,
    *,
    runway_width_m: float,
) -> float:
    """
    Lateral readout: Y shift of runway centroid in nominal vs model plane coords.
    """
    scene_model = scene_with_runway_width(scene_true, runway_width_m)
    pose_nom = nominal_pose_four_dof(pose_gt, scene_model)
    h_nom = homography_from_pose(scene_model, pose_nom)
    corners_px = project_corners_local(scene_true, pose_gt)
    pts = np.array([corners_px[n] for n in CORNER_NAMES], dtype=np.float64)
    p_nom = image_points_to_plane(pts, h_nom)
    p_model = runway_plane_xy(scene_model)
    return float((p_nom.mean(axis=0) - p_model.mean(axis=0))[1])


_ESTIMATORS = {
    "plane_y": estimate_lateral_plane_y,
    "center_shift": estimate_lateral_center_shift,
    "similarity_ty": estimate_lateral_similarity_ty,
}


def _estimator(method: str):
    if method not in _ESTIMATORS:
        raise ValueError(f"Unknown method {method!r}; choose from {sorted(_ESTIMATORS)}")
    return _ESTIMATORS[method]


def fit_runway_width(
    samples: list[tuple[RunwayScene, CameraPoseLocal]],
    *,
    width_min_m: float | None = None,
    width_max_m: float | None = None,
    n_grid: int = 80,
    method: str = "plane_y",
) -> dict:
    """
    Grid-search runway width to match GT lateral offset on training samples.

    Returns dict with best width, DB width, errors, etc.
    """
    if not samples:
        raise ValueError("No training samples")

    db_width = samples[0][0].width_m
    lat_gts = np.array([pose.lateral_offset_m for _, pose in samples])

    w_min = width_min_m if width_min_m is not None else max(10.0, 0.3 * db_width)
    w_max = width_max_m if width_max_m is not None else 3.0 * db_width
    widths = np.linspace(w_min, w_max, n_grid)

    estimator = _estimator(method)
    best_mae = float("inf")
    for w in widths:
        preds = np.array([estimator(scene, pose, runway_width_m=w) for scene, pose in samples])
        mae = float(np.mean(np.abs(preds - lat_gts)))
        if mae < best_mae:
            best_mae = mae
            best_w = float(w)

    preds_best = np.array(
        [estimator(scene, pose, runway_width_m=best_w) for scene, pose in samples]
    )
    preds_db = np.array(
        [estimator(scene, pose, runway_width_m=db_width) for scene, pose in samples]
    )

    return {
        "method": method,
        "db_width_m": db_width,
        "fitted_width_m": best_w,
        "width_scale": best_w / db_width,
        "train_mae_m": best_mae,
        "train_mae_db_width_m": float(np.mean(np.abs(preds_db - lat_gts))),
        "widths_searched": (w_min, w_max),
    }


def evaluate_width(
    samples: list[tuple[RunwayScene, CameraPoseLocal]],
    runway_width_m: float,
    *,
    method: str = "plane_y",
) -> dict:
    estimator = _estimator(method)
    lat_gts = np.array([pose.lateral_offset_m for _, pose in samples])
    preds = np.array(
        [estimator(scene, pose, runway_width_m=runway_width_m) for scene, pose in samples]
    )
    errs = preds - lat_gts
    return {
        "n": len(samples),
        "mae_m": float(np.mean(np.abs(errs))),
        "rmse_m": float(np.sqrt(np.mean(errs**2))),
        "mean_err_m": float(np.mean(errs)),
        "max_abs_err_m": float(np.max(np.abs(errs))),
    }
