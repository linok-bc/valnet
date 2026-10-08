"""Pose estimation: g_dof render-and-compare and mask-space optimization."""

from pose_estimation.g_dof import (
    GDofParams,
    GDofState,
    camera_pose_from_g_dof,
    corner_pixel_error,
    g_dof,
    g_dof_corners,
    g_dof_jacobian_corners,
    mask_iou,
    project_corners_local,
    render_runway_mask,
)
from pose_estimation.mask_corrupt import CorruptionConfig, corrupt_mask
from pose_estimation.metrics import MaskMetric, dice_loss, l2_loss, soft_dice
from pose_estimation.optimize import (
    CoarseLevel,
    OptimizerConfig,
    PoseEstimate,
    SearchBounds,
    bounds_around,
    estimate_pose_mask_space,
    hierarchical_coarse_search,
)
from pose_estimation.roi import ROIBox, roi_from_mask
from pose_estimation.runway_model import CameraPoseLocal, RunwayScene

__all__ = [
    "CameraPoseLocal",
    "CorruptionConfig",
    "GDofParams",
    "GDofState",
    "MaskMetric",
    "CoarseLevel",
    "PoseEstimate",
    "ROIBox",
    "RunwayScene",
    "SearchBounds",
    "bounds_around",
    "camera_pose_from_g_dof",
    "corner_pixel_error",
    "corrupt_mask",
    "dice_loss",
    "hierarchical_coarse_search",
    "OptimizerConfig",
    "estimate_pose_mask_space",
    "g_dof",
    "g_dof_corners",
    "g_dof_jacobian_corners",
    "l2_loss",
    "mask_iou",
    "project_corners_local",
    "render_runway_mask",
    "roi_from_mask",
    "soft_dice",
]
