"""Yaw and cross-track from runway masks: Samuel Moyal's estimator with the crab-angle EKF.

The estimator is the capstone-dof-estimation submodule (fork of samuelmoyal/Restart, branch
crab-ekf): `maskfit_analytic_lm` fits the runway's projected outline to each mask with the
along-track distance, height, pitch and roll taken as known (from the INS), and its kinematic
EKF tracks yaw, yaw rate, cross-track and the crab angle. The crab state matters in a
crosswind: without it the filter moves the aircraft along its nose and drifts sideways.

The runway's width and length come from the runway database (a real system has one); the
evaluation experiments instead fitted them per flight to the labelled corners.
"""

import math
import sys
from pathlib import Path

import numpy as np

SUBMODULE = Path(__file__).resolve().parents[1] / "capstone-dof-estimation"


def _restart():
    """Put the submodule on sys.path (its packages are evaluation, runway_detection, pose_estimation)."""
    if not (SUBMODULE / "evaluation").exists():
        raise ImportError(f"{SUBMODULE} is empty: run `git submodule update --init`")
    if str(SUBMODULE) not in sys.path:
        sys.path.insert(0, str(SUBMODULE))


class MaskSource:
    """The mask cache interface Samuel's runner reads masks from, over in-memory masks."""

    def __init__(self, seq_id, masks):
        self.seq_id, self.masks = seq_id, masks          # {frame index: uint8 probability x255}

    def available_indices(self, seq_id):
        return sorted(self.masks) if seq_id == self.seq_id else []

    def get(self, seq_id, index):
        m = self.masks.get(index) if seq_id == self.seq_id else None
        return None if m is None else m.astype(np.float32) / 255.0


def runway_calibration(flight, runways):
    """Samuel's per-flight geometry from the runway database: the logs are already in the
    runway frame (axis-rotation fixed), so every offset is zero."""
    _restart()
    from runway_detection.xp12.calibration import XP12Calibration
    rw = runways[(flight.airport, flight.runway)]
    return XP12Calibration(seq_id=flight.root.name, airport=flight.airport, runway=flight.runway,
                           width_m=rw.width_m, length_m=rw.length_m, d0_m=0.0, h0_m=0.0, c0_m=0.0,
                           psi0_deg=0.0, rms_px=math.nan, p95_px=math.nan, n_frames=0)


class Estimator:
    def __init__(self, crab=True, filter=True):
        _restart()
        from evaluation.configs import make_method
        from evaluation.runner import RunConfig
        self.method = make_method("maskfit_analytic_lm")
        self.cfg = RunConfig(source="yolo", mode="track", kf=filter, kf_crab=crab)
        self._geometry = {}                               # flight root -> (frames by index, scene)

    def run(self, flight, calibration, masks) -> list:
        """masks: {frame index: uint8 runway probability x255}. Returns one dict per frame with
        yaw_gt/yaw_est (deg, relative to the runway) and lat_gt/lat_est (m, + LEFT)."""
        from evaluation.runner import run_sequence
        from runway_detection.xp12.loader import load_sequence
        seq = load_sequence(flight.root.parent, flight.root.name)
        src = MaskSource(seq.seq_id, masks)
        return run_sequence(seq, calibration, self.method, self.cfg, yolo=src,
                            frame_indices=src.available_indices(seq.seq_id), log_seg_iou=False)

    def outline(self, flight, calibration, index, yaw_deg, lat_left_m):
        """The runway's image outline (4x2 px) for a given yaw and cross-track, or None."""
        from evaluation.video import _quad
        from pose_estimation.g_dof import GDofParams
        from runway_detection.xp12.calibration import pose_from_frame, scene_from_calibration
        from runway_detection.xp12.loader import load_sequence
        if flight.root not in self._geometry:
            frames = {f.index: f for f in load_sequence(flight.root.parent, flight.root.name).frames}
            self._geometry[flight.root] = (frames, scene_from_calibration(calibration))
        frames, scene = self._geometry[flight.root]
        params = GDofParams.from_pose(pose_from_frame(frames[index], calibration), scene)
        q = _quad(scene, params, yaw_deg, lat_left_m)
        return None if q is None else np.round(q).astype(np.int32)
