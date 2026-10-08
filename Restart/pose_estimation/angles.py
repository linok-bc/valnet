"""Angle / pose helpers from a detected line (stub)."""

from __future__ import annotations

from typing import Any


def compute_angles_from_line(line: Any) -> dict[str, float]:
    """
    Compute pose-related angles from a centerline representation.

    Args:
        line: polyline, segment, or mask-derived line (format TBD).

    Returns:
        Mapping of angle names to degrees (or radians — TBD).

    Raises:
        NotImplementedError: until pose estimation is implemented.
    """
    raise NotImplementedError(
        "pose_estimation.angles: compute pitch/yaw/roll (or heading) from centerline — TODO"
    )
