"""Group LARD images by landing scenario / temporal frame."""

from __future__ import annotations

import re
from pathlib import Path

_SCENARIO_RE = re.compile(
    r"^(?P<scenario>.+)_(?P<frame>\d{3})_(?P<source>[^.]+)\.jpg$",
    re.IGNORECASE,
)


def parse_lard_image_path(path: str | Path) -> dict[str, str]:
    """Parse ``{scenario}_{frame:03d}_{Source}.jpg`` from a LARD image path."""
    match = _SCENARIO_RE.match(Path(path).name)
    if not match:
        return {}
    return match.groupdict()


def same_landing_scene(
    path_a: str | Path,
    path_b: str | Path,
    *,
    same_frame: bool = False,
) -> bool:
    """
    Return True when two images belong to the same landing trajectory.

    Set ``same_frame=True`` to require the same timestep (e.g. flsim vs xplane).
    """
    a = parse_lard_image_path(path_a)
    b = parse_lard_image_path(path_b)
    if not a or not b or a["scenario"] != b["scenario"]:
        return False
    if same_frame:
        return a["frame"] == b["frame"]
    return True
