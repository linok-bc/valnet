"""
X-Plane 12 approach sequences (``xp12_dataset``).

Layout (one folder per approach video)::

    xp12_dataset/<seq>/meta.txt            "<ICAO> <runway>" + sim settings line
    xp12_dataset/<seq>/pose/<i>.txt        heading_rel_deg pitch_deg roll_deg
    xp12_dataset/<seq>/position/<i>.txt    x_m y_m z_m (runway frame, see below)
    xp12_dataset/<seq>/labels/<i>.txt      0 x_NL y_NL x_FL y_FL x_FR y_FR x_NR y_NR  (normalized)
    xp12_dataset/<seq>/images/<i>.jpg      1280×720 render

Conventions (recovered by fitting the GT corners, see ``calibration.py``):

- ``position``: x = along runway axis (negative before threshold), y = height (m),
  z = cross-track, **positive = right** of centerline.
- ``pose``: heading relative to runway (deg, + = nose right), pitch (+ = nose up),
  roll (deg). Rotation order yaw → pitch → roll, same as LARD.
- Camera: pinhole, HFOV = 65°, principal point at image center, square pixels.

Metadata and labels are small and read from an extracted folder; images are
read straight from the zip so the 11 GB archive never needs extracting.
"""

from __future__ import annotations

import math
import zipfile
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np

from runway_detection.lard.intrinsics import CameraIntrinsics

XP12_WIDTH = 1280
XP12_HEIGHT = 720
XP12_FOV_X_DEG = 65.0
ZIP_PREFIX = "xp12_dataset"

# Label corner order in the txt files → LARD corner names (T = far end, B = threshold).
LABEL_CORNER_ORDER = ("BL", "TL", "TR", "BR")


def xp12_intrinsics() -> CameraIntrinsics:
    fx = XP12_WIDTH / (2.0 * math.tan(math.radians(XP12_FOV_X_DEG / 2.0)))
    fov_y_deg = math.degrees(2.0 * math.atan(XP12_HEIGHT / (2.0 * fx)))
    return CameraIntrinsics.from_fov(
        width=XP12_WIDTH,
        height=XP12_HEIGHT,
        fov_x_deg=XP12_FOV_X_DEG,
        fov_y_deg=fov_y_deg,
    )


@dataclass(frozen=True)
class XP12Frame:
    seq_id: str
    index: int
    heading_rel_deg: float
    pitch_deg: float
    roll_deg: float
    x_m: float
    y_m: float
    z_m: float
    corners_px: dict[str, tuple[float, float]]  # TR, TL, BL, BR

    @property
    def name(self) -> str:
        return f"{self.index:06d}"


@dataclass
class XP12Sequence:
    seq_id: str
    airport: str
    runway: str
    meta_line: str
    frames: list[XP12Frame]
    root: Path
    zip_path: Path | None = None
    _zip: zipfile.ZipFile | None = field(default=None, repr=False)

    @property
    def label(self) -> str:
        return f"{self.seq_id} {self.airport} {self.runway}"

    def __len__(self) -> int:
        return len(self.frames)

    def image_path(self, frame: XP12Frame) -> Path:
        return self.root / self.seq_id / "images" / f"{frame.name}.jpg"

    def read_image(self, frame: XP12Frame) -> np.ndarray:
        """BGR image, from the extracted folder if present, else from the zip."""
        path = self.image_path(frame)
        if path.exists():
            img = cv2.imread(str(path))
            if img is not None:
                return img
        if self.zip_path is None:
            raise FileNotFoundError(path)
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.zip_path)
        data = self._zip.read(f"{ZIP_PREFIX}/{self.seq_id}/images/{frame.name}.jpg")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not decode {self.seq_id}/{frame.name}.jpg")
        return img

    @cached_property
    def arrays(self) -> dict[str, np.ndarray]:
        """Stacked per-frame arrays: pose (N,3), position (N,3), corners (N,4,2) TR,TL,BL,BR."""
        return {
            "pose": np.array([[f.heading_rel_deg, f.pitch_deg, f.roll_deg] for f in self.frames]),
            "position": np.array([[f.x_m, f.y_m, f.z_m] for f in self.frames]),
            "corners": np.array(
                [[f.corners_px[n] for n in ("TR", "TL", "BL", "BR")] for f in self.frames]
            ),
        }


def _read_floats(path: Path) -> list[float]:
    return [float(v) for v in path.read_text().split()]


def _parse_label(path: Path) -> dict[str, tuple[float, float]]:
    vals = path.read_text().split()
    if len(vals) < 9:
        raise ValueError(f"Malformed label {path}")
    pts = np.array(vals[1:9], dtype=float).reshape(4, 2) * [XP12_WIDTH, XP12_HEIGHT]
    return {name: (float(p[0]), float(p[1])) for name, p in zip(LABEL_CORNER_ORDER, pts)}


def list_sequences(root: str | Path) -> list[str]:
    root = Path(root)
    return sorted(p.name for p in root.iterdir() if (p / "meta.txt").exists())


def load_sequence(
    root: str | Path,
    seq_id: str,
    *,
    zip_path: str | Path | None = None,
) -> XP12Sequence:
    root = Path(root)
    d = root / seq_id
    meta = (d / "meta.txt").read_text().strip().splitlines()
    airport, runway = meta[0].split()[:2]
    frames: list[XP12Frame] = []
    for pose_path in sorted((d / "pose").glob("*.txt")):
        idx = int(pose_path.stem)
        heading, pitch, roll = _read_floats(pose_path)[:3]
        x, y, z = _read_floats(d / "position" / pose_path.name)[:3]
        corners = _parse_label(d / "labels" / pose_path.name)
        frames.append(
            XP12Frame(seq_id, idx, heading, pitch, roll, x, y, z, corners)
        )
    return XP12Sequence(
        seq_id=seq_id,
        airport=airport,
        runway=runway,
        meta_line=meta[1] if len(meta) > 1 else "",
        frames=frames,
        root=root,
        zip_path=Path(zip_path) if zip_path else None,
    )
