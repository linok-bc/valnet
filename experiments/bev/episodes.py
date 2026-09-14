"""Readers for an xp12 collector dataset, plus the runway table it was built from.

Layout produced by the collector's convert_to_yolo.py, one directory per episode:

    000001/meta.txt          "ICAO RWY" then a line of environment conditions
    000001/images/000000.jpg
    000001/labels/000000.txt "0 x1 y1 x2 y2 ..."  normalized, may be absent/empty
    000001/pose/000000.txt   "yaw pitch roll"     degrees
    000001/position/000000.txt "along height right"  metres

Sign conventions, copied from DatasetCollector.py and NOT guessed:

    yaw    positive -> nose right of the runway heading
    pitch  positive -> nose up
    roll   positive -> right wing down
    along  metres past the threshold, NEGATIVE on approach
    height metres above the runway surface at the threshold
    right  metres right of the centreline, facing down the runway

The old (pre-regeneration) dataset wrote 42-point edge polygons and .png frames;
the current one writes a 4-corner quad as .jpg. Both load here.
"""

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

DEFAULT_RUNWAYS_DAT = Path(
    "/home/linok/.local/share/Steam/steamapps/common/X-Plane 12/Resources/plugins"
    "/PythonPlugins/xplane-12-landing-collector/utils/runways.dat"
)


@dataclass(frozen=True)
class Runway:
    """One row of runways.dat. Lengths in metres, angles in degrees."""

    airport: str
    name: str
    elev_m: float
    width_m: float
    r1_lat: float
    r1_lon: float
    r2_lat: float
    r2_lon: float
    heading_deg: float

    @property
    def length_m(self) -> float:
        """Threshold-to-threshold distance, equirectangular (good to <1 m here)."""
        mlat = 111132.0
        mlon = 111320.0 * math.cos(math.radians((self.r1_lat + self.r2_lat) / 2))
        return math.hypot((self.r2_lat - self.r1_lat) * mlat,
                          (self.r2_lon - self.r1_lon) * mlon)

    def corners_runway_frame(self) -> np.ndarray:
        """The runway rectangle in runway coordinates (X down-runway, Y right).

        Order matches the collector's polygon: near-left, far-left, far-right,
        near-right, with "near" the threshold the aircraft is approaching.
        """
        w, L = self.width_m / 2, self.length_m
        return np.array([[0.0, -w], [L, -w], [L, w], [0.0, w]])


def load_runways(path=None) -> dict:
    """Parse runways.dat into {(airport, runway_name): Runway}.

    Columns: airport name elev width r1_name r1_lat r1_lon r2_name r2_lat r2_lon
    ils heading.
    """
    table = {}
    for line in Path(path or DEFAULT_RUNWAYS_DAT).read_text().splitlines():
        f = line.split()
        if len(f) < 12:
            continue
        table[(f[0], f[1])] = Runway(
            airport=f[0], name=f[1], elev_m=float(f[2]), width_m=float(f[3]),
            r1_lat=float(f[5]), r1_lon=float(f[6]),
            r2_lat=float(f[8]), r2_lon=float(f[9]), heading_deg=float(f[11]),
        )
    return table


@dataclass
class Frame:
    """One captured frame and everything logged alongside it."""

    episode: "Episode"
    stem: str
    image_path: Path
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    along_m: float
    height_m: float
    right_m: float
    polygon_norm: np.ndarray          # (n, 2) in [0, 1], empty if unlabelled

    @property
    def labelled(self) -> bool:
        return len(self.polygon_norm) >= 3

    @property
    def range_m(self) -> float:
        """Distance to the threshold along the runway. Positive on approach."""
        return -self.along_m

    def polygon_px(self, width: int, height: int) -> np.ndarray:
        return self.polygon_norm * np.array([width, height], dtype=np.float64)


class Episode:
    """One landing, lazily read."""

    def __init__(self, path):
        self.path = Path(path)
        meta = (self.path / "meta.txt").read_text().splitlines()
        self.airport, self.runway_name = meta[0].split()[:2]
        self.conditions = meta[1].split() if len(meta) > 1 else []
        self._stems = None

    def __repr__(self):
        return f"Episode({self.path.name}, {self.airport} {self.runway_name})"

    @property
    def stems(self):
        if self._stems is None:
            imgs = [p for p in (self.path / "images").iterdir()
                    if p.suffix.lower() in IMAGE_SUFFIXES]
            self._stems = sorted(p.stem for p in imgs)
        return self._stems

    def image_path(self, stem) -> Path:
        for suf in IMAGE_SUFFIXES:
            p = self.path / "images" / f"{stem}{suf}"
            if p.exists():
                return p
        raise FileNotFoundError(f"no image for {self.path.name}/{stem}")

    def frame(self, stem) -> Frame:
        yaw, pitch, roll = _read_floats(self.path / "pose" / f"{stem}.txt")
        along, height, right = _read_floats(self.path / "position" / f"{stem}.txt")
        return Frame(
            episode=self, stem=stem, image_path=self.image_path(stem),
            yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
            along_m=along, height_m=height, right_m=right,
            polygon_norm=_read_label(self.path / "labels" / f"{stem}.txt"),
        )

    def frames(self, stride: int = 1):
        for stem in self.stems[::stride]:
            yield self.frame(stem)


def find_episodes(root):
    """Every directory under `root` that looks like a collector episode."""
    root = Path(root)
    out = [p for p in sorted(root.iterdir())
           if p.is_dir() and (p / "meta.txt").exists() and (p / "images").is_dir()]
    return [Episode(p) for p in out]


def _read_floats(path) -> list:
    return [float(v) for v in Path(path).read_text().split()]


def _read_label(path) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        return np.zeros((0, 2))
    vals = p.read_text().split()
    if len(vals) < 7:                       # class id + at least 3 points
        return np.zeros((0, 2))
    return np.array([float(v) for v in vals[1:]], dtype=np.float64).reshape(-1, 2)


def runway_to_ground(points_runway: np.ndarray, along_m: float, right_m: float,
                     yaw_deg: float) -> np.ndarray:
    """Runway-frame points -> ground frame centred on the aircraft.

    The ground frame is the one experiments.bev.transform warps into: X forward
    along the aircraft's HEADING (not the runway), Y to its right. This is the
    bridge between the position/pose logs and anything measured off a BEV image.

    A threshold 1 km ahead with the aircraft 10 m right of the centreline and no
    yaw error gives (1000, -10): the runway sits 10 m to the LEFT in the image.
    """
    psi = math.radians(yaw_deg)
    d = np.asarray(points_runway, dtype=np.float64).reshape(-1, 2) - np.array([along_m, right_m])
    fwd = np.array([math.cos(psi), math.sin(psi)])      # ground +X in runway axes
    rgt = np.array([-math.sin(psi), math.cos(psi)])     # ground +Y in runway axes
    return np.column_stack([d @ fwd, d @ rgt])


def ground_to_runway(points_ground: np.ndarray, along_m: float, right_m: float,
                     yaw_deg: float) -> np.ndarray:
    """Inverse of runway_to_ground."""
    psi = math.radians(yaw_deg)
    g = np.asarray(points_ground, dtype=np.float64).reshape(-1, 2)
    fwd = np.array([math.cos(psi), math.sin(psi)])
    rgt = np.array([-math.sin(psi), math.cos(psi)])
    return g @ np.column_stack([fwd, rgt]).T + np.array([along_m, right_m])
