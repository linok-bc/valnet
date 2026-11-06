"""One recorded XP12 approach: frames, logged attitude and position, runway, weather.

A flight folder, as written by the X-Plane 12 landing collector:

    meta.txt              "<ICAO> <runway>" + a sim-settings line (visibility in statute miles, field 4)
    images/<i>.jpg        1280x720 camera frames
    pose/<i>.txt          yaw pitch roll (deg)
    position/<i>.txt      along height right (m; along < 0 before the threshold)
    labels/<i>.txt        runway polygon, YOLO-seg format (optional; used for IoU only)
    weather.json          weather regime and settings (weather flights only)
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

SM = 1609.344   # statute mile, m
W, H = 1280, 720


@dataclass
class Flight:
    root: Path
    airport: str
    runway: str
    visibility_m: float
    weather: str
    indices: list = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.root.parent.name.replace('xp12_dataset_', '')}/{self.root.name}"

    def image(self, i: int) -> np.ndarray:
        """BGR uint8 frame."""
        return cv2.imread(str(self.root / "images" / f"{i:06d}.jpg"))

    def position(self, i: int) -> tuple:
        """(along, height, right) in metres."""
        return tuple(map(float, (self.root / "position" / f"{i:06d}.txt").read_text().split()[:3]))

    def range_m(self, i: int) -> float:
        """Distance along the approach to the threshold; 0 once past it."""
        return max(0.0, -self.position(i)[0])

    def label_mask(self, i: int):
        """Binary runway mask from the label polygon(s), or None if unlabelled."""
        p = self.root / "labels" / f"{i:06d}.txt"
        if not p.exists():
            return None
        m = np.zeros((H, W), np.uint8)
        for line in p.read_text().splitlines():
            v = line.split()[1:]
            if len(v) >= 6:
                cv2.fillPoly(m, [np.round(np.array(v, np.float32).reshape(-1, 2) * [W, H]).astype(np.int32)], 1)
        return m.astype(bool)

    def frames(self, min_range=150.0, max_range=1500.0, visibility_filter=True) -> list:
        """Approach frames inside [min_range, max_range] m of the threshold.

        With visibility_filter, frames whose slant range is beyond half the visibility are
        dropped: there the runway keeps under ~22% contrast and the label can show a runway
        the camera cannot see.
        """
        out = []
        for i in self.indices:
            along, height, _ = self.position(i)
            if not (-max_range <= along <= -min_range):
                continue
            if visibility_filter and math.hypot(along, height) > 0.5 * self.visibility_m:
                continue
            out.append(i)
        return out


def load_flight(root) -> Flight:
    root = Path(root)
    meta = (root / "meta.txt").read_text().split("\n")
    airport, runway = meta[0].split()[:2]
    wj = root / "weather.json"
    weather = json.loads(wj.read_text())["regime"] if wj.exists() else "clear (easy)"
    indices = sorted(int(p.stem) for p in (root / "pose").glob("*.txt"))
    return Flight(root, airport, runway, float(meta[1].split()[3]) * SM, weather, indices)


@dataclass
class Runway:
    airport: str
    name: str
    width_m: float
    length_m: float


def load_runways(path) -> dict:
    """runways.dat -> {(airport, runway): Runway}. Columns: airport name elev width
    r1_name r1_lat r1_lon r2_name r2_lat r2_lon ils heading ..."""
    table = {}
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if len(f) < 12:
            continue
        lat1, lon1, lat2, lon2 = map(float, (f[5], f[6], f[8], f[9]))
        mlon = 111320.0 * math.cos(math.radians((lat1 + lat2) / 2))
        length = math.hypot((lat2 - lat1) * 111132.0, (lon2 - lon1) * mlon)   # equirectangular, <1 m here
        table[(f[0], f[1])] = Runway(f[0], f[1], float(f[3]), length)
    return table
