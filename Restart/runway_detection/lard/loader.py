"""Load LARD V2 metadata rows and resolve local image paths."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from runway_detection.lard.intrinsics import (
    DEFAULT_FOV_X_DEG,
    DEFAULT_FOV_Y_DEG,
    CameraIntrinsics,
)
from runway_detection.lard.projection import CORNER_NAMES

SOURCE_RUNWAYS_DB = {
    "flsim": "runways_db_V2_FLSim.json",
    "arcgis": "runways_db_V2_ArcGIS.json",
    "bingmaps": "runways_db_V2_Bing.json",
    "ges": "runways_db_V2_GEarth.json",
    "xplane": "runways_db_V2_XPlane.json",
}

RUNWAYS_DB_URL = (
    "https://raw.githubusercontent.com/deel-ai/LARD/LARD_V2/data/{filename}"
)


@dataclass(frozen=True)
class LardSample:
    """One annotated runway instance from a LARD metadata CSV row."""

    row_index: int
    image_path: Path
    width: int
    height: int
    source: str
    airport: str
    runway: str
    lat: float
    lon: float
    alt: float
    yaw: float
    pitch: float
    roll: float
    slant_distance_nm: float | None
    along_track_distance_nm: float | None
    height_above_runway: float | None
    lateral_path_angle: float | None
    vertical_path_angle: float | None
    runway_in_cone: str | None
    annotated_corners: dict[str, tuple[float, float]]
    intrinsics: CameraIntrinsics

    @property
    def fov_x_deg(self) -> float:
        return self.intrinsics.fov_x_deg

    @property
    def fov_y_deg(self) -> float:
        return self.intrinsics.fov_y_deg


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def metadata_path(data_root: Path, source: str, split: str) -> Path:
    return data_root / f"metadata_{source}_{split}.csv"


def load_metadata_rows(
    data_root: str | Path,
    *,
    source: str = "flsim",
    split: str = "train",
) -> list[dict[str, str]]:
    path = metadata_path(Path(data_root), source, split)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def iter_samples(
    data_root: str | Path,
    *,
    source: str = "flsim",
    split: str = "train",
    fov_x_deg: float = DEFAULT_FOV_X_DEG,
    fov_y_deg: float = DEFAULT_FOV_Y_DEG,
    require_image: bool = False,
) -> Iterator[LardSample]:
    data_root = Path(data_root)
    for idx, row in enumerate(load_metadata_rows(data_root, source=source, split=split)):
        rel_image = row["image"]
        image_path = data_root / rel_image
        if require_image and not image_path.exists():
            continue

        width = int(row["width"])
        height = int(row["height"])
        intrinsics = CameraIntrinsics.from_fov(
            width=width, height=height, fov_x_deg=fov_x_deg, fov_y_deg=fov_y_deg
        )
        annotated = {
            name: (float(row[f"x_{name}"]), float(row[f"y_{name}"]))
            for name in CORNER_NAMES
        }
        yield LardSample(
            row_index=idx,
            image_path=image_path,
            width=width,
            height=height,
            source=source,
            airport=row["airport"],
            runway=row["runway"],
            lat=float(row["lat"]),
            lon=float(row["lon"]),
            alt=float(row["alt"]),
            yaw=float(row["yaw"]),
            pitch=float(row["pitch"]),
            roll=float(row["roll"]),
            slant_distance_nm=_parse_float(row.get("slant_distance")),
            along_track_distance_nm=_parse_float(row.get("along_track_distance")),
            height_above_runway=_parse_float(row.get("height_above_runway")),
            lateral_path_angle=_parse_float(row.get("lateral_path_angle")),
            vertical_path_angle=_parse_float(row.get("vertical_path_angle")),
            runway_in_cone=row.get("runway_in_cone") or None,
            annotated_corners=annotated,
            intrinsics=intrinsics,
        )


def load_runways_database(
    data_root: str | Path,
    source: str,
    *,
    download_if_missing: bool = True,
) -> dict:
    data_root = Path(data_root)
    filename = SOURCE_RUNWAYS_DB.get(source)
    if filename is None:
        raise ValueError(f"No runways DB mapping for source {source!r}")

    db_path = data_root / filename
    if not db_path.exists():
        if not download_if_missing:
            raise FileNotFoundError(db_path)
        try:
            import json
            import urllib.request

            url = RUNWAYS_DB_URL.format(filename=filename)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading runways DB: {url}")
            urllib.request.urlretrieve(url, db_path)
        except Exception as exc:
            raise FileNotFoundError(
                f"Runways database not found at {db_path} and download failed."
            ) from exc

    import json

    return json.loads(db_path.read_text(encoding="utf-8"))
