"""End to end on one flight: frames -> DeepLabV3+ runway masks -> yaw and cross-track.

    from e2e_runway_dof_estimation import pipeline
    result = pipeline.run("weather_000613")          # a sample name, or a flight folder
    print(result.summary())

Cross-track is reported + RIGHT of the centreline (Samuel's estimator uses + left).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from e2e_runway_dof_estimation import assets
from e2e_runway_dof_estimation.data import Flight, load_flight, load_runways
from e2e_runway_dof_estimation.estimate import Estimator, runway_calibration

RANGE_BINS = [150, 300, 600, 1000, 1500]


@dataclass
class Result:
    flight: Flight
    calibration: object
    records: list                                        # one dict per frame, in order
    masks: dict = field(repr=False)                      # frame index -> uint8 probability x255

    def frame(self, index):
        return next(r for r in self.records if r["frame"] == index)

    def summary(self) -> dict:
        def mae(rs, k):
            v = np.abs([r[k] for r in rs if np.isfinite(r[k])])
            return float(v.mean()) if len(v) else float("nan")

        out = {"flight": self.flight.name, "airport": f"{self.flight.airport} {self.flight.runway}",
               "weather": self.flight.weather, "frames": len(self.records),
               "yaw_mae_deg": mae(self.records, "yaw_err"), "xtrack_mae_m": mae(self.records, "xt_err"),
               "mask_iou": mae(self.records, "iou"), "by_range": {}}
        for lo, hi in zip(RANGE_BINS[:-1], RANGE_BINS[1:]):
            rs = [r for r in self.records if lo <= r["range_m"] < hi]
            if rs:
                out["by_range"][f"{lo}-{hi} m"] = {"frames": len(rs), "yaw_mae_deg": mae(rs, "yaw_err"),
                                                  "xtrack_mae_m": mae(rs, "xt_err"), "mask_iou": mae(rs, "iou")}
        return out


def resolve(flight) -> Path:
    """A sample name (assets.SAMPLES) or a path to a flight folder."""
    return assets.sample_flight(flight) if str(flight) in assets.SAMPLES else Path(flight)


def run(flight, segmenter=None, estimator=None, runways=None, min_range=150.0, max_range=1500.0,
        visibility_filter=True, batch=4, progress=None) -> Result:
    """progress(done, total) is called after each segmentation batch."""
    from e2e_runway_dof_estimation.segment import Segmenter
    flight = load_flight(resolve(flight))
    segmenter = segmenter or Segmenter()
    estimator = estimator or Estimator()
    runways = runways or load_runways(assets.runways_dat())
    calibration = runway_calibration(flight, runways)

    idx = flight.frames(min_range, max_range, visibility_filter)
    if not idx:
        raise ValueError(f"{flight.name}: no frames between {min_range:.0f} and {max_range:.0f} m"
                         + (" within half the visibility" if visibility_filter else ""))
    masks = {}
    for k in range(0, len(idx), batch):
        chunk = idx[k:k + batch]
        for i, p in zip(chunk, segmenter([flight.image(i) for i in chunk])):
            masks[i] = np.round(p * 255).astype(np.uint8)
        if progress:
            progress(min(k + batch, len(idx)), len(idx))

    records = []
    for r in estimator.run(flight, calibration, masks):
        i = int(r["frame"])
        gt = flight.label_mask(i)
        pred = masks[i] > 127
        iou = float((gt & pred).sum() / max((gt | pred).sum(), 1)) if gt is not None else float("nan")
        records.append({"frame": i, "range_m": abs(r["along_m"]), "height_m": r["height_m"],
                        "yaw_gt": r["yaw_gt"], "yaw_est": r["yaw_est"], "yaw_err": r["yaw_err"],
                        "xt_gt": -r["lat_gt"], "xt_est": -r["lat_est"], "xt_err": -r["lat_err"],
                        "iou": iou, "status": r["status"]})
    return Result(flight, calibration, records, masks)
