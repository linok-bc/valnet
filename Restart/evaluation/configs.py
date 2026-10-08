"""
Named estimator variants and ablation suites for the XP12 evaluation.

A *run* = (method name, RunConfig overrides, sequence selection). Suites are
lists of runs sharing a frame selection so every method sees the same frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from evaluation.estimators import COLD_WIDE_LEVELS, CornerFit, EdgeFit, Estimator, Homography, MaskFit, PnP
from evaluation.runner import RunConfig
from pose_estimation.metrics import MaskMetric
from pose_estimation.optimize import DEFAULT_COARSE_LEVELS, WARM_START_COARSE_LEVELS, OptimizerConfig

_ORIG = OptimizerConfig()  # repo defaults: raster renderer, L2, LM (early stop at ΔL2 < 0.03)
_AN = replace(_ORIG, renderer="analytic")
_AN_LM = replace(_AN, lm_min_cost_drop=0.0, max_refine_iters=30)


def make_method(name: str) -> Estimator:
    table = {
        # --- render-and-compare (mask space)
        "maskfit_orig": lambda: MaskFit(_ORIG, COLD_WIDE_LEVELS, WARM_START_COARSE_LEVELS, name="maskfit_orig"),
        "maskfit_orig_defaultgrid": lambda: MaskFit(_ORIG, DEFAULT_COARSE_LEVELS, WARM_START_COARSE_LEVELS, name="maskfit_orig_defaultgrid"),
        "maskfit_analytic": lambda: MaskFit(_AN, COLD_WIDE_LEVELS, WARM_START_COARSE_LEVELS, name="maskfit_analytic"),
        "maskfit_analytic_lm": lambda: MaskFit(_AN_LM, COLD_WIDE_LEVELS, None, name="maskfit_analytic_lm"),
        "maskfit_analytic_lm_grid": lambda: MaskFit(_AN_LM, COLD_WIDE_LEVELS, WARM_START_COARSE_LEVELS, name="maskfit_analytic_lm_grid"),
        "maskfit_dice": lambda: MaskFit(replace(_AN_LM, metric=MaskMetric.DICE), COLD_WIDE_LEVELS, None, name="maskfit_dice"),
        "maskfit_gd": lambda: MaskFit(replace(_AN_LM, refine_method="gd"), COLD_WIDE_LEVELS, None, name="maskfit_gd"),
        "maskfit_ds32": lambda: MaskFit(replace(_AN_LM, downsample_size=32), COLD_WIDE_LEVELS, None, name="maskfit_ds32"),
        "maskfit_ds96": lambda: MaskFit(replace(_AN_LM, downsample_size=96), COLD_WIDE_LEVELS, None, name="maskfit_ds96"),
        "maskfit_noblur": lambda: MaskFit(replace(_AN_LM, render_blur_sigma=0.0), COLD_WIDE_LEVELS, None, name="maskfit_noblur"),
        # --- geometric baselines on contour / corners
        "edgefit": lambda: EdgeFit(),
        "edgefit_priorassign": lambda: EdgeFit(assign="prior", name="edgefit_priorassign"),
        "cornerfit": lambda: CornerFit(),
        "homography": lambda: Homography(iterations=1),
        "homography_it5": lambda: Homography(iterations=5, name="homography_it5"),
        "pnp": lambda: PnP(),
    }
    if name not in table:
        raise KeyError(f"Unknown method {name!r}; choose from {sorted(table)}")
    return table[name]()


@dataclass
class Run:
    label: str
    method: str
    cfg: RunConfig


@dataclass
class Suite:
    name: str
    runs: list[Run]
    sequences: str = "all"  # "all" | "yolo" (those with cached YOLO masks) | "yolo_every3"
    frame_stride: int = 1
    description: str = ""
    notes: dict = field(default_factory=dict)


NOMINAL_DOF_NOISE = {"pitch_deg": 0.2, "roll_deg": 0.2, "height_m": 2.0, "along_m": 15.0}
HIGH_DOF_NOISE = {k: 3 * v for k, v in NOMINAL_DOF_NOISE.items()}


def _runs(methods, **cfg) -> list[Run]:
    return [Run(m, m, RunConfig(**cfg)) for m in methods]


def build_suite(name: str) -> Suite:
    if name == "solver_cold":
        methods = [
            "maskfit_orig", "maskfit_orig_defaultgrid", "maskfit_analytic", "maskfit_analytic_lm",
            "maskfit_dice", "maskfit_gd", "maskfit_ds32", "maskfit_ds96", "maskfit_noblur",
            "edgefit", "cornerfit", "homography", "homography_it5", "pnp",
        ]  # cold: edgefit's two assignment rules coincide (both start from cornerfit)
        return Suite(name, _runs(methods, source="gt", mode="cold"), "all", 30,
                     "Single-frame accuracy on oracle (GT) masks, no temporal prior.")
    if name == "solver_cold_yolo":
        methods = ["maskfit_analytic", "maskfit_analytic_lm", "maskfit_dice", "edgefit", "cornerfit",
                   "homography_it5", "pnp"]
        return Suite(name, _runs(methods, source="yolo", mode="cold"), "yolo", 10,
                     "Single-frame accuracy on real YOLO masks, no temporal prior.")
    if name == "tracking":
        runs = []
        for src in ("gt", "yolo"):
            for m in ["maskfit_analytic_lm", "edgefit", "edgefit_priorassign",
                      "cornerfit", "homography", "homography_it5", "pnp"]:
                runs.append(Run(f"{m}|{src}", m, RunConfig(source=src, mode="track")))
            for m in ["maskfit_analytic_lm", "cornerfit", "edgefit", "homography_it5", "pnp"]:
                runs.append(Run(f"{m}+ekf|{src}", m, RunConfig(source=src, mode="track", kf=True)))
        return Suite(name, runs, "yolo", 2, "Sequential tracking (t−1 warm start) ± Kalman/NIS layer.")
    if name == "tracking_orig":
        runs = [Run(f"maskfit_orig|{s}", "maskfit_orig", RunConfig(source=s, mode="track")) for s in ("gt", "yolo")]
        return Suite(name, runs, "yolo_every3", 6,
                     "Original repo mask-fit (raster, hierarchical warm grid + LM) — slow, subset.")
    if name == "robustness":
        runs = []
        methods = ["maskfit_analytic_lm", "cornerfit", "edgefit", "homography_it5"]
        for m in methods:
            for src in ("gt", "corrupt", "yolo"):
                runs.append(Run(f"{m}|src={src}", m, RunConfig(source=src, mode="track", kf=True)))
            for lvl, noise in (("nominal", NOMINAL_DOF_NOISE), ("high", HIGH_DOF_NOISE)):
                runs.append(Run(f"{m}|dof={lvl}", m, RunConfig(source="yolo", mode="track", kf=True, dof_noise=noise)))
            for ws in (0.9, 0.95, 1.05, 1.1):
                runs.append(Run(f"{m}|width={ws}", m, RunConfig(source="yolo", mode="track", kf=True, width_scale=ws)))
        return Suite(name, runs, "yolo", 4,
                     "Mask source, known-DOF sensor noise and runway-width prior error (tracking + KF).")
    raise KeyError(name)


SUITES = ("solver_cold", "solver_cold_yolo", "tracking", "tracking_orig", "robustness")
