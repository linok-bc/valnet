"""What an arm is, and where the arms and grid policies are defined.

An arm is one cell of the experiment: an input representation plus a target
geometry. Three things vary and nothing else does:

    view    perspective (the raw frame) or bev (a rectified ground raster)
    target  segment (a polygon) or obb (an oriented box)
    grid    for bev arms, WHICH ground raster -- see `grids` in the config

Everything else -- the frames, the splits, the training recipe, the scoring
pipeline -- is shared, so a difference between two arms is attributable to the
cell and not to the setup.

Arms live in the config rather than in code so that adding one is a config edit
plus a rebuild. They are read through here by the builder, the trainer, the
evaluator and the visualiser, so all four always agree on what exists.
"""

from dataclasses import dataclass

import yaml

from experiments.bev.grids import GridPolicy


@dataclass(frozen=True)
class Arm:
    name: str
    view: str          # "perspective" | "bev"
    target: str        # "segment" | "obb"
    grid: str = None   # key into the config's `grids`; None for perspective arms

    @property
    def is_bev(self) -> bool:
        return self.view == "bev"


def load_arms(cfg) -> dict:
    out = {}
    for name, spec in cfg["arms"].items():
        view, target = spec["view"], spec["target"]
        if view not in ("perspective", "bev"):
            raise ValueError(f"arm {name}: unknown view {view!r}")
        if target not in ("segment", "obb", "detect"):
            raise ValueError(f"arm {name}: unknown target {target!r}")
        grid = spec.get("grid")
        if view == "bev":
            if grid is None:
                raise ValueError(f"arm {name}: a bev arm needs a `grid`")
            if grid not in cfg["grids"]:
                raise ValueError(f"arm {name}: no grid named {grid!r} in the config")
        out[name] = Arm(name, view, target, grid)
    return out


def load_grids(cfg) -> dict:
    """Only the grids some arm actually uses -- an unused grid costs a warp per frame."""
    used = {spec.get("grid") for spec in cfg["arms"].values() if spec.get("grid")}
    return {name: GridPolicy(**params)
            for name, params in cfg["grids"].items() if name in used}


def load_config(path):
    with open(path) as fp:
        cfg = yaml.safe_load(fp)
    return cfg, load_arms(cfg), load_grids(cfg)
