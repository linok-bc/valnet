"""How big a patch of ground the BEV canvas should cover, per frame.

There is no single right answer, because the runway is 3 km away at the start of
an episode and 150 m away at the end. A grid wide enough for the far field
wastes almost every pixel in the near field, and a grid sized for the near field
does not contain the runway at all early on. So the grid is a policy, chosen per
frame from quantities the aircraft actually senses (altitude and attitude) --
never from the label, which would be cheating.

    fixed      one metric window for every frame. The control: simplest thing
               that could work, and the only policy whose BEV pixels mean the
               same thing in every image.

    altitude   the window scales with height above the runway. On a 3 deg
               glideslope the threshold sits at ~19x the altitude, so a window
               spanning 2h..30h holds the runway at roughly constant apparent
               size all the way down. This is Ryu et al.'s equation 24 (ROI
               scaled by h1/h0) in a slightly more explicit form.

    horizon    the window covers exactly the ground the camera can see, from the
               range at the bottom image row out to a margin below the horizon.
               Closest to what the paper actually implements (their d_s below the
               vanishing point), and it needs no glideslope assumption.

    threshold  the window is placed on the KNOWN along-track distance and given a
               FIXED metre-per-pixel scale. This is the policy that earns its
               keep under the experiment's assumption that only yaw and
               cross-track are unknown: the threshold lands on the same canvas
               row and the runway spans the same number of pixels in every
               single image, at 3 km and at 200 m alike. A detector then never
               has to learn scale -- only the two unknowns, which show up as the
               rectangle's lean and its lateral offset. It consumes along-track,
               so it is only honest under that assumption; `altitude` and
               `horizon` are the fallbacks if along-track is ever in doubt.

All three keep SQUARE metric pixels. That is not a detail: an anisotropic scale
turns the rotated runway rectangle into a parallelogram, and the oriented box
stops fitting it -- which is the whole thing being tested.
"""

from dataclasses import dataclass
import math

import numpy as np

from experiments.bev.camera import CameraModel
from experiments.bev.transform import BEVGrid, depression_deg, horizon_v


@dataclass
class GridPolicy:
    """Per-frame BEV grid selection. See module docstring for `mode`."""

    mode: str = "threshold"
    width_px: int = 256
    height_px: int = 1024
    # fixed
    x_near_m: float = 50.0
    x_far_m: float = 1200.0
    # altitude
    x_near_h: float = 2.0
    x_far_h: float = 30.0
    # horizon
    horizon_margin_px: float = 24.0
    # threshold
    mpp_m: float = 0.75
    back_m: float = 150.0
    # shared guards
    min_range_m: float = 20.0
    max_range_m: float = 4000.0

    def for_frame(self, cam: CameraModel, roll_deg: float, pitch_deg: float,
                  height_m: float, range_m: float = None) -> BEVGrid:
        """`range_m` is the known along-track distance to the threshold. Only the
        `threshold` mode uses it; the others ignore it by design."""
        if self.mode == "threshold":
            if range_m is None:
                raise ValueError("grid mode 'threshold' needs the along-track range")
            x_near = max(range_m - self.back_m, self.min_range_m)
            return BEVGrid(self.width_px, self.height_px, self.mpp_m, x_near)
        if self.mode == "fixed":
            x_near, x_far = self.x_near_m, self.x_far_m
        elif self.mode == "altitude":
            x_near = self.x_near_h * height_m
            x_far = self.x_far_h * height_m
        elif self.mode == "horizon":
            x_near, x_far = self._horizon_span(cam, roll_deg, pitch_deg, height_m)
        else:
            raise ValueError(f"unknown grid mode {self.mode!r}")

        x_near = max(x_near, self.min_range_m)
        x_far = min(max(x_far, x_near + self.min_range_m), self.max_range_m)
        mpp = (x_far - x_near) / (self.height_px - 1)
        return BEVGrid(self.width_px, self.height_px, mpp, x_near)

    def _horizon_span(self, cam, roll_deg, pitch_deg, height_m):
        """Ground ranges visible between the bottom image row and the horizon.

        Roll is ignored here on purpose: it rotates the visible wedge but does
        not change how far down the optical axis the ground is, and using the
        rolled figure would make the canvas jitter with every wing drop.
        """
        d = math.radians(depression_deg(pitch_deg, cam.tilt_deg))
        near = _range_at_row(cam, cam.height - 1.0, d, height_m)
        v_h = horizon_v(cam, 0.0, pitch_deg)
        far_row = (v_h + self.horizon_margin_px) if v_h is not None else 0.0
        far = _range_at_row(cam, max(far_row, 0.0), d, height_m)
        near = near if near is not None else self.min_range_m
        far = far if far is not None else self.max_range_m
        return near, far


def _range_at_row(cam: CameraModel, v: float, depression_rad: float, height_m: float):
    """Forward ground range of the ray through image row `v` at the image centre.

    None when the ray points at or above the horizon.
    """
    ang = depression_rad + math.atan((v - cam.cy) / cam.fy)
    if ang <= 1e-6:
        return None
    return height_m / math.tan(ang)


def resolution_report(cam: CameraModel, grid: BEVGrid, roll_deg: float,
                      pitch_deg: float, height_m: float, target_range_m: float,
                      runway_width_m: float) -> dict:
    """How many pixels the runway width gets, in the source and in the BEV.

    The honest limit of this whole idea: warping cannot invent detail. If the
    source gives the runway 12 px across and the BEV stretches it to 40, those
    40 px carry 12 px of information. Ratios far above 1 mean the canvas is
    oversampling; far below 1 means it is throwing information away.
    """
    d = math.radians(depression_deg(pitch_deg, cam.tilt_deg))
    src_px = cam.fx * runway_width_m / math.hypot(target_range_m, height_m)
    bev_px = runway_width_m / grid.mpp
    x0, x1, half = grid.extent_m()
    return {
        "source_px": src_px,
        "bev_px": bev_px,
        "ratio": bev_px / src_px if src_px > 0 else float("inf"),
        "mpp": grid.mpp,
        "extent_m": (x0, x1, half),
        "in_canvas": x0 <= target_range_m <= x1,
        "depression_deg": math.degrees(d),
    }
