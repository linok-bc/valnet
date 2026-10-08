"""Find a valid coarse-search seed when the temporal prior has no mask overlap."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pose_estimation.g_dof import GDofState
from pose_estimation.metrics import dice_loss, soft_dice
from pose_estimation.optimize import _EvalContext


@dataclass(frozen=True)
class SeedSearchResult:
    """Outcome of expanding-ring search around a prior pose."""

    seed: GDofState
    ring: int  # 0 = prior itself was valid
    dice_loss: float
    relocated: bool
    n_samples: int


def dice_loss_at_state(ctx: _EvalContext, state: GDofState) -> float | None:
    """Return 1−Dice at ``state``, or ``None`` if render fails."""
    try:
        pred = ctx.render_downsampled(state)
        obs = ctx.observed_downsampled()
    except (ValueError, FloatingPointError):
        return None
    if float(pred.sum()) < 1e-6:
        return 1.0
    if float(obs.sum()) < 1e-8:
        return 1.0
    return dice_loss(obs, pred)


def soft_dice_at_state(ctx: _EvalContext, state: GDofState) -> float | None:
    try:
        pred = ctx.render_downsampled(state)
        obs = ctx.observed_downsampled()
    except (ValueError, FloatingPointError):
        return None
    if float(pred.sum()) < 1e-6:
        return 0.0
    return soft_dice(obs, pred)


def has_mask_overlap(
    ctx: _EvalContext,
    state: GDofState,
    *,
    min_soft_dice: float = 0.01,
) -> bool:
    """
    True when the rendered mask overlaps the observation in-frame.

    Rejects hors-cadre renders (empty prediction) and ``1−Dice = 1``.
    """
    score = soft_dice_at_state(ctx, state)
    if score is None:
        return False
    return score > min_soft_dice


def find_valid_seed_expanding_rings(
    ctx: _EvalContext,
    center: GDofState,
    *,
    yaw_radius_per_ring_deg: float = 0.5,
    lat_radius_per_ring_m: float = 2.5,
    points_per_ring: int = 16,
    max_rings: int = 40,
    min_soft_dice: float = 0.01,
) -> SeedSearchResult:
    """
    Sample poses on expanding ellipses in (yaw, lateral) until overlap exists.

    Ring ``k`` uses semi-axes ``k * yaw_radius_per_ring`` and ``k * lat_radius_per_ring``.
    Ring 0 tests ``center`` only.
    """
    n_samples = 0
    score0 = soft_dice_at_state(ctx, center)
    loss0 = 1.0 - score0 if score0 is not None else 1.0
    n_samples += 1
    if score0 is not None and score0 > min_soft_dice:
        return SeedSearchResult(
            seed=center,
            ring=0,
            dice_loss=loss0,
            relocated=False,
            n_samples=n_samples,
        )

    best: SeedSearchResult | None = None
    for ring in range(1, max_rings + 1):
        r_yaw = ring * yaw_radius_per_ring_deg
        r_lat = ring * lat_radius_per_ring_m
        for i in range(points_per_ring):
            angle = 2.0 * math.pi * i / points_per_ring
            candidate = GDofState(
                center.yaw_cam_deg + r_yaw * math.cos(angle),
                center.lateral_offset_m + r_lat * math.sin(angle),
            )
            score = soft_dice_at_state(ctx, candidate)
            n_samples += 1
            if score is None or score <= min_soft_dice:
                continue
            loss = 1.0 - score
            if best is None or ring < best.ring or (ring == best.ring and loss < best.dice_loss):
                best = SeedSearchResult(
                    seed=candidate,
                    ring=ring,
                    dice_loss=loss,
                    relocated=True,
                    n_samples=n_samples,
                )
        if best is not None:
            return best

    # No overlap found — return center anyway (caller may fall back to cold search).
    return SeedSearchResult(
        seed=center,
        ring=-1,
        dice_loss=loss0,
        relocated=False,
        n_samples=n_samples,
    )
