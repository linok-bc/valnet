"""
Exponential moving average of model weights.

Why not `ultralytics.utils.torch_utils.ModelEMA`: it iterates `state_dict()`
and updates in place, so any tensor registered under two names gets the update
applied twice and ends up tracking the live weights at a different rate than
the rest of the network. VALNetModel used to do exactly that (the backbone was
registered both whole and as the `backbone_p3/p4/p5` slices), which is fixed
now — but `state_dict()` aliasing is easy to reintroduce and silent when you do.

`named_parameters()` / `named_buffers()` deduplicate by default, so iterating
those updates each tensor exactly once regardless of how it is registered.
"""

import math
from copy import deepcopy

import torch


class ModelEMA:
    """Keep a shadow copy of the model weights, updated as an EMA.

    The decay ramps in from 0 so the average isn't anchored to the random
    initialization: decay(n) = base_decay * (1 - exp(-n / tau)).

    Args:
        model: the live model. Copied once at construction.
        decay: asymptotic decay. 0.9999 matches the Ultralytics default.
        tau: ramp-in time constant, in updates.
    """

    def __init__(self, model, decay: float = 0.9999, tau: int = 2000):
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.updates = 0
        self.decay = lambda n: decay * (1.0 - math.exp(-n / tau))

    @torch.no_grad()
    def update(self, model) -> None:
        """Pull the EMA weights one step toward the live model."""
        self.updates += 1
        d = self.decay(self.updates)

        ema_params = dict(self.ema.named_parameters())
        for name, p in model.named_parameters():
            if p.dtype.is_floating_point:
                ema_params[name].mul_(d).add_(p.detach(), alpha=1.0 - d)

        # Buffers: EMA the float ones (BN running stats), copy the rest
        # (num_batches_tracked is an integer counter).
        ema_buffers = dict(self.ema.named_buffers())
        for name, b in model.named_buffers():
            if b.dtype.is_floating_point:
                ema_buffers[name].mul_(d).add_(b.detach(), alpha=1.0 - d)
            else:
                ema_buffers[name].copy_(b.detach())
