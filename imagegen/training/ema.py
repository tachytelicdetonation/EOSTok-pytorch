"""Exponential moving average of model parameters (paper: decay 0.9999)."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMA:
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        shadow: nn.Module | None = None,
    ):
        self.decay = decay
        self.shadow = (shadow if shadow is not None else copy.deepcopy(model)).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        model_params = dict(model.named_parameters())
        for name, ema_p in self.shadow.named_parameters():
            p = model_params.get(name)
            if p is not None:
                ema_p.lerp_(p.detach(), 1 - self.decay)
        model_buffers = dict(model.named_buffers())
        for name, ema_b in self.shadow.named_buffers():
            b = model_buffers.get(name)
            if b is not None:
                ema_b.copy_(b)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd, strict: bool = True):
        self.shadow.load_state_dict(sd, strict=strict)
