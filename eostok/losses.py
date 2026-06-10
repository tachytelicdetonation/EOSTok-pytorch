"""Perceptual loss wrapper. LPIPS (VGG backbone, as in the paper) with
grayscale handling for small datasets like MNIST."""

from __future__ import annotations

import torch
import torch.nn as nn


class PerceptualLoss(nn.Module):
    def __init__(self, net: str = "vgg"):
        super().__init__()
        import lpips  # local import: heavy, downloads weights on first use

        self.lpips = lpips.LPIPS(net=net, verbose=False)
        for p in self.lpips.parameters():
            p.requires_grad_(False)
        self.lpips.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.lpips.eval()
        return self

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x, y in [-1, 1]; 1-channel inputs are replicated to RGB."""
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        return self.lpips(x.float(), y.float()).mean()
