"""GAN discriminator with hinge loss and LeCam regularization.

The paper uses the StyleGAN-T discriminator (frozen DINO features + heads),
stabilized with LeCam divergence. We ship a lighter conv (PatchGAN-style)
discriminator so the repo has no extra pretrained dependencies; it is
architecturally a deviation (documented in the README) but plays the same
role in L_VQVAE. Swap in a StyleGAN-T discriminator here for strict fidelity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config


class PatchDiscriminator(nn.Module):
    def __init__(self, channels: int = 3, base: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [nn.Conv2d(channels, base, 4, 2, 1), nn.LeakyReLU(0.2)]
        ch = base
        for i in range(1, n_layers):
            out = min(base * 2**i, 512)
            layers += [
                nn.Conv2d(ch, out, 4, 2, 1, bias=False),
                nn.GroupNorm(8, out),
                nn.LeakyReLU(0.2),
            ]
            ch = out
        layers += [nn.Conv2d(ch, 1, 4, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def hinge_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def hinge_g_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


class LeCamRegularizer(nn.Module):
    """LeCam divergence (Tseng et al., 2021): anchors D outputs to EMA of the
    opposite side, preventing the discriminator from drifting too far ahead."""

    def __init__(self, decay: float = 0.99):
        super().__init__()
        self.decay = decay
        self.register_buffer("ema_real", torch.zeros(()))
        self.register_buffer("ema_fake", torch.zeros(()))

    def forward(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.ema_real.mul_(self.decay).add_(real_logits.mean() * (1 - self.decay))
            self.ema_fake.mul_(self.decay).add_(fake_logits.mean() * (1 - self.decay))
        return ((real_logits - self.ema_fake).pow(2).mean()
                + (fake_logits - self.ema_real).pow(2).mean())


class Adversary(nn.Module):
    """Discriminator + LeCam + the ``disc_start`` gate.

    ``active(step)`` is the single source of truth for whether the GAN is on.
    The generator term keeps fake images live, while the discriminator-side
    loss detaches them so discriminator gradients never reach the generator.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.disc = PatchDiscriminator(cfg.data.channels)
        self.lecam = LeCamRegularizer()
        self.gan_weight = cfg.loss.gan
        self.lecam_weight = cfg.loss.lecam
        self.disc_start = cfg.loss.disc_start

    def active(self, step: int) -> bool:
        return self.gan_weight > 0 and step >= self.disc_start

    def g_term(self, fake: torch.Tensor, step: int) -> torch.Tensor:
        if not self.active(step):
            return fake.new_zeros(())
        return hinge_g_loss(self.disc(fake))

    def d_loss(self, real: torch.Tensor, fake: torch.Tensor, step: int) -> torch.Tensor:
        real_logits = self.disc(real)
        fake_logits = self.disc(fake.detach())
        return hinge_d_loss(real_logits, fake_logits) \
            + self.lecam_weight * self.lecam(real_logits, fake_logits)
