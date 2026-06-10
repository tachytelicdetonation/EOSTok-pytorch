"""The EOSTok training objective (Eq. 8), as a module with an interface.

  L = L_VQVAE + lambda_NTP * L_NTP + lambda_APR * L_APR
      + lambda_sem * (L_implicit + L_decoder_align)
  L_VQVAE = L2 + LPIPS + lambda_GAN * L_GAN + lambda_reg * L_reg

This file holds the *objective*, split into two sibling modules because the
generator and the discriminator optimize opposed quantities:

  - `EOSTokCriterion` — everything the generator minimizes: the cooperative
    terms (perceptual, VFM alignment, recon/APR pixels), the quantizer/AR
    regularizers surfaced by the model, and the generator-side GAN term. It
    owns the trainable cooperative sub-modules (PerceptualLoss, VFMAligner).
  - `Adversary` — the discriminator, its LeCam regularizer, and the
    `disc_start` gate. It owns the params the *discriminator* optimizer trains.

The criterion borrows a reference to the adversary (stored in a tuple so it is
NOT registered as a sub-module); that keeps Eq. 8 assembled in one place while
the discriminator's parameters stay owned by the adversary and its optimizer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .losses import PerceptualLoss
from .models.discriminator import (
    LeCamRegularizer, PatchDiscriminator, hinge_d_loss, hinge_g_loss,
)
from .models.eostok import EOSTokOutput


class Adversary(nn.Module):
    """Discriminator + LeCam + the `disc_start` gate, behind one interface.

    `active(step)` is the single source of truth for whether the GAN is on; both
    the generator term (`g_term`) and the caller's discriminator step consult it,
    so the gating rule lives in exactly one place. The detach policy lives here
    too: `d_loss` detaches the fake so the discriminator gradient never reaches
    the generator, while `g_term` keeps the fake live so the generator learns.
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
        """Generator-side GAN term -D(fake) on the LIVE recon (gradients flow to
        the generator). Zero until the gate opens."""
        if not self.active(step):
            return fake.new_zeros(())
        return hinge_g_loss(self.disc(fake))

    def d_loss(self, real: torch.Tensor, fake: torch.Tensor, step: int) -> torch.Tensor:
        """Discriminator-side loss on real vs DETACHED fake, plus LeCam. The
        caller checks `active(step)` before stepping the discriminator optimizer."""
        real_logits = self.disc(real)
        fake_logits = self.disc(fake.detach())
        return hinge_d_loss(real_logits, fake_logits) \
            + self.lecam_weight * self.lecam(real_logits, fake_logits)


class EOSTokCriterion(nn.Module):
    """The generator's objective (Eq. 8). `forward` returns (total_loss, metrics)
    for one training step; that pair is the entire test surface for the loss.

    Owns the trainable cooperative sub-modules; `criterion.parameters()` is
    exactly the extra params the *generator* optimizer must reach (the VFM
    projectors — PerceptualLoss is frozen). The discriminator is reached through
    the borrowed `adversary` reference, never registered here.
    """

    def __init__(self, cfg: Config, adversary: Adversary):
        super().__init__()
        self.w = cfg.loss
        self.qw = cfg.quantizer
        self.vw = cfg.vfm
        self.perceptual = PerceptualLoss() if cfg.loss.lpips_enabled else None

        self.vfm = None
        if cfg.vfm.enabled:
            from .models.vfm import VFMAligner
            grid = cfg.data.image_size // cfg.tokenizer.patch_size
            self.vfm = VFMAligner(
                cfg.vfm.model, cfg.tokenizer.hidden_dim, cfg.tokenizer.hidden_dim, grid,
            )

        # Borrowed, NOT owned: stored in a tuple so nn.Module does not register
        # the discriminator's params here. They belong to the adversary's optimizer.
        self._adv = (adversary,)

    def _semantic_loss(self, out: EOSTokOutput, x: torch.Tensor) -> torch.Tensor:
        if self.vfm is None:
            return x.new_zeros(())
        yv = self.vfm.features(x)
        loss = self.vfm.implicit_loss(out.activations.h_enc, yv)
        if out.activations.h_dec is not None:
            loss = loss + self.vfm.decoder_loss(out.activations.h_dec, yv.repeat(2, 1, 1))
        return loss

    def forward(self, out: EOSTokOutput, x: torch.Tensor, step: int):
        """Assemble Eq. 8 for this step. Returns (total_loss, metrics dict)."""
        adv = self._adv[0]
        px, reg = out.pixels, out.reg_losses
        zero = x.new_zeros(())

        rec_l2 = F.mse_loss(px.recon, x)
        apr_l2 = F.mse_loss(px.apr, x)
        rec_lp = self.perceptual(px.recon, x) if self.perceptual else zero
        apr_lp = self.perceptual(px.apr, x) if self.perceptual else zero
        g = adv.g_term(px.recon, step)
        sem = self._semantic_loss(out, x)

        # --- Eq. 8: weighted sum of the terms above.
        w, qw, vw = self.w, self.qw, self.vw
        loss = (
            w.recon_l2 * rec_l2 + w.recon_lpips * rec_lp
            + w.gan * g
            + qw.commit_weight * reg.commit + qw.entropy_weight * reg.entropy
            + w.ntp * reg.ntp
            + w.apr_l2 * apr_l2 + w.apr_lpips * apr_lp
            + vw.weight * sem
        )

        metrics = {
            "loss": loss.detach(),
            "rec_l2": rec_l2.detach(), "rec_lp": rec_lp.detach(),
            "apr_l2": apr_l2.detach(), "apr_lp": apr_lp.detach(),
            "g": g.detach(), "sem": sem.detach(),
            "ntp": reg.ntp.detach(), "ar_acc": out.metrics.ar_acc,
        }
        return loss, metrics
