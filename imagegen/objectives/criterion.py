"""ImageGen training objective, assembled as a first-class module.

  L = L_VQVAE + lambda_NTP * L_NTP + lambda_APR * L_APR
      + lambda_sem * (L_implicit + L_decoder_align)
  L_VQVAE = L2 + LPIPS + lambda_GAN * L_GAN + lambda_reg * L_reg

This file holds the *objective*, split into two sibling modules because the
generator and the discriminator optimize opposed quantities:

  - `ImageGenCriterion` — everything the generator minimizes: the cooperative
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

from ..config import Config
from ..models.imagegen import ImageGenOutput
from .adversary import Adversary
from .losses import PerceptualLoss


class ImageGenCriterion(nn.Module):
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
            from .vfm import VFMAligner
            grid = cfg.data.image_size // cfg.tokenizer.patch_size
            self.vfm = VFMAligner(
                cfg.vfm.model, cfg.tokenizer.hidden_dim, cfg.tokenizer.hidden_dim, grid,
            )

        # Borrowed, NOT owned: stored in a tuple so nn.Module does not register
        # the discriminator's params here. They belong to the adversary's optimizer.
        self._adv = (adversary,)

    def _semantic_loss(self, out: ImageGenOutput, x: torch.Tensor) -> torch.Tensor:
        if self.vfm is None:
            return x.new_zeros(())
        yv = self.vfm.features(x)
        loss = self.vfm.implicit_loss(out.activations.h_enc, yv)
        if out.activations.h_dec is not None:
            loss = loss + self.vfm.decoder_loss(out.activations.h_dec, yv.repeat(2, 1, 1))
        return loss

    def forward(self, out: ImageGenOutput, x: torch.Tensor, step: int):
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
