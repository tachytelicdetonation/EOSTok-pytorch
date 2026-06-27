"""ImageGen training objective, assembled as a first-class module.

  L = L_VQVAE + lambda_NTP * L_NTP + lambda_APR * L_APR
      + lambda_sem * (L_implicit + L_decoder_align)
  L_VQVAE = L2 + LPIPS + lambda_GAN * L_GAN + lambda_reg * L_reg

This file holds the generator-side objective. The discriminator remains a
sibling module owned by its optimizer; the training loop passes the already
computed generator-side adversarial term into ``ImageGenCriterion.forward``.
That keeps Eq. 8 assembled in one place without hiding an ``nn.Module`` inside a
non-registered container.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from ..models.imagegen import ImageGenOutput
from .losses import PerceptualLoss


class ImageGenCriterion(nn.Module):
    """The generator's cooperative objective plus an optional GAN term.

    ``criterion.parameters()`` is exactly the extra parameter set the generator
    optimizer must reach (VFM projectors when enabled; PerceptualLoss is frozen).
    The adversary is not stored here, so discriminator parameters cannot be
    accidentally registered under the criterion.
    """

    def __init__(self, cfg: Config):
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
                cfg.vfm.model,
                cfg.tokenizer.hidden_dim,
                cfg.tokenizer.hidden_dim,
                grid,
            )

    def _semantic_loss(self, out: ImageGenOutput, x: torch.Tensor) -> torch.Tensor:
        if self.vfm is None:
            return x.new_zeros(())
        yv = self.vfm.features(x)
        loss = self.vfm.implicit_loss(out.activations.h_enc, yv)
        if out.activations.h_dec is not None:
            loss = loss + self.vfm.decoder_loss(
                out.activations.h_dec, yv.repeat(2, 1, 1)
            )
        return loss

    def forward(
        self,
        out: ImageGenOutput,
        x: torch.Tensor,
        step: int,
        adversarial_loss: torch.Tensor | None = None,
    ):
        """Assemble Eq. 8 for this step. Returns (total_loss, metrics dict)."""
        del step  # The adversary owns the disc_start gate before passing g here.
        px, reg = out.pixels, out.reg_losses
        zero = x.new_zeros(())

        rec_l2 = F.mse_loss(px.recon, x)
        apr_l2 = F.mse_loss(px.apr, x)
        rec_lp = self.perceptual(px.recon, x) if self.perceptual is not None else zero
        apr_lp = self.perceptual(px.apr, x) if self.perceptual is not None else zero
        g = adversarial_loss if adversarial_loss is not None else zero
        sem = self._semantic_loss(out, x)

        # --- Eq. 8: weighted sum of the terms above.
        w, qw, vw = self.w, self.qw, self.vw
        loss = (
            w.recon_l2 * rec_l2
            + w.recon_lpips * rec_lp
            + w.gan * g
            + qw.commit_weight * reg.commit
            + qw.entropy_weight * reg.entropy
            + w.ntp * reg.ntp
            + w.apr_l2 * apr_l2
            + w.apr_lpips * apr_lp
            + vw.weight * sem
        )

        metrics = {
            "loss": loss.detach(),
            "rec_l2": rec_l2.detach(),
            "rec_lp": rec_lp.detach(),
            "apr_l2": apr_l2.detach(),
            "apr_lp": apr_lp.detach(),
            "g": g.detach(),
            "sem": sem.detach(),
            "ntp": reg.ntp.detach(),
            "ar_acc": out.metrics.ar_acc,
        }
        return loss, metrics
