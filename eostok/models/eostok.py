"""EOSTok = 1D tokenizer (encoder + IBQ + decoder) + AR model, trained jointly.

This module wires the full forward pass of Fig. 2:
  x -> encoder -> z -> IBQ -> (z_q, Ind, indices)
  Ind^T Embed -> AR model (teacher forcing) -> logits -> L_NTP
  softmax-STE(logits) -> codebook -> z_hat_q          (APR path)
  decoder([z_q ; z_hat_q] along batch) -> (x_recon, x_apr)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from .ar import ARModel
from .quantizer import IBQQuantizer
from .tokenizer import Decoder1D, Encoder1D


class EOSTok(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d, t, q, a = cfg.data, cfg.tokenizer, cfg.quantizer, cfg.ar
        self.num_latent_tokens = t.num_latent_tokens
        self.encoder = Encoder1D(
            d.image_size, t.patch_size, d.channels, t.hidden_dim,
            t.enc_layers, t.num_heads, t.num_latent_tokens, t.latent_dim,
        )
        self.quantizer = IBQQuantizer(q.codebook_size, t.latent_dim, q.temperature)
        align_layer = cfg.vfm.decoder_align_layer if cfg.vfm.enabled else None
        self.decoder = Decoder1D(
            d.image_size, t.patch_size, d.channels, t.hidden_dim,
            t.dec_layers, t.num_heads, t.num_latent_tokens, t.latent_dim,
            align_layer=align_layer,
        )
        self.ar = ARModel(
            q.codebook_size, d.num_classes, t.num_latent_tokens,
            a.hidden_dim, a.layers, a.num_heads,
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor, keep_tokens: int | None = None):
        """Joint training forward. keep_tokens k (nested dropout) truncates the
        latent sequence fed to the decoder; NTP always uses the full sequence."""
        z, h_enc = self.encoder(x)
        quant = self.quantizer(z)
        z_q, ind, indices = quant["z_q"], quant["ind"], quant["indices"]

        # --- NTP: AR model on soft embeddings, gradients reach encoder+codebook
        logits = self.ar(self.ar.embed_soft(ind), labels)
        ntp_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), indices.reshape(-1)
        )
        with torch.no_grad():
            ar_acc = (logits.argmax(-1) == indices).float().mean()

        # --- APR path: decode teacher-forced AR predictions to pixels (Eq. 4)
        p_ar = logits.softmax(dim=-1)
        hard_ar = F.one_hot(logits.argmax(-1), logits.shape[-1]).to(p_ar.dtype)
        ind_ar = hard_ar + p_ar - p_ar.detach()
        z_hat_q = self.quantizer.soft_codes_to_latents(ind_ar)

        k = keep_tokens or self.num_latent_tokens
        dec_in = torch.cat([z_q[:, :k], z_hat_q[:, :k]], dim=0)
        dec_out, h_dec = self.decoder(dec_in)
        x_recon, x_apr = dec_out.chunk(2, dim=0)

        return {
            "x_recon": x_recon,
            "x_apr": x_apr,
            "h_enc": h_enc,
            "h_dec": h_dec,  # (2B, N, D) at align layer, or None
            "indices": indices,
            "commit_loss": quant["commit_loss"],
            "entropy_loss": quant["entropy_loss"],
            "ntp_loss": ntp_loss,
            "ar_acc": ar_acc,
        }

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.encoder(x)
        z_q = self.quantizer(z)["z_q"]
        return self.decoder(z_q)[0]

    @torch.no_grad()
    def generate(self, labels: torch.Tensor, temperature: float = 1.0,
                 cfg_scale: float = 1.0) -> torch.Tensor:
        indices = self.ar.generate(labels, temperature, cfg_scale)
        z_q = self.quantizer.codes_to_latents(indices)
        return self.decoder(z_q)[0]
