"""ImageGen full model: 1D tokenizer + IBQ + prefix AR, trained jointly.

This module wires the EOSTok-style end-to-end image-token training loop:
  x -> encoder -> z -> IBQ -> (z_q, Ind, indices)
  Ind^T Embed -> AR model (teacher forcing) -> logits -> L_NTP
  softmax-STE(logits) -> codebook -> z_hat_q          (APR path)
  decoder([z_q ; z_hat_q] along batch) -> (x_recon, x_apr)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from ..text import Condition
from .ar import ARModel
from .quantizer import IBQQuantizer
from .tokenizer import Decoder1D, Encoder1D


@dataclass
class Pixels:
    """Decoded images, both in [-1, 1]."""
    recon: torch.Tensor  # reconstruction pass: decode(z_q)
    apr: torch.Tensor    # APR pass: decode of teacher-forced AR predictions


@dataclass
class Activations:
    """Hidden states the VFM aligner reads; both None when alignment is off."""
    h_enc: torch.Tensor          # encoder hidden patch embeddings (B, N, D)
    h_dec: torch.Tensor | None   # decoder mask-token states at the align layer (2B, N, D)


@dataclass
class RegLosses:
    """Losses the model computes internally (intrinsic to the quantizer / AR)."""
    commit: torch.Tensor
    entropy: torch.Tensor
    ntp: torch.Tensor


@dataclass
class Metrics:
    """Diagnostics — never part of the optimized objective."""
    ar_acc: torch.Tensor
    indices: torch.Tensor  # ground-truth code indices (B, L), for inspection/eval
    codebook_ppl: torch.Tensor   # effective #codes used (exp of batch-usage entropy)
    codebook_usage: torch.Tensor  # fraction of the codebook touched this batch


@dataclass
class ImageGenOutput:
    """One forward pass, grouped by role (pixels / activations / reg_losses /
    metrics) so each consumer reaches for the group it needs instead of
    categorising a flat dict by hand. The criterion reads pixels, reg_losses,
    and activations; the train loop logs pixels.recon and the metrics the
    criterion returns."""
    pixels: Pixels
    activations: Activations
    reg_losses: RegLosses
    metrics: Metrics


class ImageGen(nn.Module):
    def __init__(
        self,
        cfg: Config,
        load_text_encoder: bool = True,
        text_encoder_dim: int | None = None,
    ):
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
            q.codebook_size, t.num_latent_tokens,
            a.hidden_dim, a.layers, a.num_heads,
            text_cfg=cfg.text,
            load_text_encoder=load_text_encoder,
            text_encoder_dim=text_encoder_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: Condition,
        keep_tokens: int | None = None,
        condition_drop: torch.Tensor | None = None,
    ) -> ImageGenOutput:
        """Joint training forward. keep_tokens k (nested dropout) truncates the
        latent sequence fed to the decoder; NTP always uses the full sequence."""
        z, h_enc = self.encoder(x)
        quant = self.quantizer(z)
        z_q, ind, indices = quant["z_q"], quant["ind"], quant["indices"]

        # --- NTP: AR model on soft embeddings, gradients reach encoder+codebook
        logits = self.ar(self.ar.embed_soft(ind), condition, condition_drop)
        ntp_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), indices.reshape(-1)
        )
        with torch.no_grad():
            ar_acc = (logits.argmax(-1) == indices).float().mean()

        # --- APR path: decode teacher-forced AR predictions to pixels (Eq. 4).
        # Same straight-through one-hot the quantizer uses on the encoder path.
        ind_ar, _, _ = self.quantizer.straight_through_onehot(logits)
        z_hat_q = self.quantizer.soft_codes_to_latents(ind_ar)

        k = keep_tokens or self.num_latent_tokens
        dec_in = torch.cat([z_q[:, :k], z_hat_q[:, :k]], dim=0)
        dec_out, h_dec = self.decoder(dec_in)
        x_recon, x_apr = dec_out.chunk(2, dim=0)

        return ImageGenOutput(
            pixels=Pixels(recon=x_recon, apr=x_apr),
            activations=Activations(h_enc=h_enc, h_dec=h_dec),
            reg_losses=RegLosses(
                commit=quant["commit_loss"],
                entropy=quant["entropy_loss"],
                ntp=ntp_loss,
            ),
            metrics=Metrics(
                ar_acc=ar_acc,
                indices=indices,
                codebook_ppl=quant["perplexity"],
                codebook_usage=quant["usage"],
            ),
        )

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.encoder(x)
        z_q = self.quantizer(z)["z_q"]
        return self.decoder(z_q)[0]

    @torch.no_grad()
    def generate(
        self,
        condition: Condition,
        temperature: float = 1.0,
        cfg_scale: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
    ) -> torch.Tensor:
        indices = self.ar.generate(condition, temperature, cfg_scale, top_k, top_p)
        z_q = self.quantizer.codes_to_latents(indices)
        return self.decoder(z_q)[0]
