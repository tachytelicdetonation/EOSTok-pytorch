"""Index Backpropagation Quantization (IBQ) as used by EOSTok (Eq. 1 / Eq. 7).

logits = [z^T C_1, ..., z^T C_K] with both z and the codebook l2-normalized,
p = softmax(logits / tau), and the straight-through one-hot index

    Ind = onehot(argmax p) + p - stopgrad(p)

so that the gradient of any downstream loss reaches *all* codebook entries and
the encoder. z_q = Ind^T C is the quantized latent; Ind is also used to softly
embed tokens into the AR model (h = Ind^T Embed).

Regularizers: commitment loss + entropy loss (per-sample entropy down, batch
average entropy up) to keep codebook usage high.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IBQQuantizer(nn.Module):
    def __init__(self, codebook_size: int, latent_dim: int, temperature: float = 1.0):
        super().__init__()
        self.codebook_size = codebook_size
        self.temperature = temperature
        self.codebook = nn.Parameter(torch.randn(codebook_size, latent_dim) * 0.02)

    def forward(self, z: torch.Tensor):
        """z: (B, L, d). Returns dict with z_q, soft one-hot ind, indices, reg losses."""
        z_n = F.normalize(z, dim=-1)
        c_n = F.normalize(self.codebook, dim=-1)

        logits = torch.einsum("bld,kd->blk", z_n, c_n) / self.temperature
        ind, p, indices = self.straight_through_onehot(logits)
        z_q = torch.einsum("blk,kd->bld", ind, c_n)

        commit_loss = F.mse_loss(z_n, z_q.detach()) + 0.25 * F.mse_loss(z_n.detach(), z_q)

        # Entropy regularization (MAGVIT-v2 style): confident per-token
        # assignments, uniform usage across the batch.
        eps = 1e-8
        flat = p.reshape(-1, self.codebook_size)
        per_sample_entropy = -(flat * (flat + eps).log()).sum(-1).mean()
        avg_p = flat.mean(0)
        codebook_entropy = -(avg_p * (avg_p + eps).log()).sum()
        entropy_loss = per_sample_entropy - codebook_entropy

        return {
            "z_q": z_q,
            "ind": ind,
            "indices": indices,
            "commit_loss": commit_loss,
            "entropy_loss": entropy_loss,
        }

    def straight_through_onehot(self, logits: torch.Tensor):
        """Straight-through soft one-hot (Eq. 7) from logits (B, L, K):

            ind = onehot(argmax) + p - stopgrad(p)

        Gradients flow only through the softmax branch. Returns (ind, p, indices)
        so the entropy regularizer and the dict can reuse p/indices. This is the
        single home for the STE formulation, shared by the encoder-quantization
        path (forward) and the AR/APR path (ImageGen.forward)."""
        p = logits.softmax(dim=-1)
        indices = logits.argmax(dim=-1)
        hard = F.one_hot(indices, self.codebook_size).to(p.dtype)
        ind = hard + p - p.detach()
        return ind, p, indices

    @torch.no_grad()
    def codes_to_latents(self, indices: torch.Tensor) -> torch.Tensor:
        """Map hard code indices (B, L) to quantized latents (B, L, d) for sampling."""
        c_n = F.normalize(self.codebook, dim=-1)
        return c_n[indices]

    def soft_codes_to_latents(self, soft_ind: torch.Tensor) -> torch.Tensor:
        """Map (soft) one-hot indices (B, L, K) to latents, keeping gradients (APR path)."""
        c_n = F.normalize(self.codebook, dim=-1)
        return torch.einsum("blk,kd->bld", soft_ind, c_n)
