"""1D ViT tokenizer (encoder/decoder) following EOSTok (arXiv:2605.00503), Sec. 3.1 / A.1.1.

Encoder: image patches are flattened and concatenated with L learnable query
tokens. Attention is bidirectional among 2D patch tokens and causal along the
1D query tokens; queries may attend to patches, patches may NOT attend to
queries. The hidden patch embeddings h_enc are returned for implicit VFM
alignment; only the query outputs become the 1D latent z (projected to dim d).

Decoder: symmetric. Quantized 1D latents are concatenated with N learnable
mask tokens; attention is causal along latents (which cannot see mask tokens)
and bidirectional among mask tokens (which see everything). Mask token outputs
are unpatchified to pixels through a linear + 3x3 conv head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, attn_mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # (B, nh, T, hd)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


def _to_float_mask(allowed: torch.Tensor) -> torch.Tensor:
    """Bool 'allowed' matrix -> additive float mask for SDPA (MPS-safe)."""
    mask = torch.zeros_like(allowed, dtype=torch.float32)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask


def encoder_attn_mask(num_patches: int, num_latents: int) -> torch.Tensor:
    """Hybrid mask, sequence order [patches(N), queries(L)].

    - patches <-> patches: bidirectional
    - queries -> patches: allowed
    - queries -> queries: causal
    - patches -> queries: blocked
    """
    N, L = num_patches, num_latents
    allowed = torch.zeros(N + L, N + L, dtype=torch.bool)
    allowed[:N, :N] = True
    allowed[N:, :N] = True
    allowed[N:, N:] = torch.tril(torch.ones(L, L, dtype=torch.bool))
    return _to_float_mask(allowed)


def decoder_attn_mask(num_latents: int, num_patches: int) -> torch.Tensor:
    """Hybrid mask, sequence order [latents(k), mask tokens(N)].

    - latents -> latents: causal; latents -> mask tokens: blocked
    - mask tokens -> everything: allowed (bidirectional among patches)
    """
    k, N = num_latents, num_patches
    allowed = torch.zeros(k + N, k + N, dtype=torch.bool)
    allowed[:k, :k] = torch.tril(torch.ones(k, k, dtype=torch.bool))
    allowed[k:, :] = True
    return _to_float_mask(allowed)


class Encoder1D(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        channels: int,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        num_latent_tokens: int,
        latent_dim: int,
    ):
        super().__init__()
        assert image_size % patch_size == 0
        self.grid = image_size // patch_size
        self.num_patches = self.grid**2
        self.num_latent_tokens = num_latent_tokens

        self.patchify = nn.Conv2d(channels, hidden_dim, patch_size, patch_size)
        self.query_tokens = nn.Parameter(torch.zeros(1, num_latent_tokens, hidden_dim))
        self.patch_pos = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        self.query_pos = nn.Parameter(torch.zeros(1, num_latent_tokens, hidden_dim))
        nn.init.normal_(self.query_tokens, std=0.02)
        nn.init.normal_(self.patch_pos, std=0.02)
        nn.init.normal_(self.query_pos, std=0.02)

        self.blocks = nn.ModuleList([Block(hidden_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, latent_dim)

        self.register_buffer(
            "attn_mask",
            encoder_attn_mask(self.num_patches, num_latent_tokens),
            persistent=False,
        )

    def forward(self, x: torch.Tensor):
        """Returns (z, h_patch): 1D latents (B, L, d) and hidden patch embeddings (B, N, D)."""
        B = x.shape[0]
        p = self.patchify(x).flatten(2).transpose(1, 2) + self.patch_pos
        q = (self.query_tokens + self.query_pos).expand(B, -1, -1)
        seq = torch.cat([p, q], dim=1)
        mask = self.attn_mask.to(seq.dtype)
        for blk in self.blocks:
            seq = blk(seq, mask)
        seq = self.norm(seq)
        h_patch = seq[:, : self.num_patches]
        z = self.proj(seq[:, self.num_patches :])
        return z, h_patch


class Decoder1D(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        channels: int,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        num_latent_tokens: int,
        latent_dim: int,
        align_layer: int | None = None,
    ):
        super().__init__()
        self.grid = image_size // patch_size
        self.num_patches = self.grid**2
        self.num_latent_tokens = num_latent_tokens
        self.patch_size = patch_size
        self.channels = channels
        self.align_layer = align_layer

        self.latent_in = nn.Linear(latent_dim, hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.patch_pos = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        self.latent_pos = nn.Parameter(torch.zeros(1, num_latent_tokens, hidden_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.patch_pos, std=0.02)
        nn.init.normal_(self.latent_pos, std=0.02)

        self.blocks = nn.ModuleList([Block(hidden_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, patch_size * patch_size * channels)
        self.conv_out = nn.Conv2d(channels, channels, 3, padding=1)

        # Masks depend on the (possibly nested-dropout-truncated) latent count k.
        self._mask_cache: dict[int, torch.Tensor] = {}

    def _mask_for(self, k: int, device, dtype) -> torch.Tensor:
        if k not in self._mask_cache:
            self._mask_cache[k] = decoder_attn_mask(k, self.num_patches)
        return self._mask_cache[k].to(device=device, dtype=dtype)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        g, P, C = self.grid, self.patch_size, self.channels
        x = x.reshape(B, g, g, P, P, C).permute(0, 5, 1, 3, 2, 4)
        return x.reshape(B, C, g * P, g * P)

    def forward(self, z_q: torch.Tensor):
        """z_q: (B, k, d) with k <= L (nested dropout may truncate).

        Returns (x_rec, h_align) where h_align is the mask-token hidden state at
        `align_layer` (None if alignment is disabled).
        """
        B, k, _ = z_q.shape
        lat = self.latent_in(z_q) + self.latent_pos[:, :k]
        m = self.mask_token.expand(B, self.num_patches, -1) + self.patch_pos
        seq = torch.cat([lat, m], dim=1)
        mask = self._mask_for(k, seq.device, seq.dtype)

        h_align = None
        for i, blk in enumerate(self.blocks):
            seq = blk(seq, mask)
            if self.align_layer is not None and i == self.align_layer:
                h_align = seq[:, k:]

        out = self.norm(seq[:, k:])
        x = self.unpatchify(self.out(out))
        return self.conv_out(x), h_align
