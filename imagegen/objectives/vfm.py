"""Vision foundation model (VFM) representation alignment, EOSTok Sec. 3.3.

Two alignment losses, both cosine-similarity to frozen VFM patch features
y = f(x) through small learnable MLP projectors:

  - implicit alignment (encoder): align hidden *patch* embeddings h_enc
    (NOT the 1D latents -- that would leak the 2D raster prior, Fig. 4c)
  - decoder alignment: align mask-token hidden states at the k-th decoder layer

The VFM (default DINOv2) is frozen; only the projectors train.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

# ImageNet normalization used by DINOv2.
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class MlpProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class VFMAligner(nn.Module):
    # ponytail: the hub DINOv2 has no stubs; Any is its honest type so its
    # dynamic attrs (patch_size, embed_dim, forward_features) type-check.
    vfm: Any

    def __init__(
        self,
        model_name: str,
        enc_hidden_dim: int,
        dec_hidden_dim: int,
        tokenizer_grid: int,
    ):
        super().__init__()
        self.tokenizer_grid = tokenizer_grid
        # cast: hub.load is typed -> object, which would narrow self.vfm here and
        # mask its dynamic DINOv2 attributes; Any is the honest boundary type.
        self.vfm = cast(Any, torch.hub.load("facebookresearch/dinov2", model_name))
        self.vfm.eval()
        for p in self.vfm.parameters():
            p.requires_grad_(False)
        self.vfm_patch = self.vfm.patch_size  # 14 for DINOv2
        embed_dim = self.vfm.embed_dim
        self.proj_enc = MlpProjector(enc_hidden_dim, embed_dim)
        self.proj_dec = MlpProjector(dec_hidden_dim, embed_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vfm.eval()  # VFM stays frozen in eval mode
        return self

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """x in [-1, 1], any channel count. Returns (B, N_tok, D) patch features
        on the tokenizer's patch grid."""
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x + 1) / 2
        x = (x - _MEAN.to(x)) / _STD.to(x)
        # Resize so the VFM patch grid matches the tokenizer patch grid.
        side = self.tokenizer_grid * self.vfm_patch
        x = F.interpolate(x, size=(side, side), mode="bicubic", align_corners=False)
        return self.vfm.forward_features(x)["x_norm_patchtokens"]

    @staticmethod
    def _cosine_align(h_proj: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -F.cosine_similarity(h_proj, y, dim=-1).mean()

    def implicit_loss(self, h_enc: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Eq. 6: align encoder hidden patch embeddings to VFM features."""
        return self._cosine_align(self.proj_enc(h_enc), y)

    def decoder_loss(self, h_dec: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Align k-th-layer decoder mask-token hidden states to VFM features."""
        return self._cosine_align(self.proj_dec(h_dec), y)
