"""Autoregressive generative model, following EOSTok A.1.2.

LlamaGen-style decoder-only transformer with:
  - RMSNorm and SwiGLU activation
  - learnable positional embeddings (1D tokens; 2D RoPE removed)
  - a shared global AdaLN modulation (from the class condition) with
    per-block learnable biases (PixArt-alpha style)

During joint training the input tokens are *soft* embeddings
h = Ind^T Embed (Ind from the IBQ quantizer), so NTP/APR gradients flow back
into the tokenizer. Sampling uses a KV cache, temperature 1.0, no top-k/top-p
(paper A.3), with optional classifier-free guidance.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hidden = int(2 * (4 * dim) / 3)
        hidden = (hidden + 31) // 32 * 32
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, kv_cache: list | None = None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if kv_cache is not None:
            if len(kv_cache) == 2:  # append to existing cache
                k = torch.cat([kv_cache[0], k], dim=2)
                v = torch.cat([kv_cache[1], v], dim=2)
            kv_cache[:] = [k, v]
            # With a cache we only ever feed new tokens; they may attend to
            # everything cached, so causal masking is only needed when T > 1.
            is_causal = T > 1 and k.shape[2] == T
        else:
            is_causal = T > 1
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        return self.proj(out.transpose(1, 2).reshape(B, T, C))


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ARBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn = CausalAttention(dim, num_heads)
        self.norm2 = nn.RMSNorm(dim)
        self.mlp = SwiGLU(dim)
        # Per-block learnable bias added to the shared global AdaLN modulation.
        self.adaln_bias = nn.Parameter(torch.zeros(6 * dim))

    def forward(self, x, shared_mod, kv_cache=None):
        mod = shared_mod + self.adaln_bias
        s1, g1, sh1, s2, g2, sh2 = mod.chunk(6, dim=-1)
        x = x + g1.unsqueeze(1) * self.attn(modulate(self.norm1(x), sh1, s1), kv_cache)
        x = x + g2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), sh2, s2))
        return x


class ARModel(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        num_classes: int,
        seq_len: int,
        dim: int,
        depth: int,
        num_heads: int,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.num_classes = num_classes
        self.seq_len = seq_len

        self.tok_emb = nn.Embedding(codebook_size, dim)
        # +1 slot: the "null" class used for class dropout / CFG.
        self.cls_emb = nn.Embedding(num_classes + 1, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, dim))
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.cls_emb.weight, std=0.02)

        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.zeros_(self.adaln[1].weight)
        nn.init.zeros_(self.adaln[1].bias)

        self.blocks = nn.ModuleList([ARBlock(dim, num_heads) for _ in range(depth)])
        self.norm_f = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, codebook_size, bias=False)

    @property
    def null_class(self) -> int:
        return self.num_classes

    def embed_soft(self, ind: torch.Tensor) -> torch.Tensor:
        """Soft token embedding h = Ind^T Embed (differentiable into the tokenizer)."""
        return ind @ self.tok_emb.weight

    def forward(self, tok_embs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Teacher-forced forward.

        tok_embs: (B, L, dim) soft embeddings of the L ground-truth tokens.
        Returns logits (B, L, K) predicting tokens 1..L (input is shifted: the
        class embedding stands in for position 0).
        """
        B, L, _ = tok_embs.shape
        cond = self.cls_emb(labels)
        x = torch.cat([cond.unsqueeze(1), tok_embs[:, :-1]], dim=1) + self.pos_emb[:, :L]
        shared_mod = self.adaln(cond)
        for blk in self.blocks:
            x = blk(x, shared_mod)
        return self.head(self.norm_f(x))

    @torch.no_grad()
    def generate(
        self,
        labels: torch.Tensor,
        temperature: float = 1.0,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """Sample (B, L) code indices with a KV cache.

        cfg_scale s applies l_g = l_u + s * (l_c - l_u); s == 1 disables CFG.
        """
        B = labels.shape[0]
        device = labels.device
        use_cfg = cfg_scale != 1.0
        if use_cfg:
            labels = torch.cat([labels, torch.full_like(labels, self.null_class)])

        cond = self.cls_emb(labels)
        shared_mod = self.adaln(cond)
        caches = [[] for _ in self.blocks]
        x = cond.unsqueeze(1) + self.pos_emb[:, :1]

        tokens = []
        for t in range(self.seq_len):
            h = x
            for blk, cache in zip(self.blocks, caches):
                h = blk(h, shared_mod, kv_cache=cache)
            logits = self.head(self.norm_f(h[:, -1]))
            if use_cfg:
                lc, lu = logits.chunk(2)
                logits = lu + cfg_scale * (lc - lu)
            probs = (logits / max(temperature, 1e-6)).softmax(dim=-1)
            nxt = torch.multinomial(probs, 1).squeeze(-1)  # (B,)
            tokens.append(nxt)
            if t < self.seq_len - 1:
                nxt_in = nxt.repeat(2) if use_cfg else nxt
                x = self.tok_emb(nxt_in).unsqueeze(1) + self.pos_emb[:, t + 1 : t + 2]
        return torch.stack(tokens, dim=1)
