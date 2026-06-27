"""Autoregressive image-token model with text-prefix conditioning.

This keeps the EOSTok-style soft visual-token training path, but makes caption
conditioning AR-native:

    [text prefix tokens, <img>, previous visual codes] -> next visual code

Text is no longer a side-channel through class embeddings, AdaLN, or
cross-attention. Projected Qwen hidden states are prepended to the visual token
stream and consumed by the same causal self-attention stack as image tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import TextConfig
from ..text import EncodedText, TextCondition, TextConditioner


KVCache = tuple[torch.Tensor, torch.Tensor]


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

    def _attend(
        self,
        x: torch.Tensor,
        key_mask: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, C // self.num_heads)
        q, k_new, v_new = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        cache_len = 0
        if kv_cache is None:
            k, v = k_new, v_new
        else:
            cache_len = kv_cache[0].shape[2]
            k = torch.cat([kv_cache[0], k_new], dim=2)
            v = torch.cat([kv_cache[1], v_new], dim=2)

        total_len = k.shape[2]
        if key_mask.shape != (B, total_len):
            raise ValueError(
                f"key_mask shape {tuple(key_mask.shape)} does not match attention "
                f"keys {(B, total_len)}"
            )

        q_pos = torch.arange(cache_len, cache_len + T, device=x.device)
        k_pos = torch.arange(total_len, device=x.device)
        causal = k_pos[None, :] <= q_pos[:, None]
        allow = causal[None, None, :, :] & key_mask[:, None, None, :]
        attn_mask = torch.zeros((B, 1, T, total_len), dtype=q.dtype, device=x.device)
        attn_mask.masked_fill_(~allow, torch.finfo(q.dtype).min)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.proj(out.transpose(1, 2).reshape(B, T, C)), (k, v)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        out, _ = self._attend(x, key_mask)
        return out

    def forward_cached(
        self,
        x: torch.Tensor,
        key_mask: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        return self._attend(x, key_mask, kv_cache)


class ARBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn = CausalAttention(dim, num_heads)
        self.norm2 = nn.RMSNorm(dim)
        self.mlp = SwiGLU(dim)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_mask)
        x = x + self.mlp(self.norm2(x))
        return x

    def forward_cached(
        self,
        x: torch.Tensor,
        key_mask: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        attn_out, next_cache = self.attn.forward_cached(self.norm1(x), key_mask, kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, next_cache


class ARModel(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        seq_len: int,
        dim: int,
        depth: int,
        num_heads: int,
        text_cfg: TextConfig,
        load_text_encoder: bool = True,
        text_encoder_dim: int | None = None,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.seq_len = seq_len
        self.max_text_tokens = text_cfg.max_length

        self.tok_emb = nn.Embedding(codebook_size, dim)
        self.img_start = nn.Parameter(torch.zeros(dim))
        self.text = TextConditioner(
            text_cfg,
            dim,
            load_encoder=load_text_encoder,
            encoder_dim=text_encoder_dim,
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, self.max_text_tokens + seq_len, dim))
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.img_start, std=0.02)

        self.blocks = nn.ModuleList([ARBlock(dim, num_heads) for _ in range(depth)])
        self.norm_f = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, codebook_size, bias=False)

    @staticmethod
    def null_condition(condition: str | list[str] | tuple[str, ...]) -> list[str]:
        captions = TextConditioner.normalize_captions(condition)
        return [""] * len(captions)

    @staticmethod
    def drop_condition(condition: str | list[str] | tuple[str, ...], drop: torch.Tensor) -> list[str]:
        captions = TextConditioner.normalize_captions(condition)
        return ["" if bool(should_drop) else caption for caption, should_drop in zip(captions, drop)]

    def embed_soft(self, ind: torch.Tensor) -> torch.Tensor:
        """Soft token embedding h = Ind^T Embed (differentiable into the tokenizer)."""
        return ind @ self.tok_emb.weight

    def _match_text_batch(self, text: TextCondition, batch_size: int) -> TextCondition:
        if text.tokens.shape[0] == batch_size:
            return text
        if text.tokens.shape[0] == 1:
            return TextCondition(
                tokens=text.tokens.expand(batch_size, -1, -1),
                mask=text.mask.expand(batch_size, -1),
            )
        raise ValueError(
            f"Text batch ({text.tokens.shape[0]}) does not match image batch ({batch_size})"
        )

    def _visual_input(self, tok_embs: torch.Tensor) -> torch.Tensor:
        B = tok_embs.shape[0]
        start = self.img_start.to(dtype=tok_embs.dtype).view(1, 1, -1).expand(B, 1, -1)
        return torch.cat([start, tok_embs[:, :-1]], dim=1)

    def _add_pos(self, x: torch.Tensor, start: int) -> torch.Tensor:
        end = start + x.shape[1]
        if end > self.pos_emb.shape[1]:
            raise ValueError(
                f"Sequence length {end} exceeds positional capacity {self.pos_emb.shape[1]}"
            )
        return x + self.pos_emb[:, start:end].to(dtype=x.dtype)

    def _prefix_sequence(
        self,
        text: TextCondition,
        visual_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        text_tokens = text.tokens.to(dtype=visual_in.dtype)
        text_mask = text.mask
        visual_mask = torch.ones(
            visual_in.shape[:2],
            dtype=torch.bool,
            device=visual_in.device,
        )
        x = torch.cat([text_tokens, visual_in], dim=1)
        key_mask = torch.cat([text_mask, visual_mask], dim=1)
        x = self._add_pos(x, 0)
        return x, key_mask, text_tokens.shape[1]

    def _run(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, key_mask)
        return self.norm_f(x)

    def _run_cached(
        self,
        x: torch.Tensor,
        key_mask: torch.Tensor,
        caches: list[KVCache] | None = None,
    ) -> tuple[torch.Tensor, list[KVCache]]:
        if caches is None:
            cache_iter: list[KVCache | None] = [None] * len(self.blocks)
        else:
            if len(caches) != len(self.blocks):
                raise ValueError(f"Expected {len(self.blocks)} caches, got {len(caches)}")
            cache_iter = list(caches)

        next_caches = []
        for blk, cache in zip(self.blocks, cache_iter):
            x, next_cache = blk.forward_cached(x, key_mask, cache)
            next_caches.append(next_cache)
        return x, next_caches

    def forward(
        self,
        tok_embs: torch.Tensor,
        condition: str | list[str] | tuple[str, ...] | EncodedText,
        condition_drop: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forced forward.

        tok_embs: (B, L, dim) soft embeddings of ground-truth visual codes.
        Returns logits (B, L, K). The input visual stream is shifted as
        ``<img>, code_1, ..., code_{L-1}``, while the caption is a prefix.
        """
        B, L, _ = tok_embs.shape
        text = self._match_text_batch(self.text(condition, tok_embs.device, condition_drop), B)
        visual_in = self._visual_input(tok_embs)
        x, key_mask, text_len = self._prefix_sequence(text, visual_in)
        h = self._run(x, key_mask)
        return self.head(h[:, text_len : text_len + L])

    @torch.no_grad()
    def generate(
        self,
        condition: str | list[str] | tuple[str, ...] | EncodedText,
        temperature: float = 1.0,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """Sample (B, L) code indices.

        CFG uses the same prefix-AR path with an empty text prefix as the
        unconditional branch: l_g = l_u + s * (l_c - l_u).
        """
        if isinstance(condition, EncodedText):
            encoded = condition.to(self.pos_emb.device)
            base_batch = encoded.hidden.shape[0]
            if cfg_scale != 1.0:
                encoded = EncodedText(
                    torch.cat([encoded.hidden, encoded.hidden], dim=0),
                    torch.cat([encoded.mask, encoded.mask], dim=0),
                )
                force_empty = torch.cat([
                    torch.zeros(base_batch, dtype=torch.bool, device=self.pos_emb.device),
                    torch.ones(base_batch, dtype=torch.bool, device=self.pos_emb.device),
                ])
            else:
                force_empty = None
            text = self.text(encoded, self.pos_emb.device, force_empty)
        else:
            captions = TextConditioner.normalize_captions(condition)
            use_cfg = cfg_scale != 1.0
            if use_cfg:
                captions = captions + self.null_condition(captions)
            text = self.text(captions, self.pos_emb.device)

        use_cfg = cfg_scale != 1.0
        batch = text.tokens.shape[0]
        tokens = []
        dtype = self.tok_emb.weight.dtype
        device = self.pos_emb.device

        text_x = self._add_pos(text.tokens.to(dtype=dtype), 0)
        key_mask = text.mask
        _, caches = self._run_cached(text_x, key_mask)

        step_x = self.img_start.to(dtype=dtype).view(1, 1, -1).expand(batch, 1, -1)
        for step in range(self.seq_len):
            step_x = self._add_pos(step_x, key_mask.shape[1])
            step_mask = torch.ones((batch, 1), dtype=torch.bool, device=device)
            step_key_mask = torch.cat([key_mask, step_mask], dim=1)
            h, caches = self._run_cached(step_x, step_key_mask, caches)
            logits = self.head(self.norm_f(h[:, -1]))
            if use_cfg:
                lc, lu = logits.chunk(2)
                logits = lu + cfg_scale * (lc - lu)
            probs = (logits / max(temperature, 1e-6)).softmax(dim=-1)
            next_token = torch.multinomial(probs, 1).squeeze(-1)
            tokens.append(next_token)

            key_mask = step_key_mask
            if step < self.seq_len - 1:
                next_input = next_token.repeat(2) if use_cfg else next_token
                step_x = self.tok_emb(next_input).unsqueeze(1)

        return torch.stack(tokens, dim=1)
