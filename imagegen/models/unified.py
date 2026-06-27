"""Stage-1 multimodal graft validation: image+text -> image+text.

This is NOT the production backbone (that reuses Qwen3.5-0.8B on a GPU, via
Anole/Liquid-style early-fusion -- see project memory). It proves, offline and
RAM-safe, the load-bearing unknowns of the unified single-stream design against
our EXISTING ARBlock trunk:

  - one shared vocabulary  [ text ids | image codes | <boi> <eoi> ]
  - an interleaved stream  [ text, <boi>, L image codes, <eoi>, text ]
  - a single causal next-token cross-entropy over BOTH text and image positions
  - grafting K image-code rows onto a text embedding + LM head (Anole recipe)

It deliberately does NOT swap in Qwen, build interleaved datasets, write the
modality-switching generate loop, or add LoRA -- all of that is downstream
Stage 2/3 work once the image-only baseline lands.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ar import ARBlock


def graft_image_vocab(
    embed: nn.Embedding, head: nn.Linear, n_new: int
) -> tuple[nn.Embedding, nn.Linear]:
    """Anole-style early fusion: extend a (pretrained) text embedding table and
    LM head by ``n_new`` rows for the image codes + <boi>/<eoi>, initializing the
    new rows from the mean of the existing rows so they start near the text
    manifold instead of as noise. Mirrors HF ``resize_token_embeddings`` + a head
    graft, which is the actual operation we will run on Qwen3.5-0.8B."""
    dim = embed.embedding_dim
    old_v = embed.num_embeddings
    new_embed = nn.Embedding(old_v + n_new, dim)
    new_head = nn.Linear(dim, old_v + n_new, bias=head.bias is not None)
    with torch.no_grad():
        new_embed.weight[:old_v] = embed.weight
        new_embed.weight[old_v:] = embed.weight.mean(dim=0, keepdim=True)
        new_head.weight[:old_v] = head.weight
        new_head.weight[old_v:] = head.weight.mean(dim=0, keepdim=True)
    return new_embed, new_head


class UnifiedStream(nn.Module):
    """Causal AR over an interleaved text+image token stream, reusing the
    existing ARBlock trunk. Vocabulary layout:
        [0, text_vocab)                 text ids
        [text_vocab, text_vocab + K)    image codes (code c -> text_vocab + c)
        boi_id = text_vocab + K, eoi_id = text_vocab + K + 1
    """

    def __init__(
        self,
        text_vocab: int,
        codebook_size: int,
        dim: int,
        depth: int,
        num_heads: int,
        max_len: int = 256,
    ):
        super().__init__()
        self.text_vocab = text_vocab
        self.codebook_size = codebook_size
        self.boi_id = text_vocab + codebook_size
        self.eoi_id = text_vocab + codebook_size + 1
        self.vocab_size = text_vocab + codebook_size + 2

        # Build as a (mock-pretrained) text-only model, then graft the image
        # rows -- exactly the operation we will run on the real Qwen head.
        text_embed = nn.Embedding(text_vocab, dim)
        text_head = nn.Linear(dim, text_vocab, bias=False)
        nn.init.normal_(text_embed.weight, std=0.02)
        nn.init.normal_(text_head.weight, std=0.02)  # stands in for pretrained head
        self.embed, self.head = graft_image_vocab(
            text_embed, text_head, codebook_size + 2
        )

        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.blocks = nn.ModuleList([ARBlock(dim, num_heads) for _ in range(depth)])
        self.norm_f = nn.RMSNorm(dim)

    def image_token_id(self, code: torch.Tensor) -> torch.Tensor:
        """Map IBQ code indices to their unified-vocab token ids."""
        return code + self.text_vocab

    def forward(self, input_ids: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        """input_ids/key_mask: (B, T). key_mask True == real token (False == pad).
        Returns (B, T, vocab_size). ARBlock applies causal masking and honors the
        key_mask for padding, so no extra masking is needed here."""
        T = input_ids.shape[1]
        if T > self.pos_emb.shape[1]:
            raise ValueError(f"sequence length {T} exceeds capacity {self.pos_emb.shape[1]}")
        x = self.embed(input_ids) + self.pos_emb[:, :T]
        for blk in self.blocks:
            x = blk(x, key_mask)
        return self.head(self.norm_f(x))


Sample = tuple[list[int], torch.Tensor, list[int]]


def build_interleaved_batch(
    samples: list[Sample], model: UnifiedStream, pad_id: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a padded interleaved batch. Each sample is
    ``(text_before_ids, image_codes, text_after_ids)`` and becomes the stream
    ``text_before + <boi> + image_tokens + <eoi> + text_after``. Returns
    (input_ids, key_mask) with key_mask True on real tokens. This is the toy
    stand-in for the Stage-3 mixed image-text collator."""
    seqs: list[list[int]] = []
    for text_before, codes, text_after in samples:
        image_ids = model.image_token_id(codes).tolist()
        seqs.append(
            [*text_before, model.boi_id, *image_ids, model.eoi_id, *text_after]
        )
    length = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), length), pad_id, dtype=torch.long)
    key_mask = torch.zeros(len(seqs), length, dtype=torch.bool)
    for i, seq in enumerate(seqs):
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        key_mask[i, : len(seq)] = True
    return input_ids, key_mask


def unified_ntp_loss(
    logits: torch.Tensor, input_ids: torch.Tensor, key_mask: torch.Tensor
) -> torch.Tensor:
    """Single shifted next-token cross-entropy over the WHOLE stream -- text
    positions predict text ids and image positions predict image codes, under one
    shared softmax (Chameleon/Liquid early fusion). Pad targets are ignored."""
    pred = logits[:, :-1]
    targets = input_ids[:, 1:].clone()
    targets[~key_mask[:, 1:]] = -100
    return F.cross_entropy(
        pred.reshape(-1, pred.shape[-1]), targets.reshape(-1), ignore_index=-100
    )
