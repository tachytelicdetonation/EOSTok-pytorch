"""Text conditioning for caption-only autoregressive image generation.

The 2026 AR/T2I papers this fork follows use language-token interfaces rather
than dataset-local learned labels. This module provides a scaled-down version:
a frozen pretrained text encoder supplies prefix tokens that the AR image model
consumes through causal self-attention.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..config import TextConfig


@dataclass
class TextCondition:
    tokens: torch.Tensor
    mask: torch.Tensor


@dataclass
class EncodedText:
    """Frozen text-encoder outputs before the trainable ImageGen projection."""

    hidden: torch.Tensor
    mask: torch.Tensor

    def __len__(self) -> int:
        return int(self.batched().hidden.shape[0])

    def batched(self) -> "EncodedText":
        if self.hidden.ndim == 2:
            return EncodedText(self.hidden.unsqueeze(0), self.mask.unsqueeze(0))
        return self

    def to(self, device: torch.device) -> "EncodedText":
        batch = self.batched()
        return EncodedText(
            batch.hidden.to(device=device),
            batch.mask.to(device=device, dtype=torch.bool),
        )

    def take(self, index) -> "EncodedText":
        batch = self.batched()
        return EncodedText(batch.hidden[index], batch.mask[index]).batched()

    @staticmethod
    def collate(items: list["EncodedText"]) -> "EncodedText":
        hidden = torch.stack([item.batched().hidden.squeeze(0) for item in items])
        mask = torch.stack([item.batched().mask.squeeze(0) for item in items])
        return EncodedText(hidden, mask)


@dataclass
class _EncoderOutput:
    last_hidden_state: torch.Tensor


class _TinyTextBackbone(nn.Module):
    """Test-only text backbone used by smoke tests without network downloads."""

    def __init__(self, hidden_dim: int, vocab_size: int = 512):
        super().__init__()
        self.hidden_size = hidden_dim
        self.emb = nn.Embedding(vocab_size, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def encode(captions: list[str], max_length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.zeros(len(captions), max_length, dtype=torch.long, device=device)
        mask = torch.zeros(len(captions), max_length, dtype=torch.bool, device=device)
        for row, caption in enumerate(captions):
            values = [ord(ch) % 511 + 1 for ch in caption[:max_length]]
            if not values:
                values = [0]
            ids[row, : len(values)] = torch.tensor(values, device=device)
            mask[row, : len(values)] = True
        return ids, mask

    def forward(self, input_ids: torch.Tensor, **_) -> _EncoderOutput:
        return _EncoderOutput(last_hidden_state=self.norm(self.emb(input_ids)))


class TextConditioner(nn.Module):
    def __init__(
        self,
        cfg: TextConfig,
        out_dim: int,
        load_encoder: bool = True,
        encoder_dim: int | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.out_dim = out_dim
        self.max_length = cfg.max_length
        self.use_tiny = cfg.model_name == "__tiny__"
        self.encoder_loaded = load_encoder

        if self.use_tiny:
            self.tokenizer = None
            if encoder_dim is None:
                encoder_dim = out_dim
            self.encoder = _TinyTextBackbone(encoder_dim) if load_encoder else None
        else:
            if load_encoder:
                from transformers import AutoModel, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(
                    cfg.model_name,
                    trust_remote_code=cfg.trust_remote_code,
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.encoder = AutoModel.from_pretrained(
                    cfg.model_name,
                    trust_remote_code=cfg.trust_remote_code,
                )
                encoder_dim = self._hidden_size(self.encoder.config)
            else:
                self.tokenizer = None
                self.encoder = None
                if encoder_dim is None:
                    encoder_dim = self._resolve_hidden_size(cfg)

        if cfg.freeze and self.encoder is not None:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        self.token_proj = nn.Linear(encoder_dim, out_dim)
        self.null_token = nn.Parameter(torch.zeros(out_dim))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.cfg.freeze and self.encoder is not None:
            self.encoder.eval()
        return self

    def unload_encoder(self):
        """Drop the live text encoder/tokenizer after cached captions exist."""
        self.encoder = None
        self.tokenizer = None
        self.encoder_loaded = False
        return self

    @staticmethod
    def _hidden_size(config) -> int:
        if hasattr(config, "hidden_size"):
            return config.hidden_size
        if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
            return config.text_config.hidden_size
        raise ValueError("Text encoder config does not expose hidden_size or text_config.hidden_size")

    @classmethod
    def _resolve_hidden_size(cls, cfg: TextConfig) -> int:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            cfg.model_name,
            trust_remote_code=cfg.trust_remote_code,
        )
        return cls._hidden_size(config)

    @staticmethod
    def normalize_captions(captions: str | list[str] | tuple[str, ...]) -> list[str]:
        if isinstance(captions, str):
            return [captions]
        return [str(c) for c in captions]

    def _require_encoder(self):
        if self.encoder is None or (not self.use_tiny and self.tokenizer is None):
            raise RuntimeError(
                "Live text encoder is not loaded. Pass cached EncodedText "
                "conditions, or construct the model with load_text_encoder=True."
            )

    def _tokenize(self, captions: list[str], device: torch.device):
        if self.use_tiny:
            self._require_encoder()
            return _TinyTextBackbone.encode(captions, self.max_length, device)

        self._require_encoder()
        batch = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device=device, dtype=torch.bool)
        return input_ids, mask

    def _encode_hidden(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": mask.to(dtype=torch.long),
            "output_hidden_states": True,
            "return_dict": True,
        }
        try:
            output = self.encoder(**kwargs, use_cache=False)
        except TypeError:
            output = self.encoder(**kwargs)

        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            return output.last_hidden_state
        if hasattr(output, "hidden_states") and output.hidden_states:
            return output.hidden_states[-1]
        raise RuntimeError("Text encoder output does not include hidden states")

    @torch.no_grad()
    def encode(self, captions: str | list[str] | tuple[str, ...], device: torch.device) -> EncodedText:
        """Run only the frozen text encoder and return raw hidden states."""
        captions = self.normalize_captions(captions)
        input_ids, mask = self._tokenize(captions, device)
        hidden = self._encode_hidden(input_ids, mask)
        return EncodedText(hidden.detach().cpu(), mask.detach().cpu())

    def _project_encoded(
        self,
        encoded: EncodedText,
        device: torch.device,
        force_empty: torch.Tensor | None = None,
    ) -> TextCondition:
        encoded = encoded.to(device)
        hidden = encoded.hidden.to(self.token_proj.weight.dtype)
        mask = encoded.mask
        text_tokens = self.token_proj(hidden)

        empty = ~mask.any(dim=1)
        if force_empty is not None:
            empty = empty | force_empty.to(device=device, dtype=torch.bool)
        if empty.any():
            mask = mask.clone()
            text_tokens = text_tokens.clone()
            mask[empty] = False
            mask[empty, 0] = True
            text_tokens[empty] = 0
            text_tokens[empty, 0] = self.null_token

        return TextCondition(tokens=text_tokens, mask=mask)

    def forward(
        self,
        captions: str | list[str] | tuple[str, ...] | EncodedText,
        device: torch.device,
        force_empty: torch.Tensor | None = None,
    ) -> TextCondition:
        if isinstance(captions, EncodedText):
            return self._project_encoded(captions, device, force_empty)

        captions = self.normalize_captions(captions)
        null_caption = torch.tensor(
            [caption.strip() == "" for caption in captions],
            dtype=torch.bool,
            device=device,
        )
        if force_empty is not None:
            null_caption = null_caption | force_empty.to(device=device, dtype=torch.bool)
        input_ids, mask = self._tokenize(captions, device)

        if self.cfg.freeze:
            with torch.no_grad():
                hidden = self._encode_hidden(input_ids, mask)
        else:
            hidden = self._encode_hidden(input_ids, mask)

        hidden = hidden.to(self.token_proj.weight.dtype)
        text_tokens = self.token_proj(hidden)

        empty = null_caption | ~mask.any(dim=1)
        if empty.any():
            mask = mask.clone()
            text_tokens = text_tokens.clone()
            mask[empty] = False
            mask[empty, 0] = True
            text_tokens[empty] = 0
            text_tokens[empty, 0] = self.null_token

        return TextCondition(tokens=text_tokens, mask=mask)
