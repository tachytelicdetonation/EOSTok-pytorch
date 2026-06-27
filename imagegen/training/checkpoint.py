"""Checkpoint helpers for caption-conditioned ImageGen models."""

from __future__ import annotations

import torch.nn as nn

from ..config import Config


TEXT_ENCODER_PREFIX = "ar.text.encoder."


def omits_text_encoder(cfg: Config) -> bool:
    return cfg.text.freeze and not cfg.text.save_encoder_state


def checkpoint_state_dict(module: nn.Module, cfg: Config) -> dict:
    """State dict for checkpoints.

    The text encoder is loaded from its pretrained source at construction time.
    By default we save only the trainable caption-conditioning projections and
    AR/image modules so checkpoints do not duplicate a frozen LLM/VLM backbone.
    """
    if not cfg.text.freeze and not cfg.text.save_encoder_state:
        raise ValueError(
            "cfg.text.save_encoder_state must be true when cfg.text.freeze is false; "
            "otherwise trainable text-encoder weights would be dropped."
        )
    state = module.state_dict()
    if not omits_text_encoder(cfg):
        return state
    return {k: v for k, v in state.items() if not k.startswith(TEXT_ENCODER_PREFIX)}


def load_checkpoint_state(module: nn.Module, state: dict, cfg: Config):
    """Load a model/EMA state, allowing omitted frozen text-encoder weights."""
    if not cfg.text.freeze and not cfg.text.save_encoder_state:
        raise ValueError(
            "cfg.text.save_encoder_state must be true when cfg.text.freeze is false; "
            "otherwise trainable text-encoder weights would be missing."
        )
    if not omits_text_encoder(cfg):
        module.load_state_dict(state)
        return

    result = module.load_state_dict(state, strict=False)
    missing = set(result.missing_keys)
    unexpected = set(result.unexpected_keys)
    allowed_missing = {
        key for key in module.state_dict()
        if key.startswith(TEXT_ENCODER_PREFIX)
    }
    bad_missing = sorted(missing - allowed_missing)
    if bad_missing or unexpected:
        parts = []
        if bad_missing:
            parts.append(f"missing keys: {bad_missing[:8]}")
        if unexpected:
            parts.append(f"unexpected keys: {sorted(unexpected)[:8]}")
        raise RuntimeError("Invalid checkpoint state (" + "; ".join(parts) + ")")
