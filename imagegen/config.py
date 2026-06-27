"""Typed configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import get_type_hints

import yaml


@dataclass
class DataConfig:
    dataset: str = "hf_image_caption"
    root: str = "./data"
    image_size: int = 64
    channels: int = 3
    batch_size: int = 128
    num_workers: int = 2
    hf_name: str = ""
    hf_config: str | None = None
    revision: str | None = None  # pin a dataset commit/tag to freeze content + order
    train_split: str = "train"
    val_split: str = "validation"
    image_column: str = "image"
    caption_column: str = "caption"


@dataclass
class TextConfig:
    model_name: str = "Qwen/Qwen3.5-0.8B"
    revision: str | None = None  # pin an encoder commit/tag so cached features can't silently drift
    max_length: int = 128
    freeze: bool = True
    trust_remote_code: bool = False
    save_encoder_state: bool = False
    cache_dataset: bool = True
    cache_dir: str = "data/text_cache"
    cache_batch_size: int = 64
    cache_dtype: str = "float16"


@dataclass
class TokenizerConfig:
    patch_size: int = 4
    hidden_dim: int = 256
    enc_layers: int = 4
    dec_layers: int = 4
    num_heads: int = 4
    num_latent_tokens: int = 16  # L
    latent_dim: int = 16  # d


@dataclass
class QuantizerConfig:
    codebook_size: int = 512  # K
    temperature: float = 1.0
    commit_weight: float = 1.0e-3  # lambda_reg
    entropy_weight: float = 0.01


@dataclass
class ARConfig:
    layers: int = 4
    hidden_dim: int = 256
    num_heads: int = 4
    condition_dropout: float = 0.1


@dataclass
class VFMConfig:
    enabled: bool = False
    model: str = "dinov2_vitl14"
    decoder_align_layer: int = 2  # k-th decoder layer for decoder alignment
    weight: float = 1.0  # lambda_sem


@dataclass
class LossConfig:
    recon_l2: float = 1.0
    recon_lpips: float = 0.0
    gan: float = 0.0
    lecam: float = 0.05
    apr_l2: float = 1.0
    apr_lpips: float = 0.0
    ntp: float = 0.1
    disc_start: int = 0  # generator sees GAN loss only after this step
    lpips_enabled: bool = False


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1.0e-4
    min_lr: float = 1.0e-6
    beta1: float = 0.9
    beta2_tokenizer: float = 0.999
    beta2_ar: float = 0.95
    disc_lr: float = 1.0e-4
    ema_decay: float = 0.9999
    nested_dropout: float = 0.5  # probability of truncating decoder latents
    grad_clip: float = 1.0
    log_every: int = 50
    sample_every: int = 1000
    ckpt_every: int = 2000
    out_dir: str = "runs/default"


@dataclass
class Config:
    run_name: str = "imagegen"
    seed: int = 42
    device: str = "auto"  # auto | cuda | mps | cpu
    amp: str = "auto"  # auto (bf16 on cuda, off elsewhere) | bf16 | off
    data: DataConfig = field(default_factory=DataConfig)
    text: TextConfig = field(default_factory=TextConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    quantizer: QuantizerConfig = field(default_factory=QuantizerConfig)
    ar: ARConfig = field(default_factory=ARConfig)
    vfm: VFMConfig = field(default_factory=VFMConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _build(cls, d: dict):
    # The field's declared type is the only source of truth for nesting: any
    # field annotated as a dataclass (DataConfig, TrainConfig, ...) recurses.
    # `get_type_hints` resolves the string annotations that `from __future__
    # import annotations` leaves on the fields.
    hints = get_type_hints(cls)
    valid = {f.name for f in fields(cls)}
    kwargs = {}
    for key, val in d.items():
        if key not in valid:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}")
        ftype = hints[key]
        if is_dataclass(ftype) and isinstance(val, dict):
            kwargs[key] = _build(ftype, val)
        else:
            kwargs[key] = val
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return _build(Config, raw)
