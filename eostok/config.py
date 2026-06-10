"""Typed configuration loaded from YAML. Field defaults follow EOSTok Table 9."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    dataset: str = "mnist"  # "mnist" | "imagefolder"
    root: str = "./data"
    image_size: int = 32
    channels: int = 1
    num_classes: int = 10
    batch_size: int = 128
    num_workers: int = 2


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
    class_dropout: float = 0.1


@dataclass
class VFMConfig:
    enabled: bool = False
    model: str = "dinov2_vitl14"
    decoder_align_layer: int = 2  # k-th decoder layer for decoder alignment
    weight: float = 1.0  # lambda_sem


@dataclass
class LossConfig:
    recon_l2: float = 1.0
    recon_lpips: float = 1.0
    gan: float = 0.1
    lecam: float = 0.05
    apr_l2: float = 1.0
    apr_lpips: float = 1.0
    ntp: float = 0.1
    disc_start: int = 0  # generator sees GAN loss only after this step
    lpips_enabled: bool = True


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
    run_name: str = "eostok"
    seed: int = 42
    device: str = "auto"  # auto | cuda | mps | cpu
    amp: str = "auto"  # auto (bf16 on cuda, off elsewhere) | bf16 | off
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    quantizer: QuantizerConfig = field(default_factory=QuantizerConfig)
    ar: ARConfig = field(default_factory=ARConfig)
    vfm: VFMConfig = field(default_factory=VFMConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _build(cls, d: dict):
    kwargs = {}
    valid = {f.name: f for f in fields(cls)}
    for key, val in d.items():
        if key not in valid:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}")
        ftype = valid[key].type
        sub = {
            "data": DataConfig, "tokenizer": TokenizerConfig,
            "quantizer": QuantizerConfig, "ar": ARConfig,
            "vfm": VFMConfig, "loss": LossConfig, "train": TrainConfig,
        }.get(key)
        kwargs[key] = _build(sub, val) if sub and isinstance(val, dict) else val
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return _build(Config, raw)
