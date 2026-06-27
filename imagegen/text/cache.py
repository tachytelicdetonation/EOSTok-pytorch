"""Disk cache for frozen text-encoder caption features."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import torch
from tqdm import tqdm

from ..config import Config, DataConfig
from .conditioner import EncodedText, TextConditioner


CACHE_VERSION = 1


@dataclass
class CaptionFeatureCache:
    hidden: torch.Tensor
    mask: torch.Tensor
    captions: list[str]
    metadata: dict
    path: Path | None = None

    def __len__(self) -> int:
        return int(self.hidden.shape[0])

    @property
    def encoder_dim(self) -> int:
        return int(self.hidden.shape[-1])

    def __getitem__(self, idx: int) -> EncodedText:
        return EncodedText(self.hidden[idx], self.mask[idx])

    def take(self, indices: torch.Tensor | list[int]) -> EncodedText:
        if not torch.is_tensor(indices):
            indices = torch.tensor(indices, dtype=torch.long)
        return EncodedText(self.hidden[indices], self.mask[indices])


def _split_name(data: DataConfig, train: bool) -> str:
    return data.train_split if train else data.val_split


def _cache_metadata(cfg: Config, train: bool) -> dict:
    data, text = cfg.data, cfg.text
    return {
        "version": CACHE_VERSION,
        "dataset": data.dataset,
        "hf_name": data.hf_name,
        "hf_config": data.hf_config,
        "split": _split_name(data, train),
        "caption_column": data.caption_column,
        "model_name": text.model_name,
        "max_length": text.max_length,
        "trust_remote_code": text.trust_remote_code,
        "cache_dtype": text.cache_dtype,
    }


def caption_cache_path(cfg: Config, train: bool) -> Path:
    metadata = _cache_metadata(cfg, train)
    key = sha256(repr(sorted(metadata.items())).encode("utf-8")).hexdigest()[:16]
    split = metadata["split"].replace("/", "_")
    return Path(cfg.text.cache_dir) / f"{cfg.run_name}-{split}-{key}.pt"


def _metadata_matches(actual: dict, expected: dict) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


def load_caption_cache(path: str | Path, expected_metadata: dict | None = None) -> CaptionFeatureCache:
    path = Path(path)
    payload = torch.load(path, map_location="cpu")
    metadata = dict(payload["metadata"])
    if expected_metadata is not None and not _metadata_matches(metadata, expected_metadata):
        raise ValueError(f"Text cache metadata does not match current config: {path}")
    return CaptionFeatureCache(
        hidden=payload["hidden"],
        mask=payload["mask"].to(dtype=torch.bool),
        captions=list(payload.get("captions", [])),
        metadata=metadata,
        path=path,
    )


def _cache_dtype(name: str) -> torch.dtype:
    allowed = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in allowed:
        raise ValueError(f"Unsupported text.cache_dtype '{name}'. Use one of {sorted(allowed)}.")
    return allowed[name]


def build_caption_cache(cfg: Config, train: bool, device: torch.device, path: str | Path) -> CaptionFeatureCache:
    if cfg.data.dataset != "hf_image_caption":
        raise ValueError("Caption feature caching currently supports only hf_image_caption datasets.")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install the `datasets` package to build caption caches.") from exc

    split = _split_name(cfg.data, train)
    if cfg.data.hf_config:
        ds = load_dataset(cfg.data.hf_name, cfg.data.hf_config, split=split)
    else:
        ds = load_dataset(cfg.data.hf_name, split=split)

    captions = [str(value) for value in ds[cfg.data.caption_column]]
    conditioner = TextConditioner(cfg.text, cfg.ar.hidden_dim, load_encoder=True).to(device).eval()
    dtype = _cache_dtype(cfg.text.cache_dtype)
    hidden_parts: list[torch.Tensor] = []
    mask_parts: list[torch.Tensor] = []

    batch_size = cfg.text.cache_batch_size
    with torch.no_grad():
        for start in tqdm(range(0, len(captions), batch_size), desc=f"encoding {split} captions"):
            batch = captions[start : start + batch_size]
            encoded = conditioner.encode(batch, device)
            hidden_parts.append(encoded.hidden.to(dtype=dtype, device="cpu"))
            mask_parts.append(encoded.mask.to(dtype=torch.bool, device="cpu"))

    cache = CaptionFeatureCache(
        hidden=torch.cat(hidden_parts, dim=0),
        mask=torch.cat(mask_parts, dim=0),
        captions=captions,
        metadata=_cache_metadata(cfg, train),
        path=Path(path),
    )
    cache.path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": cache.metadata,
            "captions": cache.captions,
            "hidden": cache.hidden,
            "mask": cache.mask,
        },
        cache.path,
    )
    return cache


def ensure_caption_cache(
    cfg: Config,
    train: bool,
    device: torch.device,
    rebuild: bool = False,
) -> CaptionFeatureCache:
    metadata = _cache_metadata(cfg, train)
    path = caption_cache_path(cfg, train)
    if path.exists() and not rebuild:
        return load_caption_cache(path, metadata)
    return build_caption_cache(cfg, train, device, path)
