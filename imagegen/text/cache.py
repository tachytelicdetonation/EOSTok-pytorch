"""Disk cache for frozen text-encoder caption features."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import torch
from tqdm import tqdm

from ..config import Config, DataConfig
from .conditioner import EncodedText, TextConditioner


CACHE_VERSION = 2


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


def _identity_metadata(cfg: Config, train: bool) -> dict:
    """Config identity for a cache: everything that fixes which encoder/dataset
    produced it AND is knowable without reading the dataset. It both names the
    cache file and is checked on load. Pinning data.revision / text.revision here
    is what makes upstream drift change the key (and thus rebuild) rather than be
    silently accepted -- content/order drift under the same names is caught
    separately by the caption fingerprint."""
    data, text = cfg.data, cfg.text
    return {
        "version": CACHE_VERSION,
        "dataset": data.dataset,
        "hf_name": data.hf_name,
        "hf_config": data.hf_config,
        "dataset_revision": data.revision,
        "split": _split_name(data, train),
        "caption_column": data.caption_column,
        "model_name": text.model_name,
        "text_revision": text.revision,
        "max_length": text.max_length,
        "trust_remote_code": text.trust_remote_code,
        "cache_dtype": text.cache_dtype,
    }


def _caption_fingerprint(captions: list[str]) -> str:
    """Order-sensitive content hash of the exact encoder inputs. This is the row
    hash that the config-identity key cannot see: it changes if any caption text
    changes or the rows are reordered, which is precisely the drift that would
    pair a trained projection with a different hidden-state distribution."""
    h = sha256()
    h.update(f"{len(captions)}\n".encode("utf-8"))
    for caption in captions:
        h.update(caption.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _load_captions(cfg: Config, train: bool) -> list[str]:
    from ..data.loader import load_hf_split  # lazy: avoid a text<->data import cycle

    ds = load_hf_split(cfg.data, train)
    return [str(value) for value in ds[cfg.data.caption_column]]


def caption_cache_path(cfg: Config, train: bool) -> Path:
    metadata = _identity_metadata(cfg, train)
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

    split = _split_name(cfg.data, train)
    captions = _load_captions(cfg, train)
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

    # Record the resolved encoder commit when transformers exposes it, so a cache
    # file is self-describing about which weights produced it (forensics; the live
    # encoder is not reloaded on a cache hit to re-check this).
    encoder_commit = getattr(getattr(conditioner.encoder, "config", None), "_commit_hash", None)
    metadata = {
        **_identity_metadata(cfg, train),
        "caption_sha": _caption_fingerprint(captions),
        "encoder_commit": encoder_commit,
    }
    out_path = Path(path)
    cache = CaptionFeatureCache(
        hidden=torch.cat(hidden_parts, dim=0),
        mask=torch.cat(mask_parts, dim=0),
        captions=captions,
        metadata=metadata,
        path=out_path,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": cache.metadata,
            "captions": cache.captions,
            "hidden": cache.hidden,
            "mask": cache.mask,
        },
        out_path,
    )
    return cache


def _verify_caption_fingerprint(cfg: Config, train: bool, cache: CaptionFeatureCache):
    """Fail loudly if cached features no longer match the live dataset. The
    identity key cannot see a content/order change under a mutable split (or an
    unpinned dataset revision), so compare the recorded row hash against the
    captions on disk now. ponytail: re-reads the split on every cache hit -- cheap
    next to the encode it guards; pin data.revision to make this a no-op."""
    expected = cache.metadata.get("caption_sha")
    if expected is None:
        return  # pre-fingerprint cache: nothing to compare against
    actual = _caption_fingerprint(_load_captions(cfg, train))
    if actual != expected:
        raise ValueError(
            f"Text cache {cache.path} is stale: the dataset captions (content or "
            f"order) changed under the same config, so the cached features no "
            f"longer correspond to the current rows. Re-encode with "
            f"--rebuild-text-cache (or delete the cache file). Pin data.revision "
            f"to freeze the split and avoid this."
        )


def ensure_caption_cache(
    cfg: Config,
    train: bool,
    device: torch.device,
    rebuild: bool = False,
) -> CaptionFeatureCache:
    identity = _identity_metadata(cfg, train)
    path = caption_cache_path(cfg, train)
    if path.exists() and not rebuild:
        cache = load_caption_cache(path, identity)
        _verify_caption_fingerprint(cfg, train, cache)
        return cache
    return build_caption_cache(cfg, train, device, path)
