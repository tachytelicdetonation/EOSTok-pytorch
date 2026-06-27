"""Hugging Face image-caption datasets.

Images are normalized to [-1, 1]. Samples return ``(image, caption_text)``.
The model owns tokenization and text encoding so training and sampling use the
same prompt path.
"""

from __future__ import annotations

from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms

from ..config import DataConfig
from ..text import CaptionFeatureCache, EncodedText


def _image_transform(cfg: DataConfig, train: bool):
    norm = transforms.Normalize([0.5] * cfg.channels, [0.5] * cfg.channels)
    ops = [
        transforms.Resize(cfg.image_size),
        transforms.CenterCrop(cfg.image_size),
    ]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    return transforms.Compose(ops + [transforms.ToTensor(), norm])


class HFImageCaptionDataset:
    """Thin wrapper over a Hugging Face image-caption dataset."""

    def __init__(self, cfg: DataConfig, train: bool, text_cache: CaptionFeatureCache | None = None):
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("Install the `datasets` package for hf_image_caption data.") from exc

        split = cfg.train_split if train else cfg.val_split
        if cfg.hf_config:
            self.ds = load_dataset(cfg.hf_name, cfg.hf_config, split=split)
        else:
            self.ds = load_dataset(cfg.hf_name, split=split)
        self.cfg = cfg
        self.transform = _image_transform(cfg, train)
        self.text_cache = text_cache
        if self.text_cache is not None and len(self.text_cache) != len(self.ds):
            raise ValueError(
                f"Text cache length ({len(self.text_cache)}) does not match "
                f"dataset split length ({len(self.ds)})"
            )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        sample = self.ds[idx]
        image = sample[self.cfg.image_column]
        mode = "L" if self.cfg.channels == 1 else "RGB"
        image = image.convert(mode)
        caption = str(sample[self.cfg.caption_column])
        if self.text_cache is not None:
            return self.transform(image), self.text_cache[idx]
        return self.transform(image), caption


def build_dataset(
    cfg: DataConfig,
    train: bool = True,
    text_cache: CaptionFeatureCache | None = None,
):
    if cfg.dataset == "hf_image_caption":
        return HFImageCaptionDataset(cfg, train, text_cache)
    raise ValueError(f"Unknown caption dataset: {cfg.dataset}")


def _collate_encoded_text(batch):
    images, conditions = zip(*batch)
    return default_collate(images), EncodedText.collate(list(conditions))


def build_loader(
    cfg: DataConfig,
    train: bool = True,
    text_cache: CaptionFeatureCache | None = None,
) -> DataLoader:
    ds = build_dataset(cfg, train, text_cache)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=train,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=train,
        persistent_workers=cfg.num_workers > 0,
        collate_fn=_collate_encoded_text if text_cache is not None else None,
    )
