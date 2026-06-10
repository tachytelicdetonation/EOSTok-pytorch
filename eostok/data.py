"""Datasets: MNIST (local smoke-scale config) and ImageFolder (ImageNet-style).

Images are normalized to [-1, 1]. Every sample is (image, class_label) since
EOSTok is class-conditional.
"""

from __future__ import annotations

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import DataConfig


def build_dataset(cfg: DataConfig, train: bool = True):
    norm = transforms.Normalize([0.5] * cfg.channels, [0.5] * cfg.channels)
    if cfg.dataset == "mnist":
        tf = transforms.Compose([
            transforms.Resize(cfg.image_size),
            transforms.ToTensor(),
            norm,
        ])
        return datasets.MNIST(cfg.root, train=train, download=True, transform=tf)
    if cfg.dataset == "imagefolder":
        # ImageNet-style directory: <root>/<train|val>/<class_name>/*.JPEG
        split = "train" if train else "val"
        ops = [
            transforms.Resize(cfg.image_size),
            transforms.CenterCrop(cfg.image_size),
        ]
        if train:
            ops.append(transforms.RandomHorizontalFlip())
        tf = transforms.Compose(ops + [transforms.ToTensor(), norm])
        return datasets.ImageFolder(f"{cfg.root}/{split}", transform=tf)
    raise ValueError(f"Unknown dataset: {cfg.dataset}")


def build_loader(cfg: DataConfig, train: bool = True) -> DataLoader:
    ds = build_dataset(cfg, train)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=train,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=train,
        persistent_workers=cfg.num_workers > 0,
    )
