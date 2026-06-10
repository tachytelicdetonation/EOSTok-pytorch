"""Sample images from a trained EOSTok checkpoint (EMA weights).

Grid preview:
  python -m eostok.sample --config configs/mnist.yaml --ckpt runs/mnist/last.ckpt --out grid.png

Bulk generation for FID (one PNG per image, class-balanced):
  python -m eostok.sample --config ... --ckpt ... --out-dir gen/ --num 10000 [--cfg 1.5]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from .config import load_config
from .models import EOSTok
from .train import pick_device


def load_ema_model(cfg, ckpt_path: str, device: torch.device) -> EOSTok:
    model = EOSTok(cfg).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["ema"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None, help="grid PNG path")
    ap.add_argument("--out-dir", default=None, help="directory for bulk per-image PNGs")
    ap.add_argument("--num", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--cfg", type=float, default=1.0,
                    help="CFG scale s (1.0 = no guidance, as in headline results)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = pick_device(cfg.device)
    model = load_ema_model(cfg, args.ckpt, device)

    n_cls = cfg.data.num_classes
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        while written < args.num:
            b = min(args.batch_size, args.num - written)
            labels = (torch.arange(written, written + b) % n_cls).to(device)
            imgs = model.generate(labels, args.temperature, args.cfg)
            for i in range(b):
                save_image((imgs[i] + 1) / 2, out_dir / f"{written + i:06d}.png")
            written += b
            print(f"\r[eostok] generated {written}/{args.num}", end="", flush=True)
        print()
    else:
        labels = (torch.arange(args.num) % n_cls).to(device)
        imgs = model.generate(labels, args.temperature, args.cfg)
        out = args.out or "samples.png"
        save_image((imgs + 1) / 2, out, nrow=int(args.num**0.5))
        print(f"[eostok] wrote {out}")


if __name__ == "__main__":
    main()
