"""Sample images from a trained ImageGen checkpoint (EMA weights).

Grid preview:
  python -m imagegen.cli.sample --config configs/imagewoof_64.yaml \
    --ckpt runs/imagewoof_64/last.ckpt --prompt "a dog outdoors" --out grid.png

Bulk generation from prompts:
  python -m imagegen.cli.sample --config ... --ckpt ... --prompts-file prompts.txt \
    --out-dir gen/ --num 10000 [--cfg 1.5]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from ..config import load_config
from ..models import ImageGen
from ..training.checkpoint import TEXT_ENCODER_PREFIX, load_checkpoint_state
from ..training.train_loop import pick_device


def load_ema_model(
    cfg, ckpt_path: str, device: torch.device, load_text_encoder: bool = True
) -> ImageGen:
    """Load EMA weights into an ImageGen.

    Pass ``load_text_encoder=False`` for paths that never touch the AR/text stack
    (e.g. rFID reconstruction): the live text encoder is then never constructed,
    so the metric stays local/offline. The projection's input size is recovered
    from the checkpoint (``ar.text.token_proj.weight`` is ``(out_dim, encoder_dim)``
    and always kept), so no encoder config download is needed to size it.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt["ema"]
    text_encoder_dim = None
    if not load_text_encoder:
        text_encoder_dim = state["ar.text.token_proj.weight"].shape[1]
        # Drop any saved encoder weights so a strict load matches the encoder-less
        # model (no-op for the default frozen-and-omitted checkpoints).
        state = {k: v for k, v in state.items() if not k.startswith(TEXT_ENCODER_PREFIX)}
    model = ImageGen(
        cfg, load_text_encoder=load_text_encoder, text_encoder_dim=text_encoder_dim
    ).to(device).eval()
    load_checkpoint_state(model, state, cfg)
    return model


def _read_prompts(args) -> list[str]:
    prompts = []
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts.extend(line.strip() for line in f if line.strip())
    prompts.extend(args.prompt or [])
    return prompts or ["a dog in a natural photograph"]


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
    ap.add_argument("--prompt", action="append",
                    help="Caption prompt; repeat for multiple prompts")
    ap.add_argument("--prompts-file", default=None,
                    help="One caption prompt per line")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = pick_device(cfg.device)
    model = load_ema_model(cfg, args.ckpt, device)

    prompts = _read_prompts(args)
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        while written < args.num:
            b = min(args.batch_size, args.num - written)
            batch_prompts = [prompts[i % len(prompts)] for i in range(written, written + b)]
            imgs = model.generate(batch_prompts, args.temperature, args.cfg)
            for i in range(b):
                save_image((imgs[i] + 1) / 2, out_dir / f"{written + i:06d}.png")
            written += b
            print(f"\r[imagegen] generated {written}/{args.num}", end="", flush=True)
        print()
    else:
        grid_prompts = [prompts[i % len(prompts)] for i in range(args.num)]
        imgs = model.generate(grid_prompts, args.temperature, args.cfg)
        out = args.out or "samples.png"
        save_image((imgs + 1) / 2, out, nrow=int(args.num**0.5))
        print(f"[imagegen] wrote {out}")


if __name__ == "__main__":
    main()
