"""rFID / gFID evaluation.

Requires the `eval` extra (clean-fid):  pip install 'eostok[eval]' or pip install clean-fid

  rFID: python -m eostok.eval_fid --config ... --ckpt ... --mode recon --num 10000
  gFID: python -m eostok.eval_fid --config ... --ckpt ... --mode gen   --num 10000

Both dump real and model images to folders, then run clean-fid. The paper uses
the ADM evaluation suite (Dhariwal & Nichol) on 50k samples for headline
numbers; clean-fid values are close but not byte-identical -- for strict
reproduction export samples and use the ADM suite.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from .config import load_config
from .data import build_loader
from .sample import load_ema_model
from .train import pick_device


def dump_real(cfg, out: Path, num: int):
    loader = build_loader(cfg.data, train=False)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for x, _ in loader:
        for img in x:
            if n >= num:
                return
            save_image((img + 1) / 2, out / f"{n:06d}.png")
            n += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=["recon", "gen"], required=True)
    ap.add_argument("--num", type=int, default=10000)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--work-dir", default="fid_eval")
    args = ap.parse_args()

    try:
        from cleanfid import fid
    except ImportError:
        raise SystemExit("clean-fid not installed: pip install clean-fid")

    cfg = load_config(args.config)
    device = pick_device(cfg.device)
    model = load_ema_model(cfg, args.ckpt, device)

    work = Path(args.work_dir)
    real_dir, fake_dir = work / "real", work / f"fake_{args.mode}"
    fake_dir.mkdir(parents=True, exist_ok=True)

    if not real_dir.exists() or len(list(real_dir.glob("*.png"))) < args.num:
        print("[eostok] dumping real images...")
        dump_real(cfg, real_dir, args.num)

    print(f"[eostok] dumping {args.mode} images...")
    if args.mode == "recon":
        loader = build_loader(cfg.data, train=False)
        n = 0
        for x, _ in loader:
            if n >= args.num:
                break
            rec = model.reconstruct(x.to(device))
            for img in rec:
                if n >= args.num:
                    break
                save_image((img + 1) / 2, fake_dir / f"{n:06d}.png")
                n += 1
    else:
        n_cls = cfg.data.num_classes
        bs, n = cfg.data.batch_size, 0
        while n < args.num:
            b = min(bs, args.num - n)
            labels = (torch.arange(n, n + b) % n_cls).to(device)
            imgs = model.generate(labels, cfg_scale=args.cfg)
            for img in imgs:
                save_image((img + 1) / 2, fake_dir / f"{n:06d}.png")
                n += 1
            print(f"\r  {n}/{args.num}", end="", flush=True)
        print()

    score = fid.compute_fid(str(real_dir), str(fake_dir))
    name = "rFID" if args.mode == "recon" else "gFID"
    print(f"[eostok] {name} ({args.num} images): {score:.3f}")


if __name__ == "__main__":
    main()
