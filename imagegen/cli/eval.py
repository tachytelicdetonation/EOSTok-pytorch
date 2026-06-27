"""rFID / gFID evaluation.

Requires the `eval` extra (clean-fid): pip install 'imagegen[eval]' or pip install clean-fid

  rFID: python -m imagegen.cli.eval --config ... --ckpt ... --mode recon --num 10000
  gFID: python -m imagegen.cli.eval --config ... --ckpt ... --mode gen   --num 10000

Both dump real and model images to folders, then run clean-fid. The paper uses
the ADM evaluation suite (Dhariwal & Nichol) on 50k samples for headline
numbers; clean-fid values are close but not byte-identical -- for strict
reproduction export samples and use the ADM suite.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from torchvision.utils import save_image

from ..config import load_config
from ..data import build_dataset, build_loader
from ..training.train_loop import pick_device
from .sample import load_ema_model


def _dump(loader, make_batch, out_dir: Path, target: int, progress: bool = False) -> int:
    """Save up to `target` images as {n:06d}.png, where `make_batch(item)` turns
    one loader item into an image batch in [-1, 1]. The target guard runs BEFORE
    make_batch, so no extra (expensive) model call is made once `target` is hit."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for item in loader:
        if n >= target:
            break
        for img in make_batch(item):
            if n >= target:
                break
            save_image((img + 1) / 2, out_dir / f"{n:06d}.png")
            n += 1
        if progress:
            print(f"\r  {n}/{target}", end="", flush=True)
    if progress:
        print()
    return n


def _count_pngs(path: Path) -> int:
    return len(list(path.glob("*.png"))) if path.exists() else 0


def _clear_pngs(path: Path):
    if not path.exists():
        return
    for file in path.glob("*.png"):
        file.unlink()


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
    # recon only runs encoder->quantizer->decoder, so skip loading the live text
    # encoder; gen samples from captions and needs it.
    model = load_ema_model(cfg, args.ckpt, device, load_text_encoder=args.mode != "recon")

    val_count = len(build_dataset(cfg.data, train=False))
    target_num = min(args.num, val_count)
    if target_num < args.num:
        print(f"[imagegen] validation split has {val_count} images; "
              f"evaluating {target_num} instead of requested {args.num}")

    work = Path(args.work_dir)
    real_dir = work / f"real_{target_num}"
    fake_dir = work / f"fake_{args.mode}_{target_num}"
    fake_dir.mkdir(parents=True, exist_ok=True)

    real_count = _count_pngs(real_dir)
    if real_count < target_num:
        print("[imagegen] dumping real images...")
        real_count = _dump(build_loader(cfg.data, train=False),
                           lambda item: item[0], real_dir, target_num)
    _clear_pngs(fake_dir)

    print(f"[imagegen] dumping {args.mode} images...")
    loader = build_loader(cfg.data, train=False)
    if args.mode == "recon":
        fake_count = _dump(loader, lambda item: model.reconstruct(item[0].to(device)),
                           fake_dir, target_num)
    else:
        fake_count = _dump(loader, lambda item: model.generate(list(item[1]), cfg_scale=args.cfg),
                           fake_dir, target_num, progress=True)

    if real_count != target_num or fake_count != target_num:
        raise RuntimeError(
            f"FID dump incomplete: real={real_count}, fake={fake_count}, target={target_num}"
        )
    score = fid.compute_fid(str(real_dir), str(fake_dir))
    name = "rFID" if args.mode == "recon" else "gFID"
    print(f"[imagegen] {name} ({target_num} images): {score:.3f}")


if __name__ == "__main__":
    main()
