"""Single-stage ImageGen training loop.

  L = L_VQVAE + lambda_NTP * L_NTP + lambda_APR * L_APR
      + lambda_sem * (L_implicit + L_decoder_align)

with L_VQVAE = L2 + LPIPS + lambda_GAN * L_GAN + lambda_reg * L_reg.

Usage:
  python -m imagegen.cli.train --config configs/imagewoof_64.yaml [--max-steps N]
"""

from __future__ import annotations

import argparse
import contextlib
import math
import random
import time
from pathlib import Path
from typing import cast

import torch
from torchvision.utils import save_image

from ..config import Config, load_config
from ..data import build_loader
from ..models import ImageGen
from ..objectives import Adversary, ImageGenCriterion
from ..text import (
    condition_len,
    condition_take,
    ensure_caption_cache,
    normalize_condition,
)
from .checkpoint import checkpoint_state_dict, load_checkpoint_state
from .ema import EMA


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_dtype(mode: str, device: torch.device):
    if mode == "off":
        return None
    if mode == "bf16":
        return torch.bfloat16
    return torch.bfloat16 if device.type == "cuda" else None  # auto


def cosine_lr(step: int, total: int, base: float, minimum: float) -> float:
    t = min(step / max(total, 1), 1.0)
    return minimum + 0.5 * (base - minimum) * (1 + math.cos(math.pi * t))


def sample_keep_tokens(L: int, p: float) -> int:
    """Nested dropout: with probability p, truncate decoder latents to a
    uniform k in [1, L]; otherwise keep all L."""
    if random.random() < p:
        return random.randint(1, L)
    return L


def prepare_text_caches(cfg: Config, args, device: torch.device):
    """Build (or load) the frozen text-encoder caption caches when requested.
    Returns (train_cache, val_cache, encoder_dim); all None when caching is off."""
    cache_requested = args.text_cache == "on" or (
        args.text_cache == "auto" and cfg.text.freeze and cfg.text.cache_dataset
    )
    if cache_requested and not cfg.text.freeze:
        raise ValueError(
            "Text cache requires cfg.text.freeze=true; trainable encoders must stay live."
        )
    if not cache_requested:
        return None, None, None

    print("[imagegen] preparing frozen text-encoder caption caches")
    train_cache = ensure_caption_cache(
        cfg, train=True, device=device, rebuild=args.rebuild_text_cache
    )
    val_cache = ensure_caption_cache(
        cfg, train=False, device=device, rebuild=args.rebuild_text_cache
    )
    if train_cache.encoder_dim != val_cache.encoder_dim:
        raise ValueError(
            "Train/validation text caches have different encoder dimensions."
        )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        f"[imagegen] text cache ready: train={len(train_cache)} "
        f"val={len(val_cache)} dim={train_cache.encoder_dim}"
    )
    return train_cache, val_cache, train_cache.encoder_dim


def build_optimizers(model, criterion, adversary, tc):
    """Generator + discriminator optimizers. Adam with different beta2 for the
    tokenizer vs AR groups (Table 9); the criterion's trainable params (VFM
    projectors; PerceptualLoss is frozen) ride in the tokenizer group. The
    discriminator is the only thing the adversary owns params for."""
    tok_params = (
        list(model.encoder.parameters())
        + list(model.decoder.parameters())
        + list(model.quantizer.parameters())
        + [p for p in criterion.parameters() if p.requires_grad]
    )
    ar_params = [p for p in model.ar.parameters() if p.requires_grad]
    opt_g = torch.optim.Adam(
        [
            {"params": tok_params, "betas": (tc.beta1, tc.beta2_tokenizer)},
            {"params": ar_params, "betas": (tc.beta1, tc.beta2_ar)},
        ],
        lr=tc.lr,
    )
    opt_d = torch.optim.Adam(
        adversary.parameters(), lr=tc.disc_lr, betas=(tc.beta1, tc.beta2_tokenizer)
    )
    return opt_g, opt_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop early (overrides epochs); useful for smoke tests",
    )
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument(
        "--text-cache",
        choices=["auto", "on", "off"],
        default="auto",
        help="Precompute frozen text-encoder caption features for dataset splits",
    )
    ap.add_argument(
        "--rebuild-text-cache",
        action="store_true",
        help="Rebuild cached caption features even if a matching cache exists",
    )
    args = ap.parse_args()

    cfg: Config = load_config(args.config)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    device = pick_device(cfg.device)
    dtype = amp_dtype(cfg.amp, device)
    out_dir = Path(cfg.train.out_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    print(f"[imagegen] device={device} amp={dtype} out={out_dir}")

    train_text_cache, val_text_cache, text_encoder_dim = prepare_text_caches(
        cfg, args, device
    )

    loader = build_loader(cfg.data, train=True, text_cache=train_text_cache)
    steps_per_epoch = len(loader)
    total_steps = args.max_steps or cfg.train.epochs * steps_per_epoch

    model = ImageGen(
        cfg,
        load_text_encoder=train_text_cache is None,
        text_encoder_dim=text_encoder_dim,
    ).to(device)
    # The objective, as two sibling modules (opposed optimizers).
    adversary = Adversary(cfg).to(device)
    criterion = ImageGenCriterion(cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[imagegen] model params: {n_params / 1e6:.1f}M "
        f"(tokenizer {sum(p.numel() for m in (model.encoder, model.decoder, model.quantizer) for p in m.parameters()) / 1e6:.1f}M, "
        f"AR {sum(p.numel() for p in model.ar.parameters()) / 1e6:.1f}M)"
    )

    tc = cfg.train
    opt_g, opt_d = build_optimizers(model, criterion, adversary, tc)

    ema_shadow = None
    if train_text_cache is not None:
        ema_shadow = ImageGen(
            cfg,
            load_text_encoder=False,
            text_encoder_dim=text_encoder_dim,
        ).to(device)
        load_checkpoint_state(ema_shadow, checkpoint_state_dict(model, cfg), cfg)
    ema = EMA(model, tc.ema_decay, shadow=ema_shadow)

    def checkpoint() -> dict:
        return {
            "model": checkpoint_state_dict(model, cfg),
            "adversary": adversary.state_dict(),
            "criterion": criterion.state_dict(),
            "ema": checkpoint_state_dict(ema.shadow, cfg),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "step": step,
        }

    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        load_checkpoint_state(model, ckpt["model"], cfg)
        adversary.load_state_dict(ckpt["adversary"])
        criterion.load_state_dict(ckpt["criterion"])
        load_checkpoint_state(ema.shadow, ckpt["ema"], cfg)
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        step = ckpt["step"]
        print(f"[imagegen] resumed from {args.resume} at step {step}")

    L = cfg.tokenizer.num_latent_tokens
    fixed_conditions = None
    if val_text_cache is not None and len(val_text_cache) > 0:
        g = torch.Generator().manual_seed(cfg.seed)
        count = min(32, len(val_text_cache))
        fixed_conditions = val_text_cache.take(
            torch.randperm(len(val_text_cache), generator=g)[:count]
        )

    model.train()
    done = False
    t0 = time.time()
    while not done:
        for x, condition in loader:
            if step >= total_steps:
                done = True
                break
            x = x.to(device, non_blocking=True)
            condition = normalize_condition(condition)
            batch_conditions = condition_len(condition)
            if fixed_conditions is None:
                fixed_conditions = condition_take(
                    condition, slice(0, min(32, batch_conditions))
                )

            # Condition dropout for CFG: force_empty marks the dropped rows, which
            # the conditioner collapses to the null token. One mechanism for both
            # cached EncodedText and live captions.
            condition_drop = (
                torch.rand(batch_conditions, device=device) < cfg.ar.condition_dropout
            )

            lr = cosine_lr(step, total_steps, tc.lr, tc.min_lr)
            for group in opt_g.param_groups:
                group["lr"] = lr

            k = sample_keep_tokens(L, tc.nested_dropout)

            ctx = (
                torch.autocast(device.type, dtype=dtype)
                if dtype
                else contextlib.nullcontext()
            )
            with ctx:
                out = model(x, condition, keep_tokens=k, condition_drop=condition_drop)
                g_loss = adversary.g_term(out.pixels.recon, step)
                loss, m = criterion(out, x, step, g_loss)

            opt_g.zero_grad(set_to_none=True)
            loss.backward()
            if tc.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in opt_g.param_groups for p in g["params"]], tc.grad_clip
                )
            opt_g.step()
            ema.update(model)

            # --- Discriminator step (opposed objective; gate owned by the adversary)
            d_loss = x.new_zeros(())
            if adversary.active(step):
                with ctx:
                    d_loss = adversary.d_loss(x, out.pixels.recon, step)
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()

            if step % tc.log_every == 0:
                ips = (step + 1) * cfg.data.batch_size / (time.time() - t0)
                print(
                    f"step {step}/{total_steps} | loss {m['loss'].item():.4f} | "
                    f"rec_l2 {m['rec_l2'].item():.4f} lpips {m['rec_lp'].item():.4f} | "
                    f"ntp {m['ntp'].item():.4f} ar_acc {m['ar_acc'].item():.3f} | "
                    f"apr {m['apr_l2'].item():.4f} | d {d_loss.item():.4f} | "
                    f"k {k} lr {lr:.2e} | {ips:.0f} img/s"
                )

            if step > 0 and step % tc.sample_every == 0:
                model.eval()
                with torch.no_grad():
                    grid_x = x[:32]
                    ema_model = cast(ImageGen, ema.shadow)
                    rec = ema_model.reconstruct(grid_x)
                    # fixed_conditions is already capped at <=32 rows at every
                    # assignment, so no re-slice is needed here.
                    gen = ema_model.generate(fixed_conditions)
                save_image(
                    (torch.cat([grid_x, rec]) + 1) / 2,
                    out_dir / "samples" / f"recon_{step:07d}.png",
                    nrow=8,
                )
                save_image(
                    (gen + 1) / 2, out_dir / "samples" / f"gen_{step:07d}.png", nrow=8
                )
                model.train()

            if step > 0 and step % tc.ckpt_every == 0:
                torch.save(checkpoint(), out_dir / "last.ckpt")

            step += 1

    torch.save(checkpoint(), out_dir / "last.ckpt")
    print(f"[imagegen] done at step {step}; checkpoint -> {out_dir / 'last.ckpt'}")


if __name__ == "__main__":
    main()
