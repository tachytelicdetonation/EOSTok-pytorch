"""Single-stage end-to-end training of EOSTok (Eq. 8):

  L = L_VQVAE + lambda_NTP * L_NTP + lambda_APR * L_APR
      + lambda_sem * (L_implicit + L_decoder_align)

with L_VQVAE = L2 + LPIPS + lambda_GAN * L_GAN + lambda_reg * L_reg.

Usage:
  python -m eostok.train --config configs/mnist.yaml [--max-steps N]
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import torch
from torchvision.utils import save_image

from .config import Config, load_config
from .criterion import Adversary, EOSTokCriterion
from .data import build_loader
from .ema import EMA
from .models import EOSTok


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Stop early (overrides epochs); useful for smoke tests")
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()

    cfg: Config = load_config(args.config)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    device = pick_device(cfg.device)
    dtype = amp_dtype(cfg.amp, device)
    out_dir = Path(cfg.train.out_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    print(f"[eostok] device={device} amp={dtype} out={out_dir}")

    loader = build_loader(cfg.data, train=True)
    steps_per_epoch = len(loader)
    total_steps = args.max_steps or cfg.train.epochs * steps_per_epoch

    model = EOSTok(cfg).to(device)
    # The objective, as two sibling modules (opposed optimizers).
    adversary = Adversary(cfg).to(device)
    criterion = EOSTokCriterion(cfg, adversary).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[eostok] model params: {n_params / 1e6:.1f}M "
          f"(tokenizer {sum(p.numel() for m in (model.encoder, model.decoder, model.quantizer) for p in m.parameters()) / 1e6:.1f}M, "
          f"AR {sum(p.numel() for p in model.ar.parameters()) / 1e6:.1f}M)")

    # Adam with different beta2 for tokenizer vs AR (Table 9). The criterion's
    # trainable params (VFM projectors; PerceptualLoss is frozen) train with the
    # tokenizer, so they ride in the tokenizer group.
    tok_params = (list(model.encoder.parameters())
                  + list(model.decoder.parameters())
                  + list(model.quantizer.parameters())
                  + [p for p in criterion.parameters() if p.requires_grad])
    ar_params = list(model.ar.parameters())
    tc = cfg.train
    opt_g = torch.optim.Adam([
        {"params": tok_params, "betas": (tc.beta1, tc.beta2_tokenizer)},
        {"params": ar_params, "betas": (tc.beta1, tc.beta2_ar)},
    ], lr=tc.lr)
    # The discriminator is the only thing the adversary owns params for.
    opt_d = torch.optim.Adam(adversary.parameters(), lr=tc.disc_lr,
                             betas=(tc.beta1, tc.beta2_tokenizer))

    ema = EMA(model, tc.ema_decay)

    def checkpoint() -> dict:
        return {
            "model": model.state_dict(), "adversary": adversary.state_dict(),
            "criterion": criterion.state_dict(), "ema": ema.state_dict(),
            "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(), "step": step,
        }

    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        adversary.load_state_dict(ckpt["adversary"])
        criterion.load_state_dict(ckpt["criterion"])
        ema.load_state_dict(ckpt["ema"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        step = ckpt["step"]
        print(f"[eostok] resumed from {args.resume} at step {step}")

    L = cfg.tokenizer.num_latent_tokens
    fixed_labels = torch.arange(cfg.data.num_classes, device=device).repeat(8)[:64]

    model.train()
    done = False
    t0 = time.time()
    while not done:
        for x, y in loader:
            if step >= total_steps:
                done = True
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Class dropout for CFG: replace with the null class.
            drop = torch.rand(y.shape, device=device) < cfg.ar.class_dropout
            y_in = torch.where(drop, torch.full_like(y, model.ar.null_class), y)

            lr = cosine_lr(step, total_steps, tc.lr, tc.min_lr)
            for group in opt_g.param_groups:
                group["lr"] = lr

            k = sample_keep_tokens(L, tc.nested_dropout)

            ctx = (torch.autocast(device.type, dtype=dtype)
                   if dtype else torch.autocast(device.type, enabled=False))
            with ctx:
                out = model(x, y_in, keep_tokens=k)
                loss, m = criterion(out, x, step)

            opt_g.zero_grad(set_to_none=True)
            loss.backward()
            if tc.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in opt_g.param_groups for p in g["params"]],
                    tc.grad_clip)
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
                    rec = ema.shadow.reconstruct(grid_x)
                    gen = ema.shadow.generate(fixed_labels[:32])
                save_image((torch.cat([grid_x, rec]) + 1) / 2,
                           out_dir / "samples" / f"recon_{step:07d}.png", nrow=8)
                save_image((gen + 1) / 2,
                           out_dir / "samples" / f"gen_{step:07d}.png", nrow=8)
                model.train()

            if step > 0 and step % tc.ckpt_every == 0:
                torch.save(checkpoint(), out_dir / "last.ckpt")

            step += 1

    torch.save(checkpoint(), out_dir / "last.ckpt")
    print(f"[eostok] done at step {step}; checkpoint -> {out_dir / 'last.ckpt'}")


if __name__ == "__main__":
    main()
