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
import torch.nn.functional as F
from torchvision.utils import save_image

from .config import Config, load_config
from .data import build_loader
from .ema import EMA
from .losses import PerceptualLoss
from .models import EOSTok
from .models.discriminator import (
    LeCamRegularizer, PatchDiscriminator, hinge_d_loss, hinge_g_loss,
)


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
    disc = PatchDiscriminator(cfg.data.channels).to(device)
    lecam = LeCamRegularizer().to(device)
    perceptual = PerceptualLoss().to(device) if cfg.loss.lpips_enabled else None

    vfm = None
    if cfg.vfm.enabled:
        from .models.vfm import VFMAligner
        vfm = VFMAligner(
            cfg.vfm.model, cfg.tokenizer.hidden_dim, cfg.tokenizer.hidden_dim,
            model.encoder.grid,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[eostok] model params: {n_params / 1e6:.1f}M "
          f"(tokenizer {sum(p.numel() for m in (model.encoder, model.decoder, model.quantizer) for p in m.parameters()) / 1e6:.1f}M, "
          f"AR {sum(p.numel() for p in model.ar.parameters()) / 1e6:.1f}M)")

    # Adam with different beta2 for tokenizer vs AR (Table 9).
    tok_params = (list(model.encoder.parameters())
                  + list(model.decoder.parameters())
                  + list(model.quantizer.parameters()))
    ar_params = list(model.ar.parameters())
    if vfm is not None:
        tok_params += [p for p in vfm.parameters() if p.requires_grad]
    tc = cfg.train
    opt_g = torch.optim.Adam([
        {"params": tok_params, "betas": (tc.beta1, tc.beta2_tokenizer)},
        {"params": ar_params, "betas": (tc.beta1, tc.beta2_ar)},
    ], lr=tc.lr)
    opt_d = torch.optim.Adam(disc.parameters(), lr=tc.disc_lr,
                             betas=(tc.beta1, tc.beta2_tokenizer))

    ema = EMA(model, tc.ema_decay)

    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        disc.load_state_dict(ckpt["disc"])
        ema.load_state_dict(ckpt["ema"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        step = ckpt["step"]
        print(f"[eostok] resumed from {args.resume} at step {step}")

    lw = cfg.loss
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
                x_recon, x_apr = out["x_recon"], out["x_apr"]

                rec_l2 = F.mse_loss(x_recon, x)
                apr_l2 = F.mse_loss(x_apr, x)
                rec_lp = perceptual(x_recon, x) if perceptual else x.new_zeros(())
                apr_lp = perceptual(x_apr, x) if perceptual else x.new_zeros(())

                g_loss = x.new_zeros(())
                if lw.gan > 0 and step >= lw.disc_start:
                    g_loss = hinge_g_loss(disc(x_recon))

                sem_loss = x.new_zeros(())
                if vfm is not None:
                    yv = vfm.features(x)
                    sem_loss = vfm.implicit_loss(out["h_enc"], yv)
                    if out["h_dec"] is not None:
                        sem_loss = sem_loss + vfm.decoder_loss(
                            out["h_dec"], yv.repeat(2, 1, 1))

                loss = (
                    lw.recon_l2 * rec_l2 + lw.recon_lpips * rec_lp
                    + lw.gan * g_loss
                    + cfg.quantizer.commit_weight * out["commit_loss"]
                    + cfg.quantizer.entropy_weight * out["entropy_loss"]
                    + lw.ntp * out["ntp_loss"]
                    + lw.apr_l2 * apr_l2 + lw.apr_lpips * apr_lp
                    + cfg.vfm.weight * sem_loss
                )

            opt_g.zero_grad(set_to_none=True)
            loss.backward()
            if tc.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in opt_g.param_groups for p in g["params"]],
                    tc.grad_clip)
            opt_g.step()
            ema.update(model)

            # --- Discriminator step
            d_loss = x.new_zeros(())
            if lw.gan > 0 and step >= lw.disc_start:
                with ctx:
                    real = disc(x)
                    fake = disc(x_recon.detach())
                    d_loss = hinge_d_loss(real, fake) + lw.lecam * lecam(real, fake)
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()

            if step % tc.log_every == 0:
                ips = (step + 1) * cfg.data.batch_size / (time.time() - t0)
                print(
                    f"step {step}/{total_steps} | loss {loss.item():.4f} | "
                    f"rec_l2 {rec_l2.item():.4f} lpips {rec_lp.item():.4f} | "
                    f"ntp {out['ntp_loss'].item():.4f} ar_acc {out['ar_acc'].item():.3f} | "
                    f"apr {apr_l2.item():.4f} | d {d_loss.item():.4f} | "
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
                torch.save({
                    "model": model.state_dict(), "disc": disc.state_dict(),
                    "ema": ema.state_dict(), "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(), "step": step,
                }, out_dir / "last.ckpt")

            step += 1

    torch.save({
        "model": model.state_dict(), "disc": disc.state_dict(),
        "ema": ema.state_dict(), "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(), "step": step,
    }, out_dir / "last.ckpt")
    print(f"[eostok] done at step {step}; checkpoint -> {out_dir / 'last.ckpt'}")


if __name__ == "__main__":
    main()
