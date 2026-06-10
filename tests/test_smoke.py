"""Shape/gradient smoke tests on random data (no dataset download needed).

Run:  python tests/test_smoke.py
"""

import torch
import torch.nn.functional as F

from eostok.config import Config
from eostok.criterion import Adversary, EOSTokCriterion
from eostok.models import EOSTok


def tiny_config() -> Config:
    cfg = Config()
    cfg.data.image_size = 32
    cfg.data.channels = 1
    cfg.data.num_classes = 10
    cfg.tokenizer.patch_size = 8  # 4x4 = 16 patches, fast
    cfg.tokenizer.hidden_dim = 64
    cfg.tokenizer.enc_layers = 2
    cfg.tokenizer.dec_layers = 2
    cfg.tokenizer.num_heads = 2
    cfg.tokenizer.num_latent_tokens = 8
    cfg.tokenizer.latent_dim = 8
    cfg.quantizer.codebook_size = 64
    cfg.ar.layers = 2
    cfg.ar.hidden_dim = 64
    cfg.ar.num_heads = 2
    return cfg


def test_forward_shapes_and_grads():
    cfg = tiny_config()
    model = EOSTok(cfg)
    x = torch.randn(4, 1, 32, 32)
    y = torch.randint(0, 10, (4,))

    out = model(x, y, keep_tokens=5)
    assert out.pixels.recon.shape == (4, 1, 32, 32)
    assert out.pixels.apr.shape == (4, 1, 32, 32)
    assert out.metrics.indices.shape == (4, 8)

    loss = (out.pixels.recon.pow(2).mean() + out.pixels.apr.pow(2).mean()
            + out.reg_losses.ntp + out.reg_losses.commit
            + 0.01 * out.reg_losses.entropy)
    loss.backward()

    # End-to-end gradient flow: pixel/NTP losses must reach encoder, codebook, AR.
    assert model.encoder.patchify.weight.grad is not None
    assert model.encoder.patchify.weight.grad.abs().sum() > 0
    assert model.quantizer.codebook.grad.abs().sum() > 0
    assert model.ar.tok_emb.weight.grad.abs().sum() > 0
    print("forward/backward OK")


def test_generation():
    cfg = tiny_config()
    model = EOSTok(cfg).eval()
    labels = torch.randint(0, 10, (3,))
    imgs = model.generate(labels)
    assert imgs.shape == (3, 1, 32, 32)
    imgs_cfg = model.generate(labels, cfg_scale=2.0)
    assert imgs_cfg.shape == (3, 1, 32, 32)
    print("generation (incl. CFG) OK")


def test_criterion_assembly_and_gate():
    """The criterion is the test surface for Eq. 8: assert the weighting and the
    disc_start gate without running the training loop."""
    cfg = tiny_config()
    cfg.loss.lpips_enabled = False  # no VGG download in a smoke test
    cfg.loss.gan = 0.1
    cfg.loss.disc_start = 5
    cfg.vfm.enabled = False

    model = EOSTok(cfg)
    adversary = Adversary(cfg)
    criterion = EOSTokCriterion(cfg, adversary)

    x = torch.randn(4, 1, 32, 32)
    y = torch.randint(0, 10, (4,))
    out = model(x, y, keep_tokens=5)

    # Gate closed before disc_start -> the GAN term is exactly zero.
    assert not adversary.active(0)
    loss0, m0 = criterion(out, x, 0)
    assert m0["g"].item() == 0.0

    # Hand-assemble Eq. 8 (LPIPS off, VFM off) and compare term-for-term.
    w, qw = cfg.loss, cfg.quantizer
    expected = (
        w.recon_l2 * F.mse_loss(out.pixels.recon, x)
        + w.apr_l2 * F.mse_loss(out.pixels.apr, x)
        + qw.commit_weight * out.reg_losses.commit
        + qw.entropy_weight * out.reg_losses.entropy
        + w.ntp * out.reg_losses.ntp
    )
    assert torch.allclose(loss0, expected, atol=1e-6)

    # Gate open at disc_start -> the GAN term switches on.
    assert adversary.active(5)
    _, m1 = criterion(out, x, 5)
    assert m1["g"].item() != 0.0

    # The borrowed discriminator is NOT a param of the criterion (it belongs to
    # the adversary's optimizer).
    crit_params = {id(p) for p in criterion.parameters()}
    assert not any(id(p) in crit_params for p in adversary.parameters())
    print("criterion assembly + disc_start gate OK")


def test_reconstruct():
    cfg = tiny_config()
    model = EOSTok(cfg).eval()
    x = torch.randn(2, 1, 32, 32)
    rec = model.reconstruct(x)
    assert rec.shape == x.shape
    print("reconstruction OK")


if __name__ == "__main__":
    test_forward_shapes_and_grads()
    test_criterion_assembly_and_gate()
    test_generation()
    test_reconstruct()
    print("all smoke tests passed")
