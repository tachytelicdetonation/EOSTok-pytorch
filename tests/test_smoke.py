"""Shape/gradient smoke tests on random data (no dataset download needed).

Run:  python tests/test_smoke.py
"""

import torch

from eostok.config import Config
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
    assert out["x_recon"].shape == (4, 1, 32, 32)
    assert out["x_apr"].shape == (4, 1, 32, 32)
    assert out["indices"].shape == (4, 8)

    loss = (out["x_recon"].pow(2).mean() + out["x_apr"].pow(2).mean()
            + out["ntp_loss"] + out["commit_loss"] + 0.01 * out["entropy_loss"])
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


def test_reconstruct():
    cfg = tiny_config()
    model = EOSTok(cfg).eval()
    x = torch.randn(2, 1, 32, 32)
    rec = model.reconstruct(x)
    assert rec.shape == x.shape
    print("reconstruction OK")


if __name__ == "__main__":
    test_forward_shapes_and_grads()
    test_generation()
    test_reconstruct()
    print("all smoke tests passed")
