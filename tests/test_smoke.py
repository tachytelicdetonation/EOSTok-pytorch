"""Shape/gradient smoke tests on random data (no dataset download needed).

Run:  python tests/test_smoke.py
"""

from typing import cast

import torch
import torch.nn.functional as F

from imagegen.config import Config, load_config
from imagegen.models import ImageGen
from imagegen.models.ar import ARBlock
from imagegen.objectives import Adversary, ImageGenCriterion
from imagegen.training.checkpoint import checkpoint_state_dict, load_checkpoint_state
from imagegen.training.ema import EMA


def tiny_config() -> Config:
    cfg = Config()
    cfg.data.image_size = 32
    cfg.data.channels = 1
    cfg.text.model_name = "__tiny__"
    cfg.text.max_length = 16
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
    model = ImageGen(cfg)
    x = torch.randn(4, 1, 32, 32)
    captions = ["small dog", "large dog", "dog outside", "dog portrait"]

    out = model(x, captions, keep_tokens=5)
    assert out.pixels.recon.shape == (4, 1, 32, 32)
    assert out.pixels.apr.shape == (4, 1, 32, 32)
    assert out.metrics.indices.shape == (4, 8)

    loss = (
        out.pixels.recon.pow(2).mean()
        + out.pixels.apr.pow(2).mean()
        + out.reg_losses.ntp
        + out.reg_losses.commit
        + 0.01 * out.reg_losses.entropy
    )
    loss.backward()

    # End-to-end gradient flow: pixel/commit/entropy losses reach encoder+codebook
    # via the tokenizer path (no head in between).
    assert model.encoder.patchify.weight.grad is not None
    assert model.encoder.patchify.weight.grad.abs().sum() > 0
    assert model.quantizer.codebook.grad is not None
    assert model.quantizer.codebook.grad.abs().sum() > 0
    # The AR head learns from step 0...
    assert model.ar.head.weight.grad is not None
    assert model.ar.head.weight.grad.abs().sum() > 0
    # ...but the zero-init head means NTP/APR send no gradient below it on the
    # first step, so tok_emb (reached only through the head) is frozen one step
    # (its grad is populated with zeros, not None).
    assert model.ar.tok_emb.weight.grad is not None
    assert model.ar.tok_emb.weight.grad.abs().sum() == 0
    # Once the head is non-zero, gradient does reach tok_emb (wiring is intact).
    torch.nn.init.normal_(model.ar.head.weight, std=0.02)
    model.zero_grad(set_to_none=True)
    out2 = model(x, captions, keep_tokens=5)
    (out2.reg_losses.ntp + out2.pixels.apr.pow(2).mean()).backward()
    assert model.ar.tok_emb.weight.grad is not None
    assert model.ar.tok_emb.weight.grad.abs().sum() > 0
    print("forward/backward OK")


def test_forward_under_bf16_autocast_with_empty_condition():
    """Regression: under bf16 autocast the projected text tokens are bf16 while the
    null_token Parameter stays fp32; the in-place scatter for forced-empty rows (CFG
    dropout) has no autocast cast hook, so the dtypes must be matched explicitly.
    CPU bf16 autocast reproduces the CUDA-only crash without a GPU."""
    cfg = tiny_config()
    model = ImageGen(cfg)
    x = torch.randn(4, 1, 32, 32)
    captions = ["small dog", "large dog", "dog outside", "dog portrait"]
    # Force two rows to the null token -- the path that scatters null_token.
    drop = torch.tensor([True, False, True, False])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(x, captions, keep_tokens=5, condition_drop=drop)
    assert out.pixels.recon.shape == (4, 1, 32, 32)
    print("bf16 autocast forward OK")


def test_generation():
    cfg = tiny_config()
    model = ImageGen(cfg).eval()
    captions = ["small dog", "large dog", "dog outside"]
    imgs = model.generate(captions)
    assert imgs.shape == (3, 1, 32, 32)
    imgs_cfg = model.generate(captions, cfg_scale=2.0)
    assert imgs_cfg.shape == (3, 1, 32, 32)
    print("generation (incl. CFG) OK")


def test_cached_generation_logits_match_full_prefix():
    cfg = tiny_config()
    model = ImageGen(cfg).eval()
    captions = ["small dog", "large dog"]
    text = model.ar.text(model.ar.prepare_condition(captions, torch.device("cpu")))
    fixed_tokens = torch.arange(6).reshape(2, 3) % cfg.quantizer.codebook_size

    dtype = model.ar.tok_emb.weight.dtype
    key_mask = text.mask
    text_x = model.ar._add_pos(text.tokens.to(dtype=dtype), 0)
    _, caches = model.ar._run_cached(text_x, key_mask)
    step_x = model.ar.img_start.to(dtype=dtype).view(1, 1, -1).expand(2, 1, -1)

    cached_logits = []
    full_logits = []
    for step in range(fixed_tokens.shape[1]):
        step_x = model.ar._add_pos(step_x, key_mask.shape[1])
        step_key_mask = torch.cat(
            [
                key_mask,
                torch.ones((2, 1), dtype=torch.bool),
            ],
            dim=1,
        )
        h_cached, caches = model.ar._run_cached(step_x, step_key_mask, caches)
        cached_logits.append(model.ar.head(model.ar.norm_f(h_cached[:, -1])))

        if step == 0:
            visual_in = (
                model.ar.img_start.to(dtype=dtype).view(1, 1, -1).expand(2, 1, -1)
            )
        else:
            prev = model.ar.tok_emb(fixed_tokens[:, :step])
            start = model.ar.img_start.to(dtype=dtype).view(1, 1, -1).expand(2, 1, -1)
            visual_in = torch.cat([start, prev], dim=1)
        x, full_key_mask, text_len = model.ar._prefix_sequence(text, visual_in)
        h_full = model.ar._run(x, full_key_mask)
        full_logits.append(model.ar.head(h_full[:, text_len + visual_in.shape[1] - 1]))

        key_mask = step_key_mask
        if step < fixed_tokens.shape[1] - 1:
            step_x = model.ar.tok_emb(fixed_tokens[:, step]).unsqueeze(1)

    for cached, full in zip(cached_logits, full_logits):
        assert torch.allclose(cached, full, atol=1e-5, rtol=1e-4)
    print("cached generation logits match full prefix OK")


def test_generation_prefills_text_then_steps_visual_cache():
    cfg = tiny_config()
    model = ImageGen(cfg).eval()
    seen = []
    first_block = cast(ARBlock, model.ar.blocks[0])
    original_forward_cached = first_block.forward_cached

    def wrapped_forward_cached(x, key_mask, kv_cache=None):
        seen.append((x.shape[1], key_mask.shape[1], kv_cache is None))
        return original_forward_cached(x, key_mask, kv_cache)

    def fail_full_run(*_, **__):
        raise AssertionError("generation should use the cached path")

    first_block.forward_cached = wrapped_forward_cached  # pyrefly: ignore[bad-argument-type]  # test monkeypatch
    model.ar._run = fail_full_run

    tokens = model.ar.generate(["small dog", "large dog"], cfg_scale=2.0)

    expected = [(cfg.text.max_length, cfg.text.max_length, True)]
    expected.extend(
        (1, cfg.text.max_length + step + 1, False) for step in range(model.ar.seq_len)
    )
    assert seen == expected
    assert tokens.shape == (2, model.ar.seq_len)
    print("generation cache structure OK")


def test_caption_conditioning():
    cfg = tiny_config()
    model = ImageGen(cfg)
    x = torch.randn(4, 1, 32, 32)
    captions = ["", "large dog", "dog outside", "dog portrait"]

    out = model(x, captions, keep_tokens=5)
    assert out.pixels.recon.shape == (4, 1, 32, 32)
    assert out.pixels.apr.shape == (4, 1, 32, 32)

    # Condition dropout now runs through the conditioner's force_empty mask: a
    # dropped row collapses to the null-token prefix, a kept row keeps its caption.
    drop = torch.tensor([True, False, True, False])
    dropped = model.ar.prepare_condition(
        captions, torch.device("cpu"), force_empty=drop
    )
    dropped_text = model.ar.text(dropped)
    assert (
        dropped_text.mask[2, 0] and not dropped_text.mask[2, 1:].any()
    )  # dropped -> null
    assert dropped_text.mask[3, 1:].any()  # kept -> real caption

    text = model.ar.text(
        model.ar.prepare_condition(["", "large dog"], torch.device("cpu"))
    )
    assert text.mask.shape == (2, cfg.text.max_length)
    assert text.mask[0, 0]
    assert not text.mask[0, 1:].any()
    assert text.tokens.shape == (2, cfg.text.max_length, cfg.ar.hidden_dim)
    assert not any(name.endswith("cross") for name, _ in model.ar.named_modules())

    imgs = model.generate(captions[:2], cfg_scale=2.0)
    assert imgs.shape == (2, 1, 32, 32)
    print("caption conditioning OK")


def test_criterion_assembly_and_gate():
    """The criterion is the test surface for Eq. 8: assert the weighting and the
    disc_start gate without running the training loop."""
    cfg = tiny_config()
    cfg.loss.lpips_enabled = False  # no VGG download in a smoke test
    cfg.loss.gan = 0.1
    cfg.loss.disc_start = 5
    cfg.vfm.enabled = False

    model = ImageGen(cfg)
    adversary = Adversary(cfg)
    criterion = ImageGenCriterion(cfg)

    x = torch.randn(4, 1, 32, 32)
    captions = ["small dog", "large dog", "dog outside", "dog portrait"]
    out = model(x, captions, keep_tokens=5)

    # Gate closed before disc_start -> the GAN term is exactly zero.
    assert not adversary.active(0)
    loss0, m0 = criterion(out, x, 0, adversary.g_term(out.pixels.recon, 0))
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
    _, m1 = criterion(out, x, 5, adversary.g_term(out.pixels.recon, 5))
    assert m1["g"].item() != 0.0

    # The discriminator is NOT a param of the criterion (it belongs to the
    # adversary's optimizer).
    crit_params = {id(p) for p in criterion.parameters()}
    assert not any(id(p) in crit_params for p in adversary.parameters())
    print("criterion assembly + disc_start gate OK")


def test_reconstruct():
    cfg = tiny_config()
    model = ImageGen(cfg).eval()
    x = torch.randn(2, 1, 32, 32)
    rec = model.reconstruct(x)
    assert rec.shape == x.shape
    print("reconstruction OK")


def test_imagewoof_config_defaults():
    cfg = load_config("configs/imagewoof_64.yaml")
    assert cfg.text.model_name == "Qwen/Qwen3.5-0.8B"
    assert cfg.text.freeze is True
    assert cfg.text.save_encoder_state is False
    assert cfg.text.cache_dataset is True
    assert cfg.text.cache_dir == "data/text_cache"
    assert cfg.text.cache_dtype == "float16"
    assert cfg.vfm.enabled is False
    assert cfg.loss.gan == 0.0
    assert cfg.loss.lpips_enabled is False
    print("imagewoof config defaults OK")


def test_checkpoint_filters_frozen_text_encoder():
    cfg = tiny_config()
    cfg.text.save_encoder_state = False
    model = ImageGen(cfg)

    state = checkpoint_state_dict(model, cfg)
    assert state
    assert not any(key.startswith("ar.text.encoder.") for key in state)

    restored = ImageGen(cfg)
    load_checkpoint_state(restored, state, cfg)
    print("checkpoint frozen text filtering OK")


def test_cached_text_without_live_encoder():
    cfg = tiny_config()
    model = ImageGen(cfg)
    captions = ["small dog", "large dog"]
    encoded = model.ar.text.encode(captions, torch.device("cpu"))
    state = checkpoint_state_dict(model, cfg)

    cached = ImageGen(cfg, load_text_encoder=False, text_encoder_dim=cfg.ar.hidden_dim)
    load_checkpoint_state(cached, state, cfg)
    assert cached.ar.text.encoder is None

    x = torch.randn(2, 1, 32, 32)
    drop = torch.tensor([False, True])
    out = cached(x, encoded, keep_tokens=5, condition_drop=drop)
    assert out.pixels.recon.shape == (2, 1, 32, 32)

    imgs = cached.generate(encoded, cfg_scale=2.0)
    assert imgs.shape == (2, 1, 32, 32)

    try:
        cached.generate(captions)
    except RuntimeError as exc:
        assert "Live text encoder is not loaded" in str(exc)
    else:
        raise AssertionError("live strings should require a loaded text encoder")

    ema_shadow = ImageGen(
        cfg, load_text_encoder=False, text_encoder_dim=cfg.ar.hidden_dim
    )
    load_checkpoint_state(ema_shadow, state, cfg)
    ema = EMA(model, shadow=ema_shadow)
    ema.update(model)
    assert cast(ImageGen, ema.shadow).ar.text.encoder is None
    print("cached text without live encoder OK")


def test_caption_fingerprint_detects_reorder():
    """Issue-1 guard: the row hash must change on content OR order drift (the
    silent-corruption case the config-identity key can't see)."""
    from imagegen.text.cache import _caption_fingerprint

    captions = ["small dog", "large dog", "a cat"]
    assert _caption_fingerprint(captions) == _caption_fingerprint(
        list(captions)
    )  # stable
    assert _caption_fingerprint(captions) != _caption_fingerprint(
        captions[::-1]
    )  # order
    assert _caption_fingerprint(captions) != _caption_fingerprint(
        captions[:2]
    )  # content
    print("caption fingerprint reorder/content detection OK")


def test_load_ema_model_recon_skips_text_encoder():
    """Issue-2 guard: a recon load reconstructs without ever building the live
    text encoder, sizing the projection straight from the checkpoint."""
    import tempfile
    from pathlib import Path

    from imagegen.cli.sample import load_ema_model

    cfg = tiny_config()
    model = ImageGen(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "ema.ckpt"
        torch.save({"ema": checkpoint_state_dict(model, cfg)}, ckpt_path)
        recon_model = load_ema_model(
            cfg, str(ckpt_path), torch.device("cpu"), load_text_encoder=False
        )
    assert recon_model.ar.text.encoder is None
    rec = recon_model.reconstruct(torch.randn(2, 1, 32, 32))
    assert rec.shape == (2, 1, 32, 32)
    print("load_ema_model recon skips text encoder OK")


def test_trainable_text_encoder_must_be_saved():
    cfg = tiny_config()
    cfg.text.freeze = False
    cfg.text.save_encoder_state = False
    model = ImageGen(cfg)
    try:
        checkpoint_state_dict(model, cfg)
    except ValueError as exc:
        assert "save_encoder_state" in str(exc)
    else:
        raise AssertionError("trainable text encoder should not be silently dropped")
    print("trainable text encoder checkpoint guard OK")


def test_top_k_top_p_filter():
    from imagegen.models.ar import _filter_logits

    logits = torch.tensor([[3.0, 2.0, 1.0, -5.0]])
    # off (defaults) leaves logits untouched
    assert torch.equal(_filter_logits(logits.clone(), 0, 0.0), logits)
    # top_k=1 keeps only the argmax, everything else -inf
    out_k = _filter_logits(logits.clone(), 1, 0.0)
    assert out_k[0, 0].item() == 3.0
    assert torch.isinf(out_k[0, 1:]).all()
    # nucleus keeps at least the top token and drops the far tail (-5.0)
    out_p = _filter_logits(logits.clone(), 0, 0.9)
    assert not torch.isinf(out_p[0, 0])
    assert torch.isinf(out_p[0, 3])
    print("top-k/top-p filter OK")


if __name__ == "__main__":
    test_forward_shapes_and_grads()
    test_criterion_assembly_and_gate()
    test_generation()
    test_cached_generation_logits_match_full_prefix()
    test_generation_prefills_text_then_steps_visual_cache()
    test_caption_conditioning()
    test_reconstruct()
    test_imagewoof_config_defaults()
    test_checkpoint_filters_frozen_text_encoder()
    test_cached_text_without_live_encoder()
    test_caption_fingerprint_detects_reorder()
    test_load_ema_model_recon_skips_text_encoder()
    test_trainable_text_encoder_must_be_saved()
    test_top_k_top_p_filter()
    print("all smoke tests passed")
