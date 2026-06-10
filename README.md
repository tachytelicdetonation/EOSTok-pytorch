# EOSTok-pytorch (unofficial)

Unofficial PyTorch implementation of **"End-to-End Autoregressive Image
Generation with 1D Semantic Tokenizer"** (ByteDance Seed,
[arXiv:2605.00503](https://arxiv.org/abs/2605.00503)).

> **Disclaimer.** This is an independent reimplementation from the paper text;
> it is **not** affiliated with or endorsed by the authors or ByteDance, and no
> official code or weights were used. Architecture and hyperparameters are
> transcribed from the paper (notably Table 9). Points where the paper is
> silent are documented under [Known deviations](#known-deviations-from-the-paper).

EOSTok jointly trains a 1D ViT tokenizer (TiTok-style, IBQ quantization) and a
LlamaGen-style autoregressive model **in a single stage**, so generative
feedback shapes the tokenizer's latent space. The paper reports a
state-of-the-art **gFID of 1.48 without guidance** on ImageNet-1K 256x256
(EOSTok-H, ~1B params total).

The same pipeline runs at two scales here:

- **`configs/mnist.yaml`** — a ~5M-param config that trains on a MacBook
  (MPS) or CPU. Same code path, same four losses; for verifying the pipeline,
  not for paper numbers.
- **`configs/imagenet_{s,b,l,h}.yaml`** — direct transcriptions of the
  paper's Table 9 for anyone with the GPUs to attempt actual reproduction
  (paper setup: 8x H100, batch 256, 400 epochs ≈ 2M iterations).

## What's implemented

| Paper component | Where |
|---|---|
| 1D ViT tokenizer with hybrid attention masks (bidirectional over patches, causal over query tokens; patches blind to queries) | `eostok/models/tokenizer.py` |
| IBQ quantizer with l2-normalized codebook, softmax-STE indices, commitment + entropy regularization (Eq. 1/7) | `eostok/models/quantizer.py` |
| LlamaGen-style AR model: RMSNorm, SwiGLU, learnable pos. embeddings, shared global AdaLN with per-block biases; KV-cache sampling, CFG | `eostok/models/ar.py` |
| Soft token embedding `h = Ind^T Embed` so NTP gradients reach the encoder and codebook | `eostok/models/eostok.py` |
| APR loss: teacher-forced AR predictions decoded to pixels, batch-concatenated with the reconstruction pass (Eq. 4) | `eostok/models/eostok.py`, `eostok/train.py` |
| Implicit VFM alignment (encoder hidden patch embeddings -> DINOv2) + decoder alignment (Eq. 6) | `eostok/models/vfm.py` |
| GAN loss with LeCam regularization | `eostok/models/discriminator.py` |
| Nested dropout (p=0.5) on decoder latents, class dropout 0.1, EMA, per-module Adam betas, cosine LR | `eostok/train.py` |

## Quickstart (macOS / MPS)

```bash
uv venv && source .venv/bin/activate   # or python -m venv
uv pip install -e .

# Smoke test: a few hundred steps, ~minutes
python -m eostok.train --config configs/mnist.yaml --max-steps 300

# Real MNIST run (20 epochs)
python -m eostok.train --config configs/mnist.yaml

# Sample a grid from the EMA checkpoint
python -m eostok.sample --config configs/mnist.yaml --ckpt runs/mnist/last.ckpt --out grid.png

# FID (needs `pip install clean-fid`)
python -m eostok.eval_fid --config configs/mnist.yaml --ckpt runs/mnist/last.ckpt --mode gen --num 10000
```

Reconstruction/generation grids are also written periodically to
`runs/mnist/samples/` during training.

## Reproducing the paper

1. Get ImageNet-1K as `data/imagenet/{train,val}/<wnid>/*.JPEG`.
2. `python -m eostok.train --config configs/imagenet_l.yaml` (DINOv2-L is
   pulled via `torch.hub` on first run).
3. Generate 50k class-balanced samples with `eostok.sample --out-dir` and
   evaluate. `eostok.eval_fid` uses [clean-fid](https://github.com/GaParmar/clean-fid);
   the paper uses the [ADM evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations)
   — use that for numbers comparable to Table 3.

Single-GPU as written; wrap the model in DDP/FSDP for multi-GPU (the paper's
batch 256 across 8 GPUs). Expected headline results from the paper, all
**without** guidance:

| Model | Tokenizer / AR params | rFID | gFID |
|---|---|---|---|
| EOSTok-S | 165M / 93M  | 0.74 | 3.50 |
| EOSTok-B | 165M / 164M | 0.73 | 2.38 |
| EOSTok-L | 165M / 312M | 0.73 | 1.74 |
| EOSTok-H | 388M / 644M | 0.71 | 1.48 |

## Known deviations from the paper

- **Discriminator**: the paper uses the StyleGAN-T discriminator (frozen
  DINO backbone); we ship a small PatchGAN-style conv discriminator to avoid
  extra pretrained dependencies. LeCam regularization is included. Swap in a
  StyleGAN-T discriminator in `eostok/models/discriminator.py` for strict
  fidelity.
- **AutoGuidance**: headline L/H numbers are guidance-free, so this does not
  block reproduction; the paper's *guided* numbers for L/H use AutoGuidance
  (a small "bad" AR model replacing the unconditional logits), which is not
  implemented. Plain CFG (`--cfg`) is available, as used for S/B.
- **Nested dropout granularity**: we sample one truncation length per step
  (not per sample) to keep batches rectangular.
- **Decoder alignment layer k**: the paper does not state which decoder layer
  is aligned; configs default to the middle layer.
- **Unspecified details** (GAN warmup schedule, discriminator LR schedule,
  exact LPIPS variant, augmentations) follow common VQGAN-community practice.

## Citation

Please cite the original paper:

```bibtex
@article{chu2026eostok,
  title={End-to-End Autoregressive Image Generation with 1D Semantic Tokenizer},
  author={Chu, Wenda and Zhang, Bingliang and Han, Jiaqi and Li, Yizhuo and Yang, Linjie and Yue, Yisong and Guo, Qiushan},
  journal={arXiv preprint arXiv:2605.00503},
  year={2026}
}
```

If this specific implementation was useful, a link back to the repo is
appreciated (see `CITATION.cff` for a software entry).

## License

[MIT](LICENSE) for the code in this repository. The paper, its figures, and any
trademarks remain the property of their respective owners.
