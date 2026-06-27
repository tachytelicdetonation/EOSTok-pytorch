# ImageGen

ImageGen is a caption-conditioned autoregressive image-generation research
project. It keeps an EOSTok-style end-to-end image-token training loop, but uses
a frozen Qwen text encoder as prefix conditioning for the AR image-token model.

The default target is 64x64 Imagewoof captions from
`tachytelicdetonation/imagewoof-gemma4-31b-captions-multires`.

## Architecture

| Component | Where |
|---|---|
| 1D ViT tokenizer with hybrid attention masks | `imagegen/models/tokenizer.py` |
| IBQ quantizer with softmax-STE indices | `imagegen/models/quantizer.py` |
| Prefix-conditioned AR image-token model | `imagegen/models/ar.py` |
| Full ImageGen model and APR/NTP path | `imagegen/models/imagegen.py` |
| Frozen Qwen text-prefix conditioner | `imagegen/text/conditioner.py` |
| APR/NTP/reconstruction objective | `imagegen/objectives/criterion.py` |
| Optional VFM alignment | `imagegen/objectives/vfm.py` |
| Optional GAN/adversary loss | `imagegen/objectives/adversary.py` |
| Hugging Face image-caption data | `imagegen/data/loader.py` |
| Training loop, EMA, checkpoints, AMP/device helpers | `imagegen/training/` |
| Train, sample, eval entrypoints | `imagegen/cli/` |

## Caption Conditioning

Captions are encoded by a frozen pretrained text model, projected to the AR
width, and prepended to the visual-token stream:

```text
text tokens, <img>, previous image codes -> next image code
```

The unconditional branch for CFG is the same sequence with an empty caption.
By default the text encoder is `Qwen/Qwen3.5-0.8B`, `freeze: true`, and
`save_encoder_state: false`, so checkpoints do not duplicate frozen Qwen
weights.

For dataset training, frozen text-encoder outputs are cached once under
`data/text_cache/`. Training then constructs ImageGen without a live Qwen
encoder and feeds cached hidden states through the trainable `token_proj`
layer. The full text encoder is loaded only when building or rebuilding the
cache, or for live-prompt sampling where the user supplies new text at runtime.

## Default Config

The default config is `configs/imagewoof_64.yaml`.

- Imagewoof captions at 64x64
- frozen `Qwen/Qwen3.5-0.8B`
- prefix-token AR conditioning
- cached train/validation caption encoder outputs
- `vfm.enabled: false`
- `loss.gan: 0.0`
- `loss.lpips_enabled: false`

VFM alignment and GAN/adversary loss remain first-class modules for later
experiments, but they are disabled in the smoke-friendly default path.

## Quickstart

```bash
uv venv && source .venv/bin/activate
uv pip install -e .

uv run python tests/test_smoke.py

python -m imagegen.cli.train \
  --config configs/imagewoof_64.yaml \
  --max-steps 300

python -m imagegen.cli.sample \
  --config configs/imagewoof_64.yaml \
  --ckpt runs/imagewoof_64/last.ckpt \
  --prompt "a terrier dog standing outside" \
  --out imagewoof_prompt.png
```

Training downloads the selected Hugging Face split on first run.
With the default frozen encoder config, training also builds matching train and
validation caption caches before the model is moved into the main training path.
Use `--text-cache off` to keep the live encoder in memory, or
`--rebuild-text-cache` to force cache regeneration after changing text settings.

## Research References

This is a new ImageGen codebase. The main training recipe remains influenced by
EOSTok, and the text-prefix AR direction is informed by recent autoregressive
multimodal/image-generation work:

- [EOSTok](https://arxiv.org/abs/2605.00503)
- [OmniGen-AR](https://arxiv.org/abs/2606.09156)
- [ARM](https://arxiv.org/abs/2606.11188)
- [UniAR](https://arxiv.org/abs/2606.18249)
- [LINA](https://arxiv.org/abs/2601.22630)
- [ViQ](https://arxiv.org/abs/2606.27313)

## License

[MIT](LICENSE) for the code in this repository. Papers, pretrained models,
datasets, and trademarks remain the property of their respective owners.
