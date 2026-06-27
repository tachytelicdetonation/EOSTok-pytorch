# ImageGen Architecture

ImageGen combines an EOSTok-style end-to-end image-token training loop with
frozen Qwen prefix conditioning for caption-conditioned autoregressive image
generation.

The image path is:

```text
image -> encoder -> IBQ codes -> decoder
```

The AR path is:

```text
caption -> frozen Qwen -> projection -> text prefix
text prefix, <img>, previous image codes -> next image code
```

During training, the model optimizes reconstruction, next-token prediction, and
APR decoded-pixel losses. Optional VFM alignment and GAN/adversary objectives
are implemented under `imagegen/objectives/`, but disabled in
`configs/imagewoof_64.yaml`.
