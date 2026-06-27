# ImageGen

Domain vocabulary for the ImageGen research codebase. The project keeps an
EOSTok-style end-to-end image-token training loop and adds frozen Qwen prefix
conditioning for caption-conditioned AR image generation.

## Architecture

**ImageGen**:
The full model in `imagegen/models/imagegen.py`: encoder, IBQ quantizer,
decoder, and prefix-conditioned AR model.

**Text prefix**:
Projected frozen-Qwen hidden states prepended to the AR visual-token stream.
The AR sequence is `text tokens, <img>, previous image codes`.

**ImageGenOutput**:
The typed result of one model forward, grouped by role: `pixels`,
`activations`, `reg_losses`, and `metrics`. The criterion and logger read these
groups instead of categorising a flat dict.

## Objective

**ImageGenCriterion**:
The generator's training objective in `imagegen/objectives/criterion.py`. It
owns cooperative terms such as reconstruction, APR pixels, perceptual loss, VFM
alignment, and the quantizer/AR regularizers surfaced by the model.

**Adversary**:
The discriminator side of the optional GAN objective in
`imagegen/objectives/adversary.py`, including LeCam regularization and the
`disc_start` gate. It is a sibling of the criterion because the discriminator
optimizer owns its parameters.

**Gate** (`Adversary.active(step)`):
The single source of truth for whether GAN loss is active at a given step. Both
the generator term and discriminator step consult it.

**Borrowed reference**:
A module a criterion calls but does not own, held in a tuple so `nn.Module`
does not register its parameters. The criterion borrows the adversary this way.
