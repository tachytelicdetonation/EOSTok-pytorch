# EOSTok

Domain vocabulary for this unofficial EOSTok implementation. The model modules
(tokenizer, AR, quantizer, VFM aligner) name the paper's architecture; the terms
below name the *training objective*, which this codebase keeps as its own seam.

## The objective

**Criterion**:
The generator's training objective (Eq. 8) as a module: it owns the cooperative
terms (perceptual, VFM alignment, recon/APR pixels), folds in the regularizers
the model surfaces, and returns `(total_loss, metrics)` for one step. Lives in
`eostok/criterion.py`.
_Avoid_: loss function, loss module, objective function.

**Adversary**:
The discriminator side of the GAN, with its LeCam regularizer and the
`disc_start` gate, behind one interface. A *sibling* of the criterion (not a
part of it) because the discriminator maximizes what the generator minimizes.
The criterion borrows a reference to it for the generator-side GAN term.
_Avoid_: discriminator (that is the bare conv net the adversary owns), GAN loss.

**Gate** (`Adversary.active(step)`):
The single source of truth for whether the GAN is on at a given step. Both the
generator term and the discriminator step consult it, so the `disc_start` rule
lives in exactly one place.

**EOSTokOutput**:
The typed result of one model forward, grouped by role — `pixels`,
`activations`, `reg_losses`, `metrics` — so the criterion and the logger each
read one group instead of categorising a flat dict. Lives in
`eostok/models/eostok.py`.
_Avoid_: output dict, forward dict.

## Roles around the objective

**Cooperative term**:
Any objective term that pushes the same direction the generator optimizer pulls
(recon, APR, perceptual, VFM alignment, the quantizer/AR regularizers). The
criterion owns these. Contrast with the adversary's *opposed* objective.

**Borrowed reference**:
A module a criterion calls but does not own — held in a tuple so `nn.Module`
does not register its parameters. The criterion borrows the adversary this way,
keeping the discriminator's params owned by the discriminator's optimizer.
