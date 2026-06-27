"""Stage-1 multimodal graft validation (offline, RAM-safe, no Qwen/downloads).

Proves the unified single-stream design works against the existing ARBlock trunk:
interleaved [text, <boi>, image codes, <eoi>, text], one shared vocab, a single
next-token loss over both modalities, and a trainable image-code graft.

Run:  python tests/test_multimodal.py
"""

import torch

from imagegen.models.unified import (
    UnifiedStream,
    build_interleaved_batch,
    graft_image_vocab,
    unified_ntp_loss,
)


def _tiny_model() -> UnifiedStream:
    torch.manual_seed(0)
    return UnifiedStream(text_vocab=50, codebook_size=32, dim=64, depth=2, num_heads=2)


def test_graft_extends_vocab_with_mean_init():
    import torch.nn as nn

    embed = nn.Embedding(50, 64)
    head = nn.Linear(64, 50, bias=False)
    nn.init.normal_(embed.weight, std=0.02)
    nn.init.normal_(head.weight, std=0.02)
    new_embed, new_head = graft_image_vocab(embed, head, n_new=34)  # 32 codes + boi/eoi

    assert new_embed.num_embeddings == 84
    assert new_head.out_features == 84
    # Old rows are copied verbatim; new rows all start at the mean of the old.
    assert torch.allclose(new_embed.weight[:50], embed.weight)
    assert torch.allclose(new_embed.weight[50], embed.weight.mean(dim=0))
    assert torch.allclose(new_embed.weight[50], new_embed.weight[83])
    print("graft vocab/mean-init OK")


def test_interleaved_batch_layout():
    model = _tiny_model()
    codes = torch.tensor([0, 1, 2, 3])
    input_ids, key_mask = build_interleaved_batch(
        [([3, 4, 5], codes, [6, 7]), ([1, 2], codes, [8])], model
    )
    # Row 0: text(3) + <boi> + 4 image ids + <eoi> + text(2) = 11 real tokens.
    assert input_ids.shape == (2, 11)
    assert input_ids[0, :3].tolist() == [3, 4, 5]
    assert input_ids[0, 3].item() == model.boi_id
    assert input_ids[0, 4:8].tolist() == [50, 51, 52, 53]  # codes 0..3 -> text_vocab + c
    assert input_ids[0, 8].item() == model.eoi_id
    assert input_ids[0, 9:11].tolist() == [6, 7]
    # Row 1: text(2)+<boi>+4+<eoi>+text(1) = 9 real tokens -> 2 trailing pads.
    assert key_mask[1, :9].all() and not key_mask[1, 9:].any()
    print("interleaved layout OK")


def test_unified_stream_forward_and_graft_trains():
    model = _tiny_model()
    assert model.vocab_size == 50 + 32 + 2

    codes = torch.randint(0, 32, (8,))
    samples = [([3, 4, 5], codes, [6, 7]), ([1, 2], codes, [8, 9, 10])]
    input_ids, key_mask = build_interleaved_batch(samples, model)

    logits = model(input_ids, key_mask)
    assert logits.shape == (2, input_ids.shape[1], model.vocab_size)

    # One unified loss over BOTH text and image positions.
    loss = unified_ntp_loss(logits, input_ids, key_mask)
    assert torch.isfinite(loss)
    loss.backward()

    # The grafted image-code rows (embedding AND head) must receive gradient,
    # i.e. the image vocabulary actually trains in the unified stream.
    img = slice(model.text_vocab, model.text_vocab + model.codebook_size)
    assert model.embed.weight.grad is not None
    assert model.embed.weight.grad[img].abs().sum() > 0
    assert model.head.weight.grad is not None
    assert model.head.weight.grad[img].abs().sum() > 0
    print("unified forward/backward + image-graft gradient OK")


if __name__ == "__main__":
    test_graft_extends_vocab_with_mean_init()
    test_interleaved_batch_layout()
    test_unified_stream_forward_and_graft_trains()
    print("all multimodal graft tests passed")
