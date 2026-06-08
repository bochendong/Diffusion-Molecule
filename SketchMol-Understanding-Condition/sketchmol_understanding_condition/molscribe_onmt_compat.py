"""Runtime compatibility patches for SketchMol MolScribe + onmt220 overlay."""

from __future__ import annotations

import inspect

_PATCHED = False


def format_attention_mask_onmt220(module, mask):
    """Format attention masks for onmt220 ``MultiHeadedAttention``.

    The vendored overlay only applies ``mask.unsqueeze(1)`` once and expects a
    3D mask ``[batch, query_len, key_len]`` (or ``[batch, 1, key_len]`` for
    padding). The stock MolScribe helper incorrectly squeezes ``[B, 1, T]`` down
    to ``[B, T]``, which then broadcasts against ``[B, heads, Q, T]`` scores as
    ``[B, 1, T]`` and fails once ``batch != heads`` (e.g. batch 16, heads 8).
    """

    if mask is None:
        return None

    supported = inspect.signature(module.forward).parameters
    if "step" in supported:
        if mask.dim() == 3:
            return mask.unsqueeze(1)
        return mask

    if mask.dim() == 4 and mask.size(1) == 1 and mask.size(2) == 1:
        return mask.squeeze(2)
    if mask.dim() == 3 and mask.size(1) == 1:
        return mask
    return mask


def apply_onmt_attention_mask_patch() -> None:
    """Patch MolScribe decoder mask formatting for onmt220 at import time."""

    global _PATCHED
    if _PATCHED:
        return

    import molscribe.transformer.decoder as decoder_module

    decoder_module.format_attention_mask = format_attention_mask_onmt220
    _PATCHED = True
