"""Lightweight text feature extraction for instruction prompts."""

from __future__ import annotations

import hashlib

import numpy as np


def hashed_text_vector(text: str, dim: int = 128) -> np.ndarray:
    """Signed hashing vector over simple word tokens."""

    vec = np.zeros(dim, dtype=np.float32)
    for token in _tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if int.from_bytes(digest[4:], "little") % 2 == 0 else -1.0
        vec[bucket] += sign
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def _tokens(text: str) -> list[str]:
    tokens = []
    cur = []
    for ch in str(text or "").lower():
        if ch.isalnum() or ch in {"_", "-"}:
            cur.append(ch)
        elif cur:
            tokens.append("".join(cur))
            cur = []
    if cur:
        tokens.append("".join(cur))
    return tokens
