"""SMI-Editor-inspired fragment corruption and edit supervision."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import random
from typing import Iterable

from .hierarchy import build_hierarchy
from .tokenization import MASK, detokenize_smiles, tokenize_smiles


@dataclass(frozen=True)
class EditOperation:
    """A token-level edit operation."""

    tag: str
    source_start: int
    source_end: int
    target_start: int
    target_end: int
    source_text: str
    target_text: str


@dataclass(frozen=True)
class EditExample:
    """A pure-SMILES edit training example."""

    clean_smiles: str
    corrupted_smiles: str
    clean_tokens: tuple[str, ...]
    corrupted_tokens: tuple[str, ...]
    operations: tuple[EditOperation, ...]
    policy: str

    @property
    def operation_counts(self) -> dict[str, int]:
        counts = {"equal": 0, "replace": 0, "insert": 0, "delete": 0}
        for operation in self.operations:
            counts[operation.tag] = counts.get(operation.tag, 0) + 1
        return counts


def make_edit_example(
    smiles: str,
    *,
    seed: int = 0,
    identity_probability: float = 0.15,
    policies: tuple[str, ...] = ("mask", "delete", "shuffle"),
) -> EditExample:
    """Create a fragment-level corruption example from one clean SMILES."""

    rng = random.Random(seed)
    clean_tokens = tokenize_smiles(smiles)
    if not clean_tokens or rng.random() < identity_probability:
        corrupted = list(clean_tokens)
        policy = "identity"
    else:
        hierarchy = build_hierarchy(clean_tokens)
        spans = list(hierarchy.fragment_spans) or [hierarchy.molecule_span]
        span = rng.choice(spans)
        policy = rng.choice(policies)
        corrupted = _apply_policy(clean_tokens, span.start, span.end, policy, rng)

    operations = tuple(edit_operations(corrupted, clean_tokens))
    return EditExample(
        clean_smiles=detokenize_smiles(clean_tokens),
        corrupted_smiles=detokenize_smiles(corrupted),
        clean_tokens=tuple(clean_tokens),
        corrupted_tokens=tuple(corrupted),
        operations=operations,
        policy=policy,
    )


def edit_operations(source_tokens: Iterable[str], target_tokens: Iterable[str]) -> list[EditOperation]:
    """Return SequenceMatcher edit operations between two token sequences."""

    source = list(source_tokens)
    target = list(target_tokens)
    matcher = SequenceMatcher(a=source, b=target, autojunk=False)
    operations: list[EditOperation] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        operations.append(
            EditOperation(
                tag=tag,
                source_start=i1,
                source_end=i2,
                target_start=j1,
                target_end=j2,
                source_text=detokenize_smiles(source[i1:i2]),
                target_text=detokenize_smiles(target[j1:j2]),
            )
        )
    return operations


def levenshtein_distance(left: Iterable[str], right: Iterable[str]) -> int:
    """Compute token-level Levenshtein distance."""

    a = list(left)
    b = list(right)
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, left_token in enumerate(a, start=1):
        curr = [i]
        for j, right_token in enumerate(b, start=1):
            substitution = 0 if left_token == right_token else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + substitution))
        prev = curr
    return int(prev[-1])


def operation_counts(operations: Iterable[EditOperation]) -> dict[str, int]:
    counts = {"equal": 0, "replace": 0, "insert": 0, "delete": 0}
    for operation in operations:
        counts[operation.tag] = counts.get(operation.tag, 0) + 1
    return counts


def _apply_policy(tokens: list[str], start: int, end: int, policy: str, rng: random.Random) -> list[str]:
    if start >= end:
        return list(tokens)
    before = tokens[:start]
    span = tokens[start:end]
    after = tokens[end:]
    if policy == "mask":
        return before + [MASK] + after
    if policy == "delete":
        return before + after
    if policy == "shuffle":
        shuffled = list(span)
        rng.shuffle(shuffled)
        return before + shuffled + after
    raise ValueError(f"Unknown corruption policy: {policy}")

