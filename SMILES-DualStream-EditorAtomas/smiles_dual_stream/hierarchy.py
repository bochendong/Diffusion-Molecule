"""Atomas-inspired hierarchy views for pure SMILES strings."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

from .tokenization import is_atom_token, tokenize_smiles


@dataclass(frozen=True)
class TokenSpan:
    """A half-open token span."""

    start: int
    end: int
    label: str

    def text(self, tokens: list[str]) -> str:
        return "".join(tokens[self.start : self.end])


@dataclass(frozen=True)
class HierarchyView:
    """Three semantic levels used by the alignment stream."""

    tokens: tuple[str, ...]
    atom_spans: tuple[TokenSpan, ...]
    fragment_spans: tuple[TokenSpan, ...]
    molecule_span: TokenSpan

    def fragment_texts(self) -> list[str]:
        values = [span.text(list(self.tokens)) for span in self.fragment_spans]
        return [value for value in values if value]


def build_hierarchy(smiles_or_tokens: str | Iterable[str]) -> HierarchyView:
    """Build token, fragment, and molecule levels for a SMILES string."""

    if isinstance(smiles_or_tokens, str):
        tokens = tokenize_smiles(smiles_or_tokens)
    else:
        tokens = list(smiles_or_tokens)
    atom_spans = tuple(TokenSpan(i, i + 1, "atom") for i, token in enumerate(tokens) if is_atom_token(token))
    fragment_spans = tuple(_fragment_spans(tokens))
    molecule_span = TokenSpan(0, len(tokens), "molecule")
    return HierarchyView(tuple(tokens), atom_spans, fragment_spans, molecule_span)


def hierarchy_alignment(left: str | Iterable[str], right: str | Iterable[str]) -> dict[str, float]:
    """Compute dependency-free hierarchical alignment diagnostics."""

    left_view = build_hierarchy(left)
    right_view = build_hierarchy(right)
    left_tokens = list(left_view.tokens)
    right_tokens = list(right_view.tokens)
    return {
        "token_jaccard": _jaccard(left_tokens, right_tokens),
        "atom_jaccard": _jaccard(_span_texts(left_tokens, left_view.atom_spans), _span_texts(right_tokens, right_view.atom_spans)),
        "fragment_jaccard": _jaccard(left_view.fragment_texts(), right_view.fragment_texts()),
        "token_lcs_ratio": _sequence_ratio(left_tokens, right_tokens),
        "fragment_count_ratio": _count_ratio(len(left_view.fragment_spans), len(right_view.fragment_spans)),
        "atom_count_ratio": _count_ratio(len(left_view.atom_spans), len(right_view.atom_spans)),
    }


def adaptive_polymerization_levels(smiles_or_tokens: str | Iterable[str]) -> dict[str, list[str]]:
    """Return Atomas-like local-to-global groups for pure SMILES.

    The names mirror the Atomas paper's semantic levels, but here all groups are
    derived from SMILES only.
    """

    view = build_hierarchy(smiles_or_tokens)
    tokens = list(view.tokens)
    return {
        "atom": _span_texts(tokens, view.atom_spans),
        "fragment": view.fragment_texts(),
        "molecule": ["".join(tokens)] if tokens else [],
    }


def _fragment_spans(tokens: list[str]) -> list[TokenSpan]:
    spans: list[TokenSpan] = []
    start: int | None = None
    branch_depth = 0

    for index, token in enumerate(tokens):
        boundary = token == "." or token in {"(", ")"}
        if token == "(":
            branch_depth += 1
        elif token == ")" and branch_depth > 0:
            branch_depth -= 1

        if boundary:
            _append_fragment_if_atom(tokens, spans, start, index)
            start = None
            continue

        if start is None:
            start = index

        if branch_depth == 0 and token in {"=", "#", "/", "\\"}:
            continue

    _append_fragment_if_atom(tokens, spans, start, len(tokens))

    if not spans and tokens:
        spans.append(TokenSpan(0, len(tokens), "fragment"))

    return _chunk_long_spans(tokens, spans)


def _append_fragment_if_atom(tokens: list[str], spans: list[TokenSpan], start: int | None, end: int) -> None:
    if start is None or start >= end:
        return
    if any(is_atom_token(token) for token in tokens[start:end]):
        spans.append(TokenSpan(start, end, "fragment"))


def _chunk_long_spans(tokens: list[str], spans: list[TokenSpan], max_tokens: int = 12) -> list[TokenSpan]:
    chunked: list[TokenSpan] = []
    for span in spans:
        length = span.end - span.start
        if length <= max_tokens:
            chunked.append(span)
            continue
        cursor = span.start
        while cursor < span.end:
            end = min(span.end, cursor + max_tokens)
            while end < span.end and not any(is_atom_token(token) for token in tokens[cursor:end]):
                end += 1
            chunked.append(TokenSpan(cursor, end, span.label))
            cursor = end
    return chunked


def _span_texts(tokens: list[str], spans: Iterable[TokenSpan]) -> list[str]:
    return [span.text(tokens) for span in spans if span.start < span.end]


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = {item for item in left if item}
    right_set = {item for item in right if item}
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _sequence_ratio(left: list[str], right: list[str]) -> float:
    if not left and not right:
        return 1.0
    return float(SequenceMatcher(a=left, b=right).ratio())


def _count_ratio(left_count: int, right_count: int) -> float:
    if left_count == 0 and right_count == 0:
        return 1.0
    if left_count == 0 or right_count == 0:
        return 0.0
    return min(left_count, right_count) / max(left_count, right_count)

