"""SMILES tokenization utilities.

The tokenizer is deliberately small and dependency-free. It keeps bracket atoms
and common two-character atoms as atomic tokens so edit spans are chemically less
chaotic than character-level spans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
MASK = "<mask>"

SPECIAL_TOKENS = [PAD, BOS, EOS, UNK, MASK]

SMILES_TOKEN_RE = re.compile(
    r"(\[[^\]]+\]|"
    r"Br|Cl|Si|Se|Na|Li|Mg|Ca|Al|Fe|Zn|Cu|Mn|"
    r"@@?|%\d{2}|\d|"
    r"\.|=|#|-|/|\\|\+|:|~|\(|\)|"
    r"[BCNOFPSIHK]|[bcnops]|.)"
)


def tokenize_smiles(smiles: str) -> list[str]:
    """Split a SMILES string into chemistry-aware tokens."""

    text = str(smiles or "").strip()
    if not text:
        return []
    return [token for token in SMILES_TOKEN_RE.findall(text) if token]


def detokenize_smiles(tokens: Iterable[str]) -> str:
    """Join tokens back into a SMILES-like string."""

    return "".join(token for token in tokens if token not in {PAD, BOS, EOS})


def is_atom_token(token: str) -> bool:
    """Return whether a token looks like an atom token."""

    if not token:
        return False
    if token.startswith("[") and token.endswith("]"):
        return True
    if token in {"B", "C", "N", "O", "P", "S", "F", "I", "H", "K"}:
        return True
    if token in {"Br", "Cl", "Si", "Se", "Na", "Li", "Mg", "Ca", "Al", "Fe", "Zn", "Cu", "Mn"}:
        return True
    return token in {"b", "c", "n", "o", "p", "s"}


@dataclass
class SmilesVocabulary:
    """A deterministic token vocabulary."""

    token_to_id: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for token in SPECIAL_TOKENS:
            self.add(token)

    @property
    def id_to_token(self) -> list[str]:
        ordered = sorted(self.token_to_id.items(), key=lambda item: item[1])
        return [token for token, _ in ordered]

    def add(self, token: str) -> int:
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.token_to_id)
        return self.token_to_id[token]

    def update(self, token_sequences: Iterable[Iterable[str]]) -> None:
        for tokens in token_sequences:
            for token in tokens:
                self.add(token)

    def encode(self, tokens: Iterable[str], *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        values: list[int] = []
        if add_bos:
            values.append(self.token_to_id[BOS])
        unk = self.token_to_id[UNK]
        values.extend(self.token_to_id.get(token, unk) for token in tokens)
        if add_eos:
            values.append(self.token_to_id[EOS])
        return values

    def decode(self, ids: Iterable[int]) -> list[str]:
        tokens = self.id_to_token
        result: list[str] = []
        for value in ids:
            if 0 <= int(value) < len(tokens):
                result.append(tokens[int(value)])
            else:
                result.append(UNK)
        return result

    def to_dict(self) -> dict[str, int]:
        return dict(self.token_to_id)

    @classmethod
    def from_dict(cls, payload: dict[str, int]) -> "SmilesVocabulary":
        vocab = cls()
        vocab.token_to_id = dict(payload)
        return vocab

