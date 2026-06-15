"""MLLM-conditioned direct SMILES generation utilities."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
import torch.nn as nn


PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]

SMILES_TOKEN_RE = re.compile(
    r"(\[[^\]]+\]|"
    r"Br|Cl|Si|Se|Na|Li|Mg|Ca|Al|Fe|Zn|Cu|Mn|"
    r"@@?|%\d{2}|\d|"
    r"\.|=|#|-|/|\\|\+|:|~|\(|\)|"
    r"[BCNOFPSIHK]|[bcnops]|.)"
)


def tokenize_smiles(smiles: str) -> list[str]:
    text = str(smiles or "").strip()
    if not text:
        return []
    return [token for token in SMILES_TOKEN_RE.findall(text) if token]


def detokenize_smiles(tokens: Iterable[str]) -> str:
    skip = {PAD, BOS, EOS}
    return "".join(token for token in tokens if token not in skip)


@dataclass
class SmilesVocabulary:
    token_to_id: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for token in SPECIAL_TOKENS:
            self.add(token)

    @property
    def id_to_token(self) -> list[str]:
        return [token for token, _ in sorted(self.token_to_id.items(), key=lambda item: item[1])]

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS]

    def add(self, token: str) -> int:
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.token_to_id)
        return self.token_to_id[token]

    def update(self, token_sequences: Iterable[Iterable[str]]) -> None:
        for tokens in token_sequences:
            for token in tokens:
                self.add(token)

    def encode(self, tokens: Iterable[str], *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        unk = self.token_to_id[UNK]
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.token_to_id.get(token, unk) for token in tokens)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int]) -> list[str]:
        tokens = self.id_to_token
        out = []
        for value in ids:
            idx = int(value)
            if idx == self.eos_id:
                break
            out.append(tokens[idx] if 0 <= idx < len(tokens) else UNK)
        return out

    def to_dict(self) -> dict[str, int]:
        return dict(self.token_to_id)

    @classmethod
    def from_dict(cls, payload: dict[str, int]) -> "SmilesVocabulary":
        vocab = cls()
        vocab.token_to_id = dict(payload)
        return vocab


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512) -> None:
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / max(dim, 1)))
        pe = torch.zeros(max_len, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(positions * div_term)
        if dim > 1:
            pe[:, 1::2] = torch.cos(positions * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.pe[:, : values.shape[1]].to(dtype=values.dtype, device=values.device)


class ConditionedSmilesDecoder(nn.Module):
    """Autoregressive SMILES decoder conditioned on frozen MLLM query tokens."""

    def __init__(
        self,
        *,
        vocab_size: int,
        condition_dim: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pad_id: int = 0,
        max_length: int = 192,
    ) -> None:
        super().__init__()
        self.pad_id = int(pad_id)
        self.max_length = int(max_length)
        self.condition_proj = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position = PositionalEncoding(d_model, max_len=max_length + 8)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        condition_tokens: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        *,
        condition_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory = self.condition_proj(condition_tokens)
        tgt = self.position(self.token_embedding(decoder_input_ids))
        seq_len = decoder_input_ids.shape[1]
        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=decoder_input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        tgt_key_padding_mask = decoder_input_ids.eq(self.pad_id)
        memory_key_padding_mask = None if condition_mask is None else ~condition_mask.bool()
        decoded = self.decoder(
            tgt,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output(decoded)

    @torch.no_grad()
    def generate(
        self,
        condition_tokens: torch.Tensor,
        *,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int,
        condition_mask: torch.Tensor | None = None,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        batch = condition_tokens.shape[0]
        device = condition_tokens.device
        generated = torch.full((batch, 1), int(bos_id), dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        for _ in range(max(1, int(max_new_tokens))):
            logits = self(condition_tokens, generated, condition_mask=condition_mask)[:, -1, :]
            if temperature and temperature > 0:
                logits = logits / float(temperature)
                if top_k > 0 and top_k < logits.shape[-1]:
                    threshold = torch.topk(logits, int(top_k), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, -torch.inf)
                next_ids = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).squeeze(1)
            else:
                next_ids = logits.argmax(dim=-1)
            next_ids = torch.where(finished, torch.full_like(next_ids, int(eos_id)), next_ids)
            generated = torch.cat([generated, next_ids[:, None]], dim=1)
            finished |= next_ids.eq(int(eos_id))
            if bool(finished.all()):
                break
        return generated


def build_vocabulary(smiles_values: Sequence[str]) -> SmilesVocabulary:
    vocab = SmilesVocabulary()
    vocab.update(tokenize_smiles(value) for value in smiles_values)
    return vocab
