"""Condition-token adapters for SketchMol-style cross-attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ConditionEncoderOutput:
    """Condition tokens plus an attention mask."""

    tokens: torch.Tensor
    attention_mask: torch.Tensor


def masked_mean(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean-pool hidden states with an optional `[B, L]` mask."""

    if attention_mask is None:
        return hidden_states.mean(dim=1)
    mask = attention_mask.to(dtype=hidden_states.dtype, device=hidden_states.device).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden_states * mask).sum(dim=1) / denom


class MolecularQueryProjector(nn.Module):
    """Project MLLM hidden states into SketchMol condition query tokens.

    This is intentionally lightweight: it can sit after a frozen MLLM and before
    SketchMol's UNet cross-attention, producing `[B, num_queries, context_dim]`.
    """

    def __init__(
        self,
        mllm_hidden_dim: int,
        context_dim: int = 256,
        num_queries: int = 32,
        hidden_dim: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.context_dim = context_dim
        self.query_tokens = nn.Parameter(torch.randn(num_queries, context_dim) * 0.02)
        self.global_projector = nn.Sequential(
            nn.LayerNorm(mllm_hidden_dim),
            nn.Linear(mllm_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, context_dim),
        )
        self.token_projector = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )

    def forward(
        self,
        mllm_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> ConditionEncoderOutput:
        if mllm_hidden_states.ndim != 3:
            raise ValueError(
                "mllm_hidden_states must be [batch, sequence, hidden_dim], "
                f"got {tuple(mllm_hidden_states.shape)}"
            )

        batch_size = mllm_hidden_states.shape[0]
        pooled = masked_mean(mllm_hidden_states, attention_mask)
        global_token = self.global_projector(pooled).unsqueeze(1)
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        tokens = self.token_projector(queries + global_token)
        mask = torch.ones(
            batch_size,
            self.num_queries,
            dtype=torch.bool,
            device=mllm_hidden_states.device,
        )
        return ConditionEncoderOutput(tokens=tokens, attention_mask=mask)


class HybridConditionEncoder(nn.Module):
    """Concatenate existing SketchMol property tokens with MLLM query tokens."""

    def __init__(self, mllm_projector: MolecularQueryProjector) -> None:
        super().__init__()
        self.mllm_projector = mllm_projector

    def forward(
        self,
        mllm_hidden_states: torch.Tensor,
        mllm_attention_mask: Optional[torch.Tensor] = None,
        property_tokens: Optional[torch.Tensor] = None,
        property_attention_mask: Optional[torch.Tensor] = None,
    ) -> ConditionEncoderOutput:
        mllm_out = self.mllm_projector(mllm_hidden_states, mllm_attention_mask)

        if property_tokens is None:
            return mllm_out
        if property_tokens.ndim != 3:
            raise ValueError(
                "property_tokens must be [batch, condition_len, context_dim], "
                f"got {tuple(property_tokens.shape)}"
            )
        if property_tokens.shape[0] != mllm_out.tokens.shape[0]:
            raise ValueError("property_tokens and mllm_hidden_states batch sizes differ")
        if property_tokens.shape[-1] != mllm_out.tokens.shape[-1]:
            raise ValueError("property token dim must match projected MLLM context dim")

        if property_attention_mask is None:
            property_attention_mask = torch.ones(
                property_tokens.shape[:2],
                dtype=torch.bool,
                device=property_tokens.device,
            )

        tokens = torch.cat([property_tokens, mllm_out.tokens], dim=1)
        mask = torch.cat(
            [
                property_attention_mask.to(device=tokens.device, dtype=torch.bool),
                mllm_out.attention_mask,
            ],
            dim=1,
        )
        return ConditionEncoderOutput(tokens=tokens, attention_mask=mask)
