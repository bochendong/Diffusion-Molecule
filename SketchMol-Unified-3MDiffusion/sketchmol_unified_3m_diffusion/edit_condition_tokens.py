"""Models for molecule-language alignment and edit-aware condition tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import ConditionEncoderOutput, MolecularQueryProjector
from .unified_condition_dataset import PROPERTY_COLUMNS


@dataclass
class EditConditionOutput:
    """Condition tokens and auxiliary edit predictions."""

    tokens: torch.Tensor
    attention_mask: torch.Tensor
    pooled: torch.Tensor
    target_properties: torch.Tensor
    property_deltas: torch.Tensor
    active_logits: torch.Tensor
    direction_logits: torch.Tensor
    target_fingerprint_logits: torch.Tensor
    similarity_bin_logits: torch.Tensor


class MoleculeLanguageAlignmentModel(nn.Module):
    """Contrastively align molecule/image features with language features."""

    def __init__(
        self,
        molecule_dim: int,
        text_dim: int,
        *,
        image_dim: int = 0,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.image_dim = int(image_dim)
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))
        self.molecule_projector = _mlp(molecule_dim, hidden_dim, embed_dim)
        self.text_projector = _mlp(text_dim, hidden_dim, embed_dim)
        self.image_projector = _mlp(image_dim, hidden_dim, embed_dim) if image_dim > 0 else None

    def encode_molecule(self, molecule_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.molecule_projector(molecule_features), dim=-1)

    def encode_text(self, text_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_projector(text_features), dim=-1)

    def encode_image(self, image_features: torch.Tensor) -> torch.Tensor | None:
        if self.image_projector is None:
            return None
        return F.normalize(self.image_projector(image_features), dim=-1)

    def contrastive_loss(
        self,
        molecule_features: torch.Tensor,
        text_features: torch.Tensor,
        image_features: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        molecule_emb = self.encode_molecule(molecule_features)
        text_emb = self.encode_text(text_features)
        temp = self.temperature.clamp_min(1e-3)
        losses = [symmetric_infonce(molecule_emb, text_emb, temperature=temp)]
        logs = {"structure_text_loss": losses[0].detach()}

        if image_features is not None and self.image_projector is not None:
            image_emb = self.encode_image(image_features)
            assert image_emb is not None
            if image_mask is not None:
                keep = image_mask.to(dtype=torch.bool, device=image_emb.device)
                if bool(keep.any()):
                    image_loss = symmetric_infonce(image_emb[keep], text_emb[keep], temperature=temp)
                    losses.append(image_loss)
                    logs["image_text_loss"] = image_loss.detach()
            else:
                image_loss = symmetric_infonce(image_emb, text_emb, temperature=temp)
                losses.append(image_loss)
                logs["image_text_loss"] = image_loss.detach()

        loss = torch.stack(losses).mean()
        logs["loss"] = loss.detach()
        return loss, logs


class EditConditionTokenConnector(nn.Module):
    """Map VLM/source-instruction hidden states to edit-aware condition tokens."""

    def __init__(
        self,
        input_hidden_dim: int,
        *,
        context_dim: int = 256,
        num_queries: int = 16,
        hidden_dim: int = 512,
        fingerprint_dim: int = 512,
        num_similarity_bins: int = 4,
        num_properties: int = len(PROPERTY_COLUMNS),
    ) -> None:
        super().__init__()
        self.input_hidden_dim = input_hidden_dim
        self.context_dim = context_dim
        self.num_queries = num_queries
        self.fingerprint_dim = fingerprint_dim
        self.num_properties = num_properties
        self.query_projector = MolecularQueryProjector(
            mllm_hidden_dim=input_hidden_dim,
            context_dim=context_dim,
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            dropout=0.1,
        )
        self.pool = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.target_property_head = nn.Linear(hidden_dim, num_properties)
        self.delta_head = nn.Linear(hidden_dim, num_properties)
        self.active_head = nn.Linear(hidden_dim, num_properties)
        self.direction_head = nn.Linear(hidden_dim, num_properties * 3)
        self.fingerprint_head = nn.Linear(hidden_dim, fingerprint_dim)
        self.similarity_bin_head = nn.Linear(hidden_dim, num_similarity_bins)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> EditConditionOutput:
        projected = self.query_projector(hidden_states, attention_mask)
        pooled_tokens = projected.tokens.mean(dim=1)
        pooled = self.pool(pooled_tokens)
        return EditConditionOutput(
            tokens=projected.tokens,
            attention_mask=projected.attention_mask,
            pooled=pooled,
            target_properties=self.target_property_head(pooled),
            property_deltas=self.delta_head(pooled),
            active_logits=self.active_head(pooled),
            direction_logits=self.direction_head(pooled).view(-1, self.num_properties, 3),
            target_fingerprint_logits=self.fingerprint_head(pooled),
            similarity_bin_logits=self.similarity_bin_head(pooled),
        )


def edit_condition_loss(
    output: EditConditionOutput,
    *,
    target_properties: torch.Tensor,
    property_deltas: torch.Tensor,
    active_mask: torch.Tensor,
    direction_labels: torch.Tensor,
    target_fingerprint: torch.Tensor,
    similarity_bin: torch.Tensor,
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Multi-task loss for edit-aware condition tokens."""

    weights = weights or {}
    losses = {
        "target_property_mse": F.mse_loss(output.target_properties, target_properties),
        "delta_mse": F.mse_loss(output.property_deltas, property_deltas),
        "active_bce": F.binary_cross_entropy_with_logits(output.active_logits, active_mask),
        "direction_ce": F.cross_entropy(output.direction_logits.reshape(-1, 3), direction_labels.reshape(-1)),
        "fingerprint_bce": F.binary_cross_entropy_with_logits(output.target_fingerprint_logits, target_fingerprint),
        "similarity_ce": F.cross_entropy(output.similarity_bin_logits, similarity_bin),
    }
    total = sum(float(weights.get(name, 1.0)) * loss for name, loss in losses.items())
    logs = {name: loss.detach() for name, loss in losses.items()}
    logs["loss"] = total.detach()
    return total, logs


def symmetric_infonce(left: torch.Tensor, right: torch.Tensor, *, temperature: torch.Tensor) -> torch.Tensor:
    """Symmetric in-batch contrastive loss."""

    if left.shape[0] != right.shape[0]:
        raise ValueError("left and right batch sizes must match")
    if left.shape[0] == 0:
        raise ValueError("contrastive batch is empty")
    logits = left @ right.T / temperature
    labels = torch.arange(left.shape[0], device=left.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )
