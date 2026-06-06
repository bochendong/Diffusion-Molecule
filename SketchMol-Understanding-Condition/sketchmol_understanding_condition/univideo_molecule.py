"""UniVideo-style dual-stream molecular editing modules.

The understanding stream consumes frozen MLLM/VLM hidden states and produces
SketchMol-compatible cross-attention condition tokens. The generation stream
consumes a source molecular latent and denoises a target molecular latent under
those condition tokens.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import MolecularQueryProjector, masked_mean
from .latent_diffusion_generation import extract, make_beta_schedule
from .unified_condition_dataset import PROPERTY_COLUMNS, UnifiedConditionSample
from .unified_featurization import (
    active_property_vector,
    direction_label_vector,
    hidden_sequence_for_sample,
    molecule_feature,
    property_delta_vector,
    similarity_bin_label,
    target_latent_vector,
    target_property_vector,
)


@dataclass
class UniVideoConditionOutput:
    """Projected diffusion condition plus auxiliary edit predictions."""

    tokens: torch.Tensor
    attention_mask: torch.Tensor
    pooled: torch.Tensor
    target_latent: torch.Tensor
    target_properties: torch.Tensor
    property_deltas: torch.Tensor
    active_logits: torch.Tensor
    direction_logits: torch.Tensor
    similarity_bin_logits: torch.Tensor


@dataclass
class UniVideoDiffusionPrediction:
    """Noise and clean-latent predictions from the source-conditioned denoiser."""

    pred_noise: torch.Tensor
    pred_x0: torch.Tensor


class UniVideoMoleculeConnector(nn.Module):
    """Map frozen MLLM condition states to diffusion-readable edit tokens."""

    def __init__(
        self,
        input_hidden_dim: int,
        *,
        latent_dim: int,
        context_dim: int = 256,
        num_queries: int = 32,
        hidden_dim: int = 512,
        num_similarity_bins: int = 4,
        num_properties: int = len(PROPERTY_COLUMNS),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_hidden_dim = int(input_hidden_dim)
        self.latent_dim = int(latent_dim)
        self.context_dim = int(context_dim)
        self.num_queries = int(num_queries)
        self.projector = MolecularQueryProjector(
            mllm_hidden_dim=input_hidden_dim,
            context_dim=context_dim,
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.pool = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.target_latent_head = nn.Linear(hidden_dim, latent_dim)
        self.target_property_head = nn.Linear(hidden_dim, num_properties)
        self.delta_head = nn.Linear(hidden_dim, num_properties)
        self.active_head = nn.Linear(hidden_dim, num_properties)
        self.direction_head = nn.Linear(hidden_dim, num_properties * 3)
        self.similarity_bin_head = nn.Linear(hidden_dim, num_similarity_bins)

    def forward(
        self,
        mllm_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> UniVideoConditionOutput:
        projected = self.projector(mllm_hidden_states, attention_mask)
        pooled_tokens = masked_mean(projected.tokens, projected.attention_mask)
        pooled = self.pool(pooled_tokens)
        return UniVideoConditionOutput(
            tokens=projected.tokens,
            attention_mask=projected.attention_mask,
            pooled=pooled,
            target_latent=self.target_latent_head(pooled),
            target_properties=self.target_property_head(pooled),
            property_deltas=self.delta_head(pooled),
            active_logits=self.active_head(pooled),
            direction_logits=self.direction_head(pooled).view(-1, len(PROPERTY_COLUMNS), 3),
            similarity_bin_logits=self.similarity_bin_head(pooled),
        )


class SourceConditionedEditDenoiser(nn.Module):
    """Denoise target latents from source latents and understanding tokens."""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        *,
        hidden_dim: int = 512,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.source_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
        )
        self.context_proj = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
        )
        layers = []
        in_dim = latent_dim + hidden_dim + hidden_dim + hidden_dim
        for idx in range(depth):
            layers.append(nn.Linear(in_dim if idx == 0 else hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        source_latent: torch.Tensor,
        condition_tokens: torch.Tensor,
        condition_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = masked_mean(condition_tokens, condition_mask)
        hidden = torch.cat(
            [
                noisy_latent,
                self.time_mlp(timesteps.float()),
                self.source_proj(source_latent),
                self.context_proj(context),
            ],
            dim=-1,
        )
        return self.output(self.net(hidden))


class SourceConditionedGaussianLatentDiffusion(nn.Module):
    """Gaussian latent diffusion with explicit source-latent conditioning."""

    def __init__(
        self,
        denoiser: SourceConditionedEditDenoiser,
        *,
        timesteps: int = 1000,
        objective: str = "pred_noise",
        beta_schedule: str = "linear",
        condition_dropout: float = 0.0,
        source_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if objective not in {"pred_noise", "pred_x0"}:
            raise ValueError(f"Unsupported objective: {objective}")
        self.denoiser = denoiser
        self.timesteps = int(timesteps)
        self.objective = objective
        self.condition_dropout = float(condition_dropout)
        self.source_dropout = float(source_dropout)
        betas = make_beta_schedule(self.timesteps, beta_schedule)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

    def q_sample(self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return extract(self.sqrt_alphas_cumprod, timesteps, x0.shape) * x0 + extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x0.shape
        ) * noise

    def predict_x0_from_noise(self, xt: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return extract(self.sqrt_recip_alphas_cumprod, timesteps, xt.shape) * xt - extract(
            self.sqrt_recipm1_alphas_cumprod, timesteps, xt.shape
        ) * noise

    def predict_noise_from_x0(self, xt: torch.Tensor, timesteps: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        return (extract(self.sqrt_recip_alphas_cumprod, timesteps, xt.shape) * xt - x0) / extract(
            self.sqrt_recipm1_alphas_cumprod, timesteps, xt.shape
        )

    def model_predictions(
        self,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
        source_latent: torch.Tensor,
        condition_tokens: torch.Tensor,
        condition_mask: torch.Tensor | None = None,
    ) -> UniVideoDiffusionPrediction:
        model_out = self.denoiser(xt, timesteps, source_latent, condition_tokens, condition_mask)
        if self.objective == "pred_noise":
            pred_noise = model_out
            pred_x0 = self.predict_x0_from_noise(xt, timesteps, pred_noise)
        else:
            pred_x0 = model_out
            pred_noise = self.predict_noise_from_x0(xt, timesteps, pred_x0)
        return UniVideoDiffusionPrediction(pred_noise=pred_noise, pred_x0=pred_x0)

    def loss(
        self,
        target_latent: torch.Tensor,
        source_latent: torch.Tensor,
        condition_tokens: torch.Tensor,
        condition_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = target_latent.shape[0]
        timesteps = torch.randint(0, self.timesteps, (batch,), device=target_latent.device)
        noise = torch.randn_like(target_latent)
        xt = self.q_sample(target_latent, timesteps, noise)
        source_latent, condition_tokens, condition_mask = self._apply_dropout(source_latent, condition_tokens, condition_mask)
        predictions = self.model_predictions(xt, timesteps, source_latent, condition_tokens, condition_mask)
        target = noise if self.objective == "pred_noise" else target_latent
        pred = predictions.pred_noise if self.objective == "pred_noise" else predictions.pred_x0
        return F.mse_loss(pred, target)

    @torch.no_grad()
    def sample(
        self,
        source_latent: torch.Tensor,
        condition_tokens: torch.Tensor,
        condition_mask: torch.Tensor | None = None,
        *,
        steps: int | None = None,
    ) -> torch.Tensor:
        steps = int(steps or self.timesteps)
        latent = torch.randn_like(source_latent)
        times = torch.linspace(self.timesteps - 1, 0, steps, device=source_latent.device).long()
        for timestep in times:
            t = timestep.repeat(source_latent.shape[0])
            predictions = self.model_predictions(latent, t, source_latent, condition_tokens, condition_mask)
            latent = predictions.pred_x0
            if int(timestep.item()) > 0:
                latent = latent + extract(self.sqrt_one_minus_alphas_cumprod, t, latent.shape) * torch.randn_like(latent)
        return latent

    def _apply_dropout(
        self,
        source_latent: torch.Tensor,
        condition_tokens: torch.Tensor,
        condition_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if not self.training:
            return source_latent, condition_tokens, condition_mask
        if self.source_dropout > 0:
            keep = torch.rand(source_latent.shape[0], 1, device=source_latent.device) >= self.source_dropout
            source_latent = source_latent * keep.to(source_latent.dtype)
        if self.condition_dropout > 0:
            keep = torch.rand(condition_tokens.shape[0], 1, 1, device=condition_tokens.device) >= self.condition_dropout
            condition_tokens = condition_tokens * keep.to(condition_tokens.dtype)
            if condition_mask is not None:
                condition_keep = keep.squeeze(-1).to(dtype=torch.bool)
                condition_mask = condition_mask & condition_keep
        return source_latent, condition_tokens, condition_mask


def univideo_connector_alignment_loss(
    output: UniVideoConditionOutput,
    *,
    target_latent: torch.Tensor,
    target_properties: torch.Tensor,
    property_deltas: torch.Tensor,
    active_mask: torch.Tensor,
    direction_labels: torch.Tensor,
    similarity_bin: torch.Tensor,
    weights: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Auxiliary connector loss used before/alongside diffusion training."""

    weights = weights or {}
    losses = {
        "target_latent_mse": F.mse_loss(output.target_latent, target_latent),
        "target_property_mse": F.mse_loss(output.target_properties, target_properties),
        "delta_mse": F.mse_loss(output.property_deltas, property_deltas),
        "active_bce": F.binary_cross_entropy_with_logits(output.active_logits, active_mask),
        "direction_ce": F.cross_entropy(output.direction_logits.reshape(-1, 3), direction_labels.reshape(-1)),
        "similarity_ce": F.cross_entropy(output.similarity_bin_logits, similarity_bin),
    }
    total = sum(float(weights.get(name, 1.0)) * loss for name, loss in losses.items())
    logs = {name: loss.detach() for name, loss in losses.items()}
    logs["loss"] = total.detach()
    return total, logs


class FrozenConditionFeatureStore:
    """Load exported frozen VLM features by condition id."""

    def __init__(self, feature_dir: str | Path, *, array_name: str = "query_tokens", variant: str = "full") -> None:
        self.feature_dir = Path(feature_dir)
        self.array_name = array_name
        self.variant = variant
        array_path = self.feature_dir / f"{array_name}.npy"
        index_path = self.feature_dir / "index.csv"
        if not array_path.exists():
            raise FileNotFoundError(f"Missing condition feature array: {array_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"Missing condition feature index: {index_path}")
        self.features = np.load(array_path).astype(np.float32)
        self.by_condition_id: dict[str, np.ndarray] = {}
        with index_path.open(newline="", encoding="utf-8") as handle:
            for idx, row in enumerate(csv.DictReader(handle)):
                if idx >= len(self.features):
                    break
                if variant and row.get("variant", "") not in {"", variant}:
                    continue
                condition_id = row.get("condition_id", "")
                if condition_id and condition_id not in self.by_condition_id:
                    self.by_condition_id[condition_id] = _as_sequence(self.features[idx])

    @property
    def input_hidden_dim(self) -> int:
        if self.features.ndim == 3:
            return int(self.features.shape[-1])
        if self.features.ndim == 2:
            return int(self.features.shape[-1])
        raise ValueError(f"Unsupported feature shape: {self.features.shape}")

    def get(self, condition_id: str) -> np.ndarray | None:
        value = self.by_condition_id.get(condition_id)
        return None if value is None else value.copy()


def source_latent_vector(sample: UnifiedConditionSample, *, fingerprint_dim: int = 512) -> np.ndarray:
    """Fallback vector latent backend for fast sanity checks."""

    source = sample.source_smiles or sample.molecule_smiles or sample.target_smiles
    fp = molecule_feature(source, fingerprint_dim)
    props = _resize(np.asarray([sample.source_properties.get(prop, 0.0) for prop in PROPERTY_COLUMNS], dtype=np.float32), 32)
    deltas = np.zeros(32, dtype=np.float32)
    active = np.zeros(16, dtype=np.float32)
    return np.concatenate([fp, props, deltas, active]).astype(np.float32)


def univideo_training_arrays(
    sample: UnifiedConditionSample,
    *,
    feature_store: FrozenConditionFeatureStore | None = None,
    fallback_token_dim: int = 512,
    fingerprint_dim: int = 512,
) -> dict[str, np.ndarray | int]:
    """Build one training row for the UniVideo-style molecule model."""

    condition_id = sample.metadata.get("condition_id", "")
    hidden = feature_store.get(condition_id) if feature_store is not None else None
    if hidden is None:
        hidden = hidden_sequence_for_sample(sample, token_dim=fallback_token_dim)
    target_latent = target_latent_vector(sample, fingerprint_dim=fingerprint_dim)
    return {
        "mllm_hidden": _as_sequence(hidden).astype(np.float32),
        "source_latent": source_latent_vector(sample, fingerprint_dim=fingerprint_dim),
        "target_latent": target_latent,
        "target_properties": target_property_vector(sample),
        "property_deltas": property_delta_vector(sample),
        "active_mask": active_property_vector(sample),
        "direction_labels": direction_label_vector(sample).astype(np.int64),
        "similarity_bin": np.int64(similarity_bin_label(sample)),
    }


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=timesteps.device, dtype=torch.float32) * -scale)
        emb = timesteps[:, None] * freqs[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


def _as_sequence(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 1:
        return value[None, :]
    if value.ndim == 2:
        return value
    raise ValueError(f"Expected 1D or 2D condition features, got shape {value.shape}")


def _resize(vec: np.ndarray, dim: int) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    if vec.shape[0] == dim:
        return vec
    if vec.shape[0] > dim:
        return vec[:dim]
    out = np.zeros(dim, dtype=np.float32)
    out[: vec.shape[0]] = vec
    return out
