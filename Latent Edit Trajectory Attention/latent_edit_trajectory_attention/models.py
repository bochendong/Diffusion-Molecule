"""History-conditioned latent diffusion modules.

The model keeps the core research idea explicit:

    trajectory tokens -> transformer context -> diffusion noise predictor

Inputs are latent molecular states rather than raw graphs or sketches. This
keeps the first prototype independent of a particular molecular encoder while
leaving a clean interface for SketchMol, graph, or SMILES encoders later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _load_torch() -> Any:
    try:
        import torch

        return torch
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("Latent Edit Trajectory Attention requires PyTorch.") from exc


@dataclass(frozen=True)
class TrajectoryDiffusionConfig:
    """Configuration for the trajectory-conditioned latent diffusion editor."""

    latent_dim: int = 128
    property_dim: int = 4
    target_dim: int = 4
    edit_type_count: int = 16
    hidden_dim: int = 256
    transformer_layers: int = 4
    attention_heads: int = 8
    diffusion_steps: int = 100
    max_history: int = 16
    dropout: float = 0.1

    def validate(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if self.property_dim < 0:
            raise ValueError("property_dim cannot be negative.")
        if self.property_dim > self.latent_dim:
            raise ValueError("property_dim cannot exceed latent_dim in the first prototype.")
        if self.target_dim < 0:
            raise ValueError("target_dim cannot be negative.")
        if self.edit_type_count <= 0:
            raise ValueError("edit_type_count must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.transformer_layers <= 0:
            raise ValueError("transformer_layers must be positive.")
        if self.attention_heads <= 0:
            raise ValueError("attention_heads must be positive.")
        if self.hidden_dim % self.attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads.")
        if self.diffusion_steps <= 0:
            raise ValueError("diffusion_steps must be positive.")
        if self.max_history <= 0:
            raise ValueError("max_history must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")


class TrajectoryConditionedDiffusionEditor:
    """Factory that creates a PyTorch module without importing torch at module load."""

    def __new__(cls, config: TrajectoryDiffusionConfig | None = None, **kwargs: Any) -> Any:
        torch = _load_torch()
        nn = torch.nn

        cfg = config or TrajectoryDiffusionConfig(**kwargs)
        cfg.validate()

        class _Model(nn.Module):
            def __init__(self, model_config: TrajectoryDiffusionConfig) -> None:
                super().__init__()
                self.config = model_config
                token_in_dim = model_config.latent_dim + model_config.property_dim
                target_in_dim = max(1, model_config.target_dim)

                self.token_projection = nn.Linear(token_in_dim, model_config.hidden_dim)
                self.edit_embedding = nn.Embedding(model_config.edit_type_count, model_config.hidden_dim)
                self.position_embedding = nn.Embedding(model_config.max_history, model_config.hidden_dim)
                self.time_embedding = nn.Embedding(model_config.diffusion_steps + 1, model_config.hidden_dim)
                self.target_projection = nn.Linear(target_in_dim, model_config.hidden_dim)

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=model_config.hidden_dim,
                    nhead=model_config.attention_heads,
                    dim_feedforward=model_config.hidden_dim * 4,
                    dropout=model_config.dropout,
                    batch_first=True,
                    norm_first=True,
                )
                self.trajectory_encoder = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=model_config.transformer_layers,
                )
                self.context_norm = nn.LayerNorm(model_config.hidden_dim)

                editor_in_dim = model_config.latent_dim + model_config.hidden_dim * 3
                self.editor = nn.Sequential(
                    nn.Linear(editor_in_dim, model_config.hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(model_config.dropout),
                    nn.Linear(model_config.hidden_dim * 2, model_config.hidden_dim * 2),
                    nn.GELU(),
                    nn.Linear(model_config.hidden_dim * 2, model_config.latent_dim),
                )

            def _default_property_delta(self, z_history: Any) -> Any:
                batch, steps, _ = z_history.shape
                return z_history.new_zeros((batch, steps, self.config.property_dim))

            def _default_target(self, z_history: Any) -> Any:
                batch = z_history.shape[0]
                return z_history.new_zeros((batch, max(1, self.config.target_dim)))

            def _default_edit_types(self, z_history: Any) -> Any:
                batch, steps = z_history.shape[:2]
                return torch.zeros((batch, steps), dtype=torch.long, device=z_history.device)

            def _default_history_mask(self, z_history: Any) -> Any:
                batch, steps = z_history.shape[:2]
                return torch.ones((batch, steps), dtype=torch.bool, device=z_history.device)

            def encode_trajectory(
                self,
                z_history: Any,
                property_delta: Any | None = None,
                edit_type_ids: Any | None = None,
                history_mask: Any | None = None,
            ) -> tuple[Any, Any]:
                """Return ``(history_context, encoded_tokens)`` for a latent edit history."""

                if z_history.ndim != 3:
                    raise ValueError("z_history must have shape [batch, history, latent_dim].")
                batch, steps, latent_dim = z_history.shape
                if latent_dim != self.config.latent_dim:
                    raise ValueError(f"Expected latent_dim={self.config.latent_dim}, got {latent_dim}.")
                if steps > self.config.max_history:
                    z_history = z_history[:, -self.config.max_history :, :]
                    steps = self.config.max_history
                    if property_delta is not None:
                        property_delta = property_delta[:, -self.config.max_history :, :]
                    if edit_type_ids is not None:
                        edit_type_ids = edit_type_ids[:, -self.config.max_history :]
                    if history_mask is not None:
                        history_mask = history_mask[:, -self.config.max_history :]

                if property_delta is None:
                    property_delta = self._default_property_delta(z_history)
                if edit_type_ids is None:
                    edit_type_ids = self._default_edit_types(z_history)
                if history_mask is None:
                    history_mask = self._default_history_mask(z_history)

                if property_delta.shape[:2] != (batch, steps):
                    raise ValueError("property_delta must match z_history batch/history dimensions.")
                if property_delta.shape[-1] != self.config.property_dim:
                    raise ValueError(f"Expected property_dim={self.config.property_dim}.")
                if edit_type_ids.shape != (batch, steps):
                    raise ValueError("edit_type_ids must have shape [batch, history].")
                if history_mask.shape != (batch, steps):
                    raise ValueError("history_mask must have shape [batch, history].")

                token_features = torch.cat([z_history, property_delta], dim=-1)
                positions = torch.arange(steps, device=z_history.device).unsqueeze(0).expand(batch, -1)
                edit_type_ids = edit_type_ids.clamp(0, self.config.edit_type_count - 1).long()
                hidden = (
                    self.token_projection(token_features)
                    + self.edit_embedding(edit_type_ids)
                    + self.position_embedding(positions)
                )

                padding_mask = ~history_mask.bool()
                encoded = self.trajectory_encoder(hidden, src_key_padding_mask=padding_mask)
                lengths = history_mask.long().sum(dim=1).clamp(min=1)
                last_indices = (lengths - 1).view(batch, 1, 1).expand(batch, 1, self.config.hidden_dim)
                context = encoded.gather(dim=1, index=last_indices).squeeze(1)
                return self.context_norm(context), encoded

            def forward(
                self,
                noisy_next_z: Any,
                noise_step: Any,
                z_history: Any,
                property_delta: Any | None = None,
                edit_type_ids: Any | None = None,
                history_mask: Any | None = None,
                target: Any | None = None,
            ) -> tuple[Any, Any]:
                """Predict diffusion noise for the next latent state."""

                if noisy_next_z.ndim != 2:
                    raise ValueError("noisy_next_z must have shape [batch, latent_dim].")
                if noisy_next_z.shape[-1] != self.config.latent_dim:
                    raise ValueError(f"Expected latent_dim={self.config.latent_dim}.")

                context, _ = self.encode_trajectory(
                    z_history=z_history,
                    property_delta=property_delta,
                    edit_type_ids=edit_type_ids,
                    history_mask=history_mask,
                )
                if target is None:
                    target = self._default_target(z_history)
                if target.shape[-1] != max(1, self.config.target_dim):
                    raise ValueError(f"Expected target_dim={self.config.target_dim}.")

                t = noise_step.clamp(0, self.config.diffusion_steps).long()
                time_context = self.time_embedding(t)
                target_context = self.target_projection(target)
                editor_input = torch.cat([noisy_next_z, context, time_context, target_context], dim=-1)
                predicted_noise = self.editor(editor_input)
                return predicted_noise, context

            def denoising_loss(
                self,
                noisy_next_z: Any,
                noise: Any,
                noise_step: Any,
                z_history: Any,
                property_delta: Any | None = None,
                edit_type_ids: Any | None = None,
                history_mask: Any | None = None,
                target: Any | None = None,
            ) -> Any:
                predicted_noise, _ = self.forward(
                    noisy_next_z=noisy_next_z,
                    noise_step=noise_step,
                    z_history=z_history,
                    property_delta=property_delta,
                    edit_type_ids=edit_type_ids,
                    history_mask=history_mask,
                    target=target,
                )
                return torch.nn.functional.mse_loss(predicted_noise, noise)

        return _Model(cfg)


class CurrentStateDiffusionEditor:
    """Current-state-only baseline: p(z_next | z_current, target)."""

    def __new__(cls, config: TrajectoryDiffusionConfig | None = None, **kwargs: Any) -> Any:
        torch = _load_torch()
        nn = torch.nn

        cfg = config or TrajectoryDiffusionConfig(**kwargs)
        cfg.validate()

        class _Model(nn.Module):
            def __init__(self, model_config: TrajectoryDiffusionConfig) -> None:
                super().__init__()
                self.config = model_config
                target_in_dim = max(1, model_config.target_dim)
                self.current_projection = nn.Linear(model_config.latent_dim, model_config.hidden_dim)
                self.time_embedding = nn.Embedding(model_config.diffusion_steps + 1, model_config.hidden_dim)
                self.target_projection = nn.Linear(target_in_dim, model_config.hidden_dim)
                self.context_norm = nn.LayerNorm(model_config.hidden_dim)

                editor_in_dim = model_config.latent_dim + model_config.hidden_dim * 3
                self.editor = nn.Sequential(
                    nn.Linear(editor_in_dim, model_config.hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(model_config.dropout),
                    nn.Linear(model_config.hidden_dim * 2, model_config.hidden_dim * 2),
                    nn.GELU(),
                    nn.Linear(model_config.hidden_dim * 2, model_config.latent_dim),
                )

            def _default_target(self, z_history: Any) -> Any:
                batch = z_history.shape[0]
                return z_history.new_zeros((batch, max(1, self.config.target_dim)))

            def _current_z(self, z_history: Any, history_mask: Any | None = None) -> Any:
                if z_history.ndim != 3:
                    raise ValueError("z_history must have shape [batch, history, latent_dim].")
                batch, steps, latent_dim = z_history.shape
                if latent_dim != self.config.latent_dim:
                    raise ValueError(f"Expected latent_dim={self.config.latent_dim}, got {latent_dim}.")
                if history_mask is None:
                    return z_history[:, -1, :]
                if history_mask.shape != (batch, steps):
                    raise ValueError("history_mask must have shape [batch, history].")
                lengths = history_mask.long().sum(dim=1).clamp(min=1)
                last_indices = (lengths - 1).view(batch, 1, 1).expand(batch, 1, latent_dim)
                return z_history.gather(dim=1, index=last_indices).squeeze(1)

            def forward(
                self,
                noisy_next_z: Any,
                noise_step: Any,
                z_history: Any,
                property_delta: Any | None = None,
                edit_type_ids: Any | None = None,
                history_mask: Any | None = None,
                target: Any | None = None,
            ) -> tuple[Any, Any]:
                """Predict diffusion noise using only the latest molecular state."""

                del property_delta, edit_type_ids
                if noisy_next_z.ndim != 2:
                    raise ValueError("noisy_next_z must have shape [batch, latent_dim].")
                if noisy_next_z.shape[-1] != self.config.latent_dim:
                    raise ValueError(f"Expected latent_dim={self.config.latent_dim}.")

                current_z = self._current_z(z_history, history_mask=history_mask)
                if target is None:
                    target = self._default_target(z_history)
                if target.shape[-1] != max(1, self.config.target_dim):
                    raise ValueError(f"Expected target_dim={self.config.target_dim}.")

                t = noise_step.clamp(0, self.config.diffusion_steps).long()
                current_context = self.context_norm(self.current_projection(current_z))
                time_context = self.time_embedding(t)
                target_context = self.target_projection(target)
                editor_input = torch.cat([noisy_next_z, current_context, time_context, target_context], dim=-1)
                predicted_noise = self.editor(editor_input)
                return predicted_noise, current_context

            def denoising_loss(
                self,
                noisy_next_z: Any,
                noise: Any,
                noise_step: Any,
                z_history: Any,
                property_delta: Any | None = None,
                edit_type_ids: Any | None = None,
                history_mask: Any | None = None,
                target: Any | None = None,
            ) -> Any:
                predicted_noise, _ = self.forward(
                    noisy_next_z=noisy_next_z,
                    noise_step=noise_step,
                    z_history=z_history,
                    property_delta=property_delta,
                    edit_type_ids=edit_type_ids,
                    history_mask=history_mask,
                    target=target,
                )
                return torch.nn.functional.mse_loss(predicted_noise, noise)

        return _Model(cfg)


def add_diffusion_noise(clean_z: Any, noise_step: Any, diffusion_steps: int) -> tuple[Any, Any]:
    """Simple variance-preserving noise schedule for latent smoke training."""

    torch = _load_torch()
    if diffusion_steps <= 0:
        raise ValueError("diffusion_steps must be positive.")
    noise = torch.randn_like(clean_z)
    alpha = 1.0 - (noise_step.float().clamp(0, diffusion_steps) / float(diffusion_steps))
    alpha = alpha.view(-1, 1).clamp(0.0, 1.0)
    noisy = alpha.sqrt() * clean_z + (1.0 - alpha).sqrt() * noise
    return noisy, noise

