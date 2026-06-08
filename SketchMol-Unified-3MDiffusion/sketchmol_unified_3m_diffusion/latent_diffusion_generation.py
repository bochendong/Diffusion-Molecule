"""Latent diffusion generation stream for molecular edit targets."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiffusionPrediction:
    pred_noise: torch.Tensor
    pred_x0: torch.Tensor


class EditLatentDenoiser(nn.Module):
    """Denoise target molecular latents conditioned on edit condition tokens."""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        *,
        hidden_dim: int = 512,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_proj = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
        )
        layers = []
        in_dim = latent_dim + hidden_dim + hidden_dim
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
        condition_tokens: torch.Tensor,
        condition_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = masked_mean(condition_tokens, condition_mask)
        time_emb = self.time_mlp(timesteps.float())
        context_emb = self.context_proj(context)
        hidden = torch.cat([noisy_latent, time_emb, context_emb], dim=-1)
        return self.output(self.net(hidden))


class GaussianLatentDiffusion(nn.Module):
    """Simple Gaussian diffusion over fixed molecular latent vectors."""

    def __init__(
        self,
        denoiser: EditLatentDenoiser,
        *,
        timesteps: int = 1000,
        objective: str = "pred_noise",
        beta_schedule: str = "linear",
    ) -> None:
        super().__init__()
        if objective not in {"pred_noise", "pred_x0"}:
            raise ValueError(f"Unsupported objective: {objective}")
        self.denoiser = denoiser
        self.timesteps = int(timesteps)
        self.objective = objective
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
        condition_tokens: torch.Tensor,
        condition_mask: torch.Tensor | None = None,
    ) -> DiffusionPrediction:
        model_out = self.denoiser(xt, timesteps, condition_tokens, condition_mask)
        if self.objective == "pred_noise":
            pred_noise = model_out
            pred_x0 = self.predict_x0_from_noise(xt, timesteps, pred_noise)
        else:
            pred_x0 = model_out
            pred_noise = self.predict_noise_from_x0(xt, timesteps, pred_x0)
        return DiffusionPrediction(pred_noise=pred_noise, pred_x0=pred_x0)

    def loss(
        self,
        target_latent: torch.Tensor,
        condition_tokens: torch.Tensor,
        condition_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss, _ = self.loss_and_pred_x0(target_latent, condition_tokens, condition_mask)
        return loss

    def loss_and_pred_x0(
        self,
        target_latent: torch.Tensor,
        condition_tokens: torch.Tensor,
        condition_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = target_latent.shape[0]
        timesteps = torch.randint(0, self.timesteps, (batch,), device=target_latent.device)
        noise = torch.randn_like(target_latent)
        xt = self.q_sample(target_latent, timesteps, noise)
        predictions = self.model_predictions(xt, timesteps, condition_tokens, condition_mask)
        target = noise if self.objective == "pred_noise" else target_latent
        pred = predictions.pred_noise if self.objective == "pred_noise" else predictions.pred_x0
        return F.mse_loss(pred, target), predictions.pred_x0

    @torch.no_grad()
    def sample(
        self,
        condition_tokens: torch.Tensor,
        condition_mask: torch.Tensor | None = None,
        *,
        steps: int | None = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        steps = max(1, min(int(steps or self.timesteps), self.timesteps))
        eta = float(eta)
        device = condition_tokens.device
        latent = torch.randn(condition_tokens.shape[0], self.denoiser.latent_dim, device=device)
        times = torch.linspace(self.timesteps - 1, 0, steps, device=device).long()
        time_values = times.tolist()
        time_pairs = list(zip(time_values, [*time_values[1:], -1]))
        for timestep, next_timestep in time_pairs:
            t = torch.full((condition_tokens.shape[0],), int(timestep), device=device, dtype=torch.long)
            predictions = self.model_predictions(latent, t, condition_tokens, condition_mask)
            if int(next_timestep) < 0:
                latent = predictions.pred_x0
                continue
            next_t = torch.full((condition_tokens.shape[0],), int(next_timestep), device=device, dtype=torch.long)
            alpha_t = extract(self.alphas_cumprod, t, latent.shape)
            alpha_next = extract(self.alphas_cumprod, next_t, latent.shape)
            sigma = eta * torch.sqrt(
                ((1.0 - alpha_next) / (1.0 - alpha_t).clamp_min(1e-8))
                * (1.0 - alpha_t / alpha_next).clamp_min(0.0)
            )
            noise_scale = torch.sqrt((1.0 - alpha_next - sigma.pow(2)).clamp_min(0.0))
            latent = torch.sqrt(alpha_next) * predictions.pred_x0 + noise_scale * predictions.pred_noise
            if eta > 0:
                latent = latent + sigma * torch.randn_like(latent)
        return latent


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


def masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values.mean(dim=1)
    weights = mask.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (values * weights).sum(dim=1) / denom


def make_beta_schedule(timesteps: int, schedule: str) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(1e-4, 2e-2, timesteps, dtype=torch.float32)
    if schedule == "cosine":
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return betas.clamp(0, 0.999).float()
    raise ValueError(f"Unsupported beta schedule: {schedule}")


def extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    out = values.gather(0, timesteps)
    return out.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))
