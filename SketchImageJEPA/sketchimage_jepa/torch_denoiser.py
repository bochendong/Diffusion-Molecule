"""Optional PyTorch conditional latent denoising backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TorchDenoiserConfig:
    feature_dim: int = 256
    latent_dim: int = 4096
    hidden_dim: int = 1024
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    diffusion_steps: int = 16
    train_noise: float = 0.35
    direct_loss_weight: float = 1.0
    cosine_loss_weight: float = 1.0
    positive_loss_weight: float = 8.0
    contrastive_loss_weight: float = 0.25
    contrastive_temperature: float = 0.10
    device: str = "auto"
    seed: int = 7


class TorchLatentDenoiser:
    """A GPU-ready conditional denoising model for molecular latents.

    The backend learns p(target_latent | condition, source_latent) by denoising
    noisy target latents. At inference time it starts from noise for de novo rows
    and from the source latent for source-conditioned rows, then iteratively
    denoises to produce a target latent for retrieval/decoding.
    """

    backend_name = "torch_latent_denoiser"

    def __init__(self, config: TorchDenoiserConfig | None = None):
        self.config = config or TorchDenoiserConfig()
        self.model: Any | None = None
        self.device_name = "cpu"
        self.target_mean_: np.ndarray | None = None
        self.history: list[dict[str, float | str]] = []

    def fit(self, conditions: np.ndarray, target_latents: np.ndarray, source_latents: np.ndarray) -> "TorchLatentDenoiser":
        torch, nn, DataLoader, TensorDataset = _torch_deps()
        _set_torch_seed(torch, self.config.seed)
        conditions = np.asarray(conditions, dtype=np.float32)
        target_latents = np.asarray(target_latents, dtype=np.float32)
        source_latents = np.asarray(source_latents, dtype=np.float32)
        self.config.feature_dim = int(conditions.shape[1])
        self.config.latent_dim = int(target_latents.shape[1])
        device = _resolve_device(torch, self.config.device)
        self.device_name = str(device)
        target_mean = torch.mean(torch.from_numpy(target_latents), dim=0, keepdim=True).to(device)
        self.target_mean_ = target_mean.detach().cpu().numpy().astype(np.float32)

        model = _DenoisingMLP(self.config.feature_dim, self.config.latent_dim, self.config.hidden_dim, nn).to(device)
        dataset = TensorDataset(
            torch.from_numpy(conditions),
            torch.from_numpy(target_latents),
            torch.from_numpy(source_latents),
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        model.train()
        history: list[dict[str, float | str]] = []
        for epoch in range(1, self.config.epochs + 1):
            total = 0.0
            recon_total = 0.0
            recon_cosine_total = 0.0
            recon_contrastive_total = 0.0
            direct_total = 0.0
            direct_cosine_total = 0.0
            direct_contrastive_total = 0.0
            seen = 0
            for batch_conditions, batch_targets, batch_sources in loader:
                batch_conditions = batch_conditions.to(device)
                batch_targets = batch_targets.to(device)
                batch_sources = batch_sources.to(device)
                batch_size = int(batch_targets.shape[0])
                source_mask = (torch.linalg.norm(batch_sources, dim=1, keepdim=True) > 1e-8).float()
                t = torch.rand(batch_size, 1, device=device)
                noise = torch.randn_like(batch_targets)
                noisy = (1.0 - self.config.train_noise * t) * batch_targets + (self.config.train_noise * t) * noise

                pred = model(noisy, batch_conditions, batch_sources, source_mask, t)
                recon_mse = _positive_weighted_mse(torch, pred, batch_targets, self.config.positive_loss_weight)
                recon_cosine = _cosine_loss(torch, pred, batch_targets)
                recon_contrastive = _batch_contrastive_loss(torch, pred, batch_targets, self.config.contrastive_temperature)
                recon_loss = recon_mse + self.config.cosine_loss_weight * recon_cosine
                delta_loss = torch.sum(((pred - batch_sources) - (batch_targets - batch_sources)) ** 2 * source_mask) / max(
                    1.0, float(torch.sum(source_mask).detach().cpu()) * self.config.latent_dim
                )
                mean_anchor = target_mean.expand(batch_size, -1)
                direct_input = source_mask * batch_sources + (1.0 - source_mask) * mean_anchor
                direct_t = torch.ones(batch_size, 1, device=device)
                direct_pred = model(direct_input, batch_conditions, batch_sources, source_mask, direct_t)
                direct_mse = _positive_weighted_mse(torch, direct_pred, batch_targets, self.config.positive_loss_weight)
                direct_cosine = _cosine_loss(torch, direct_pred, batch_targets)
                direct_contrastive = _batch_contrastive_loss(torch, direct_pred, batch_targets, self.config.contrastive_temperature)
                direct_loss = direct_mse + self.config.cosine_loss_weight * direct_cosine
                contrastive_loss = 0.5 * recon_contrastive + direct_contrastive
                loss = recon_loss + 0.2 * delta_loss + self.config.direct_loss_weight * direct_loss + self.config.contrastive_loss_weight * contrastive_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                total += float(loss.detach().cpu()) * batch_size
                recon_total += float(recon_mse.detach().cpu()) * batch_size
                recon_cosine_total += float(recon_cosine.detach().cpu()) * batch_size
                recon_contrastive_total += float(recon_contrastive.detach().cpu()) * batch_size
                direct_total += float(direct_mse.detach().cpu()) * batch_size
                direct_cosine_total += float(direct_cosine.detach().cpu()) * batch_size
                direct_contrastive_total += float(direct_contrastive.detach().cpu()) * batch_size
                seen += batch_size
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": total / max(1, seen),
                    "recon_loss": recon_total / max(1, seen),
                    "recon_cosine_loss": recon_cosine_total / max(1, seen),
                    "recon_contrastive_loss": recon_contrastive_total / max(1, seen),
                    "direct_loss": direct_total / max(1, seen),
                    "direct_cosine_loss": direct_cosine_total / max(1, seen),
                    "direct_contrastive_loss": direct_contrastive_total / max(1, seen),
                    "backend": self.backend_name,
                    "device": self.device_name,
                }
            )

        self.model = model
        self.history = history
        return self

    def predict(self, conditions: np.ndarray, source_latents: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.zeros((len(conditions), self.config.latent_dim), dtype=np.float32)
        torch, _, _, _ = _torch_deps()
        _set_torch_seed(torch, self.config.seed)
        device = torch.device(self.device_name)
        conditions_tensor = torch.from_numpy(np.asarray(conditions, dtype=np.float32)).to(device)
        sources_tensor = torch.from_numpy(np.asarray(source_latents, dtype=np.float32)).to(device)
        source_mask = (torch.linalg.norm(sources_tensor, dim=1, keepdim=True) > 1e-8).float()
        model = self.model.to(device)
        model.eval()
        with torch.no_grad():
            target_mean = self._target_mean_tensor(torch, device, len(conditions_tensor))
            x = source_mask * sources_tensor + (1.0 - source_mask) * target_mean
            steps = max(1, int(self.config.diffusion_steps))
            for step in range(steps, 0, -1):
                t = torch.full((len(conditions_tensor), 1), step / steps, device=device)
                pred = model(x, conditions_tensor, sources_tensor, source_mask, t)
                blend = 1.0 / float(step)
                x = (1.0 - blend) * x + blend * pred
            return x.detach().cpu().numpy().astype(np.float32)

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(asdict(self.config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (out_dir / "metadata.json").write_text(
            json.dumps({"model_type": self.backend_name, "device": self.device_name, "history": self.history}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if self.model is not None:
            torch, _, _, _ = _torch_deps()
            torch.save(self.model.state_dict(), out_dir / "model.pt")
        if self.target_mean_ is not None:
            np.save(out_dir / "target_mean.npy", self.target_mean_)

    def _target_mean_tensor(self, torch: Any, device: Any, rows: int):
        if self.target_mean_ is None:
            mean = torch.zeros((1, self.config.latent_dim), device=device)
        else:
            mean = torch.from_numpy(self.target_mean_).to(device)
        return mean.expand(rows, -1)


def _DenoisingMLP(feature_dim: int, latent_dim: int, hidden_dim: int, nn: Any):
    input_dim = latent_dim + feature_dim + latent_dim + 1 + 1

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, latent_dim),
            )

        def forward(self, noisy, conditions, sources, source_mask, t):
            return self.net(torch_cat([noisy, conditions, sources, source_mask, t], dim=1))

    torch_cat = _torch_deps()[0].cat
    return Model()


def _resolve_device(torch: Any, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Torch backend requested CUDA, but torch.cuda.is_available() is false.")
    return device


def _set_torch_seed(torch: Any, seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _positive_weighted_mse(torch: Any, pred: Any, target: Any, positive_weight: float):
    weights = 1.0 + float(positive_weight) * (torch.abs(target) > 1e-8).float()
    return torch.sum(weights * (pred - target) ** 2) / torch.clamp(torch.sum(weights), min=1.0)


def _cosine_loss(torch: Any, pred: Any, target: Any):
    pred_norm = torch.nn.functional.normalize(pred, p=2, dim=1, eps=1e-8)
    target_norm = torch.nn.functional.normalize(target, p=2, dim=1, eps=1e-8)
    return 1.0 - torch.mean(torch.sum(pred_norm * target_norm, dim=1))


def _batch_contrastive_loss(torch: Any, pred: Any, target: Any, temperature: float):
    pred_norm = torch.nn.functional.normalize(pred, p=2, dim=1, eps=1e-8)
    target_norm = torch.nn.functional.normalize(target, p=2, dim=1, eps=1e-8)
    logits = (pred_norm @ target_norm.T) / max(float(temperature), 1e-6)
    target_self_similarity = target_norm @ target_norm.T
    positive_mask = target_self_similarity > 0.999
    positive_mask = positive_mask | torch.eye(target_norm.shape[0], dtype=torch.bool, device=target_norm.device)
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    masked_log_probs = log_probs.masked_fill(~positive_mask, -1e9)
    return -torch.mean(torch.logsumexp(masked_log_probs, dim=1))


def _torch_deps():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:  # pragma: no cover - depends on optional install.
        raise RuntimeError(
            "The torch backend requires PyTorch. Install/load a CUDA-enabled torch environment, "
            "then run with SKETCHIMAGE_BACKEND=torch_denoiser."
        ) from exc
    return torch, nn, DataLoader, TensorDataset
