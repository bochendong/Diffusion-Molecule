"""Stage 2 multimodal understanding alignment.

UniVideo separates an MLLM understanding stream from the diffusion model. This
module mirrors that separation for molecules: the deterministic understanding
features are trained to predict target molecular latents before diffusion sees
them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .artifacts import ensure_dir, load_json, save_json

try:  # pragma: no cover - server path.
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    F = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False


@dataclass
class AlignmentConfig:
    input_dim: int = 256
    latent_dim: int = 64
    hidden_dim: int = 512
    layers: int = 3
    epochs: int = 20
    batch_size: int = 512
    lr: float = 3e-4
    contrastive_weight: float = 0.05
    ridge: float = 1e-3
    seed: int = 7


if TORCH_AVAILABLE:  # pragma: no cover - server path.

    class _AlignmentMLP(nn.Module):
        def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int, layers: int):
            super().__init__()
            dims = [input_dim] + [hidden_dim] * layers + [latent_dim]
            blocks = []
            for left, right in zip(dims[:-2], dims[1:-1]):
                blocks.extend([nn.Linear(left, right), nn.LayerNorm(right), nn.SiLU()])
            blocks.append(nn.Linear(dims[-2], dims[-1]))
            self.net = nn.Sequential(*blocks)

        def forward(self, x):
            return self.net(x)


class UnderstandingAlignment:
    def __init__(self, config: AlignmentConfig | None = None):
        self.config = config or AlignmentConfig()
        self.model = None
        self.weights_: np.ndarray | None = None
        self.history: list[dict[str, float | str]] = []

    def fit(self, conditions: np.ndarray, target_latents: np.ndarray) -> "UnderstandingAlignment":
        conditions = np.asarray(conditions, dtype=np.float32)
        target_latents = np.asarray(target_latents, dtype=np.float32)
        self.config.input_dim = int(conditions.shape[1])
        self.config.latent_dim = int(target_latents.shape[1])
        if TORCH_AVAILABLE:
            self._fit_torch(conditions, target_latents)
        else:
            self._fit_ridge(conditions, target_latents)
        return self

    def predict(self, conditions: np.ndarray) -> np.ndarray:
        conditions = np.asarray(conditions, dtype=np.float32)
        if TORCH_AVAILABLE and self.model is not None:
            self.model.eval()
            chunks = []
            with torch.no_grad():
                device = next(self.model.parameters()).device
                for start in range(0, len(conditions), max(1, self.config.batch_size)):
                    batch = torch.tensor(conditions[start : start + self.config.batch_size], dtype=torch.float32, device=device)
                    chunks.append(self.model(batch).cpu().numpy())
            return np.concatenate(chunks, axis=0).astype(np.float32)
        if self.weights_ is not None:
            x = np.concatenate([conditions, np.ones((len(conditions), 1), dtype=np.float32)], axis=1)
            return (x @ self.weights_).astype(np.float32)
        return np.zeros((len(conditions), self.config.latent_dim), dtype=np.float32)

    def save(self, out_dir: str | Path) -> None:
        out_dir = ensure_dir(out_dir)
        save_json(asdict(self.config), out_dir / "config.json")
        save_json({"history": self.history, "backend": self.backend}, out_dir / "metadata.json")
        if self.weights_ is not None:
            np.save(out_dir / "ridge_weights.npy", self.weights_)
        if TORCH_AVAILABLE and self.model is not None:
            torch.save(self.model.state_dict(), out_dir / "alignment.pt")

    @classmethod
    def load(cls, out_dir: str | Path) -> "UnderstandingAlignment":
        out_dir = Path(out_dir)
        cfg = AlignmentConfig(**load_json(out_dir / "config.json"))
        obj = cls(cfg)
        if (out_dir / "ridge_weights.npy").exists():
            obj.weights_ = np.load(out_dir / "ridge_weights.npy")
        if TORCH_AVAILABLE and (out_dir / "alignment.pt").exists():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            obj.model = _AlignmentMLP(cfg.input_dim, cfg.latent_dim, cfg.hidden_dim, cfg.layers)
            obj.model.load_state_dict(torch.load(out_dir / "alignment.pt", map_location="cpu"))
            obj.model.to(device)
            obj.model.eval()
        if (out_dir / "metadata.json").exists():
            obj.history = list(load_json(out_dir / "metadata.json").get("history", []))
        return obj

    @property
    def backend(self) -> str:
        if TORCH_AVAILABLE and self.model is not None:
            return "torch_alignment_mlp"
        if self.weights_ is not None:
            return "numpy_ridge_alignment"
        return "untrained_alignment"

    def _fit_torch(self, conditions: np.ndarray, target_latents: np.ndarray) -> None:
        cfg = self.config
        torch.manual_seed(cfg.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _AlignmentMLP(cfg.input_dim, cfg.latent_dim, cfg.hidden_dim, cfg.layers).to(device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=1e-4)
        loader = DataLoader(
            TensorDataset(torch.tensor(conditions, dtype=torch.float32), torch.tensor(target_latents, dtype=torch.float32)),
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        self.history = []
        for epoch in range(cfg.epochs):
            losses = []
            mse_losses = []
            contrastive_losses = []
            for cond, target in loader:
                cond = cond.to(device)
                target = target.to(device)
                pred = self.model(cond)
                mse = F.mse_loss(pred, target)
                contrastive = _info_nce(pred, target) if len(pred) > 1 else pred.new_tensor(0.0)
                loss = mse + cfg.contrastive_weight * contrastive
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
                mse_losses.append(float(mse.detach()))
                contrastive_losses.append(float(contrastive.detach()))
            self.history.append(
                {
                    "epoch": float(epoch + 1),
                    "loss": float(np.mean(losses)),
                    "mse": float(np.mean(mse_losses)),
                    "contrastive": float(np.mean(contrastive_losses)),
                    "backend": "torch",
                }
            )

    def _fit_ridge(self, conditions: np.ndarray, target_latents: np.ndarray) -> None:
        x = np.concatenate([conditions, np.ones((len(conditions), 1), dtype=np.float32)], axis=1)
        reg = self.config.ridge * np.eye(x.shape[1], dtype=np.float32)
        reg[-1, -1] = 0.0
        self.weights_ = np.linalg.solve(x.T @ x + reg, x.T @ target_latents).astype(np.float32)
        pred = x @ self.weights_
        loss = float(np.mean((pred - target_latents) ** 2))
        self.history = [{"epoch": 0.0, "loss": loss, "mse": loss, "contrastive": 0.0, "backend": "numpy_ridge"}]


def _info_nce(pred, target):
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = pred @ target.T / 0.07
    labels = torch.arange(pred.shape[0], device=pred.device)
    return F.cross_entropy(logits, labels)
