"""JEPA-style molecular latent predictor.

MolPilot uses this as the research-facing Stage 2 model. Inspired by I-JEPA and
V-JEPA2, the model does not reconstruct raw SMILES or pixels. It predicts the
target molecule embedding from a context embedding made of the source molecule,
the grounded instruction, and optional multimodal features.
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
class JEPAConfig:
    input_dim: int = 256
    latent_dim: int = 64
    hidden_dim: int = 512
    layers: int = 3
    epochs: int = 20
    batch_size: int = 512
    lr: float = 3e-4
    contrastive_weight: float = 0.05
    delta_weight: float = 0.25
    sigreg_weight: float = 0.01
    ridge: float = 1e-3
    seed: int = 7


if TORCH_AVAILABLE:  # pragma: no cover - exercised on GPU server.

    class _MolecularJEPA(nn.Module):
        def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int, layers: int):
            super().__init__()
            dims = [input_dim + latent_dim + 1] + [hidden_dim] * layers
            blocks = []
            for left, right in zip(dims[:-1], dims[1:]):
                blocks.extend([nn.Linear(left, right), nn.LayerNorm(right), nn.SiLU()])
            self.context_encoder = nn.Sequential(*blocks)
            self.delta_head = nn.Linear(hidden_dim, latent_dim)
            self.direct_head = nn.Linear(hidden_dim, latent_dim)

        def forward(self, condition, source_latent, source_mask):
            h = self.context_encoder(torch.cat([condition, source_latent, source_mask], dim=-1))
            delta = self.delta_head(h)
            direct = self.direct_head(h)
            pred = source_mask * (source_latent + delta) + (1.0 - source_mask) * direct
            return pred, delta


class MolecularJEPAPredictor:
    expects_source_latents = True

    def __init__(self, config: JEPAConfig | None = None):
        self.config = config or JEPAConfig()
        self.model = None
        self.weights_: np.ndarray | None = None
        self.history: list[dict[str, float | str]] = []

    def fit(self, conditions: np.ndarray, target_latents: np.ndarray, source_latents: np.ndarray) -> "MolecularJEPAPredictor":
        conditions = np.asarray(conditions, dtype=np.float32)
        target_latents = np.asarray(target_latents, dtype=np.float32)
        source_latents = np.asarray(source_latents, dtype=np.float32)
        self.config.input_dim = int(conditions.shape[1])
        self.config.latent_dim = int(target_latents.shape[1])
        if TORCH_AVAILABLE:
            self._fit_torch(conditions, target_latents, source_latents)
        else:
            self._fit_ridge(conditions, target_latents, source_latents)
        return self

    def predict(self, conditions: np.ndarray, source_latents: np.ndarray) -> np.ndarray:
        conditions = np.asarray(conditions, dtype=np.float32)
        source_latents = np.asarray(source_latents, dtype=np.float32)
        source_mask = _source_mask(source_latents)
        if TORCH_AVAILABLE and self.model is not None:
            self.model.eval()
            chunks = []
            with torch.no_grad():
                device = next(self.model.parameters()).device
                batch_size = max(1, self.config.batch_size)
                for start in range(0, len(conditions), batch_size):
                    cond = torch.tensor(conditions[start : start + batch_size], dtype=torch.float32, device=device)
                    src = torch.tensor(source_latents[start : start + batch_size], dtype=torch.float32, device=device)
                    mask = torch.tensor(source_mask[start : start + batch_size], dtype=torch.float32, device=device)
                    pred, _ = self.model(cond, src, mask)
                    chunks.append(pred.cpu().numpy())
            return np.concatenate(chunks, axis=0).astype(np.float32)
        if self.weights_ is not None:
            x = _design_matrix(conditions, source_latents, source_mask)
            return (x @ self.weights_).astype(np.float32)
        return np.zeros((len(conditions), self.config.latent_dim), dtype=np.float32)

    def save(self, out_dir: str | Path) -> None:
        out_dir = ensure_dir(out_dir)
        save_json(asdict(self.config), out_dir / "config.json")
        save_json({"model_type": "molecular_jepa", "history": self.history, "backend": self.backend}, out_dir / "metadata.json")
        if self.weights_ is not None:
            np.save(out_dir / "ridge_weights.npy", self.weights_)
        if TORCH_AVAILABLE and self.model is not None:
            torch.save(self.model.state_dict(), out_dir / "jepa.pt")

    @classmethod
    def load(cls, out_dir: str | Path) -> "MolecularJEPAPredictor":
        out_dir = Path(out_dir)
        cfg = JEPAConfig(**load_json(out_dir / "config.json"))
        obj = cls(cfg)
        if (out_dir / "ridge_weights.npy").exists():
            obj.weights_ = np.load(out_dir / "ridge_weights.npy")
        if TORCH_AVAILABLE and (out_dir / "jepa.pt").exists():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            obj.model = _MolecularJEPA(cfg.input_dim, cfg.latent_dim, cfg.hidden_dim, cfg.layers)
            obj.model.load_state_dict(torch.load(out_dir / "jepa.pt", map_location="cpu"))
            obj.model.to(device)
            obj.model.eval()
        if (out_dir / "metadata.json").exists():
            obj.history = list(load_json(out_dir / "metadata.json").get("history", []))
        return obj

    @property
    def backend(self) -> str:
        if TORCH_AVAILABLE and self.model is not None:
            return "torch_molecular_jepa"
        if self.weights_ is not None:
            return "numpy_ridge_molecular_jepa"
        return "untrained_molecular_jepa"

    def _fit_torch(self, conditions: np.ndarray, target_latents: np.ndarray, source_latents: np.ndarray) -> None:
        cfg = self.config
        torch.manual_seed(cfg.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        source_masks = _source_mask(source_latents)
        self.model = _MolecularJEPA(cfg.input_dim, cfg.latent_dim, cfg.hidden_dim, cfg.layers).to(device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=1e-4)
        loader = DataLoader(
            TensorDataset(
                torch.tensor(conditions, dtype=torch.float32),
                torch.tensor(target_latents, dtype=torch.float32),
                torch.tensor(source_latents, dtype=torch.float32),
                torch.tensor(source_masks, dtype=torch.float32),
            ),
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        self.history = []
        for epoch in range(cfg.epochs):
            losses = []
            pred_losses = []
            delta_losses = []
            contrastive_losses = []
            sigreg_losses = []
            for cond, target, source, mask in loader:
                cond = cond.to(device)
                target = target.to(device)
                source = source.to(device)
                mask = mask.to(device)
                pred, delta = self.model(cond, source, mask)
                pred_loss = F.smooth_l1_loss(pred, target)
                source_rows = mask.squeeze(-1) > 0.5
                if source_rows.any():
                    delta_loss = F.smooth_l1_loss(delta[source_rows], (target - source)[source_rows])
                else:
                    delta_loss = pred.new_tensor(0.0)
                contrastive = _info_nce(pred, target) if len(pred) > 1 else pred.new_tensor(0.0)
                sigreg = _sigreg(pred)
                loss = pred_loss + cfg.delta_weight * delta_loss + cfg.contrastive_weight * contrastive + cfg.sigreg_weight * sigreg
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach()))
                pred_losses.append(float(pred_loss.detach()))
                delta_losses.append(float(delta_loss.detach()))
                contrastive_losses.append(float(contrastive.detach()))
                sigreg_losses.append(float(sigreg.detach()))
            self.history.append(
                {
                    "epoch": float(epoch + 1),
                    "loss": float(np.mean(losses)),
                    "prediction": float(np.mean(pred_losses)),
                    "delta": float(np.mean(delta_losses)),
                    "contrastive": float(np.mean(contrastive_losses)),
                    "sigreg": float(np.mean(sigreg_losses)),
                    "backend": "torch",
                }
            )

    def _fit_ridge(self, conditions: np.ndarray, target_latents: np.ndarray, source_latents: np.ndarray) -> None:
        source_masks = _source_mask(source_latents)
        x = _design_matrix(conditions, source_latents, source_masks)
        reg = self.config.ridge * np.eye(x.shape[1], dtype=np.float32)
        reg[-1, -1] = 0.0
        self.weights_ = np.linalg.solve(x.T @ x + reg, x.T @ target_latents).astype(np.float32)
        pred = x @ self.weights_
        loss = float(np.mean((pred - target_latents) ** 2))
        self.history = [{"epoch": 0.0, "loss": loss, "prediction": loss, "delta": 0.0, "contrastive": 0.0, "sigreg": 0.0, "backend": "numpy_ridge"}]


def _source_mask(source_latents: np.ndarray) -> np.ndarray:
    source_latents = np.asarray(source_latents, dtype=np.float32)
    return (np.linalg.norm(source_latents, axis=1, keepdims=True) > 1e-8).astype(np.float32)


def _design_matrix(conditions: np.ndarray, source_latents: np.ndarray, source_masks: np.ndarray) -> np.ndarray:
    return np.concatenate([conditions, source_latents, source_masks, np.ones((len(conditions), 1), dtype=np.float32)], axis=1)


def _info_nce(pred, target):
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = pred @ target.T / 0.07
    labels = torch.arange(pred.shape[0], device=pred.device)
    return F.cross_entropy(logits, labels)


def _sigreg(z):
    z = F.normalize(z, dim=-1)
    mean_penalty = z.mean(dim=0).pow(2).mean()
    std_penalty = (z.std(dim=0) - 1.0 / np.sqrt(max(1, z.shape[-1]))).pow(2).mean()
    return mean_penalty + std_penalty
