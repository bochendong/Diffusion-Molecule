"""Lightweight molecular latent diffusion scaffold.

This module is deliberately small. It gives the new project a real training and
sampling path for smoke tests while leaving room to replace the nearest-latent
decoder with a graph or SELFIES decoder.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .artifacts import ensure_dir, load_json, save_json
from .features import molecule_feature_vector

try:  # pragma: no cover - server path.
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False


@dataclass
class DiffusionConfig:
    latent_dim: int = 64
    condition_dim: int = 256
    timesteps: int = 100
    epochs: int = 20
    batch_size: int = 256
    hidden_dim: int = 512
    layers: int = 4
    lr: float = 3e-4
    seed: int = 7


class NearestLatentCodec:
    """Encode molecules to fixed latents and decode by nearest training latent.

    This is only a baseline decoder. It is useful for verifying the pipeline
    before a learned molecular decoder is added.
    """

    def __init__(self, latent_dim: int = 64):
        self.latent_dim = latent_dim
        self.train_latents: np.ndarray | None = None
        self.train_smiles: list[str] = []

    def encode(self, smiles: str | None) -> np.ndarray:
        return molecule_feature_vector(smiles, self.latent_dim)

    def fit(self, smiles: list[str]) -> "NearestLatentCodec":
        self.train_smiles = list(smiles)
        self.train_latents = np.asarray([self.encode(smi) for smi in smiles], dtype=np.float32)
        return self

    def decode(self, latent: np.ndarray, top_k: int = 8) -> list[str]:
        if self.train_latents is None or len(self.train_smiles) == 0:
            return []
        distances = np.mean(np.square(self.train_latents - latent[None, :]), axis=1)
        k = min(top_k, len(self.train_smiles))
        chosen = np.argpartition(distances, k - 1)[:k]
        chosen = chosen[np.argsort(distances[chosen])]
        return [self.train_smiles[int(idx)] for idx in chosen]


if TORCH_AVAILABLE:  # pragma: no cover - exercised on GPU server.

    class _Denoiser(nn.Module):
        def __init__(self, latent_dim: int, condition_dim: int, hidden_dim: int, layers: int):
            super().__init__()
            dims = [latent_dim + condition_dim + 1] + [hidden_dim] * layers + [latent_dim]
            blocks = []
            for left, right in zip(dims[:-2], dims[1:-1]):
                blocks.extend([nn.Linear(left, right), nn.LayerNorm(right), nn.SiLU()])
            blocks.append(nn.Linear(dims[-2], dims[-1]))
            self.net = nn.Sequential(*blocks)

        def forward(self, noisy_latent, condition, t):
            if t.ndim == 1:
                t = t[:, None]
            return self.net(torch.cat([noisy_latent, condition, t], dim=-1))


class MolecularLatentDiffusion:
    def __init__(self, config: DiffusionConfig | None = None):
        self.config = config or DiffusionConfig()
        self.codec = NearestLatentCodec(self.config.latent_dim)
        self.history: list[dict[str, float]] = []
        self.model = None

    def fit(self, target_smiles: list[str], conditions: np.ndarray) -> "MolecularLatentDiffusion":
        self.codec.fit(target_smiles)
        target_latents = np.asarray([self.codec.encode(smi) for smi in target_smiles], dtype=np.float32)
        return self.fit_latents(target_latents, conditions, train_smiles=target_smiles, codec=self.codec)

    def fit_latents(
        self,
        target_latents: np.ndarray,
        conditions: np.ndarray,
        train_smiles: list[str] | None = None,
        codec=None,
    ) -> "MolecularLatentDiffusion":
        target_latents = np.asarray(target_latents, dtype=np.float32)
        conditions = np.asarray(conditions, dtype=np.float32)
        self.config.latent_dim = int(target_latents.shape[1])
        self.config.condition_dim = int(conditions.shape[1])
        if codec is not None:
            self.codec = codec
        elif train_smiles is not None:
            self.codec.fit(train_smiles)
        if not TORCH_AVAILABLE:
            self.history = [{"epoch": 0.0, "loss": 0.0, "backend": "nearest_only"}]
            return self
        cfg = self.config
        torch.manual_seed(cfg.seed)
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _Denoiser(cfg.latent_dim, cfg.condition_dim, cfg.hidden_dim, cfg.layers).to(self.device_)
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
        dataset = TensorDataset(torch.tensor(target_latents, dtype=torch.float32), torch.tensor(conditions.astype(np.float32)))
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
        for epoch in range(cfg.epochs):
            losses = []
            for latent, cond in loader:
                latent = latent.to(self.device_)
                cond = cond.to(self.device_)
                t_int = torch.randint(1, cfg.timesteps + 1, (latent.shape[0],), dtype=torch.float32)
                t = (t_int / float(cfg.timesteps)).to(self.device_)
                alpha = torch.cos((t + 0.008) / 1.008 * np.pi / 2).pow(2)[:, None]
                noise = torch.randn_like(latent)
                noisy = alpha.sqrt() * latent + (1.0 - alpha).sqrt() * noise
                pred = self.model(noisy, cond, t)
                loss = torch.mean((pred - noise) ** 2)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
            self.history.append({"epoch": float(epoch + 1), "loss": float(np.mean(losses)), "backend": "torch"})
        return self

    def sample_latents(self, conditions: np.ndarray, n_per_condition: int = 8, guidance_noise: float = 0.20) -> np.ndarray:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed + 101)
        out = []
        if self.model is None or not TORCH_AVAILABLE:
            if self.codec.train_latents is None:
                return np.zeros((0, cfg.latent_dim), dtype=np.float32)
            for _ in range(len(conditions) * n_per_condition):
                idx = int(rng.integers(0, len(self.codec.train_latents)))
                out.append(self.codec.train_latents[idx])
            return np.asarray(out, dtype=np.float32)
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            for condition in conditions:
                cond = torch.tensor(condition[None, :].astype(np.float32), device=device)
                for _ in range(n_per_condition):
                    z = torch.randn(1, cfg.latent_dim, device=device)
                    for step in range(cfg.timesteps, 0, -1):
                        t = torch.tensor([step / float(cfg.timesteps)], dtype=torch.float32, device=device)
                        alpha = torch.cos((t + 0.008) / 1.008 * np.pi / 2).pow(2)[:, None]
                        eps = self.model(z, cond, t)
                        z0 = (z - (1.0 - alpha).sqrt() * eps) / alpha.sqrt().clamp_min(1e-6)
                        if step > 1:
                            next_t = torch.tensor([(step - 1) / float(cfg.timesteps)], dtype=torch.float32, device=device)
                            next_alpha = torch.cos((next_t + 0.008) / 1.008 * np.pi / 2).pow(2)[:, None]
                            z = next_alpha.sqrt() * z0 + guidance_noise * (1.0 - next_alpha).sqrt() * torch.randn_like(z)
                        else:
                            z = z0
                    out.append(z.cpu().numpy()[0])
        return np.asarray(out, dtype=np.float32)

    def sample_smiles(self, conditions: np.ndarray, n_per_condition: int = 8, top_k: int = 1) -> list[list[str]]:
        latents = self.sample_latents(conditions, n_per_condition=n_per_condition)
        results = []
        idx = 0
        for _ in range(len(conditions)):
            rows = []
            for _ in range(n_per_condition):
                rows.extend(self.codec.decode(latents[idx], top_k=top_k))
                idx += 1
            results.append(_dedupe(rows))
        return results

    def save_history(self, path: str | Path) -> None:
        import json

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.history, indent=2), encoding="utf-8")

    def save(self, out_dir: str | Path) -> None:
        out_dir = ensure_dir(out_dir)
        save_json(asdict(self.config), out_dir / "config.json")
        save_json({"history": self.history, "backend": self.backend}, out_dir / "metadata.json")
        if TORCH_AVAILABLE and self.model is not None:
            torch.save(self.model.state_dict(), out_dir / "diffusion.pt")

    @classmethod
    def load(cls, out_dir: str | Path, codec=None) -> "MolecularLatentDiffusion":
        out_dir = Path(out_dir)
        cfg = DiffusionConfig(**load_json(out_dir / "config.json"))
        obj = cls(cfg)
        if codec is not None:
            obj.codec = codec
        if (out_dir / "metadata.json").exists():
            obj.history = list(load_json(out_dir / "metadata.json").get("history", []))
        if TORCH_AVAILABLE and (out_dir / "diffusion.pt").exists():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            obj.model = _Denoiser(cfg.latent_dim, cfg.condition_dim, cfg.hidden_dim, cfg.layers)
            obj.model.load_state_dict(torch.load(out_dir / "diffusion.pt", map_location="cpu"))
            obj.model.to(device)
            obj.model.eval()
        return obj

    @property
    def backend(self) -> str:
        if TORCH_AVAILABLE and self.model is not None:
            return "torch_latent_diffusion"
        return "nearest_only"


def _dedupe(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out
