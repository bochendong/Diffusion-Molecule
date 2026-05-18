"""Stage 1 molecular latent autoencoder.

SketchMol first learns an image latent space before training diffusion. MolPilot
does the molecular analogue here: train a reusable molecular latent space before
any instruction conditioning is introduced.

The first implementation trains an autoencoder over deterministic molecular
features. It intentionally keeps a nearest-latent decoder as a baseline so the
pipeline can run before a SELFIES or graph decoder is added.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .artifacts import ensure_dir, load_json, load_lines, save_json, save_lines
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
class AutoencoderConfig:
    feature_dim: int = 256
    latent_dim: int = 64
    hidden_dim: int = 512
    layers: int = 3
    epochs: int = 20
    batch_size: int = 512
    lr: float = 1e-3
    seed: int = 7


if TORCH_AVAILABLE:  # pragma: no cover - server path.

    class _FeatureAutoencoder(nn.Module):
        def __init__(self, feature_dim: int, latent_dim: int, hidden_dim: int, layers: int):
            super().__init__()
            enc = []
            dim = feature_dim
            for _ in range(layers):
                enc.extend([nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()])
                dim = hidden_dim
            enc.append(nn.Linear(dim, latent_dim))
            dec = []
            dim = latent_dim
            for _ in range(layers):
                dec.extend([nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()])
                dim = hidden_dim
            dec.append(nn.Linear(dim, feature_dim))
            self.encoder = nn.Sequential(*enc)
            self.decoder = nn.Sequential(*dec)

        def forward(self, x):
            z = self.encoder(x)
            recon = self.decoder(z)
            return recon, z


class MolecularAutoencoder:
    def __init__(self, config: AutoencoderConfig | None = None):
        self.config = config or AutoencoderConfig()
        self.history: list[dict[str, float | str]] = []
        self.model = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.train_smiles: list[str] = []
        self.train_latents: np.ndarray | None = None

    def fit(self, smiles: list[str]) -> "MolecularAutoencoder":
        self.train_smiles = list(smiles)
        features = self._feature_matrix(smiles)
        self.mean_ = features.mean(axis=0)
        self.std_ = features.std(axis=0) + 1e-6
        x = ((features - self.mean_) / self.std_).astype(np.float32)
        if TORCH_AVAILABLE:
            self._fit_torch(x)
            self.train_latents = self.encode_many(smiles)
        else:
            self._fit_svd(x)
            self.train_latents = self.encode_many(smiles)
        return self

    def encode(self, smiles: str | None) -> np.ndarray:
        return self.encode_many([smiles or ""])[0]

    def encode_many(self, smiles: list[str]) -> np.ndarray:
        x = self._feature_matrix(smiles)
        if self.mean_ is not None and self.std_ is not None:
            x = ((x - self.mean_) / self.std_).astype(np.float32)
        if TORCH_AVAILABLE and self.model is not None:
            self.model.eval()
            device = next(self.model.parameters()).device
            chunks = []
            with torch.no_grad():
                for start in range(0, len(x), max(1, self.config.batch_size)):
                    tensor = torch.tensor(x[start : start + self.config.batch_size], dtype=torch.float32, device=device)
                    chunks.append(self.model.encoder(tensor).cpu().numpy())
            return np.concatenate(chunks, axis=0).astype(np.float32)
        if self.components_ is not None:
            z = x @ self.components_.T
            return _pad_or_trim(z, self.config.latent_dim)
        return _pad_or_trim(x, self.config.latent_dim)

    def decode(self, latent: np.ndarray, top_k: int = 8) -> list[str]:
        if self.train_latents is None or not self.train_smiles:
            return []
        latent = np.asarray(latent, dtype=np.float32)
        distances = np.mean(np.square(self.train_latents - latent[None, :]), axis=1)
        k = min(max(1, top_k), len(self.train_smiles))
        chosen = np.argpartition(distances, k - 1)[:k]
        chosen = chosen[np.argsort(distances[chosen])]
        return [self.train_smiles[int(idx)] for idx in chosen]

    def save(self, out_dir: str | Path) -> None:
        out_dir = ensure_dir(out_dir)
        save_json(asdict(self.config), out_dir / "config.json")
        save_json({"history": self.history, "backend": self.backend}, out_dir / "metadata.json")
        save_lines(self.train_smiles, out_dir / "train_smiles.txt")
        if self.train_latents is not None:
            np.save(out_dir / "train_latents.npy", self.train_latents)
        if self.mean_ is not None:
            np.save(out_dir / "feature_mean.npy", self.mean_)
        if self.std_ is not None:
            np.save(out_dir / "feature_std.npy", self.std_)
        if self.components_ is not None:
            np.save(out_dir / "svd_components.npy", self.components_)
        if TORCH_AVAILABLE and self.model is not None:
            torch.save(self.model.state_dict(), out_dir / "autoencoder.pt")

    @classmethod
    def load(cls, out_dir: str | Path) -> "MolecularAutoencoder":
        out_dir = Path(out_dir)
        cfg = AutoencoderConfig(**load_json(out_dir / "config.json"))
        obj = cls(cfg)
        obj.train_smiles = load_lines(out_dir / "train_smiles.txt") if (out_dir / "train_smiles.txt").exists() else []
        obj.train_latents = np.load(out_dir / "train_latents.npy") if (out_dir / "train_latents.npy").exists() else None
        obj.mean_ = np.load(out_dir / "feature_mean.npy") if (out_dir / "feature_mean.npy").exists() else None
        obj.std_ = np.load(out_dir / "feature_std.npy") if (out_dir / "feature_std.npy").exists() else None
        obj.components_ = np.load(out_dir / "svd_components.npy") if (out_dir / "svd_components.npy").exists() else None
        if TORCH_AVAILABLE and (out_dir / "autoencoder.pt").exists():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            obj.model = _FeatureAutoencoder(cfg.feature_dim, cfg.latent_dim, cfg.hidden_dim, cfg.layers)
            obj.model.load_state_dict(torch.load(out_dir / "autoencoder.pt", map_location="cpu"))
            obj.model.to(device)
            obj.model.eval()
        if (out_dir / "metadata.json").exists():
            obj.history = list(load_json(out_dir / "metadata.json").get("history", []))
        return obj

    @property
    def backend(self) -> str:
        if TORCH_AVAILABLE and self.model is not None:
            return "torch_feature_autoencoder"
        if self.components_ is not None:
            return "numpy_svd_autoencoder"
        return "deterministic_feature_codec"

    def _feature_matrix(self, smiles: list[str]) -> np.ndarray:
        return np.asarray([molecule_feature_vector(smi, self.config.feature_dim) for smi in smiles], dtype=np.float32)

    def _fit_torch(self, x: np.ndarray) -> None:
        cfg = self.config
        torch.manual_seed(cfg.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _FeatureAutoencoder(cfg.feature_dim, cfg.latent_dim, cfg.hidden_dim, cfg.layers).to(device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=1e-4)
        loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32)), batch_size=cfg.batch_size, shuffle=True)
        mse = nn.MSELoss()
        self.history = []
        for epoch in range(cfg.epochs):
            losses = []
            for (xb,) in loader:
                xb = xb.to(device)
                recon, z = self.model(xb)
                loss = mse(recon, xb) + 1e-4 * torch.mean(z.pow(2))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
            self.history.append({"epoch": float(epoch + 1), "loss": float(np.mean(losses)), "backend": "torch"})

    def _fit_svd(self, x: np.ndarray) -> None:
        if len(x) <= 1:
            self.components_ = np.eye(self.config.latent_dim, self.config.feature_dim, dtype=np.float32)
            self.history = [{"epoch": 0.0, "loss": 0.0, "backend": "numpy_svd"}]
            return
        _, singular_values, vt = np.linalg.svd(x, full_matrices=False)
        keep = min(self.config.latent_dim, vt.shape[0])
        self.components_ = np.zeros((self.config.latent_dim, self.config.feature_dim), dtype=np.float32)
        self.components_[:keep] = vt[:keep].astype(np.float32)
        explained = float(np.sum(singular_values[:keep] ** 2) / max(1e-8, np.sum(singular_values**2)))
        self.history = [{"epoch": 0.0, "loss": float(1.0 - explained), "backend": "numpy_svd", "explained_variance": explained}]


def _pad_or_trim(x: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape[1] == dim:
        return x
    if x.shape[1] > dim:
        return x[:, :dim]
    out = np.zeros((x.shape[0], dim), dtype=np.float32)
    out[:, : x.shape[1]] = x
    return out


def load_autoencoder(out_dir: str | Path):
    """Load either the feature baseline codec or the sequence molecular codec."""

    out_dir = Path(out_dir)
    metadata_path = out_dir / "metadata.json"
    if metadata_path.exists():
        metadata = load_json(metadata_path)
        if metadata.get("codec_type") == "sequence":
            from .sequence_autoencoder import MolecularSequenceAutoencoder

            return MolecularSequenceAutoencoder.load(out_dir)
    return MolecularAutoencoder.load(out_dir)
