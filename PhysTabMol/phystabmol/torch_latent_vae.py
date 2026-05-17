"""PyTorch molecular/table VAE + latent diffusion backend.

UniVideo and SketchMol both denoise in a learned latent space instead of raw
pixels. This backend mirrors that design for PhysTabMol: a VAE compresses the
molecular design table into a compact latent, and the conditional diffusion
model generates that latent from source/reference/instruction context.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import os
from pathlib import Path

import numpy as np

NearestNeighbors = None
StandardScaler = None
SKLEARN_AVAILABLE = False

from .diffusion import clip_table_row
from .progress import iter_progress
from .schema import BOUNDS, INTEGER_COLUMNS, TABLE_COLUMNS


TARGET_SCALES = np.asarray([120.0, 3.0, 0.35, 60.0, 3.0, 5.0, 5.0, 2.5], dtype=np.float32)
TABLE_SCALES = np.asarray(
    [
        120.0,
        3.0,
        0.35,
        60.0,
        3.0,
        5.0,
        5.0,
        2.5,
        8.0,
        3.0,
        3.0,
        2.0,
        3.0,
        2.0,
        2.0,
        1.0,
        3.0,
        4.0,
        3.0,
        3.0,
        3.0,
        4.0,
        3.0,
    ],
    dtype=np.float32,
)
STRUCTURAL_ANCHOR_COLUMNS = ("ring_count", "scaffold_class")
ATOM_ANCHOR_COLUMNS = ("C", "N", "O", "S")
HALOGEN_ANCHOR_COLUMNS = ("F", "Cl", "Br", "I")

try:  # pragma: no cover - exercised on GPU servers.
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, TensorDataset

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    Dataset = None
    TensorDataset = None
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:  # pragma: no cover - server path.

    class _TableVAE(nn.Module):
        def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int, layers: int, dropout: float):
            super().__init__()
            enc_blocks = []
            dim = input_dim
            for _ in range(layers):
                enc_blocks.extend([nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout)])
                dim = hidden_dim
            self.encoder = nn.Sequential(*enc_blocks)
            self.mu = nn.Linear(dim, latent_dim)
            self.logvar = nn.Linear(dim, latent_dim)

            dec_blocks = []
            dim = latent_dim
            for _ in range(layers):
                dec_blocks.extend([nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout)])
                dim = hidden_dim
            dec_blocks.append(nn.Linear(dim, input_dim))
            self.decoder = nn.Sequential(*dec_blocks)

        def encode(self, x):
            h = self.encoder(x)
            return self.mu(h), self.logvar(h).clamp(min=-8.0, max=8.0)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std

        def decode(self, z):
            return self.decoder(z)

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar


    class _Denoiser(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, layers: int, dropout: float):
            super().__init__()
            blocks = []
            dim = in_dim
            for _ in range(layers):
                blocks.extend([nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout)])
                dim = hidden_dim
            blocks.append(nn.Linear(dim, out_dim))
            self.net = nn.Sequential(*blocks)

        def forward(self, x):
            return self.net(x)


    class _LatentNoisyDataset(Dataset):
        def __init__(self, z: np.ndarray, c: np.ndarray, timesteps: int, noise_repeats: int):
            self.z = torch.tensor(z, dtype=torch.float32)
            self.c = torch.tensor(c, dtype=torch.float32)
            self.timesteps = int(timesteps)
            self.noise_repeats = int(noise_repeats)
            steps = np.arange(self.timesteps + 1, dtype=np.float32) / self.timesteps
            self.alpha_bars = torch.tensor(np.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2, dtype=torch.float32)

        def __len__(self) -> int:
            return int(len(self.z) * self.noise_repeats)

        def __getitem__(self, idx: int):
            row_idx = idx // self.noise_repeats
            t = int(torch.randint(1, self.timesteps + 1, (1,)).item())
            alpha_bar = self.alpha_bars[t]
            eps = torch.randn_like(self.z[row_idx])
            noisy = torch.sqrt(alpha_bar) * self.z[row_idx] + torch.sqrt(1.0 - alpha_bar) * eps
            x = torch.cat([noisy, self.c[row_idx], torch.tensor([t / self.timesteps], dtype=torch.float32)])
            return x, eps


@dataclass
class TorchLatentVAEDiffusion:
    timesteps: int = 100
    noise_repeats: int = 12
    hidden_dim: int = 1024
    layers: int = 6
    dropout: float = 0.05
    epochs: int = 80
    batch_size: int = 256
    lr: float = 2e-4
    seed: int = 11
    device: str = "auto"
    target_condition_start: int = 12
    target_anchor: float = 1.0
    anchor_neighbors: int = 128
    count_anchor_weight: float = 0.8
    vae_latent_dim: int = 16
    vae_hidden_dim: int = 512
    vae_layers: int = 3
    vae_dropout: float = 0.05
    vae_epochs: int = 60
    vae_batch_size: int = 1024
    vae_lr: float = 1e-3
    vae_beta: float = 1e-3
    source_anchor_weight: float = 0.35
    source_count_anchor_weight: float = 0.15
    source_anchor_neighbors: int = 32

    def fit(self, table_y: np.ndarray, condition_x: np.ndarray) -> "TorchLatentVAEDiffusion":
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed. Latent VAE diffusion requires --backend torch.")
        if not _ensure_sklearn():
            raise RuntimeError("scikit-learn is required for latent VAE diffusion scaling and anchor retrieval.")

        self.y_scaler = StandardScaler().fit(table_y)
        self.c_scaler = StandardScaler().fit(condition_x)
        self.train_table_y_ = np.asarray(table_y, dtype=np.float32)
        self.train_targets_ = self.train_table_y_[:, : len(TARGET_SCALES)]
        self.anchor_nn_ = NearestNeighbors(n_neighbors=min(self.anchor_neighbors, len(self.train_targets_)))
        self.anchor_nn_.fit(self.train_targets_ / TARGET_SCALES)
        y = self.y_scaler.transform(table_y).astype(np.float32)
        c = self.c_scaler.transform(condition_x).astype(np.float32)
        self.device_ = self._resolve_device()
        torch.manual_seed(self.seed)
        self.vae = _TableVAE(
            input_dim=len(TABLE_COLUMNS),
            latent_dim=self.vae_latent_dim,
            hidden_dim=self.vae_hidden_dim,
            layers=self.vae_layers,
            dropout=self.vae_dropout,
        ).to(self.device_)
        self.vae_history_ = self._fit_vae(y)
        z = self._encode_table_latents(y)
        self.z_scaler = StandardScaler().fit(z)
        z_scaled = self.z_scaler.transform(z).astype(np.float32)
        self.diffusion_history_ = self._fit_denoiser(z_scaled, c)
        self.history_ = {
            "vae": self.vae_history_,
            "diffusion": self.diffusion_history_,
        }
        return self

    def _fit_vae(self, y: np.ndarray) -> list[dict[str, float]]:
        dataset = TensorDataset(torch.tensor(y, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.vae_batch_size, shuffle=True, drop_last=False)
        opt = torch.optim.AdamW(self.vae.parameters(), lr=self.vae_lr, weight_decay=1e-4)
        mse = nn.MSELoss(reduction="mean")
        history = []
        self.vae.train()
        for epoch in iter_progress(range(self.vae_epochs), total=self.vae_epochs, label="training VAE epochs"):
            losses = []
            recon_losses = []
            kl_losses = []
            for (yb,) in loader:
                yb = yb.to(self.device_)
                opt.zero_grad(set_to_none=True)
                recon, mu, logvar = self.vae(yb)
                recon_loss = mse(recon, yb)
                kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + self.vae_beta * kl
                loss.backward()
                nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach().cpu()))
                recon_losses.append(float(recon_loss.detach().cpu()))
                kl_losses.append(float(kl.detach().cpu()))
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "loss": float(np.mean(losses)),
                    "recon_loss": float(np.mean(recon_losses)),
                    "kl_loss": float(np.mean(kl_losses)),
                }
            )
        return history

    @torch.no_grad() if TORCH_AVAILABLE else (lambda fn: fn)
    def _encode_table_latents(self, y: np.ndarray) -> np.ndarray:
        self.vae.eval()
        latents = []
        loader = DataLoader(TensorDataset(torch.tensor(y, dtype=torch.float32)), batch_size=self.vae_batch_size)
        for (yb,) in loader:
            yb = yb.to(self.device_)
            mu, _ = self.vae.encode(yb)
            latents.append(mu.detach().cpu().numpy())
        return np.concatenate(latents, axis=0).astype(np.float32)

    def _fit_denoiser(self, z: np.ndarray, c: np.ndarray) -> list[dict[str, float]]:
        dataset = _LatentNoisyDataset(z, c, timesteps=self.timesteps, noise_repeats=self.noise_repeats)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.model = _Denoiser(
            in_dim=self.vae_latent_dim + c.shape[1] + 1,
            out_dim=self.vae_latent_dim,
            hidden_dim=self.hidden_dim,
            layers=self.layers,
            dropout=self.dropout,
        ).to(self.device_)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        history = []
        self.model.train()
        for epoch in iter_progress(range(self.epochs), total=self.epochs, label="training latent diffusion epochs"):
            losses = []
            for xb, eps in loader:
                xb = xb.to(self.device_)
                eps = eps.to(self.device_)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(self.model(xb), eps)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach().cpu()))
            history.append({"epoch": float(epoch + 1), "loss": float(np.mean(losses))})
        return history

    @torch.no_grad() if TORCH_AVAILABLE else (lambda fn: fn)
    def sample(self, condition_x: np.ndarray, n: int = 32, guidance_noise: float = 0.25) -> list[dict[str, float]]:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed.")
        if condition_x.ndim == 1:
            condition_x = condition_x[None, :]
        rng = np.random.default_rng(self.seed + 101)
        cond_scaled = self.c_scaler.transform(condition_x).astype(np.float32)
        rows = []
        self.model.eval()
        self.vae.eval()
        row_iter = iter_progress(range(n), total=n, label="sampling latent diffusion rows") if n >= 100 else range(n)
        for idx in row_iter:
            c_np = cond_scaled[idx % len(cond_scaled)]
            z = rng.normal(size=self.vae_latent_dim).astype(np.float32)
            for t in range(self.timesteps, 0, -1):
                alpha_bar = self._alpha_bar(t)
                model_x = np.concatenate([z, c_np, [t / self.timesteps]], dtype=np.float32)
                xb = torch.tensor(model_x[None, :], device=self.device_)
                eps_hat = self.model(xb).detach().cpu().numpy()[0]
                z0 = (z - np.sqrt(1.0 - alpha_bar) * eps_hat) / max(1e-6, np.sqrt(alpha_bar))
                if t > 1:
                    next_alpha = self._alpha_bar(t - 1)
                    noise = rng.normal(size=self.vae_latent_dim).astype(np.float32)
                    z = np.sqrt(next_alpha) * z0 + guidance_noise * np.sqrt(1.0 - next_alpha) * noise
                else:
                    z = z0
            raw = self._decode_latent_row(z)
            source_values = None
            if condition_x.shape[1] >= len(TABLE_COLUMNS):
                source_values = condition_x[idx % len(condition_x), : len(TABLE_COLUMNS)]
                raw = self._preserve_source_structure(raw, source_values)
            if condition_x.shape[1] >= self.target_condition_start + len(TARGET_SCALES):
                target_values = condition_x[
                    idx % len(condition_x),
                    self.target_condition_start : self.target_condition_start + len(TARGET_SCALES),
                ]
                raw[: len(TARGET_SCALES)] = (1.0 - self.target_anchor) * raw[: len(TARGET_SCALES)] + self.target_anchor * target_values
                raw = self._calibrate_counts(raw, target_values, source_values=source_values, sample_idx=idx, rng=rng)
            rows.append(clip_table_row(dict(zip(TABLE_COLUMNS, raw))))
        return rows

    def _decode_latent_row(self, z_scaled: np.ndarray) -> np.ndarray:
        z = self.z_scaler.inverse_transform(z_scaled[None, :]).astype(np.float32)
        tensor = torch.tensor(z, dtype=torch.float32, device=self.device_)
        y_scaled = self.vae.decode(tensor).detach().cpu().numpy()
        return self.y_scaler.inverse_transform(y_scaled)[0]

    def _calibrate_counts(
        self,
        raw: np.ndarray,
        target_values: np.ndarray,
        source_values: np.ndarray | None,
        sample_idx: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if not hasattr(self, "anchor_nn_"):
            return raw
        target_distances, indices = self.anchor_nn_.kneighbors((target_values / TARGET_SCALES)[None, :], return_distance=True)
        choices = indices[0]
        if source_values is not None and len(choices) > 1:
            source_scaled = source_values[len(TARGET_SCALES) :] / TABLE_SCALES[len(TARGET_SCALES) :]
            choice_scaled = self.train_table_y_[choices, len(TARGET_SCALES) :] / TABLE_SCALES[len(TARGET_SCALES) :]
            source_distances = np.mean(np.abs(choice_scaled - source_scaled[None, :]), axis=1)
            combined = target_distances[0] + self.source_anchor_weight * source_distances
            keep = max(1, min(self.source_anchor_neighbors, len(choices)))
            choices = choices[np.argsort(combined)[:keep]]
        anchor = self.train_table_y_[choices[(sample_idx * 9973 + self.seed) % len(choices)]]
        out = np.array(raw, copy=True)
        for col_idx, col in enumerate(TABLE_COLUMNS[len(TARGET_SCALES) :], start=len(TARGET_SCALES)):
            lo, hi = BOUNDS[col]
            value = out[col_idx]
            if value < lo - 0.5 or value > hi + 0.5:
                value = anchor[col_idx]
            else:
                value = self.count_anchor_weight * anchor[col_idx] + (1.0 - self.count_anchor_weight) * value
            if col in INTEGER_COLUMNS and rng.random() < 0.18:
                value += rng.choice([-1.0, 1.0])
            out[col_idx] = value
        return out

    def _preserve_source_structure(self, raw: np.ndarray, source_values: np.ndarray) -> np.ndarray:
        out = np.array(raw, copy=True)
        for col in STRUCTURAL_ANCHOR_COLUMNS:
            idx = TABLE_COLUMNS.index(col)
            out[idx] = self.source_anchor_weight * source_values[idx] + (1.0 - self.source_anchor_weight) * out[idx]
        atom_weight = self.source_count_anchor_weight
        for col in ATOM_ANCHOR_COLUMNS:
            idx = TABLE_COLUMNS.index(col)
            out[idx] = atom_weight * source_values[idx] + (1.0 - atom_weight) * out[idx]
        halogen_weight = 0.5 * atom_weight
        for col in HALOGEN_ANCHOR_COLUMNS:
            idx = TABLE_COLUMNS.index(col)
            out[idx] = halogen_weight * source_values[idx] + (1.0 - halogen_weight) * out[idx]
        return out

    def save(self, path: str | Path) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed.")
        torch.save(
            {
                "config": {field.name: getattr(self, field.name) for field in fields(self)},
                "vae_state_dict": self.vae.state_dict(),
                "denoiser_state_dict": self.model.state_dict(),
                "y_scaler": self.y_scaler,
                "z_scaler": self.z_scaler,
                "c_scaler": self.c_scaler,
                "train_table_y": getattr(self, "train_table_y_", None),
                "history": getattr(self, "history_", {}),
            },
            path,
        )

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _alpha_bar(self, t: int) -> float:
        frac = t / self.timesteps
        return float(np.cos((frac + 0.008) / 1.008 * np.pi / 2) ** 2)


def _ensure_sklearn() -> bool:
    global NearestNeighbors, StandardScaler, SKLEARN_AVAILABLE
    if SKLEARN_AVAILABLE:
        return True
    if os.environ.get("PHYSTABMOL_DISABLE_SKLEARN", "0") == "1":
        return False
    try:  # pragma: no cover - server path.
        from sklearn.neighbors import NearestNeighbors as _NearestNeighbors
        from sklearn.preprocessing import StandardScaler as _StandardScaler

        NearestNeighbors = _NearestNeighbors
        StandardScaler = _StandardScaler
        SKLEARN_AVAILABLE = True
        return True
    except Exception:  # pragma: no cover
        return False
