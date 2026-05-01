"""PyTorch conditional tabular diffusion backend for server experiments."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .diffusion import clip_table_row
from .schema import BOUNDS, INTEGER_COLUMNS, TABLE_COLUMNS


TARGET_SCALES = np.asarray([120.0, 3.0, 0.35, 60.0, 3.0, 5.0, 5.0, 2.5], dtype=np.float32)

try:  # pragma: no cover - this workstation has no torch; server path uses it.
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    Dataset = None
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:  # pragma: no cover - exercised on torch servers.

    class _Denoiser(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, layers: int, dropout: float):
            super().__init__()
            blocks = []
            dim = in_dim
            for _ in range(layers):
                blocks.extend([nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout)])
                dim = hidden_dim
            blocks.append(nn.Linear(dim, out_dim))
            self.net = nn.Sequential(*blocks)

        def forward(self, x):
            return self.net(x)


    class _NoisyDiffusionDataset(Dataset):
        def __init__(self, y: np.ndarray, c: np.ndarray, timesteps: int, noise_repeats: int):
            self.y = torch.tensor(y, dtype=torch.float32)
            self.c = torch.tensor(c, dtype=torch.float32)
            self.timesteps = int(timesteps)
            self.noise_repeats = int(noise_repeats)
            steps = np.arange(self.timesteps + 1, dtype=np.float32) / self.timesteps
            self.alpha_bars = torch.tensor(np.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2, dtype=torch.float32)

        def __len__(self) -> int:
            return int(len(self.y) * self.noise_repeats)

        def __getitem__(self, idx: int):
            row_idx = idx // self.noise_repeats
            t = int(torch.randint(1, self.timesteps + 1, (1,)).item())
            alpha_bar = self.alpha_bars[t]
            eps = torch.randn_like(self.y[row_idx])
            noisy = torch.sqrt(alpha_bar) * self.y[row_idx] + torch.sqrt(1.0 - alpha_bar) * eps
            x = torch.cat([noisy, self.c[row_idx], torch.tensor([t / self.timesteps], dtype=torch.float32)])
            return x, eps


@dataclass
class TorchTabularDiffusion:
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

    def fit(self, table_y: np.ndarray, condition_x: np.ndarray) -> "TorchTabularDiffusion":
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed. Use --backend sklearn or install torch on the server.")

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
        dataset = _NoisyDiffusionDataset(y, c, timesteps=self.timesteps, noise_repeats=self.noise_repeats)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.model = _Denoiser(
            in_dim=len(TABLE_COLUMNS) + c.shape[1] + 1,
            out_dim=len(TABLE_COLUMNS),
            hidden_dim=self.hidden_dim,
            layers=self.layers,
            dropout=self.dropout,
        ).to(self.device_)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        self.history_ = []
        self.model.train()
        for epoch in range(self.epochs):
            losses = []
            for xb, yb in loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach().cpu()))
            self.history_.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
        return self

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
        for idx in range(n):
            c_np = cond_scaled[idx % len(cond_scaled)]
            y = rng.normal(size=len(TABLE_COLUMNS)).astype(np.float32)
            for t in range(self.timesteps, 0, -1):
                alpha_bar = self._alpha_bar(t)
                model_x = np.concatenate([y, c_np, [t / self.timesteps]], dtype=np.float32)
                xb = torch.tensor(model_x[None, :], device=self.device_)
                eps_hat = self.model(xb).detach().cpu().numpy()[0]
                y0 = (y - np.sqrt(1.0 - alpha_bar) * eps_hat) / max(1e-6, np.sqrt(alpha_bar))
                if t > 1:
                    next_alpha = self._alpha_bar(t - 1)
                    z = rng.normal(size=len(TABLE_COLUMNS)).astype(np.float32)
                    y = np.sqrt(next_alpha) * y0 + guidance_noise * np.sqrt(1.0 - next_alpha) * z
                else:
                    y = y0
            raw = self.y_scaler.inverse_transform(y[None, :])[0]
            if condition_x.shape[1] >= self.target_condition_start + 8:
                target_values = condition_x[idx % len(condition_x), self.target_condition_start : self.target_condition_start + 8]
                raw[:8] = (1.0 - self.target_anchor) * raw[:8] + self.target_anchor * target_values
                raw = self._calibrate_counts(raw, target_values, sample_idx=idx, rng=rng)
            rows.append(clip_table_row(dict(zip(TABLE_COLUMNS, raw))))
        return rows

    def _calibrate_counts(
        self,
        raw: np.ndarray,
        target_values: np.ndarray,
        sample_idx: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if not hasattr(self, "anchor_nn_"):
            return raw
        _, indices = self.anchor_nn_.kneighbors((target_values / TARGET_SCALES)[None, :], return_distance=True)
        choices = indices[0]
        anchor = self.train_table_y_[choices[(sample_idx * 9973 + self.seed) % len(choices)]]
        out = np.array(raw, copy=True)
        for col_idx, col in enumerate(TABLE_COLUMNS[8:], start=8):
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

    def save(self, path: str | Path) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed.")
        torch.save(
            {
                "config": {field.name: getattr(self, field.name) for field in fields(self)},
                "state_dict": self.model.state_dict(),
                "y_scaler": self.y_scaler,
                "c_scaler": self.c_scaler,
                "train_table_y": getattr(self, "train_table_y_", None),
                "history": getattr(self, "history_", []),
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
