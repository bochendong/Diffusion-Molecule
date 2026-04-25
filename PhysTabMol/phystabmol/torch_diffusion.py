"""PyTorch conditional tabular diffusion backend for server experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from .diffusion import clip_table_row
from .schema import TABLE_COLUMNS

try:  # pragma: no cover - this workstation has no torch; server path uses it.
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


@dataclass
class TorchTabularDiffusion:
    timesteps: int = 100
    noise_repeats: int = 12
    hidden_dim: int = 256
    layers: int = 4
    dropout: float = 0.05
    epochs: int = 80
    batch_size: int = 256
    lr: float = 2e-4
    seed: int = 11
    device: str = "auto"
    target_condition_start: int = 12
    target_anchor: float = 1.0

    def fit(self, table_y: np.ndarray, condition_x: np.ndarray) -> "TorchTabularDiffusion":
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed. Use --backend sklearn or install torch on the server.")

        self.y_scaler = StandardScaler().fit(table_y)
        self.c_scaler = StandardScaler().fit(condition_x)
        y = self.y_scaler.transform(table_y).astype(np.float32)
        c = self.c_scaler.transform(condition_x).astype(np.float32)
        self.device_ = self._resolve_device()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        train_x = []
        train_eps = []
        for row_idx in range(len(y)):
            for _ in range(self.noise_repeats):
                t = rng.integers(1, self.timesteps + 1)
                alpha_bar = self._alpha_bar(t)
                eps = rng.normal(size=y.shape[1]).astype(np.float32)
                noisy = np.sqrt(alpha_bar) * y[row_idx] + np.sqrt(1.0 - alpha_bar) * eps
                train_x.append(np.concatenate([noisy, c[row_idx], [t / self.timesteps]], dtype=np.float32))
                train_eps.append(eps)

        dataset = TensorDataset(torch.tensor(np.asarray(train_x)), torch.tensor(np.asarray(train_eps)))
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
    def sample(self, condition_x: np.ndarray, n: int = 32, guidance_noise: float = 0.55) -> list[dict[str, float]]:
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
            rows.append(clip_table_row(dict(zip(TABLE_COLUMNS, raw))))
        return rows

    def save(self, path: str | Path) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed.")
        torch.save(
            {
                "config": self.__dict__ | {"model": None},
                "state_dict": self.model.state_dict(),
                "y_scaler": self.y_scaler,
                "c_scaler": self.c_scaler,
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
