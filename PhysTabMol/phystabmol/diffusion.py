"""Conditional tabular diffusion implemented with scikit-learn."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .schema import BOUNDS, INTEGER_COLUMNS, TABLE_COLUMNS


@dataclass
class TabularDiffusion:
    timesteps: int = 40
    noise_repeats: int = 24
    hidden: tuple[int, int] = (96, 96)
    seed: int = 11
    target_condition_start: int = 12
    target_anchor: float = 1.0

    def fit(self, table_y: np.ndarray, condition_x: np.ndarray) -> "TabularDiffusion":
        self.y_scaler = StandardScaler().fit(table_y)
        self.c_scaler = StandardScaler().fit(condition_x)
        y = self.y_scaler.transform(table_y)
        c = self.c_scaler.transform(condition_x)
        rng = np.random.default_rng(self.seed)
        train_x = []
        train_eps = []
        for row_idx in range(len(y)):
            for _ in range(self.noise_repeats):
                t = rng.integers(1, self.timesteps + 1)
                alpha_bar = self._alpha_bar(t)
                eps = rng.normal(size=y.shape[1])
                noisy = np.sqrt(alpha_bar) * y[row_idx] + np.sqrt(1.0 - alpha_bar) * eps
                train_x.append(np.concatenate([noisy, c[row_idx], [t / self.timesteps]]))
                train_eps.append(eps)
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden,
            activation="relu",
            alpha=1e-4,
            learning_rate_init=2e-3,
            max_iter=900,
            random_state=self.seed,
            early_stopping=True,
            n_iter_no_change=30,
        )
        self.model.fit(np.asarray(train_x), np.asarray(train_eps))
        return self

    def sample(self, condition_x: np.ndarray, n: int = 8, guidance_noise: float = 0.55) -> list[dict[str, float]]:
        if condition_x.ndim == 1:
            condition_x = condition_x[None, :]
        rng = np.random.default_rng(self.seed + 101)
        rows = []
        cond_scaled = self.c_scaler.transform(condition_x)
        for idx in range(n):
            c = cond_scaled[idx % len(cond_scaled)]
            y = rng.normal(size=len(TABLE_COLUMNS))
            for t in range(self.timesteps, 0, -1):
                alpha_bar = self._alpha_bar(t)
                model_x = np.concatenate([y, c, [t / self.timesteps]])[None, :]
                eps_hat = self.model.predict(model_x)[0]
                y0 = (y - np.sqrt(1.0 - alpha_bar) * eps_hat) / max(1e-6, np.sqrt(alpha_bar))
                if t > 1:
                    next_alpha = self._alpha_bar(t - 1)
                    z = rng.normal(size=len(TABLE_COLUMNS))
                    y = np.sqrt(next_alpha) * y0 + guidance_noise * np.sqrt(1.0 - next_alpha) * z
                else:
                    y = y0
            raw = self.y_scaler.inverse_transform(y[None, :])[0]
            if condition_x.shape[1] >= self.target_condition_start + 8:
                target_values = condition_x[idx % len(condition_x), self.target_condition_start : self.target_condition_start + 8]
                raw[:8] = (1.0 - self.target_anchor) * raw[:8] + self.target_anchor * target_values
            rows.append(clip_table_row(dict(zip(TABLE_COLUMNS, raw))))
        return rows

    def _alpha_bar(self, t: int) -> float:
        frac = t / self.timesteps
        return float(np.cos((frac + 0.008) / 1.008 * np.pi / 2) ** 2)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "TabularDiffusion":
        with open(path, "rb") as f:
            return pickle.load(f)


def clip_table_row(row: dict[str, float]) -> dict[str, float]:
    clipped = {}
    for col in TABLE_COLUMNS:
        lo, hi = BOUNDS[col]
        value = float(np.clip(row.get(col, 0.0), lo, hi))
        if col in INTEGER_COLUMNS:
            value = float(int(round(value)))
        clipped[col] = value
    return clipped
