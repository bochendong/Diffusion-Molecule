"""Conditional tabular diffusion implemented with scikit-learn."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import pickle

import numpy as np

if os.environ.get("PHYSTABMOL_DISABLE_SKLEARN", "0") == "1":  # pragma: no cover - local smoke path.
    MLPRegressor = None
    NearestNeighbors = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False
else:
    try:  # pragma: no cover - server path has sklearn; local macOS wheels can be broken.
        from sklearn.neural_network import MLPRegressor
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler

        SKLEARN_AVAILABLE = True
    except Exception:  # pragma: no cover
        MLPRegressor = None
        NearestNeighbors = None
        StandardScaler = None
        SKLEARN_AVAILABLE = False

from .schema import BOUNDS, INTEGER_COLUMNS, TABLE_COLUMNS
from .progress import iter_progress


TARGET_SCALES = np.asarray([120.0, 3.0, 0.35, 60.0, 3.0, 5.0, 5.0, 2.5], dtype=np.float32)


@dataclass
class TabularDiffusion:
    timesteps: int = 40
    noise_repeats: int = 24
    hidden: tuple[int, int] = (96, 96)
    seed: int = 11
    target_condition_start: int = 12
    target_anchor: float = 1.0
    anchor_neighbors: int = 128
    count_anchor_weight: float = 0.8

    def fit(self, table_y: np.ndarray, condition_x: np.ndarray) -> "TabularDiffusion":
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is not available. Use --backend torch or install a working sklearn/scipy stack.")
        self.y_scaler = StandardScaler().fit(table_y)
        self.c_scaler = StandardScaler().fit(condition_x)
        self.train_table_y_ = np.asarray(table_y, dtype=np.float32)
        self.train_targets_ = self.train_table_y_[:, : len(TARGET_SCALES)]
        self.anchor_nn_ = NearestNeighbors(n_neighbors=min(self.anchor_neighbors, len(self.train_targets_)))
        self.anchor_nn_.fit(self.train_targets_ / TARGET_SCALES)
        y = self.y_scaler.transform(table_y)
        c = self.c_scaler.transform(condition_x)
        rng = np.random.default_rng(self.seed)
        train_x = []
        train_eps = []
        for row_idx in iter_progress(range(len(y)), total=len(y), label="building sklearn diffusion noise"):
            for _ in range(self.noise_repeats):
                t = rng.integers(1, self.timesteps + 1)
                alpha_bar = self._alpha_bar(t)
                eps = rng.normal(size=y.shape[1])
                noisy = np.sqrt(alpha_bar) * y[row_idx] + np.sqrt(1.0 - alpha_bar) * eps
                train_x.append(np.concatenate([noisy, c[row_idx], [t / self.timesteps]]))
                train_eps.append(eps)
        use_early_stopping = len(train_x) >= 50
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden,
            activation="relu",
            alpha=1e-4,
            learning_rate_init=2e-3,
            max_iter=900,
            random_state=self.seed,
            early_stopping=use_early_stopping,
            n_iter_no_change=30,
        )
        self.model.fit(np.asarray(train_x), np.asarray(train_eps))
        return self

    def sample(self, condition_x: np.ndarray, n: int = 8, guidance_noise: float = 0.25) -> list[dict[str, float]]:
        if condition_x.ndim == 1:
            condition_x = condition_x[None, :]
        rng = np.random.default_rng(self.seed + 101)
        rows = []
        cond_scaled = self.c_scaler.transform(condition_x)
        row_iter = iter_progress(range(n), total=n, label="sampling sklearn diffusion rows") if n >= 100 else range(n)
        for idx in row_iter:
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
