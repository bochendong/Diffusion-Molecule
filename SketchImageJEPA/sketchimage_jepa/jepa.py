"""JEPA-style predictor from context/source embeddings to target embeddings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class JEPAConfig:
    feature_dim: int = 96
    latent_dim: int = 48
    ridge: float = 1e-3


class SketchImageJEPAPredictor:
    """A small ridge JEPA baseline.

    The model predicts target molecular latents from context features and an
    optional source latent. It is intentionally numpy-only so the first project
    smoke test works before server/GPU setup.
    """

    expects_source_latents = True

    def __init__(self, config: JEPAConfig | None = None):
        self.config = config or JEPAConfig()
        self.weights_: np.ndarray | None = None
        self.history: list[dict[str, float | str]] = []

    def fit(self, conditions: np.ndarray, target_latents: np.ndarray, source_latents: np.ndarray) -> "SketchImageJEPAPredictor":
        conditions = np.asarray(conditions, dtype=np.float32)
        target_latents = np.asarray(target_latents, dtype=np.float32)
        source_latents = np.asarray(source_latents, dtype=np.float32)
        self.config.feature_dim = int(conditions.shape[1])
        self.config.latent_dim = int(target_latents.shape[1])
        source_mask = _source_mask(source_latents)
        x = _design_matrix(conditions, source_latents, source_mask)
        self.weights_ = _ridge_solve(x, target_latents, self.config.ridge)
        pred = x @ self.weights_
        loss = float(np.mean((pred - target_latents) ** 2))
        delta_loss = _masked_delta_loss(pred, target_latents, source_latents, source_mask)
        self.history = [{"epoch": 0.0, "loss": loss, "delta": delta_loss, "backend": "numpy_ridge_jepa"}]
        return self

    def predict(self, conditions: np.ndarray, source_latents: np.ndarray) -> np.ndarray:
        conditions = np.asarray(conditions, dtype=np.float32)
        source_latents = np.asarray(source_latents, dtype=np.float32)
        if self.weights_ is None:
            return np.zeros((len(conditions), self.config.latent_dim), dtype=np.float32)
        source_mask = _source_mask(source_latents)
        return (_design_matrix(conditions, source_latents, source_mask) @ self.weights_).astype(np.float32)

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(_json_dumps(asdict(self.config)), encoding="utf-8")
        (out_dir / "metadata.json").write_text(_json_dumps({"model_type": "sketchimage_jepa", "history": self.history}), encoding="utf-8")
        if self.weights_ is not None:
            np.save(out_dir / "ridge_weights.npy", self.weights_)

    @classmethod
    def load(cls, out_dir: str | Path) -> "SketchImageJEPAPredictor":
        import json

        out_dir = Path(out_dir)
        config = JEPAConfig(**json.loads((out_dir / "config.json").read_text(encoding="utf-8")))
        obj = cls(config)
        weights_path = out_dir / "ridge_weights.npy"
        if weights_path.exists():
            obj.weights_ = np.load(weights_path)
        metadata_path = out_dir / "metadata.json"
        if metadata_path.exists():
            obj.history = list(json.loads(metadata_path.read_text(encoding="utf-8")).get("history", []))
        return obj


def _source_mask(source_latents: np.ndarray) -> np.ndarray:
    return (np.linalg.norm(source_latents, axis=1, keepdims=True) > 1e-8).astype(np.float32)


def _design_matrix(conditions: np.ndarray, source_latents: np.ndarray, source_mask: np.ndarray) -> np.ndarray:
    delta_context = conditions[:, : source_latents.shape[1]]
    return np.concatenate([conditions, source_latents, source_mask, source_mask * delta_context, np.ones((len(conditions), 1), dtype=np.float32)], axis=1)


def _ridge_solve(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    """Solve ridge regression with a dual path for high-dimensional latents."""

    if x.shape[0] < x.shape[1]:
        reg = ridge * np.eye(x.shape[0], dtype=np.float32)
        alpha = np.linalg.solve(x @ x.T + reg, y)
        return (x.T @ alpha).astype(np.float32)
    reg = ridge * np.eye(x.shape[1], dtype=np.float32)
    reg[-1, -1] = 0.0
    return np.linalg.solve(x.T @ x + reg, x.T @ y).astype(np.float32)


def _masked_delta_loss(pred: np.ndarray, target: np.ndarray, source: np.ndarray, mask: np.ndarray) -> float:
    rows = mask.squeeze(-1) > 0.5
    if not np.any(rows):
        return 0.0
    pred_delta = pred[rows] - source[rows]
    target_delta = target[rows] - source[rows]
    return float(np.mean((pred_delta - target_delta) ** 2))


def _json_dumps(payload: object) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
