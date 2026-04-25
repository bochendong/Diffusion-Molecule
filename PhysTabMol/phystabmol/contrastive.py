"""Tiny NumPy contrastive aligner for image/table features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class ContrastiveAligner:
    embedding_dim: int = 12
    temperature: float = 0.12
    lr: float = 0.05
    epochs: int = 400
    seed: int = 7

    def fit(self, image_x: np.ndarray, table_x: np.ndarray) -> "ContrastiveAligner":
        self.image_scaler = StandardScaler().fit(image_x)
        self.table_scaler = StandardScaler().fit(table_x)
        xi = self.image_scaler.transform(image_x)
        xt = self.table_scaler.transform(table_x)
        rng = np.random.default_rng(self.seed)
        self.w_image = rng.normal(0, 0.08, size=(xi.shape[1], self.embedding_dim))
        self.w_table = rng.normal(0, 0.08, size=(xt.shape[1], self.embedding_dim))

        for _ in range(self.epochs):
            zi = _normalize(xi @ self.w_image)
            zt = _normalize(xt @ self.w_table)
            logits = zi @ zt.T / self.temperature
            p_i = _softmax(logits)
            p_t = _softmax(logits.T)
            eye = np.eye(len(xi))
            grad_logits = ((p_i - eye) + (p_t - eye).T) / (2 * len(xi) * self.temperature)
            grad_zi = grad_logits @ zt
            grad_zt = grad_logits.T @ zi
            self.w_image -= self.lr * xi.T @ grad_zi
            self.w_table -= self.lr * xt.T @ grad_zt
        return self

    def transform_image(self, image_x: np.ndarray) -> np.ndarray:
        xi = self.image_scaler.transform(image_x)
        return _normalize(xi @ self.w_image)

    def transform_table(self, table_x: np.ndarray) -> np.ndarray:
        xt = self.table_scaler.transform(table_x)
        return _normalize(xt @ self.w_table)

    def retrieval_accuracy(self, image_x: np.ndarray, table_x: np.ndarray) -> float:
        zi = self.transform_image(image_x)
        zt = self.transform_table(table_x)
        pred = np.argmax(zi @ zt.T, axis=1)
        return float((pred == np.arange(len(pred))).mean())

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "ContrastiveAligner":
        with open(path, "rb") as f:
            return pickle.load(f)


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(1e-8, np.linalg.norm(x, axis=1, keepdims=True))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=1, keepdims=True)
