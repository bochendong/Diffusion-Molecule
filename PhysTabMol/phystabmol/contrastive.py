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
    batch_size: int = 512
    max_pairs: int | None = 20000
    retrieval_eval_size: int = 2048
    seed: int = 7

    def fit(self, image_x: np.ndarray, table_x: np.ndarray) -> "ContrastiveAligner":
        self.image_scaler = StandardScaler().fit(image_x)
        self.table_scaler = StandardScaler().fit(table_x)
        xi = self.image_scaler.transform(image_x).astype(np.float32)
        xt = self.table_scaler.transform(table_x).astype(np.float32)
        rng = np.random.default_rng(self.seed)
        self.w_image = rng.normal(0, 0.08, size=(xi.shape[1], self.embedding_dim)).astype(np.float32)
        self.w_table = rng.normal(0, 0.08, size=(xt.shape[1], self.embedding_dim)).astype(np.float32)

        train_idx = np.arange(len(xi))
        if self.max_pairs and len(train_idx) > self.max_pairs:
            train_idx = rng.choice(train_idx, size=self.max_pairs, replace=False)
        xi_train = xi[train_idx]
        xt_train = xt[train_idx]
        batch_size = max(2, min(int(self.batch_size), len(xi_train)))

        for _ in range(self.epochs):
            order = rng.permutation(len(xi_train))
            for start in range(0, len(xi_train), batch_size):
                idx = order[start : start + batch_size]
                if len(idx) < 2:
                    continue
                xb = xi_train[idx]
                tb = xt_train[idx]
                zi = _normalize(xb @ self.w_image)
                zt = _normalize(tb @ self.w_table)
                logits = zi @ zt.T / self.temperature
                p_i = _softmax(logits)
                p_t = _softmax(logits.T)
                eye = np.eye(len(idx), dtype=np.float32)
                grad_logits = ((p_i - eye) + (p_t - eye).T) / (2 * len(idx) * self.temperature)
                grad_zi = grad_logits @ zt
                grad_zt = grad_logits.T @ zi
                self.w_image -= self.lr * xb.T @ grad_zi
                self.w_table -= self.lr * tb.T @ grad_zt
        return self

    def transform_image(self, image_x: np.ndarray) -> np.ndarray:
        xi = self.image_scaler.transform(image_x).astype(np.float32)
        return _normalize(xi @ self.w_image)

    def transform_table(self, table_x: np.ndarray) -> np.ndarray:
        xt = self.table_scaler.transform(table_x).astype(np.float32)
        return _normalize(xt @ self.w_table)

    def retrieval_accuracy(self, image_x: np.ndarray, table_x: np.ndarray) -> float:
        if self.retrieval_eval_size and len(image_x) > self.retrieval_eval_size:
            rng = np.random.default_rng(self.seed + 1001)
            idx = rng.choice(len(image_x), size=self.retrieval_eval_size, replace=False)
            image_x = image_x[idx]
            table_x = table_x[idx]
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
