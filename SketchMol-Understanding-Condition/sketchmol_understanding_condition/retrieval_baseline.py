"""Lightweight retrieval baseline for condition-token ablations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RetrievalConfig:
    hidden_dim: int = 256
    epochs: int = 80
    batch_size: int = 64
    learning_rate: float = 1e-3
    seed: int = 7
    device: str = "auto"
    ridge_alpha: float = 1.0


def train_and_eval_ridge_variant(matrices, config: RetrievalConfig) -> dict[str, float | int | str]:
    """Train a closed-form ridge baseline for one variant."""

    x = _standardize_train_eval(matrices.train_x, matrices.eval_x)
    train_x, eval_x = x
    train_y = matrices.train_y.astype(np.float32)
    eye = np.eye(train_x.shape[1], dtype=np.float32)
    weights = np.linalg.solve(
        train_x.T @ train_x + config.ridge_alpha * eye,
        train_x.T @ train_y,
    )
    pred = _sigmoid(eval_x @ weights)
    metrics = evaluate_retrieval(
        pred,
        matrices.eval_y,
        eval_target_smiles=matrices.eval_target_smiles,
    )
    metrics.update(
        {
            "variant": matrices.variant,
            "train_examples": int(matrices.train_x.shape[0]),
            "eval_examples": int(matrices.eval_x.shape[0]),
            "feature_dim": int(matrices.feature_dim),
            "target_dim": int(matrices.target_dim),
            "ridge_alpha": float(config.ridge_alpha),
            "model": "ridge",
        }
    )
    return metrics


def train_and_eval_variant(matrices, config: RetrievalConfig) -> dict[str, float | int | str]:
    """Train a Torch MLP for one variant and evaluate fingerprint retrieval."""

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)
    model = nn.Sequential(
        nn.Linear(matrices.feature_dim, config.hidden_dim),
        nn.GELU(),
        nn.Linear(config.hidden_dim, config.hidden_dim),
        nn.GELU(),
        nn.Linear(config.hidden_dim, matrices.target_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(
        torch.from_numpy(matrices.train_x),
        torch.from_numpy(matrices.train_y),
    )
    loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    final_loss = 0.0
    for _ in range(config.epochs):
        model.train()
        losses = []
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses)) if losses else 0.0

    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(torch.from_numpy(matrices.eval_x).to(device))).cpu().numpy()

    metrics = evaluate_retrieval(
        pred,
        matrices.eval_y,
        eval_target_smiles=matrices.eval_target_smiles,
    )
    metrics.update(
        {
            "variant": matrices.variant,
            "train_examples": int(matrices.train_x.shape[0]),
            "eval_examples": int(matrices.eval_x.shape[0]),
            "feature_dim": int(matrices.feature_dim),
            "target_dim": int(matrices.target_dim),
            "final_train_loss": final_loss,
            "device": str(device),
        }
    )
    return metrics


def evaluate_retrieval(
    predicted: np.ndarray,
    target_pool: np.ndarray,
    *,
    eval_target_smiles: list[str],
) -> dict[str, float | int]:
    """Evaluate retrieval against the eval target pool."""

    similarities = _soft_tanimoto_matrix(predicted, target_pool)
    ranks = []
    top1_tanimotos = []
    true_tanimotos = []
    for idx in range(similarities.shape[0]):
        order = np.argsort(-similarities[idx])
        rank = int(np.where(order == idx)[0][0]) + 1
        ranks.append(rank)
        top1_tanimotos.append(float(similarities[idx, order[0]]))
        true_tanimotos.append(float(similarities[idx, idx]))

    exact_unique_targets = len(set(eval_target_smiles))
    return {
        "top1_hit": _hit_at(ranks, 1),
        "top5_hit": _hit_at(ranks, 5),
        "top10_hit": _hit_at(ranks, 10),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_top1_pred_target_tanimoto": float(np.mean(top1_tanimotos)),
        "mean_true_pred_target_tanimoto": float(np.mean(true_tanimotos)),
        "eval_unique_targets": int(exact_unique_targets),
    }


def _soft_tanimoto_matrix(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    a = np.clip(a.astype(np.float32), 0.0, 1.0)
    b = np.clip(b.astype(np.float32), 0.0, 1.0)
    inter = a @ b.T
    denom = a.sum(axis=1, keepdims=True) + b.sum(axis=1, keepdims=True).T - inter
    return inter / np.maximum(denom, eps)


def _hit_at(ranks: list[int], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for rank in ranks if rank <= k) / len(ranks)


def _resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _standardize_train_eval(train_x: np.ndarray, eval_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (eval_x - mean) / std


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))
