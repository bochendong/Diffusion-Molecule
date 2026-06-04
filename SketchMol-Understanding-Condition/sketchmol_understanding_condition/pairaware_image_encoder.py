"""Pair-aware trainable image encoder for edit-condition probes."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .image_encoder_v2 import load_grayscale_image


def train_pairaware_image_encoder(
    *,
    targets_npz: str | Path,
    output_dir: str | Path,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    embedding_dim: int = 256,
    image_size: int = 96,
    seed: int = 7,
) -> dict[str, object]:
    """Train a CNN on edit-aware labels: delta bucket plus property deltas."""

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    torch.manual_seed(seed)
    data = np.load(targets_npz, allow_pickle=True)
    image_paths = data["image_paths"].astype(str)
    splits = data["splits"].astype(str)
    labels = data["labels"].astype(np.int64)
    property_deltas = data["property_deltas"].astype(np.float32)
    label_names = data["label_names"].astype(str).tolist()

    train_mask = splits == "train"
    eval_mask = splits == "eval"
    delta_mean = property_deltas[train_mask].mean(axis=0, keepdims=True)
    delta_std = property_deltas[train_mask].std(axis=0, keepdims=True)
    delta_std = np.where(delta_std < 1e-6, 1.0, delta_std)
    deltas_z = (property_deltas - delta_mean) / delta_std

    class PairDataset(Dataset):
        def __init__(self, mask: np.ndarray) -> None:
            self.indices = np.flatnonzero(mask)

        def __len__(self) -> int:
            return int(self.indices.shape[0])

        def __getitem__(self, item: int):
            idx = int(self.indices[item])
            image = load_grayscale_image(image_paths[idx], image_size=image_size)
            return (
                torch.from_numpy(image),
                torch.tensor(labels[idx], dtype=torch.long),
                torch.from_numpy(deltas_z[idx]),
            )

    model = PairAwareImageEncoder(embedding_dim=embedding_dim, num_classes=len(label_names), delta_dim=property_deltas.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_loader = DataLoader(PairDataset(train_mask), batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(PairDataset(eval_mask), batch_size=batch_size, shuffle=False)

    history = []
    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        total = 0
        for images, y, deltas in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, delta_pred, _ = model(images.float())
            cls_loss = F.cross_entropy(logits, y)
            delta_loss = F.mse_loss(delta_pred, deltas.float())
            loss = cls_loss + 0.25 * delta_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(images.shape[0])
            total += int(images.shape[0])
        metrics = _evaluate(model, eval_loader)
        metrics["train_loss"] = total_loss / max(1, total)
        history.append(metrics)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "pairaware_image_encoder.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "embedding_dim": embedding_dim,
            "num_classes": len(label_names),
            "delta_dim": int(property_deltas.shape[1]),
            "image_size": image_size,
            "label_names": label_names,
            "delta_mean": delta_mean.astype(np.float32),
            "delta_std": delta_std.astype(np.float32),
            "history": history,
        },
        checkpoint_path,
    )
    return {
        "checkpoint": str(checkpoint_path),
        "train_examples": int(train_mask.sum()),
        "eval_examples": int(eval_mask.sum()),
        "epochs": epochs,
        "label_names": label_names,
        "history": history,
    }


def load_pairaware_image_encoder(checkpoint_path: str | Path):
    """Load a pair-aware image encoder checkpoint."""

    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = PairAwareImageEncoder(
        embedding_dim=int(payload["embedding_dim"]),
        num_classes=int(payload["num_classes"]),
        delta_dim=int(payload["delta_dim"]),
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, int(payload["image_size"])


def encode_image_path(model, image_path: str | Path, image_size: int) -> np.ndarray:
    """Encode one image path using a pair-aware model."""

    import torch

    image = torch.from_numpy(load_grayscale_image(image_path, image_size=image_size)).unsqueeze(0)
    with torch.no_grad():
        _, _, embedding = model(image.float())
    vec = embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


class PairAwareImageEncoder:  # pragma: no cover - optional torch wrapper
    def __new__(cls, *args, **kwargs):
        import torch
        from torch import nn

        class _PairAwareImageEncoder(nn.Module):
            def __init__(self, embedding_dim: int, num_classes: int, delta_dim: int) -> None:
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((4, 4)),
                    nn.Flatten(),
                )
                self.proj = nn.Sequential(nn.Linear(64 * 4 * 4, embedding_dim), nn.ReLU(inplace=True))
                self.class_head = nn.Linear(embedding_dim, num_classes)
                self.delta_head = nn.Linear(embedding_dim, delta_dim)

            def forward(self, images: torch.Tensor):
                embedding = self.proj(self.backbone(images))
                return self.class_head(embedding), self.delta_head(embedding), embedding

        return _PairAwareImageEncoder(*args, **kwargs)


def _evaluate(model, loader) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    model.eval()
    losses = []
    delta_losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for images, y, deltas in loader:
            logits, delta_pred, _ = model(images.float())
            losses.append(float(F.cross_entropy(logits, y).item()))
            delta_losses.append(float(F.mse_loss(delta_pred, deltas.float()).item()))
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.shape[0])
    return {
        "eval_class_ce": float(np.mean(losses)) if losses else 0.0,
        "eval_delta_mse": float(np.mean(delta_losses)) if delta_losses else 0.0,
        "eval_accuracy": correct / total if total else 0.0,
    }
