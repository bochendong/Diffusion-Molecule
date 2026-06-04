"""Trainable image+text fusion encoder for edit-aware condition tokens."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .image_encoder_v2 import load_grayscale_image
from .text_features import hashed_text_vector


def train_fusion_image_text_encoder(
    *,
    targets_npz: str | Path,
    output_dir: str | Path,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    embedding_dim: int = 256,
    image_size: int = 96,
    text_dim: int = 256,
    contrastive_weight: float = 0.0,
    contrastive_temperature: float = 0.2,
    seed: int = 7,
) -> dict[str, object]:
    """Train a fusion encoder on source image plus instruction prompt."""

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    torch.manual_seed(seed)
    data = np.load(targets_npz, allow_pickle=True)
    image_paths = data["image_paths"].astype(str)
    prompts = data["prompts"].astype(str)
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
    text_vectors = np.stack([hashed_text_vector(prompt, text_dim) for prompt in prompts]).astype(np.float32)

    class FusionDataset(Dataset):
        def __init__(self, mask: np.ndarray) -> None:
            self.indices = np.flatnonzero(mask)

        def __len__(self) -> int:
            return int(self.indices.shape[0])

        def __getitem__(self, item: int):
            idx = int(self.indices[item])
            image = load_grayscale_image(image_paths[idx], image_size=image_size)
            return (
                torch.from_numpy(image),
                torch.from_numpy(text_vectors[idx]),
                torch.tensor(labels[idx], dtype=torch.long),
                torch.from_numpy(deltas_z[idx]),
            )

    model = FusionImageTextEncoder(
        embedding_dim=embedding_dim,
        text_dim=text_dim,
        num_classes=len(label_names),
        delta_dim=property_deltas.shape[1],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_loader = DataLoader(FusionDataset(train_mask), batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(FusionDataset(eval_mask), batch_size=batch_size, shuffle=False)

    history = []
    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_delta_loss = 0.0
        total_contrastive_loss = 0.0
        total = 0
        for images, text, y, deltas in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, delta_pred, embedding = model(images.float(), text.float())
            cls_loss = F.cross_entropy(logits, y)
            delta_loss = F.mse_loss(delta_pred, deltas.float())
            contrastive_loss = _supervised_contrastive_loss(
                embedding,
                y,
                temperature=contrastive_temperature,
            )
            loss = cls_loss + 0.25 * delta_loss + contrastive_weight * contrastive_loss
            loss.backward()
            optimizer.step()
            batch_count = int(images.shape[0])
            total_loss += float(loss.item()) * batch_count
            total_cls_loss += float(cls_loss.item()) * batch_count
            total_delta_loss += float(delta_loss.item()) * batch_count
            total_contrastive_loss += float(contrastive_loss.item()) * batch_count
            total += batch_count
        metrics = _evaluate(model, eval_loader, contrastive_temperature=contrastive_temperature)
        metrics["train_loss"] = total_loss / max(1, total)
        metrics["train_class_ce"] = total_cls_loss / max(1, total)
        metrics["train_delta_mse"] = total_delta_loss / max(1, total)
        metrics["train_contrastive_loss"] = total_contrastive_loss / max(1, total)
        history.append(metrics)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "fusion_image_text_encoder.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "embedding_dim": embedding_dim,
            "text_dim": text_dim,
            "num_classes": len(label_names),
            "delta_dim": int(property_deltas.shape[1]),
            "image_size": image_size,
            "label_names": label_names,
            "delta_mean": delta_mean.astype(np.float32),
            "delta_std": delta_std.astype(np.float32),
            "contrastive_weight": float(contrastive_weight),
            "contrastive_temperature": float(contrastive_temperature),
            "history": history,
        },
        checkpoint_path,
    )
    return {
        "checkpoint": str(checkpoint_path),
        "train_examples": int(train_mask.sum()),
        "eval_examples": int(eval_mask.sum()),
        "epochs": epochs,
        "contrastive_weight": float(contrastive_weight),
        "contrastive_temperature": float(contrastive_temperature),
        "label_names": label_names,
        "history": history,
    }


def load_fusion_image_text_encoder(checkpoint_path: str | Path):
    """Load a trained image+text fusion encoder."""

    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = FusionImageTextEncoder(
        embedding_dim=int(payload["embedding_dim"]),
        text_dim=int(payload["text_dim"]),
        num_classes=int(payload["num_classes"]),
        delta_dim=int(payload["delta_dim"]),
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, int(payload["image_size"]), int(payload["text_dim"])


def encode_image_text(
    model,
    *,
    image_path: str | Path | None,
    prompt: str,
    image_size: int,
    text_dim: int,
    use_image: bool = True,
    use_text: bool = True,
) -> np.ndarray:
    """Encode one image/prompt pair using a trained fusion model."""

    import torch

    if use_image and image_path:
        image_arr = load_grayscale_image(image_path, image_size=image_size)
    else:
        image_arr = np.zeros((1, image_size, image_size), dtype=np.float32)
    text_vec = hashed_text_vector(prompt if use_text else "", text_dim).astype(np.float32)
    image = torch.from_numpy(image_arr).unsqueeze(0)
    text = torch.from_numpy(text_vec).unsqueeze(0)
    with torch.no_grad():
        _, _, embedding = model(image.float(), text.float())
    vec = embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


class FusionImageTextEncoder:  # pragma: no cover - optional torch wrapper
    def __new__(cls, *args, **kwargs):
        import torch
        from torch import nn

        class _FusionImageTextEncoder(nn.Module):
            def __init__(self, embedding_dim: int, text_dim: int, num_classes: int, delta_dim: int) -> None:
                super().__init__()
                self.image_backbone = nn.Sequential(
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
                self.image_proj = nn.Sequential(nn.Linear(64 * 4 * 4, embedding_dim), nn.ReLU(inplace=True))
                self.text_proj = nn.Sequential(nn.Linear(text_dim, embedding_dim), nn.ReLU(inplace=True))
                self.fusion = nn.Sequential(
                    nn.Linear(embedding_dim * 2, embedding_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.ReLU(inplace=True),
                )
                self.class_head = nn.Linear(embedding_dim, num_classes)
                self.delta_head = nn.Linear(embedding_dim, delta_dim)

            def forward(self, images: torch.Tensor, text: torch.Tensor):
                image_embedding = self.image_proj(self.image_backbone(images))
                text_embedding = self.text_proj(text)
                embedding = self.fusion(torch.cat([image_embedding, text_embedding], dim=1))
                return self.class_head(embedding), self.delta_head(embedding), embedding

        return _FusionImageTextEncoder(*args, **kwargs)


def _supervised_contrastive_loss(embedding, labels, *, temperature: float = 0.2):
    """Pull same-label embeddings together and push different labels apart."""

    import torch
    import torch.nn.functional as F

    if int(embedding.shape[0]) <= 1:
        return embedding.sum() * 0.0

    labels = labels.view(-1)
    positive_mask = labels[:, None].eq(labels[None, :])
    self_mask = torch.eye(int(labels.shape[0]), dtype=torch.bool, device=labels.device)
    positive_mask = positive_mask & ~self_mask
    valid = positive_mask.sum(dim=1) > 0
    if not bool(valid.any()):
        return embedding.sum() * 0.0

    normalized = F.normalize(embedding, dim=1)
    logits = normalized @ normalized.T
    logits = logits / max(float(temperature), 1e-6)
    masked_logits = logits.masked_fill(self_mask, torch.finfo(logits.dtype).min)
    log_prob = logits - torch.logsumexp(masked_logits, dim=1, keepdim=True)
    mean_log_prob = (log_prob * positive_mask.float()).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1)
    return -mean_log_prob[valid].mean()


def _evaluate(model, loader, *, contrastive_temperature: float = 0.2) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    model.eval()
    losses = []
    delta_losses = []
    contrastive_losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for images, text, y, deltas in loader:
            logits, delta_pred, embedding = model(images.float(), text.float())
            losses.append(float(F.cross_entropy(logits, y).item()))
            delta_losses.append(float(F.mse_loss(delta_pred, deltas.float()).item()))
            contrastive_losses.append(
                float(_supervised_contrastive_loss(embedding, y, temperature=contrastive_temperature).item())
            )
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.shape[0])
    return {
        "eval_class_ce": float(np.mean(losses)) if losses else 0.0,
        "eval_delta_mse": float(np.mean(delta_losses)) if delta_losses else 0.0,
        "eval_contrastive_loss": float(np.mean(contrastive_losses)) if contrastive_losses else 0.0,
        "eval_accuracy": correct / total if total else 0.0,
    }
