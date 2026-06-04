"""Trainable image encoder v2 utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_grayscale_image(path: str | Path, image_size: int = 128) -> np.ndarray:
    """Load a molecule image as a normalized [1, H, W] array."""

    from PIL import Image

    image = Image.open(path).convert("L").resize((image_size, image_size))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr[None, :, :]


def train_image_encoder_v2(
    *,
    targets_npz: str | Path,
    output_dir: str | Path,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    embedding_dim: int = 256,
    image_size: int = 128,
    seed: int = 7,
) -> dict[str, object]:
    """Train a small CNN to predict source fingerprints/properties from images."""

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    torch.manual_seed(seed)
    data = np.load(targets_npz, allow_pickle=True)
    image_paths = data["image_paths"].astype(str)
    splits = data["splits"].astype(str)
    fingerprints = data["fingerprints"].astype(np.float32)
    properties = data["properties"].astype(np.float32)
    train_mask = splits == "train"
    eval_mask = splits == "eval"
    prop_mean = properties[train_mask].mean(axis=0, keepdims=True)
    prop_std = properties[train_mask].std(axis=0, keepdims=True)
    prop_std = np.where(prop_std < 1e-6, 1.0, prop_std)
    properties_z = (properties - prop_mean) / prop_std

    class ImageTargetDataset(Dataset):
        def __init__(self, mask: np.ndarray) -> None:
            self.indices = np.flatnonzero(mask)

        def __len__(self) -> int:
            return int(self.indices.shape[0])

        def __getitem__(self, item: int):
            idx = int(self.indices[item])
            image = load_grayscale_image(image_paths[idx], image_size=image_size)
            return (
                torch.from_numpy(image),
                torch.from_numpy(fingerprints[idx]),
                torch.from_numpy(properties_z[idx]),
            )

    model = ImageEncoderV2(embedding_dim=embedding_dim, fingerprint_dim=fingerprints.shape[1], property_dim=properties.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_loader = DataLoader(ImageTargetDataset(train_mask), batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(ImageTargetDataset(eval_mask), batch_size=batch_size, shuffle=False)

    history = []
    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        total = 0
        for images, fp, props in train_loader:
            optimizer.zero_grad(set_to_none=True)
            fp_logits, prop_pred, _ = model(images.float())
            fp_loss = F.binary_cross_entropy_with_logits(fp_logits, fp.float())
            prop_loss = F.mse_loss(prop_pred, props.float())
            loss = fp_loss + 0.25 * prop_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(images.shape[0])
            total += int(images.shape[0])
        eval_metrics = _evaluate(model, eval_loader)
        eval_metrics["train_loss"] = total_loss / max(1, total)
        history.append(eval_metrics)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "image_encoder_v2.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "embedding_dim": embedding_dim,
            "fingerprint_dim": int(fingerprints.shape[1]),
            "property_dim": int(properties.shape[1]),
            "image_size": image_size,
            "property_mean": prop_mean.astype(np.float32),
            "property_std": prop_std.astype(np.float32),
            "history": history,
        },
        checkpoint_path,
    )
    return {
        "checkpoint": str(checkpoint_path),
        "train_examples": int(train_mask.sum()),
        "eval_examples": int(eval_mask.sum()),
        "epochs": epochs,
        "history": history,
    }


def load_trained_image_encoder(checkpoint_path: str | Path):
    """Load a trained image encoder checkpoint."""

    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ImageEncoderV2(
        embedding_dim=int(payload["embedding_dim"]),
        fingerprint_dim=int(payload["fingerprint_dim"]),
        property_dim=int(payload["property_dim"]),
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, int(payload["image_size"])


def encode_image_path(model, image_path: str | Path, image_size: int) -> np.ndarray:
    """Encode one image path using a trained model."""

    import torch

    image = torch.from_numpy(load_grayscale_image(image_path, image_size=image_size)).unsqueeze(0)
    with torch.no_grad():
        _, _, embedding = model(image.float())
    vec = embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


class ImageEncoderV2:  # pragma: no cover - thin wrapper around optional torch
    def __new__(cls, *args, **kwargs):
        import torch
        from torch import nn

        class _TinyImageEncoderV2(nn.Module):
            def __init__(self, embedding_dim: int, fingerprint_dim: int, property_dim: int) -> None:
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
                self.fp_head = nn.Linear(embedding_dim, fingerprint_dim)
                self.prop_head = nn.Linear(embedding_dim, property_dim)

            def forward(self, images: torch.Tensor):
                embedding = self.proj(self.backbone(images))
                return self.fp_head(embedding), self.prop_head(embedding), embedding

        return _TinyImageEncoderV2(*args, **kwargs)


def _evaluate(model, loader) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    model.eval()
    fp_losses = []
    prop_losses = []
    with torch.no_grad():
        for images, fp, props in loader:
            fp_logits, prop_pred, _ = model(images.float())
            fp_losses.append(float(F.binary_cross_entropy_with_logits(fp_logits, fp.float()).item()))
            prop_losses.append(float(F.mse_loss(prop_pred, props.float()).item()))
    return {
        "eval_fp_bce": float(np.mean(fp_losses)) if fp_losses else 0.0,
        "eval_property_mse": float(np.mean(prop_losses)) if prop_losses else 0.0,
    }
