"""Optional image-only pretrained understanding encoders.

This deliberately avoids video. On a server with torch/transformers installed,
CLIP can provide a learned image-understanding stream. The rest of PhysTabMol
still works without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .features import descriptor_image


@dataclass
class PretrainedImageUnderstanding:
    model_name: str = "openai/clip-vit-base-patch32"
    device: str = "auto"
    batch_size: int = 64

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except Exception as exc:  # pragma: no cover - server optional path.
            raise RuntimeError(
                "Pretrained understanding requires torch and transformers. "
                "Install them or use --understanding-backbone handcrafted."
            ) from exc

        self.torch = torch
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.device_ = self._resolve_device()
        self.model.to(self.device_)
        self.model.eval()

    def encode_dataframe(self, df, image_column: str | None = None) -> np.ndarray:
        images = []
        for _, row in df.iterrows():
            image = None
            if image_column and image_column in row and not _is_missing(row[image_column]):
                path = Path(str(row[image_column]))
                if path.exists():
                    image = Image.open(path).convert("RGB")
            if image is None:
                image = Image.fromarray(descriptor_image(str(row["smiles"]))).convert("RGB")
            images.append(image)
        return self.encode_images(images)

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        chunks = []
        with self.torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                batch = images[start : start + self.batch_size]
                inputs = self.processor(images=batch, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device_) for k, v in inputs.items()}
                features = self.model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                chunks.append(features.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(chunks, axis=0)

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if self.torch.cuda.is_available() else "cpu"


def _is_missing(value) -> bool:
    try:
        import pandas as pd

        return pd.isna(value)
    except Exception:
        return value is None
