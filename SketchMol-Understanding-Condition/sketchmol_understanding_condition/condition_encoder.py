"""Unified condition encoder interfaces.

The proxy encoder wraps the current non-neural features behind the same API we
want for a future MLLM encoder:

    row + variant -> pooled_condition, query_tokens
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .retrieval_data import condition_features_for_row
from .image_features import image_feature_vector, image_patch_feature_vector
from .text_features import hashed_text_vector


@dataclass(frozen=True)
class ConditionEncoding:
    """Condition representation consumed by downstream classifiers/generators."""

    pooled: np.ndarray
    query_tokens: np.ndarray
    variant: str
    condition_mode: str


class ConditionEncoder(Protocol):
    """Interface for proxy, text, image, or MLLM condition encoders."""

    name: str
    pooled_dim: int
    num_queries: int
    query_dim: int

    def encode_row(self, row: dict[str, str]) -> ConditionEncoding:
        """Encode one manifest row."""


class ProxyConditionEncoder:
    """Deterministic encoder based on Morgan fingerprints and hashed text.

    This is not the final research model. It makes the experimental plumbing
    match the planned MLLM path while keeping smoke tests cheap and reproducible.
    """

    name = "proxy"

    def __init__(
        self,
        *,
        fingerprint_bits: int = 512,
        text_dim: int = 128,
        random_dim: int = 128,
        pooled_dim: int = 768,
        num_queries: int = 16,
        query_dim: int = 256,
    ) -> None:
        self.fingerprint_bits = fingerprint_bits
        self.text_dim = text_dim
        self.random_dim = random_dim
        self.pooled_dim = pooled_dim
        self.num_queries = num_queries
        self.query_dim = query_dim
        self._projection = _fixed_projection(pooled_dim, num_queries * query_dim, seed=17)

    def encode_row(self, row: dict[str, str]) -> ConditionEncoding:
        variant = row.get("variant", "full")
        features = condition_features_for_row(
            row,
            variant=variant,
            fingerprint_bits=self.fingerprint_bits,
            text_dim=self.text_dim,
            random_dim=self.random_dim,
        )
        pooled = _resize_vector(features, self.pooled_dim)
        query_flat = np.tanh(pooled @ self._projection)
        query_tokens = query_flat.reshape(self.num_queries, self.query_dim).astype(np.float32)
        return ConditionEncoding(
            pooled=pooled.astype(np.float32),
            query_tokens=query_tokens,
            variant=variant,
            condition_mode=row.get("condition_mode", variant),
        )


class MultimodalV0ConditionEncoder:
    """OCR-free rendered-image statistics plus hashed instruction text."""

    name = "multimodal_v0"

    def __init__(
        self,
        *,
        image_dim: int = 256,
        text_dim: int = 256,
        random_dim: int = 128,
        pooled_dim: int = 768,
        num_queries: int = 16,
        query_dim: int = 256,
    ) -> None:
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.random_dim = random_dim
        self.pooled_dim = pooled_dim
        self.num_queries = num_queries
        self.query_dim = query_dim
        self._projection = _fixed_projection(pooled_dim, num_queries * query_dim, seed=29)

    def encode_row(self, row: dict[str, str]) -> ConditionEncoding:
        variant = row.get("variant", "full")
        image_vec = image_feature_vector(row.get("source_image", ""), self.image_dim)
        text_vec = hashed_text_vector(row.get("prompt", ""), self.text_dim)
        if variant == "full":
            features = np.concatenate([image_vec, text_vec])
        elif variant == "text_only":
            features = np.concatenate([np.zeros_like(image_vec), text_vec])
        elif variant == "image_only":
            features = np.concatenate([image_vec, np.zeros_like(text_vec)])
        elif variant == "caption_bottleneck":
            features = np.concatenate([np.zeros_like(image_vec), text_vec])
        elif variant == "random_query":
            features = np.concatenate(
                [
                    np.zeros_like(image_vec),
                    np.zeros_like(text_vec),
                    _deterministic_random_vector(row.get("variant_id", ""), self.random_dim),
                ]
            )
        else:
            raise ValueError(f"Unsupported variant: {variant}")
        pooled = _resize_vector(features, self.pooled_dim)
        query_flat = np.tanh(pooled @ self._projection)
        query_tokens = query_flat.reshape(self.num_queries, self.query_dim).astype(np.float32)
        return ConditionEncoding(
            pooled=pooled.astype(np.float32),
            query_tokens=query_tokens,
            variant=variant,
            condition_mode=row.get("condition_mode", variant),
        )


class MultimodalCnnV1ConditionEncoder(MultimodalV0ConditionEncoder):
    """Fixed convolution and patch-layout image branch plus hashed text."""

    name = "multimodal_cnn_v1"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._projection = _fixed_projection(self.pooled_dim, self.num_queries * self.query_dim, seed=31)

    def encode_row(self, row: dict[str, str]) -> ConditionEncoding:
        variant = row.get("variant", "full")
        image_vec = image_patch_feature_vector(row.get("source_image", ""), self.image_dim)
        text_vec = hashed_text_vector(row.get("prompt", ""), self.text_dim)
        if variant == "full":
            features = np.concatenate([image_vec, text_vec])
        elif variant == "text_only":
            features = np.concatenate([np.zeros_like(image_vec), text_vec])
        elif variant == "image_only":
            features = np.concatenate([image_vec, np.zeros_like(text_vec)])
        elif variant == "caption_bottleneck":
            features = np.concatenate([np.zeros_like(image_vec), text_vec])
        elif variant == "random_query":
            features = np.concatenate(
                [
                    np.zeros_like(image_vec),
                    np.zeros_like(text_vec),
                    _deterministic_random_vector(row.get("variant_id", ""), self.random_dim),
                ]
            )
        else:
            raise ValueError(f"Unsupported variant: {variant}")
        pooled = _resize_vector(features, self.pooled_dim)
        query_flat = np.tanh(pooled @ self._projection)
        query_tokens = query_flat.reshape(self.num_queries, self.query_dim).astype(np.float32)
        return ConditionEncoding(
            pooled=pooled.astype(np.float32),
            query_tokens=query_tokens,
            variant=variant,
            condition_mode=row.get("condition_mode", variant),
        )


class MultimodalTrainedV2ConditionEncoder(MultimodalV0ConditionEncoder):
    """Trainable CNN image embedding plus hashed instruction text."""

    name = "multimodal_trained_v2"

    def __init__(self, *, image_encoder_checkpoint: str, **kwargs) -> None:
        super().__init__(**kwargs)
        from .image_encoder_v2 import encode_image_path, load_trained_image_encoder

        self._encode_image_path = encode_image_path
        self._image_model, self._image_size = load_trained_image_encoder(image_encoder_checkpoint)
        self._image_cache: dict[str, np.ndarray] = {}
        self._projection = _fixed_projection(self.pooled_dim, self.num_queries * self.query_dim, seed=37)

    def encode_row(self, row: dict[str, str]) -> ConditionEncoding:
        variant = row.get("variant", "full")
        image_path = row.get("source_image", "")
        image_vec = self._cached_image_embedding(image_path)
        text_vec = hashed_text_vector(row.get("prompt", ""), self.text_dim)
        if variant == "full":
            features = np.concatenate([image_vec, text_vec])
        elif variant == "text_only":
            features = np.concatenate([np.zeros_like(image_vec), text_vec])
        elif variant == "image_only":
            features = np.concatenate([image_vec, np.zeros_like(text_vec)])
        elif variant == "caption_bottleneck":
            features = np.concatenate([np.zeros_like(image_vec), text_vec])
        elif variant == "random_query":
            features = np.concatenate(
                [
                    np.zeros_like(image_vec),
                    np.zeros_like(text_vec),
                    _deterministic_random_vector(row.get("variant_id", ""), self.random_dim),
                ]
            )
        else:
            raise ValueError(f"Unsupported variant: {variant}")
        pooled = _resize_vector(features, self.pooled_dim)
        query_flat = np.tanh(pooled @ self._projection)
        query_tokens = query_flat.reshape(self.num_queries, self.query_dim).astype(np.float32)
        return ConditionEncoding(
            pooled=pooled.astype(np.float32),
            query_tokens=query_tokens,
            variant=variant,
            condition_mode=row.get("condition_mode", variant),
        )

    def _cached_image_embedding(self, image_path: str) -> np.ndarray:
        if not image_path:
            return np.zeros(self.image_dim, dtype=np.float32)
        if image_path not in self._image_cache:
            self._image_cache[image_path] = _resize_vector(
                self._encode_image_path(self._image_model, image_path, self._image_size),
                self.image_dim,
            )
        return self._image_cache[image_path]


class MultimodalPairAwareV2ConditionEncoder(MultimodalTrainedV2ConditionEncoder):
    """Pair-aware CNN image embedding plus hashed instruction text."""

    name = "multimodal_pairaware_v2"

    def __init__(self, *, image_encoder_checkpoint: str, **kwargs) -> None:
        MultimodalV0ConditionEncoder.__init__(self, **kwargs)
        from .pairaware_image_encoder import encode_image_path, load_pairaware_image_encoder

        self._encode_image_path = encode_image_path
        self._image_model, self._image_size = load_pairaware_image_encoder(image_encoder_checkpoint)
        self._image_cache: dict[str, np.ndarray] = {}
        self._projection = _fixed_projection(self.pooled_dim, self.num_queries * self.query_dim, seed=41)


class MultimodalFusionV2ConditionEncoder(MultimodalV0ConditionEncoder):
    """Trainable image+text fusion embedding for edit-aware conditions."""

    name = "multimodal_fusion_v2"

    def __init__(self, *, image_encoder_checkpoint: str, **kwargs) -> None:
        MultimodalV0ConditionEncoder.__init__(self, **kwargs)
        from .fusion_image_text_encoder import encode_image_text, load_fusion_image_text_encoder

        self._encode_image_text = encode_image_text
        self._fusion_model, self._image_size, self._text_dim = load_fusion_image_text_encoder(image_encoder_checkpoint)
        self._cache: dict[tuple[str, str, bool, bool], np.ndarray] = {}
        self._projection = _fixed_projection(self.pooled_dim, self.num_queries * self.query_dim, seed=47)

    def encode_row(self, row: dict[str, str]) -> ConditionEncoding:
        variant = row.get("variant", "full")
        image_path = row.get("source_image", "")
        prompt = row.get("prompt", "")
        if variant == "full":
            features = self._cached_embedding(image_path, prompt, True, True)
        elif variant == "text_only":
            features = self._cached_embedding(image_path, prompt, False, True)
        elif variant == "image_only":
            features = self._cached_embedding(image_path, prompt, True, False)
        elif variant == "caption_bottleneck":
            features = self._cached_embedding(image_path, prompt, False, True)
        elif variant == "random_query":
            features = _deterministic_random_vector(row.get("variant_id", ""), self.random_dim)
        else:
            raise ValueError(f"Unsupported variant: {variant}")
        pooled = _resize_vector(features, self.pooled_dim)
        query_flat = np.tanh(pooled @ self._projection)
        query_tokens = query_flat.reshape(self.num_queries, self.query_dim).astype(np.float32)
        return ConditionEncoding(
            pooled=pooled.astype(np.float32),
            query_tokens=query_tokens,
            variant=variant,
            condition_mode=row.get("condition_mode", variant),
        )

    def encode_rows(self, rows: list[dict[str, str]]) -> list[ConditionEncoding]:
        import torch

        from .image_encoder_v2 import load_grayscale_image

        image_arrays: dict[str, np.ndarray] = {}
        pending: list[tuple[int, dict[str, str], bool]] = []
        features = [None] * len(rows)
        for idx, row in enumerate(rows):
            variant = row.get("variant", "full")
            if variant == "random_query":
                features[idx] = _deterministic_random_vector(row.get("variant_id", ""), self.random_dim)
                continue
            use_image = variant in {"full", "image_only"}
            use_text = variant in {"full", "text_only", "caption_bottleneck"}
            image_path = row.get("source_image", "")
            if use_image and image_path and image_path not in image_arrays:
                image_arrays[image_path] = load_grayscale_image(image_path, image_size=self._image_size)
            pending.append((idx, row, use_image and bool(image_path)))

        batch_size = 64
        self._fusion_model.eval()
        with torch.no_grad():
            for start in range(0, len(pending), batch_size):
                chunk = pending[start : start + batch_size]
                images = []
                texts = []
                for _, row, use_image in chunk:
                    variant = row.get("variant", "full")
                    image_path = row.get("source_image", "")
                    if use_image:
                        images.append(image_arrays[image_path])
                    else:
                        images.append(np.zeros((1, self._image_size, self._image_size), dtype=np.float32))
                    use_text = variant in {"full", "text_only", "caption_bottleneck"}
                    texts.append(hashed_text_vector(row.get("prompt", "") if use_text else "", self._text_dim))
                image_tensor = torch.from_numpy(np.stack(images).astype(np.float32))
                text_tensor = torch.from_numpy(np.stack(texts).astype(np.float32))
                _, _, embeddings = self._fusion_model(image_tensor.float(), text_tensor.float())
                embeddings_np = embeddings.detach().cpu().numpy().astype(np.float32)
                norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
                embeddings_np = embeddings_np / np.where(norms > 0, norms, 1.0)
                for (idx, _, _), embedding in zip(chunk, embeddings_np):
                    features[idx] = embedding

        out = []
        for row, feature in zip(rows, features):
            if feature is None:
                raise RuntimeError("Internal error: missing fusion feature")
            pooled = _resize_vector(np.asarray(feature, dtype=np.float32), self.pooled_dim)
            query_flat = np.tanh(pooled @ self._projection)
            query_tokens = query_flat.reshape(self.num_queries, self.query_dim).astype(np.float32)
            out.append(
                ConditionEncoding(
                    pooled=pooled.astype(np.float32),
                    query_tokens=query_tokens,
                    variant=row.get("variant", "full"),
                    condition_mode=row.get("condition_mode", row.get("variant", "full")),
                )
            )
        return out

    def _cached_embedding(self, image_path: str, prompt: str, use_image: bool, use_text: bool) -> np.ndarray:
        key = (image_path, prompt, use_image, use_text)
        if key not in self._cache:
            self._cache[key] = _resize_vector(
                self._encode_image_text(
                    self._fusion_model,
                    image_path=image_path,
                    prompt=prompt,
                    image_size=self._image_size,
                    text_dim=self._text_dim,
                    use_image=use_image,
                    use_text=use_text,
                ),
                self.image_dim,
            )
        return self._cache[key]


def build_condition_encoder(name: str = "proxy", **kwargs) -> ConditionEncoder:
    """Factory for condition encoders."""

    if name == "proxy":
        return ProxyConditionEncoder(**kwargs)
    if name == "multimodal_v0":
        return MultimodalV0ConditionEncoder(**kwargs)
    if name == "multimodal_cnn_v1":
        return MultimodalCnnV1ConditionEncoder(**kwargs)
    if name == "multimodal_trained_v2":
        return MultimodalTrainedV2ConditionEncoder(**kwargs)
    if name == "multimodal_pairaware_v2":
        return MultimodalPairAwareV2ConditionEncoder(**kwargs)
    if name == "multimodal_fusion_v2":
        return MultimodalFusionV2ConditionEncoder(**kwargs)
    raise ValueError(f"Unsupported condition encoder: {name}")


def _resize_vector(values: np.ndarray, dim: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] == dim:
        return values
    if values.shape[0] > dim:
        return values[:dim]
    output = np.zeros(dim, dtype=np.float32)
    output[: values.shape[0]] = values
    return output


def _fixed_projection(in_dim: int, out_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scale = 1.0 / max(1.0, np.sqrt(in_dim))
    return rng.normal(0.0, scale, size=(in_dim, out_dim)).astype(np.float32)


def _deterministic_random_vector(key: str, dim: int) -> np.ndarray:
    import hashlib

    seed = int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "little")
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=dim).astype(np.float32)
