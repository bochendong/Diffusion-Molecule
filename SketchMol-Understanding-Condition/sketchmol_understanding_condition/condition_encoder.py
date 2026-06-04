"""Unified condition encoder interfaces.

The proxy encoder wraps the current non-neural features behind the same API we
want for a future MLLM encoder:

    row + variant -> pooled_condition, query_tokens
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


class HfVlmConditionEncoder:
    """Frozen HuggingFace VLM hidden states for big-model conditioning."""

    name = "hf_vlm"

    def __init__(
        self,
        *,
        hf_model_name_or_path: str,
        hf_device_map: str = "auto",
        hf_dtype: str = "auto",
        hf_batch_size: int = 1,
        hf_max_length: int = 2048,
        hf_trust_remote_code: bool = True,
        hf_attn_implementation: str | None = None,
        hf_prompt_style: str = "auto",
        hf_render_image_size: int = 256,
        pooled_dim: int = 4096,
        num_queries: int = 32,
        query_dim: int = 256,
        **_: object,
    ) -> None:
        self.hf_model_name_or_path = hf_model_name_or_path
        self.hf_device_map = hf_device_map
        self.hf_dtype = hf_dtype
        self.hf_batch_size = max(1, int(hf_batch_size))
        self.hf_max_length = int(hf_max_length)
        self.hf_trust_remote_code = bool(hf_trust_remote_code)
        self.hf_attn_implementation = hf_attn_implementation
        self.hf_prompt_style = hf_prompt_style
        self.hf_render_image_size = int(hf_render_image_size)
        self.pooled_dim = pooled_dim
        self.num_queries = num_queries
        self.query_dim = query_dim
        self._processor, self._model = self._load_hf_model()
        self._projection = _fixed_projection(self.pooled_dim, self.num_queries * self.query_dim, seed=53)

    def encode_row(self, row: dict[str, str]) -> ConditionEncoding:
        return self.encode_rows([row])[0]

    def encode_rows(self, rows: list[dict[str, str]]) -> list[ConditionEncoding]:
        features: list[np.ndarray | None] = [None] * len(rows)
        pending_image: list[tuple[int, dict[str, str]]] = []
        pending_text: list[tuple[int, dict[str, str]]] = []

        for idx, row in enumerate(rows):
            variant = row.get("variant", "full")
            if variant == "random_query":
                features[idx] = _deterministic_random_vector(row.get("variant_id", ""), self.pooled_dim)
                continue
            if _variant_uses_image(row):
                pending_image.append((idx, row))
            else:
                pending_text.append((idx, row))

        self._encode_pending(pending_image, features, use_images=True)
        self._encode_pending(pending_text, features, use_images=False)

        out = []
        for row, feature in zip(rows, features):
            if feature is None:
                raise RuntimeError("Internal error: missing VLM feature")
            pooled = _resize_vector(np.asarray(feature, dtype=np.float32), self.pooled_dim)
            norm = float(np.linalg.norm(pooled))
            if norm > 0:
                pooled = pooled / norm
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

    def _load_hf_model(self):
        import torch
        import transformers
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            self.hf_model_name_or_path,
            trust_remote_code=self.hf_trust_remote_code,
        )
        model_kwargs: dict[str, object] = {"trust_remote_code": self.hf_trust_remote_code}
        dtype = _torch_dtype(torch, self.hf_dtype)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if self.hf_device_map and self.hf_device_map != "none":
            model_kwargs["device_map"] = self.hf_device_map
        if self.hf_attn_implementation:
            model_kwargs["attn_implementation"] = self.hf_attn_implementation

        errors = []
        for class_name in (
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForCausalLM",
            "AutoModel",
        ):
            model_cls = getattr(transformers, class_name, None)
            if model_cls is None:
                continue
            try:
                model = model_cls.from_pretrained(self.hf_model_name_or_path, **model_kwargs)
                model.eval()
                if not self.hf_device_map or self.hf_device_map == "none":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model.to(device)
                return processor, model
            except Exception as exc:  # pragma: no cover - depends on installed transformers/model
                errors.append(f"{class_name}: {exc}")
        detail = "\n".join(errors[-4:])
        raise RuntimeError(f"Could not load HuggingFace VLM {self.hf_model_name_or_path!r}.\n{detail}")

    def _encode_pending(
        self,
        pending: list[tuple[int, dict[str, str]]],
        features: list[np.ndarray | None],
        *,
        use_images: bool,
    ) -> None:
        if not pending:
            return
        import torch
        from PIL import Image

        input_device = _first_parameter_device(self._model)
        with torch.inference_mode():
            for start in range(0, len(pending), self.hf_batch_size):
                chunk = pending[start : start + self.hf_batch_size]
                prompts = [_vlm_prompt_for_row(row) for _, row in chunk]
                images = None
                if use_images:
                    images = [
                        _load_or_render_rgb_image(
                            row,
                            image_module=Image,
                            render_size=self.hf_render_image_size,
                        )
                        for _, row in chunk
                    ]
                inputs = _processor_call(
                    self._processor,
                    prompts=prompts,
                    images=images,
                    max_length=self.hf_max_length,
                    prompt_style=self.hf_prompt_style,
                    model_name=self.hf_model_name_or_path,
                )
                inputs = {
                    key: value.to(input_device) if torch.is_tensor(value) else value
                    for key, value in inputs.items()
                }
                try:
                    outputs = self._model(
                        **inputs,
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=False,
                    )
                except TypeError:
                    outputs = self._model(
                        **inputs,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                hidden = _last_hidden_state(outputs)
                attention_mask = inputs.get("attention_mask")
                pooled = _masked_mean_torch(hidden, attention_mask).float().cpu().numpy().astype(np.float32)
                for (idx, _), vec in zip(chunk, pooled):
                    features[idx] = vec


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
    if name == "hf_vlm":
        return HfVlmConditionEncoder(**kwargs)
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


def _variant_uses_image(row: dict[str, str]) -> bool:
    variant = row.get("variant", "full")
    if variant in {"full", "image_only"}:
        return bool(row.get("source_image") or row.get("source_smiles"))
    return False


def _vlm_prompt_for_row(row: dict[str, str]) -> str:
    variant = row.get("variant", "full")
    if variant == "image_only":
        return "Represent the molecular structure in this image for scaffold-preserving molecular editing."
    prompt = row.get("prompt") or row.get("instruction") or ""
    if variant == "caption_bottleneck":
        return prompt
    if variant == "text_only":
        return prompt
    return prompt or "Represent this molecule image and edit instruction for molecular generation."


def _torch_dtype(torch_module, name: str):
    if not name or name == "auto":
        return None
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch_module.float16
    if normalized in {"fp32", "float32"}:
        return torch_module.float32
    raise ValueError(f"Unsupported hf_dtype={name!r}")


def _first_parameter_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        import torch

        return torch.device("cpu")


def _load_or_render_rgb_image(row: dict[str, str], *, image_module, render_size: int):
    image_path_text = row.get("source_image", "")
    if image_path_text:
        image_path = Path(image_path_text)
        if image_path.exists():
            with image_module.open(image_path) as image:
                return image.convert("RGB").copy()
    smiles = row.get("source_smiles", "")
    if not smiles:
        raise FileNotFoundError(
            "Missing source image and source_smiles for hf_vlm export: "
            f"variant_id={row.get('variant_id', '')}"
        )
    return _render_smiles_to_pil(smiles, image_module=image_module, render_size=render_size)


def _render_smiles_to_pil(smiles: str, *, image_module, render_size: int):
    from rdkit import Chem
    from rdkit.Chem import Draw

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot render invalid source_smiles for hf_vlm export: {smiles!r}")
    image = Draw.MolToImage(mol, size=(render_size, render_size))
    if isinstance(image, image_module.Image):
        return image.convert("RGB").copy()
    raise TypeError("RDKit Draw.MolToImage did not return a PIL image")


def _processor_call(
    processor,
    *,
    prompts: list[str],
    images,
    max_length: int,
    prompt_style: str,
    model_name: str,
):
    prompts = _format_vlm_prompts(
        processor,
        prompts=prompts,
        images=images,
        prompt_style=prompt_style,
        model_name=model_name,
    )
    kwargs = {
        "text": prompts,
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
        "max_length": max_length,
    }
    if images is not None:
        kwargs["images"] = images
    try:
        return processor(**kwargs)
    except TypeError:
        kwargs.pop("truncation", None)
        kwargs.pop("max_length", None)
        return processor(**kwargs)


def _format_vlm_prompts(processor, *, prompts: list[str], images, prompt_style: str, model_name: str) -> list[str]:
    style = prompt_style
    if style == "auto":
        lower_name = model_name.lower()
        if "qwen" in lower_name and hasattr(processor, "apply_chat_template"):
            style = "qwen_chat"
        elif "llava" in lower_name:
            style = "llava"
        else:
            style = "plain"

    if style == "qwen_chat" and hasattr(processor, "apply_chat_template"):
        formatted = []
        for idx, prompt in enumerate(prompts):
            content = []
            if images is not None:
                content.append({"type": "image", "image": images[idx]})
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            formatted.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        return formatted
    if style == "llava" and images is not None:
        return [f"<image>\n{prompt}" for prompt in prompts]
    return prompts


def _last_hidden_state(outputs):
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states:
        return hidden_states[-1]
    last_hidden = getattr(outputs, "last_hidden_state", None)
    if last_hidden is not None:
        return last_hidden
    raise RuntimeError("The VLM output did not include hidden_states or last_hidden_state")


def _masked_mean_torch(hidden_states, attention_mask):
    if attention_mask is None:
        return hidden_states.mean(dim=1)
    mask = attention_mask.to(dtype=hidden_states.dtype, device=hidden_states.device).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden_states * mask).sum(dim=1) / denom
