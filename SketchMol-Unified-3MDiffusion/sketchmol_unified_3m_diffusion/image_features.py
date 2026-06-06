"""OCR-free molecule image feature extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np


IMAGE_STATS_DIM = 32


def image_feature_vector(path: str | None, dim: int = 128) -> np.ndarray:
    """Return deterministic visual features from a rendered molecule image."""

    out = np.zeros(dim, dtype=np.float32)
    stats = _image_stats(path)
    if stats is None:
        return out
    values = np.asarray(stats, dtype=np.float32)
    out[: min(dim, len(values))] = values[:dim]
    return _normalize(out)


def image_patch_feature_vector(path: str | None, dim: int = 256) -> np.ndarray:
    """Return richer fixed-convolution patch features from a molecule image."""

    arr = _load_grayscale(path, size=128)
    if arr is None:
        return np.zeros(dim, dtype=np.float32)

    ink = 1.0 - arr
    sobel_x = _conv2d(arr, np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32))
    sobel_y = _conv2d(arr, np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32))
    lap = _conv2d(arr, np.asarray([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32))
    diag_a = _conv2d(arr, np.asarray([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype=np.float32))
    diag_b = _conv2d(arr, np.asarray([[-1, 0, 1], [0, 0, 0], [1, 0, -1]], dtype=np.float32))
    grad = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)

    channels = [arr, ink, np.abs(sobel_x), np.abs(sobel_y), np.abs(lap), np.abs(diag_a), np.abs(diag_b), grad]
    patch_feats = []
    for channel in channels:
        patch_feats.extend(_patch_stats(channel, grid=8))
    lowres = _patch_means(ink, grid=16)
    global_stats = np.asarray(_image_stats(path) or [], dtype=np.float32)
    raw = np.concatenate([np.asarray(patch_feats, dtype=np.float32), lowres, global_stats])
    return _normalize(_project_or_pad(raw, dim, seed=43))


def _image_stats(path: str | None) -> list[float] | None:
    arr = _load_grayscale(path, size=128)
    if arr is None:
        return None

    gy, gx = np.gradient(arr)
    grad = np.sqrt(gx * gx + gy * gy)
    hist, _ = np.histogram(arr, bins=8, range=(0.0, 1.0), density=False)
    hist = hist.astype(np.float32) / max(1, int(hist.sum()))
    patch_means = _patch_means(arr, grid=4)
    ink = 1.0 - arr
    ink_mass = float(ink.mean())
    ys, xs = np.nonzero(ink > np.percentile(ink, 80))
    if len(xs) > 0 and len(ys) > 0:
        bbox_w = float((xs.max() - xs.min() + 1) / arr.shape[1])
        bbox_h = float((ys.max() - ys.min() + 1) / arr.shape[0])
    else:
        bbox_w = 0.0
        bbox_h = 0.0

    return [
        float(arr.mean()),
        float(arr.std()),
        float(arr.max() - arr.min()),
        float(grad.mean()),
        float(grad.std()),
        float((grad > np.percentile(grad, 80)).mean()),
        float((arr < 0.2).mean()),
        float((arr > 0.8).mean()),
        ink_mass,
        bbox_w,
        bbox_h,
        float(patch_means.min()),
        float(patch_means.max()),
        float(patch_means.std()),
        *[float(v) for v in hist],
        *[float(v) for v in patch_means[:10]],
    ]


def _patch_means(arr: np.ndarray, grid: int) -> np.ndarray:
    h, w = arr.shape
    vals = []
    for y in range(grid):
        for x in range(grid):
            y0 = y * h // grid
            y1 = (y + 1) * h // grid
            x0 = x * w // grid
            x1 = (x + 1) * w // grid
            vals.append(float(arr[y0:y1, x0:x1].mean()))
    return np.asarray(vals, dtype=np.float32)


def _patch_stats(arr: np.ndarray, grid: int) -> list[float]:
    h, w = arr.shape
    vals = []
    for y in range(grid):
        for x in range(grid):
            y0 = y * h // grid
            y1 = (y + 1) * h // grid
            x0 = x * w // grid
            x1 = (x + 1) * w // grid
            patch = arr[y0:y1, x0:x1]
            vals.extend([float(patch.mean()), float(patch.std())])
    return vals


def _load_grayscale(path: str | None, *, size: int) -> np.ndarray | None:
    if not path:
        return None
    image_path = Path(path)
    if not image_path.exists():
        return None
    try:
        from PIL import Image

        image = Image.open(image_path).convert("L").resize((size, size))
        return np.asarray(image, dtype=np.float32) / 255.0
    except Exception:
        return None


def _conv2d(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2
    padded = np.pad(arr, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
    out = np.zeros_like(arr, dtype=np.float32)
    for y in range(kernel.shape[0]):
        for x in range(kernel.shape[1]):
            out += kernel[y, x] * padded[y : y + arr.shape[0], x : x + arr.shape[1]]
    return out


def _project_or_pad(values: np.ndarray, dim: int, *, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] == dim:
        return values
    if values.shape[0] < dim:
        out = np.zeros(dim, dtype=np.float32)
        out[: values.shape[0]] = values
        return out
    rng = np.random.default_rng(seed)
    projection = rng.normal(0.0, 1.0 / np.sqrt(values.shape[0]), size=(values.shape[0], dim)).astype(np.float32)
    return (values @ projection).astype(np.float32)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return (vec / norm).astype(np.float32) if norm > 0 else vec.astype(np.float32)
