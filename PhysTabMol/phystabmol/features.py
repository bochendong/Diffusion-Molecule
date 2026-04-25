"""Image and table feature builders."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .chem import molecular_descriptors
from .schema import TABLE_COLUMNS

IMAGE_FEATURE_COLUMNS = [
    "img_mean",
    "img_std",
    "img_contrast",
    "img_edge_density",
    "img_texture",
    "hist_0",
    "hist_1",
    "hist_2",
    "hist_3",
    "patch_mean_min",
    "patch_mean_max",
    "patch_mean_std",
]


def extract_image_features(image_path: str | Path) -> dict[str, float]:
    arr = _load_grayscale(image_path)
    return image_array_features(arr)


def image_array_features(arr: np.ndarray) -> dict[str, float]:
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    gy, gx = np.gradient(arr)
    grad = np.sqrt(gx * gx + gy * gy)
    hist, _ = np.histogram(arr, bins=4, range=(0.0, 1.0), density=False)
    hist = hist.astype(np.float32) / max(1, hist.sum())
    patch_means = _patch_means(arr, grid=4)
    return {
        "img_mean": float(arr.mean()),
        "img_std": float(arr.std()),
        "img_contrast": float(arr.max() - arr.min()),
        "img_edge_density": float((grad > np.percentile(grad, 80)).mean()),
        "img_texture": float(grad.std()),
        "hist_0": float(hist[0]),
        "hist_1": float(hist[1]),
        "hist_2": float(hist[2]),
        "hist_3": float(hist[3]),
        "patch_mean_min": float(patch_means.min()),
        "patch_mean_max": float(patch_means.max()),
        "patch_mean_std": float(patch_means.std()),
    }


def table_row_from_smiles(smiles: str) -> dict[str, float]:
    record = molecular_descriptors(smiles)
    if not record.valid:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return {col: float(record.descriptors.get(col, 0.0)) for col in TABLE_COLUMNS}


def descriptor_image(smiles: str, size: int = 96) -> np.ndarray:
    """Create a deterministic proxy molecular image when RDKit drawing is absent.

    The image is not meant to be chemically faithful. It gives the contrastive
    module an image-conditioned signal in environments without RDKit.
    """

    row = table_row_from_smiles(smiles)
    seed = int(hashlib.sha256(smiles.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    bg = int(np.clip(230 - row["LogP"] * 10, 170, 245))
    img = Image.new("L", (size, size), color=bg)
    draw = ImageDraw.Draw(img)

    rings = max(1, int(row["ring_count"]) + 1)
    radius = max(8, min(30, int(row["MW"] / 18)))
    center = np.array([size // 2, size // 2], dtype=float)
    for idx in range(rings):
        sides = 5 + ((int(row["scaffold_class"]) + idx) % 3)
        angle0 = rng.uniform(0, np.pi)
        points = []
        local_radius = radius - idx * 4
        shift = rng.normal(0, 5, size=2)
        for k in range(sides):
            angle = angle0 + 2 * np.pi * k / sides
            p = center + shift + local_radius * np.array([np.cos(angle), np.sin(angle)])
            points.append(tuple(p))
        draw.line(points + [points[0]], fill=30, width=2)

    branches = int(row["HBA"] + row["HBD"] + row["fg_halogen"])
    for _ in range(max(2, branches)):
        start = center + rng.normal(0, radius / 2, size=2)
        length = rng.uniform(10, 28)
        angle = rng.uniform(0, 2 * np.pi)
        end = start + length * np.array([np.cos(angle), np.sin(angle)])
        draw.line([tuple(start), tuple(end)], fill=20, width=2)
        if rng.random() < 0.35:
            draw.ellipse([end[0] - 2, end[1] - 2, end[0] + 2, end[1] + 2], fill=0)

    noise = rng.normal(0, 4, size=(size, size))
    return np.clip(np.asarray(img, dtype=np.float32) + noise, 0, 255).astype(np.uint8)


def _load_grayscale(image_path: str | Path) -> np.ndarray:
    with Image.open(image_path) as img:
        return np.asarray(img.convert("L"), dtype=np.float32)


def _patch_means(arr: np.ndarray, grid: int) -> np.ndarray:
    h, w = arr.shape
    vals = []
    for y in range(grid):
        for x in range(grid):
            y0, y1 = int(y * h / grid), int((y + 1) * h / grid)
            x0, x1 = int(x * w / grid), int((x + 1) * w / grid)
            vals.append(float(arr[y0:y1, x0:x1].mean()))
    return np.asarray(vals, dtype=np.float32)

