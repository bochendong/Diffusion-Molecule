"""MolScribe-oriented molecule image rendering and preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def preprocess_image_for_molscribe(image_rgb: np.ndarray) -> np.ndarray:
    """Convert soft or colored molecule renders to high-contrast black-on-white arrays."""

    import cv2

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {image_rgb.shape}")

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    foreground = gray < 250

    if float(foreground.mean()) < 0.005:
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if float(otsu.mean()) < 127.0:
            otsu = 255 - otsu
        foreground = otsu < 127

    binary = np.where(foreground, 0, 255).astype(np.uint8)
    if foreground.any():
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.erode(binary, kernel, iterations=1)

    cropped = _crop_white(binary, pad=5)
    return cv2.cvtColor(cropped, cv2.COLOR_GRAY2RGB)


def _crop_white(image: np.ndarray, *, pad: int = 5) -> np.ndarray:
    """Trim large white margins and add a small border, similar to MolScribe CropWhite."""

    import cv2

    if image.ndim == 3:
        mask = (image != 255).any(axis=2)
    else:
        mask = image != 255
    if not mask.any():
        return image

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    cropped = image[top:bottom, left:right]
    if pad > 0:
        if cropped.ndim == 3:
            cropped = cv2.copyMakeBorder(cropped, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        else:
            cropped = cv2.copyMakeBorder(cropped, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    return cropped


def load_rgb_image(path: str | Path) -> np.ndarray:
    """Load an image file as an HWC RGB uint8 array."""

    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_preprocessed_rgb_image(path: str | Path, *, preprocess: bool = True) -> np.ndarray:
    """Load an image and optionally convert it to MolScribe-friendly black-on-white."""

    image = load_rgb_image(path)
    if preprocess:
        return preprocess_image_for_molscribe(image)
    return image
