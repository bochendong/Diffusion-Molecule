"""Runtime helpers for large unified 3M training jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve a user requested device string into a concrete torch device."""

    requested = str(requested or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {requested!r}, but torch.cuda.is_available() is false")
    if requested == "mps" and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):
        raise RuntimeError("Requested device 'mps', but torch.backends.mps.is_available() is false")
    return torch.device(requested)


def move_batch_to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move a dataloader batch of tensors onto the active device."""

    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def device_report(device: torch.device) -> dict[str, object]:
    """Return compact device metadata for logs and metrics files."""

    report: dict[str, object] = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        report.update(
            {
                "cuda_device_count": int(torch.cuda.device_count()),
                "cuda_device_name": torch.cuda.get_device_name(device if device.type == "cuda" else 0),
            }
        )
    return report


def checkpoint_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir) / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def latest_checkpoint_path(output_dir: str | Path) -> Path:
    return checkpoint_dir(output_dir) / "latest.pt"
