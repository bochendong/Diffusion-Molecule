"""Optional adapter for the original SketchMol AutoencoderKL first-stage VAE."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


class SketchMolVAEAdapter(nn.Module):
    """Load SketchMol's original AutoencoderKL without modifying its repository."""

    def __init__(
        self,
        *,
        sketchmol_root: str | Path,
        config_path: str | Path,
        checkpoint_path: str | Path,
        scale_factor: float = 1.0,
    ) -> None:
        super().__init__()
        root = Path(sketchmol_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"SketchMol root does not exist: {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from omegaconf import OmegaConf
        from ldm.util import instantiate_from_config

        config = OmegaConf.load(str(config_path))
        first_stage_config = _first_stage_config(config)
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        first_stage_config.params.ckpt_path = None
        first_stage_config.params.lossconfig = {"target": "torch.nn.Identity"}

        self.model = instantiate_from_config(first_stage_config).eval()
        self.scale_factor = float(scale_factor)
        self.loaded_key_count = _load_checkpoint(self.model, checkpoint)
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def latent_dim(self) -> int:
        return 4 * 32 * 32

    def encode(self, images: torch.Tensor, *, sample: bool = False) -> torch.Tensor:
        posterior = self.model.encode(images)
        if hasattr(posterior, "sample") and sample:
            latent = posterior.sample()
        elif hasattr(posterior, "mode"):
            latent = posterior.mode()
        elif torch.is_tensor(posterior):
            latent = posterior
        else:
            raise TypeError(f"Unsupported SketchMol VAE posterior type: {type(posterior)}")
        return latent * self.scale_factor

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.model.decode(latents / self.scale_factor)


def load_sketchmol_vae_adapter(
    *,
    sketchmol_root: str | Path,
    config_path: str | Path,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
    scale_factor: float = 1.0,
) -> SketchMolVAEAdapter:
    """Load and freeze the original SketchMol first-stage VAE."""

    adapter = SketchMolVAEAdapter(
        sketchmol_root=sketchmol_root,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        scale_factor=scale_factor,
    )
    adapter.to(map_location)
    adapter.eval()
    return adapter


def _first_stage_config(config):
    model_config = config.model
    params = model_config.get("params", {})
    if "first_stage_config" in params:
        return params.first_stage_config
    return model_config


def _load_checkpoint(model: nn.Module, checkpoint_path: Path) -> int:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SketchMol VAE checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)}")

    model_keys = set(model.state_dict().keys())
    candidates: list[dict[str, torch.Tensor]] = []
    for prefix in ("first_stage_model.", "model.first_stage_model.", ""):
        stripped = {
            key[len(prefix) :] if prefix and key.startswith(prefix) else key: value
            for key, value in state.items()
            if not prefix or key.startswith(prefix)
        }
        matched = {key: value for key, value in stripped.items() if key in model_keys}
        if matched:
            candidates.append(matched)
    if not candidates:
        raise ValueError(f"No AutoencoderKL keys matched checkpoint: {checkpoint_path}")

    selected = max(candidates, key=len)
    model.load_state_dict(selected, strict=False)
    return len(selected)
