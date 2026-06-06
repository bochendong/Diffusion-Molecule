"""Molecule image VAE backend with SketchMol-style latent shape."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoleculeImageVAEOutput:
    """Reconstruction and posterior statistics."""

    reconstruction: torch.Tensor
    latent: torch.Tensor
    mean: torch.Tensor
    logvar: torch.Tensor


class MoleculeImageVAE(nn.Module):
    """Lightweight image VAE whose latent matches SketchMol's 4x32x32 interface."""

    def __init__(
        self,
        *,
        image_size: int = 256,
        latent_channels: int = 4,
        latent_size: int = 32,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        if image_size % latent_size != 0:
            raise ValueError("image_size must be divisible by latent_size")
        downsample_factor = image_size // latent_size
        if downsample_factor != 8:
            raise ValueError("This VAE currently expects image_size / latent_size == 8")
        self.image_size = int(image_size)
        self.latent_channels = int(latent_channels)
        self.latent_size = int(latent_size)
        self.base_channels = int(base_channels)

        ch = base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1),
            nn.SiLU(),
            ResidualBlock(ch),
            Downsample(ch, ch * 2),
            ResidualBlock(ch * 2),
            Downsample(ch * 2, ch * 4),
            ResidualBlock(ch * 4),
            Downsample(ch * 4, ch * 4),
            ResidualBlock(ch * 4),
        )
        self.to_posterior = nn.Conv2d(ch * 4, latent_channels * 2, 1)
        self.from_latent = nn.Conv2d(latent_channels, ch * 4, 1)
        self.decoder = nn.Sequential(
            ResidualBlock(ch * 4),
            Upsample(ch * 4, ch * 4),
            ResidualBlock(ch * 4),
            Upsample(ch * 4, ch * 2),
            ResidualBlock(ch * 2),
            Upsample(ch * 2, ch),
            ResidualBlock(ch),
            nn.Conv2d(ch, 3, 3, padding=1),
            nn.Tanh(),
        )

    @property
    def latent_dim(self) -> int:
        return self.latent_channels * self.latent_size * self.latent_size

    def encode_distribution(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(images)
        mean, logvar = self.to_posterior(hidden).chunk(2, dim=1)
        return mean, logvar.clamp(-30.0, 20.0)

    def encode(self, images: torch.Tensor, *, sample: bool = False) -> torch.Tensor:
        mean, logvar = self.encode_distribution(images)
        if not sample:
            return mean
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.from_latent(latents))

    def forward(self, images: torch.Tensor, *, sample: bool = True) -> MoleculeImageVAEOutput:
        mean, logvar = self.encode_distribution(images)
        if sample:
            std = torch.exp(0.5 * logvar)
            latent = mean + std * torch.randn_like(std)
        else:
            latent = mean
        reconstruction = self.decode(latent)
        return MoleculeImageVAEOutput(reconstruction=reconstruction, latent=latent, mean=mean, logvar=logvar)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(value)


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(value)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value, scale_factor=2, mode="nearest")
        return self.conv(value)


def vae_loss(
    output: MoleculeImageVAEOutput,
    target: torch.Tensor,
    *,
    kl_weight: float = 1e-6,
    foreground_weight: float = 8.0,
    foreground_gamma: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Foreground-aware reconstruction plus small KL regularization.

    Molecule drawings are mostly white canvas. Plain pixel L1 can look good while
    the model learns to output blank white images, so dark/ink pixels receive
    extra reconstruction weight.
    """

    abs_error = (output.reconstruction - target).abs()
    target_ink = molecule_ink_mask(target)
    weights = 1.0 + float(foreground_weight) * target_ink.pow(float(foreground_gamma))
    reconstruction_loss = (abs_error * weights).sum() / weights.expand_as(abs_error).sum().clamp_min(1e-6)
    kl = -0.5 * torch.mean(1.0 + output.logvar - output.mean.pow(2) - output.logvar.exp())
    loss = reconstruction_loss + float(kl_weight) * kl
    foreground_mask = target_ink > 0.05
    foreground_error = abs_error[foreground_mask.expand_as(abs_error)]
    background_error = abs_error[(~foreground_mask).expand_as(abs_error)]
    return loss, {
        "loss": loss.detach(),
        "reconstruction_l1": reconstruction_loss.detach(),
        "foreground_l1": foreground_error.mean().detach() if foreground_error.numel() else target.new_tensor(0.0),
        "background_l1": background_error.mean().detach() if background_error.numel() else target.new_tensor(0.0),
        "blank_canvas_l1": F.l1_loss(torch.ones_like(target), target).detach(),
        "target_ink_fraction": (target_ink > 0.05).to(dtype=target.dtype).mean().detach(),
        "reconstruction_ink_fraction": (molecule_ink_mask(output.reconstruction) > 0.05).to(dtype=target.dtype).mean().detach(),
        "kl": kl.detach(),
    }


def molecule_ink_mask(images: torch.Tensor) -> torch.Tensor:
    """Return a soft ink mask where dark molecule strokes are near 1."""

    luminance = ((images.clamp(-1.0, 1.0) + 1.0) * 0.5).mean(dim=1, keepdim=True)
    return (1.0 - luminance).clamp(0.0, 1.0)


def save_molecule_image_vae(
    path: str | Path,
    model: MoleculeImageVAE,
    *,
    config: dict[str, object] | None = None,
    metrics: dict[str, object] | None = None,
) -> None:
    """Save a VAE checkpoint with shape metadata."""

    payload = {
        "model_state": model.state_dict(),
        "config": {
            "image_size": model.image_size,
            "latent_channels": model.latent_channels,
            "latent_size": model.latent_size,
            "base_channels": model.base_channels,
            **(config or {}),
        },
        "metrics": metrics or {},
    }
    torch.save(payload, path)


def load_molecule_image_vae(path: str | Path, *, map_location: str | torch.device = "cpu") -> MoleculeImageVAE:
    """Load a SUCC molecule image VAE checkpoint."""

    payload = torch.load(path, map_location=map_location)
    config = dict(payload.get("config", {}))
    model = MoleculeImageVAE(
        image_size=int(config.get("image_size", 256)),
        latent_channels=int(config.get("latent_channels", 4)),
        latent_size=int(config.get("latent_size", 32)),
        base_channels=int(config.get("base_channels", 64)),
    )
    state = payload.get("model_state", payload)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def image_to_tensor(
    *,
    image_path: str = "",
    smiles: str = "",
    image_size: int = 256,
) -> torch.Tensor:
    """Load or render a molecule image as `[3,H,W]` in `[-1,1]`."""

    image = _load_or_render_pil(image_path=image_path, smiles=smiles, image_size=image_size)
    arr = np.asarray(image.convert("RGB").resize((image_size, image_size)), dtype=np.float32)
    arr = arr / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a decoded `[-1,1]` image tensor to uint8 HWC."""

    tensor = tensor.detach().float().cpu().clamp(-1.0, 1.0)
    arr = ((tensor.permute(1, 2, 0).numpy() + 1.0) * 127.5).round()
    return arr.clip(0, 255).astype(np.uint8)


def _load_or_render_pil(*, image_path: str, smiles: str, image_size: int):
    from PIL import Image

    if image_path:
        path = Path(image_path)
        if path.exists():
            with Image.open(path) as image:
                return image.convert("RGB")
    if smiles:
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
        except ImportError as exc:
            raise RuntimeError("RDKit is required to render SMILES when image_path is missing") from exc
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Cannot render invalid SMILES: {smiles!r}")
        image = Draw.MolToImage(mol, size=(image_size, image_size))
        if not isinstance(image, Image.Image):
            raise TypeError("RDKit Draw.MolToImage did not return a PIL image")
        return image.convert("RGB")
    raise ValueError("Either image_path or smiles must be provided")


def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1
