"""Optional RDKit-rendered molecule image context."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from .chem import RDKIT_AVAILABLE, render_molecule_image
from .schema import BenchmarkExample


def attach_rendered_image_context(examples: list[BenchmarkExample], out_dir: str | Path) -> tuple[list[BenchmarkExample], dict[str, object]]:
    """Render source/target molecules and attach image paths when possible.

    This is a lightweight proxy for SketchMol-style visual context. If RDKit or
    Pillow is unavailable, examples are returned unchanged and the run remains
    usable with descriptor/text features only.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    updated: list[BenchmarkExample] = []
    rendered = 0
    masked = 0
    skipped = 0
    for example in examples:
        if example.image_path:
            updated.append(example)
            skipped += 1
            continue
        smiles = example.source_smiles
        if not smiles:
            updated.append(example)
            skipped += 1
            continue
        raw_path = out_dir / f"{_safe_name(example.task_id)}_context.png"
        rendered_path = render_molecule_image(smiles, raw_path)
        if rendered_path is None:
            updated.append(example)
            skipped += 1
            continue
        rendered += 1
        final_path = rendered_path
        if example.mask_hint:
            masked_path = out_dir / f"{_safe_name(example.task_id)}_masked_context.png"
            if _apply_deterministic_mask(rendered_path, masked_path, example.mask_hint):
                final_path = str(masked_path)
                masked += 1
        updated.append(replace(example, image_path=final_path))
    return updated, {"rdkit_available": RDKIT_AVAILABLE, "rendered_images": rendered, "masked_images": masked, "skipped_images": skipped}


def _apply_deterministic_mask(image_path: str | Path, out_path: str | Path, hint: str) -> bool:
    try:
        from PIL import Image, ImageDraw

        image = Image.open(image_path).convert("RGBA")
        width, height = image.size
        digest = hashlib.sha256(hint.encode("utf-8")).digest()
        box_w = max(width // 5, width // 4 + digest[0] % max(1, width // 8))
        box_h = max(height // 5, height // 4 + digest[1] % max(1, height // 8))
        left = digest[2] % max(1, width - box_w)
        top = digest[3] % max(1, height - box_h)
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rectangle((left, top, left + box_w, top + box_h), fill=(0, 0, 0, 72), outline=(0, 0, 0, 180), width=2)
        Image.alpha_composite(image, overlay).convert("RGB").save(out_path)
        return True
    except Exception:
        return False


def _safe_name(text: str) -> str:
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
    return "".join(out).strip("_") or "example"
