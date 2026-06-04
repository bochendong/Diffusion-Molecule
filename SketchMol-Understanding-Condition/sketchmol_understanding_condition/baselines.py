"""Baseline manifest builders for understanding-condition experiments."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path


BASELINE_VARIANTS = (
    "full",
    "text_only",
    "image_only",
    "random_query",
    "caption_bottleneck",
)


@dataclass(frozen=True)
class BaselineVariantRow:
    variant_id: str
    pair_id: str
    split: str
    variant: str
    condition_mode: str
    use_source_image: bool
    use_instruction: bool
    source_image: str
    target_image: str
    source_smiles: str
    target_smiles: str
    instruction: str
    prompt: str
    scaffold: str
    similarity: str
    property_name: str
    property_delta: str
    objective: str
    direction: str


def build_baseline_rows(edit_pair_rows: list[dict[str, str]]) -> list[BaselineVariantRow]:
    """Expand edit-pair rows into baseline-specific condition rows."""

    rows: list[BaselineVariantRow] = []
    for row in edit_pair_rows:
        for variant in BASELINE_VARIANTS:
            rows.append(_build_variant_row(row, variant))
    return rows


def read_edit_pair_rows(path: str | Path) -> list[dict[str, str]]:
    """Read `build_edit_dataset.py` output rows."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_baseline_rows(path: str | Path, rows: list[BaselineVariantRow]) -> None:
    """Write baseline variant rows to CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(BaselineVariantRow.__dataclass_fields__.keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _build_variant_row(row: dict[str, str], variant: str) -> BaselineVariantRow:
    pair_id = row["pair_id"]
    instruction = row.get("instruction", "")
    source_smiles = row.get("source_smiles", "")
    target_smiles = row.get("target_smiles", "")
    scaffold = row.get("scaffold", "")
    property_name = row.get("property_name", "")
    property_delta = row.get("property_delta", "")

    if variant == "full":
        condition_mode = "mllm_image_text"
        use_source_image = True
        use_instruction = True
        prompt = instruction
    elif variant == "text_only":
        condition_mode = "mllm_text_only"
        use_source_image = False
        use_instruction = True
        prompt = instruction
    elif variant == "image_only":
        condition_mode = "mllm_image_only"
        use_source_image = True
        use_instruction = False
        prompt = "Preserve the visible molecular scaffold and make a valid local edit."
    elif variant == "random_query":
        condition_mode = "random_query_tokens"
        use_source_image = False
        use_instruction = False
        prompt = ""
    elif variant == "caption_bottleneck":
        condition_mode = "caption_bottleneck"
        use_source_image = False
        use_instruction = True
        prompt = _caption_bottleneck_prompt(
            source_smiles=source_smiles,
            target_smiles=target_smiles,
            scaffold=scaffold,
            instruction=instruction,
            property_name=property_name,
            property_delta=property_delta,
        )
    else:
        raise ValueError(f"Unsupported baseline variant: {variant}")

    return BaselineVariantRow(
        variant_id=f"{pair_id}:{variant}",
        pair_id=pair_id,
        split=row.get("split", ""),
        variant=variant,
        condition_mode=condition_mode,
        use_source_image=use_source_image,
        use_instruction=use_instruction,
        source_image=row.get("source_image", ""),
        target_image=row.get("target_image", ""),
        source_smiles=source_smiles,
        target_smiles=target_smiles,
        instruction=instruction,
        prompt=prompt,
        scaffold=scaffold,
        similarity=row.get("similarity", ""),
        property_name=property_name,
        property_delta=property_delta,
        objective=row.get("objective", property_name),
        direction=row.get("direction", ""),
    )


def _caption_bottleneck_prompt(
    *,
    source_smiles: str,
    target_smiles: str,
    scaffold: str,
    instruction: str,
    property_name: str,
    property_delta: str,
) -> str:
    parts = [
        f"Source molecule SMILES: {source_smiles}.",
        f"Shared scaffold: {scaffold or 'unknown'}.",
        f"Requested edit: {instruction}",
    ]
    if property_name:
        parts.append(f"The requested property objective is {property_name}.")
    parts.append("Generate an edited molecule image consistent with this textual caption.")
    return " ".join(parts)
