"""Input/output helpers for the isolated pure-SMILES experiment."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class SmilesPair:
    """One training pair for either source-target editing or self-supervision."""

    sample_id: str
    source_smiles: str
    target_smiles: str
    instruction: str = ""
    split: str = "train"
    metadata: dict[str, str] = field(default_factory=dict)


def read_smiles_pairs(
    csv_path: str | Path,
    *,
    source_column: str = "source_smiles",
    target_column: str = "target_smiles",
    smiles_column: str = "smiles",
    instruction_column: str = "instruction",
    id_column: str = "sample_id",
    split_column: str = "split",
    limit: int | None = None,
) -> list[SmilesPair]:
    """Read source-target pairs or self-supervised SMILES rows from CSV."""

    path = Path(csv_path)
    rows: list[SmilesPair] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        has_pair = source_column in fieldnames and target_column in fieldnames
        has_single = smiles_column in fieldnames
        if not has_pair and not has_single:
            raise ValueError(
                f"{path} must contain either {source_column}/{target_column} or {smiles_column}"
            )
        for index, row in enumerate(reader):
            if limit is not None and len(rows) >= limit:
                break
            if has_pair:
                source = (row.get(source_column) or "").strip()
                target = (row.get(target_column) or "").strip()
            else:
                source = (row.get(smiles_column) or "").strip()
                target = source
            if not source or not target:
                continue
            sample_id = (row.get(id_column) or row.get("condition_id") or f"row_{index:06d}").strip()
            rows.append(
                SmilesPair(
                    sample_id=sample_id,
                    source_smiles=source,
                    target_smiles=target,
                    instruction=(row.get(instruction_column) or row.get("prompt") or "").strip(),
                    split=(row.get(split_column) or "train").strip() or "train",
                    metadata={
                        key: str(value)
                        for key, value in row.items()
                        if value is not None and not _is_image_derived_column(key)
                    },
                )
            )
    return rows


def write_jsonl(rows: Iterable[Mapping[str, object]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_summary(summary: Mapping[str, object], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(summary), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _is_image_derived_column(name: str) -> bool:
    lower = str(name or "").strip().lower()
    if not lower:
        return False
    if "image" in lower:
        return True
    return lower.startswith(("img_", "hist_", "patch_"))
