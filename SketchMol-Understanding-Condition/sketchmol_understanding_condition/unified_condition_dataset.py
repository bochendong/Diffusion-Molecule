"""Unified dataset helpers for molecular understanding and edit generation."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping


PROPERTY_COLUMNS = ("MW", "LogP", "QED", "TPSA", "HBD", "HBA", "RB")
DESCRIPTION_PRETRAIN = "description_pretrain"
EDIT_GENERATION = "edit_generation"


@dataclass
class UnifiedConditionSample:
    """One row for alignment, edit-token, or latent-diffusion training."""

    sample_id: str
    task_type: str
    split: str
    prompt: str
    target_smiles: str
    source_smiles: str = ""
    molecule_smiles: str = ""
    source_image: str = ""
    target_image: str = ""
    description: str = ""
    instruction: str = ""
    condition_properties: str = ""
    property_count: str = ""
    source_tanimoto: str = ""
    source_similarity_bin: str = ""
    source_properties: dict[str, float] = field(default_factory=dict)
    target_properties: dict[str, float] = field(default_factory=dict)
    property_deltas: dict[str, float] = field(default_factory=dict)
    active_properties: dict[str, bool] = field(default_factory=dict)
    directions: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "task_type": self.task_type,
            "split": self.split,
            "prompt": self.prompt,
            "target_smiles": self.target_smiles,
            "source_smiles": self.source_smiles,
            "molecule_smiles": self.molecule_smiles,
            "source_image": self.source_image,
            "target_image": self.target_image,
            "description": self.description,
            "instruction": self.instruction,
            "condition_properties": self.condition_properties,
            "property_count": self.property_count,
            "source_tanimoto": self.source_tanimoto,
            "source_similarity_bin": self.source_similarity_bin,
            "source_properties": self.source_properties,
            "target_properties": self.target_properties,
            "property_deltas": self.property_deltas,
            "active_properties": self.active_properties,
            "directions": self.directions,
            "metadata": self.metadata,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "UnifiedConditionSample":
        return cls(
            sample_id=str(value.get("sample_id", "")),
            task_type=str(value.get("task_type", "")),
            split=str(value.get("split", "")),
            prompt=str(value.get("prompt", "")),
            target_smiles=str(value.get("target_smiles", "")),
            source_smiles=str(value.get("source_smiles", "")),
            molecule_smiles=str(value.get("molecule_smiles", "")),
            source_image=str(value.get("source_image", "")),
            target_image=str(value.get("target_image", "")),
            description=str(value.get("description", "")),
            instruction=str(value.get("instruction", "")),
            condition_properties=str(value.get("condition_properties", "")),
            property_count=str(value.get("property_count", "")),
            source_tanimoto=str(value.get("source_tanimoto", "")),
            source_similarity_bin=str(value.get("source_similarity_bin", "")),
            source_properties=_float_dict(value.get("source_properties", {})),
            target_properties=_float_dict(value.get("target_properties", {})),
            property_deltas=_float_dict(value.get("property_deltas", {})),
            active_properties=_bool_dict(value.get("active_properties", {})),
            directions={str(k): str(v) for k, v in _dict_like(value.get("directions", {})).items()},
            metadata={str(k): str(v) for k, v in _dict_like(value.get("metadata", {})).items()},
        )


def read_jsonl(path: str | Path) -> list[UnifiedConditionSample]:
    """Read unified samples from JSONL."""

    samples = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append(UnifiedConditionSample.from_mapping(json.loads(line)))
    return samples


def write_jsonl(path: str | Path, samples: Iterable[UnifiedConditionSample]) -> None:
    """Write unified samples to JSONL."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_json_dict(), ensure_ascii=True, sort_keys=True) + "\n")


def read_3m_description_samples(
    path: str | Path,
    *,
    split: str,
    dataset_name: str,
    limit: int | None = None,
) -> list[UnifiedConditionSample]:
    """Read 3M-Diffusion style CID/SMILES/description rows."""

    samples = []
    with Path(path).open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for idx, row in enumerate(reader):
            smiles = (row.get("SMILES") or "").strip()
            description = (row.get("description") or "").strip()
            if not smiles or not description:
                continue
            cid = (row.get("CID") or str(idx)).strip()
            samples.append(
                UnifiedConditionSample(
                    sample_id=f"desc:{dataset_name}:{split}:{cid}",
                    task_type=DESCRIPTION_PRETRAIN,
                    split=split,
                    prompt=description,
                    target_smiles=smiles,
                    molecule_smiles=smiles,
                    description=description,
                    metadata={"dataset": dataset_name, "cid": cid},
                )
            )
            if limit is not None and len(samples) >= limit:
                break
    return samples


def read_edit_generation_samples(
    path: str | Path,
    *,
    dataset_name: str = "multiproperty_edit",
    limit: int | None = None,
) -> list[UnifiedConditionSample]:
    """Read diffusion edit manifest rows as source-conditioned edit samples."""

    samples = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            source_smiles = (row.get("source_smiles") or "").strip()
            target_smiles = (row.get("target_smiles") or "").strip()
            instruction = (row.get("instruction") or row.get("prompt") or "").strip()
            if not source_smiles or not target_smiles or not instruction:
                continue
            sample_id = row.get("sample_id") or row.get("condition_id") or f"edit:{idx}"
            samples.append(
                UnifiedConditionSample(
                    sample_id=f"edit:{dataset_name}:{sample_id}",
                    task_type=EDIT_GENERATION,
                    split=row.get("split", "train") or "train",
                    prompt=instruction,
                    target_smiles=target_smiles,
                    source_smiles=source_smiles,
                    molecule_smiles=source_smiles,
                    source_image=row.get("source_image", ""),
                    target_image=row.get("target_image", ""),
                    instruction=instruction,
                    condition_properties=row.get("condition_properties", ""),
                    property_count=row.get("property_count", ""),
                    source_tanimoto=row.get("source_tanimoto", ""),
                    source_similarity_bin=row.get("source_similarity_bin", ""),
                    source_properties=_properties_from_prefix(row, "source"),
                    target_properties=_properties_from_prefix(row, "target"),
                    property_deltas=_properties_from_prefix(row, "delta"),
                    active_properties={prop: _truthy(row.get(f"{prop}_active", "")) for prop in PROPERTY_COLUMNS},
                    directions={prop: row.get(f"{prop}_direction", "") for prop in PROPERTY_COLUMNS},
                    metadata={"dataset": dataset_name, "condition_id": row.get("condition_id", "")},
                )
            )
            if limit is not None and len(samples) >= limit:
                break
    return samples


def split_samples(
    samples: Iterable[UnifiedConditionSample],
    *,
    train_output: str | Path,
    eval_output: str | Path,
) -> dict[str, object]:
    """Write train/eval JSONL files and return a compact summary."""

    train = []
    eval_rows = []
    for sample in samples:
        if sample.split in {"eval", "valid", "validation", "test"}:
            eval_rows.append(sample)
        else:
            train.append(sample)
    write_jsonl(train_output, train)
    write_jsonl(eval_output, eval_rows)
    return summarize_samples([*train, *eval_rows], train_rows=len(train), eval_rows=len(eval_rows))


def summarize_samples(
    samples: list[UnifiedConditionSample],
    *,
    train_rows: int | None = None,
    eval_rows: int | None = None,
) -> dict[str, object]:
    """Build a summary suitable for dataset build logs."""

    by_task: dict[str, int] = {}
    by_split: dict[str, int] = {}
    by_property_count: dict[str, int] = {}
    unique_targets = set()
    unique_sources = set()
    for sample in samples:
        by_task[sample.task_type] = by_task.get(sample.task_type, 0) + 1
        by_split[sample.split] = by_split.get(sample.split, 0) + 1
        if sample.property_count:
            by_property_count[str(sample.property_count)] = by_property_count.get(str(sample.property_count), 0) + 1
        if sample.target_smiles:
            unique_targets.add(sample.target_smiles)
        if sample.source_smiles:
            unique_sources.add(sample.source_smiles)
    summary = {
        "rows": len(samples),
        "tasks": by_task,
        "splits": by_split,
        "unique_target_smiles": len(unique_targets),
        "unique_source_smiles": len(unique_sources),
        "property_counts": by_property_count,
    }
    if train_rows is not None:
        summary["train_rows"] = train_rows
    if eval_rows is not None:
        summary["eval_rows"] = eval_rows
    return summary


def _properties_from_prefix(row: Mapping[str, str], prefix: str) -> dict[str, float]:
    out = {}
    for prop in PROPERTY_COLUMNS:
        value = _to_float(row.get(f"{prefix}_{prop}", ""))
        if not math.isnan(value):
            out[prop] = value
    return out


def _dict_like(value: object) -> Mapping[object, object]:
    return value if isinstance(value, Mapping) else {}


def _float_dict(value: object) -> dict[str, float]:
    out = {}
    for key, item in _dict_like(value).items():
        parsed = _to_float(item)
        if not math.isnan(parsed):
            out[str(key)] = parsed
    return out


def _bool_dict(value: object) -> dict[str, bool]:
    return {str(key): _truthy(item) for key, item in _dict_like(value).items()}


def _truthy(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "active", "increase", "decrease"}


def _to_float(value: object) -> float:
    try:
        return float(str(value if value is not None else "").strip())
    except ValueError:
        return math.nan

