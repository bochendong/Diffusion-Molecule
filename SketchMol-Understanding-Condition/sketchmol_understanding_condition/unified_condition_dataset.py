"""Unified dataset helpers for molecular understanding and edit generation."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping


PROPERTY_COLUMNS = ("MW", "LogP", "QED", "TPSA", "HBD", "HBA", "RB", "SA")
TABLE1_TASK_SPECS = {
    frozenset({("GSK3B", "increase")}): "GSK3B:increase",
    frozenset({("RB", "decrease")}): "RB:decrease",
    frozenset({("MW", "increase")}): "MW:increase",
    frozenset({("SA", "decrease")}): "SA:decrease",
    frozenset({("HBA", "decrease"), ("SA", "decrease")}): "HBA:decrease+SA:decrease",
    frozenset({("QED", "increase"), ("SA", "decrease")}): "QED:increase+SA:decrease",
    frozenset({("HBA", "decrease"), ("LogP", "increase")}): "HBA:decrease+LogP:increase",
    frozenset({("HBA", "decrease"), ("MW", "decrease")}): "HBA:decrease+MW:decrease",
    frozenset({("DRD2", "decrease"), ("MW", "decrease"), ("SA", "decrease")}): (
        "DRD2:decrease+MW:decrease+SA:decrease"
    ),
    frozenset({("HBA", "increase"), ("MW", "increase"), ("QED", "decrease")}): (
        "HBA:increase+MW:increase+QED:decrease"
    ),
}
TABLE1_TASK_KEYS = set(TABLE1_TASK_SPECS.values())
DESCRIPTION_PRETRAIN = "description_pretrain"
EDIT_GENERATION = "edit_generation"
EDIT_QUALITY_METADATA_FIELDS = (
    "source_scaffold",
    "target_scaffold",
    "same_scaffold",
    "scaffold_relation",
    "pair_quality_tier",
    "selection_reason",
    "same_scaffold_neighbor_count",
    "source_neighbor_count_t04",
    "source_neighbor_count_t05",
    "source_neighbor_count_t06",
    "target_neighbor_rank_by_tanimoto",
    "candidate_pool_size_t04",
    "candidate_pool_size_t05",
    "candidate_pool_size_t06",
    "strict_candidate_count_t04",
    "strict_candidate_count_t05",
    "strict_candidate_count_t06",
    "oracle_candidate_smiles_t04",
    "oracle_source_tanimoto_t04",
    "oracle_strict_success_t04",
    "oracle_property_error_t04",
    "oracle_property_errors_json_t04",
    "source_identity_strict_success",
    "instruction_template_id",
    "instruction_style",
    "preservation_constraint",
    "property_constraints_json",
)


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
    min_source_tanimoto: float | None = None,
    require_quality_columns: bool = False,
    require_eval_oracle_strict: bool = False,
) -> list[UnifiedConditionSample]:
    """Read diffusion edit manifest rows as source-conditioned edit samples."""

    samples = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if require_quality_columns:
            missing = set(EDIT_QUALITY_METADATA_FIELDS) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Edit manifest missing quality columns: {sorted(missing)}")
        for idx, row in enumerate(reader):
            source_smiles = (row.get("source_smiles") or "").strip()
            target_smiles = (row.get("target_smiles") or "").strip()
            instruction = (row.get("instruction") or row.get("prompt") or "").strip()
            if not source_smiles or not target_smiles or not instruction:
                continue
            source_tanimoto = _to_float(row.get("source_tanimoto", ""))
            if min_source_tanimoto is not None and (
                math.isnan(source_tanimoto) or source_tanimoto < float(min_source_tanimoto)
            ):
                continue
            if (
                require_eval_oracle_strict
                and (row.get("split", "") or "train") in {"eval", "valid", "validation", "test"}
                and row.get("oracle_strict_success_t04", "").strip().lower() not in {"1", "true", "yes", "y"}
            ):
                raise ValueError(
                    "Eval edit row is not source-neighbor oracle-feasible: "
                    f"{row.get('condition_id') or row.get('sample_id') or idx}"
                )
            sample_id = row.get("sample_id") or row.get("condition_id") or f"edit:{idx}"
            metadata = {"dataset": dataset_name, "condition_id": row.get("condition_id", "")}
            metadata.update({field: row.get(field, "") for field in EDIT_QUALITY_METADATA_FIELDS if field in row})
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
                    metadata=metadata,
                )
            )
            if limit is not None and len(samples) >= limit:
                break
    return samples


def read_moledit_generation_samples(
    path: str | Path,
    *,
    split: str,
    dataset_name: str = "moledit_instruct",
    limit: int | None = None,
    min_source_tanimoto: float | None = None,
    table1_tasks_only: bool = False,
) -> list[UnifiedConditionSample]:
    """Read enhanced MolEdit-Instruct split rows as source-conditioned edits."""

    samples = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"example_id", "instruction", "source_smiles", "target_smiles"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"MolEdit split missing columns: {sorted(missing)}")
        for idx, row in enumerate(reader):
            source_smiles = (row.get("source_smiles") or "").strip()
            target_smiles = (row.get("target_smiles") or "").strip()
            instruction = (row.get("instruction") or "").strip()
            if not source_smiles or not target_smiles or not instruction:
                continue
            source_tanimoto = _to_float(row.get("source_target_tanimoto", ""))
            if min_source_tanimoto is not None and (
                math.isnan(source_tanimoto) or source_tanimoto < float(min_source_tanimoto)
            ):
                continue

            instruction_tasks = _parse_instruction_tasks(row.get("instruction_tasks", ""))
            task_specs = _task_specs_from_instruction(row, instruction_tasks)
            task_key = _task_key(task_specs)
            if table1_tasks_only and task_key not in TABLE1_TASK_KEYS:
                continue

            source_props = _properties_from_prefix(row, "source")
            target_props = _properties_from_prefix(row, "target")
            deltas = _properties_from_prefix(row, "delta")
            _fill_missing_computed_properties(source_props, target_props, deltas, source_smiles, target_smiles, task_specs)
            active_props = _active_props_from_moledit(row, task_specs, deltas)
            directions = _directions_from_moledit(row, task_specs, deltas)
            condition_props = [prop for prop in (spec.get("property", "") for spec in task_specs) if prop]
            model_props = [prop for prop in condition_props if prop in PROPERTY_COLUMNS]
            if not model_props:
                model_props = [prop for prop in PROPERTY_COLUMNS if active_props.get(prop, False)]

            example_id = row.get("example_id") or row.get("pair_hash") or str(idx)
            metadata = {
                "dataset": dataset_name,
                "condition_id": example_id,
                "example_id": example_id,
                "pair_hash": row.get("pair_hash", ""),
                "moledit_task_key": task_key,
                "moledit_tasks": json.dumps(task_specs, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
                "instruction_task_properties": row.get("instruction_task_properties", ""),
                "instruction_task_directions": row.get("instruction_task_directions", ""),
                "computed_active_properties": row.get("computed_active_properties", ""),
                "computed_active_count": row.get("computed_active_count", ""),
                "pair_quality_tier": row.get("pair_quality", ""),
                "difficulty_bucket": row.get("difficulty_bucket", ""),
                "source_valid": row.get("source_valid", ""),
                "target_valid": row.get("target_valid", ""),
                "source_canonical_smiles": row.get("source_canonical_smiles", ""),
                "target_canonical_smiles": row.get("target_canonical_smiles", ""),
                "source_scaffold": row.get("source_scaffold_smiles", ""),
                "target_scaffold": row.get("target_scaffold_smiles", ""),
                "same_scaffold": row.get("scaffold_match", ""),
                "preservation_constraint": "source_tanimoto",
            }
            samples.append(
                UnifiedConditionSample(
                    sample_id=f"edit:{dataset_name}:{example_id}",
                    task_type=EDIT_GENERATION,
                    split=split,
                    prompt=instruction,
                    target_smiles=target_smiles,
                    source_smiles=source_smiles,
                    molecule_smiles=source_smiles,
                    instruction=instruction,
                    condition_properties=",".join(model_props),
                    property_count=str(len(model_props)),
                    source_tanimoto=row.get("source_target_tanimoto", ""),
                    source_similarity_bin=row.get("difficulty_bucket", ""),
                    source_properties=source_props,
                    target_properties=target_props,
                    property_deltas=deltas,
                    active_properties=active_props,
                    directions=directions,
                    metadata=metadata,
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
    by_source_similarity_bin: dict[str, int] = {}
    by_pair_quality_tier: dict[str, int] = {}
    edit_source_tanimoto_values = []
    edit_eval_oracle_strict_false = 0
    unique_targets = set()
    unique_sources = set()
    for sample in samples:
        by_task[sample.task_type] = by_task.get(sample.task_type, 0) + 1
        by_split[sample.split] = by_split.get(sample.split, 0) + 1
        if sample.property_count:
            by_property_count[str(sample.property_count)] = by_property_count.get(str(sample.property_count), 0) + 1
        if sample.source_similarity_bin:
            by_source_similarity_bin[str(sample.source_similarity_bin)] = (
                by_source_similarity_bin.get(str(sample.source_similarity_bin), 0) + 1
            )
        pair_quality = sample.metadata.get("pair_quality_tier", "")
        if pair_quality:
            by_pair_quality_tier[pair_quality] = by_pair_quality_tier.get(pair_quality, 0) + 1
        if sample.task_type == EDIT_GENERATION:
            source_tanimoto = _to_float(sample.source_tanimoto)
            if not math.isnan(source_tanimoto):
                edit_source_tanimoto_values.append(source_tanimoto)
            if sample.split in {"eval", "valid", "validation", "test"} and sample.metadata.get(
                "oracle_strict_success_t04", ""
            ).lower() in {"0", "false", "no", "n"}:
                edit_eval_oracle_strict_false += 1
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
        "source_similarity_bins": by_source_similarity_bin,
        "pair_quality_tiers": by_pair_quality_tier,
        "edit_eval_oracle_strict_false": edit_eval_oracle_strict_false,
    }
    if edit_source_tanimoto_values:
        summary["edit_source_tanimoto"] = {
            "min": min(edit_source_tanimoto_values),
            "mean": sum(edit_source_tanimoto_values) / len(edit_source_tanimoto_values),
            "max": max(edit_source_tanimoto_values),
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


def _fill_missing_computed_properties(
    source_props: dict[str, float],
    target_props: dict[str, float],
    deltas: dict[str, float],
    source_smiles: str,
    target_smiles: str,
    task_specs: list[dict[str, str]],
) -> None:
    requested = {spec.get("property", "") for spec in task_specs}
    needed = {prop for prop in requested if prop in PROPERTY_COLUMNS}
    if not needed:
        return
    source_computed = _computed_properties(source_smiles) if any(prop not in source_props for prop in needed) else {}
    target_computed = _computed_properties(target_smiles) if any(prop not in target_props for prop in needed) else {}
    for prop in needed:
        if prop not in source_props and prop in source_computed:
            source_props[prop] = source_computed[prop]
        if prop not in target_props and prop in target_computed:
            target_props[prop] = target_computed[prop]
        if prop not in deltas and prop in source_props and prop in target_props:
            deltas[prop] = float(target_props[prop]) - float(source_props[prop])


def _computed_properties(smiles: str) -> dict[str, float]:
    try:
        from .chem import molecular_properties
    except Exception:
        return {}
    try:
        props = molecular_properties(smiles)
    except RuntimeError:
        return {}
    if not props:
        return {}
    out = {
        "MW": props.get("MolWt", math.nan),
        "LogP": props.get("LogP", math.nan),
        "QED": props.get("QED", math.nan),
        "TPSA": props.get("TPSA", math.nan),
        "HBD": props.get("HBD", math.nan),
        "HBA": props.get("HBA", math.nan),
        "RB": props.get("rotatable", math.nan),
        "SA": props.get("SA", math.nan),
    }
    return {key: float(value) for key, value in out.items() if value is not None and not math.isnan(float(value))}


def _parse_instruction_tasks(value: object) -> list[dict[str, str]]:
    try:
        parsed = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    out = []
    for item in parsed:
        if not isinstance(item, Mapping):
            continue
        prop = str(item.get("property", "")).strip()
        direction = str(item.get("direction", "")).strip().lower()
        if prop:
            out.append({"property": prop, "direction": direction or "unknown"})
    return out


def _task_specs_from_instruction(
    row: Mapping[str, str],
    instruction_tasks: list[dict[str, str]],
) -> list[dict[str, str]]:
    if instruction_tasks:
        return instruction_tasks
    specs = []
    raw_props = row.get("instruction_task_properties", "") or row.get("computed_active_properties", "")
    directions = _parse_json_mapping(row.get("instruction_task_directions", ""))
    for prop in [item for item in _split_props(raw_props) if item]:
        direction = str(directions.get(prop, "") or row.get(f"{prop}_direction", "") or "unknown").lower()
        specs.append({"property": prop, "direction": direction})
    return specs


def _split_props(value: object) -> list[str]:
    text = str(value or "").replace(",", "|")
    return [part.strip() for part in text.split("|") if part.strip()]


def _parse_json_mapping(value: object) -> dict[str, object]:
    try:
        parsed = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _task_key(task_specs: list[dict[str, str]]) -> str:
    pairs = [
        (spec.get("property", ""), str(spec.get("direction", "") or "unknown").lower())
        for spec in task_specs
        if spec.get("property")
    ]
    canonical = TABLE1_TASK_SPECS.get(frozenset(pairs))
    if canonical:
        return canonical
    return "+".join(f"{prop}:{direction}" for prop, direction in sorted(pairs))


def _active_props_from_moledit(
    row: Mapping[str, str],
    task_specs: list[dict[str, str]],
    deltas: Mapping[str, float],
) -> dict[str, bool]:
    task_props = {spec.get("property", "") for spec in task_specs}
    active = {}
    for prop in PROPERTY_COLUMNS:
        value = row.get(f"{prop}_active", "")
        if value != "":
            active[prop] = _truthy(value)
        elif prop in task_props:
            active[prop] = True
        else:
            active[prop] = prop in deltas and abs(float(deltas[prop])) > 1e-8
    return active


def _directions_from_moledit(
    row: Mapping[str, str],
    task_specs: list[dict[str, str]],
    deltas: Mapping[str, float],
) -> dict[str, str]:
    task_directions = {
        spec.get("property", ""): str(spec.get("direction", "") or "").lower()
        for spec in task_specs
    }
    out = {}
    for prop in PROPERTY_COLUMNS:
        direction = task_directions.get(prop) or row.get(f"{prop}_direction", "")
        if not direction and prop in deltas:
            direction = "increase" if float(deltas[prop]) >= 0 else "decrease"
        out[prop] = direction
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
