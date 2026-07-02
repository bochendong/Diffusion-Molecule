#!/usr/bin/env python3
"""Export source-conditioned external multi-property benchmark rows.

This adapter targets GeLLMO/MuMOInstruct and GeLLMO-C/C-MuMOInstruct style
source-conditioned optimization tasks.  It intentionally keeps the official
task properties in metadata even when the local SUCC numeric property program
does not yet have a native oracle for them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sketchmol_understanding_condition.chem import canonical_smiles, molecular_properties  # noqa: E402
from sketchmol_understanding_condition.unified_condition_dataset import PROPERTY_COLUMNS  # noqa: E402


LOCAL_PROXY_PROPERTY_MAP = {
    "plogp": "LogP",
    "qed": "QED",
    "sas": "SA",
}
LOCAL_PROPERTY_VALUE_KEYS = {
    "MW": "MolWt",
    "LogP": "LogP",
    "QED": "QED",
    "TPSA": "TPSA",
    "HBD": "HBD",
    "HBA": "HBA",
    "RB": "rotatable",
    "SA": "SA",
}
MUMO_THRESHOLDS = {
    "bbbp": 0.2,
    "drd2": 0.2,
    "hia": 0.1,
    "mutagenicity": 0.1,
    "plogp": 1.0,
    "qed": 0.1,
}
CMUMO_THRESHOLDS = {
    "ampa": 0.1,
    "bbbp": 0.1,
    "carc": 0.1,
    "drd2": 0.2,
    "erg": 0.2,
    "hia": 0.1,
    "liver": 0.1,
    "mutagenicity": 0.1,
    "plogp": 1.0,
    "qed": 0.1,
}
DEFAULT_DIRECTION = {
    "ampa": "increase",
    "bbbp": "increase",
    "drd2": "increase",
    "hia": "increase",
    "plogp": "increase",
    "qed": "increase",
    "carc": "decrease",
    "erg": "decrease",
    "liver": "decrease",
    "mutagenicity": "decrease",
}
DEFAULT_OBJECTIVE = {
    "ampa": "improve",
    "bbbp": "improve",
    "carc": "improve",
    "drd2": "improve",
    "erg": "improve",
    "hia": "improve",
    "liver": "improve",
    "mutagenicity": "improve",
    "plogp": "improve",
    "qed": "improve",
}
PROPERTY_DISPLAY_NAMES = {
    "ampa": "membrane permeability",
    "bbbp": "BBB permeability",
    "carc": "carcinogenicity",
    "drd2": "DRD2 inhibition",
    "erg": "hERG inhibition",
    "hia": "human intestinal absorption",
    "liver": "liver injury risk",
    "mutagenicity": "mutagenicity",
    "plogp": "penalized logP",
    "qed": "QED",
    "sas": "synthetic accessibility",
}


@dataclass(frozen=True)
class ExternalTaskSpec:
    suite: str
    task_id: str
    task_key: str
    split: str
    properties: tuple[str, ...]
    description: str

    @property
    def thresholds(self) -> Mapping[str, float]:
        return CMUMO_THRESHOLDS if self.suite == "cmumo" else MUMO_THRESHOLDS

    @property
    def directions(self) -> dict[str, str]:
        return {prop: DEFAULT_DIRECTION[prop] for prop in self.properties}

    @property
    def objectives(self) -> dict[str, str]:
        return {prop: DEFAULT_OBJECTIVE[prop] for prop in self.properties}


TASK_SPECS: tuple[ExternalTaskSpec, ...] = (
    ExternalTaskSpec("mumo", "BDP", "bbbp+drd2+plogp", "ind", ("bbbp", "drd2", "plogp"), "MuMO in-domain"),
    ExternalTaskSpec("mumo", "BDQ", "bbbp+drd2+qed", "ind", ("bbbp", "drd2", "qed"), "MuMO in-domain"),
    ExternalTaskSpec("mumo", "BPQ", "bbbp+plogp+qed", "ind", ("bbbp", "plogp", "qed"), "MuMO in-domain"),
    ExternalTaskSpec("mumo", "DPQ", "drd2+plogp+qed", "ind", ("drd2", "plogp", "qed"), "MuMO in-domain"),
    ExternalTaskSpec(
        "mumo",
        "BDPQ",
        "bbbp+drd2+plogp+qed",
        "ind",
        ("bbbp", "drd2", "plogp", "qed"),
        "MuMO in-domain",
    ),
    ExternalTaskSpec(
        "mumo",
        "MPQ",
        "mutagenicity+plogp+qed",
        "ood",
        ("mutagenicity", "plogp", "qed"),
        "MuMO OOD",
    ),
    ExternalTaskSpec(
        "mumo",
        "BDMQ",
        "bbbp+drd2+mutagenicity+qed",
        "ood",
        ("bbbp", "drd2", "mutagenicity", "qed"),
        "MuMO OOD",
    ),
    ExternalTaskSpec(
        "mumo",
        "BHMQ",
        "bbbp+hia+mutagenicity+qed",
        "ood",
        ("bbbp", "hia", "mutagenicity", "qed"),
        "MuMO OOD",
    ),
    ExternalTaskSpec(
        "mumo",
        "BMPQ",
        "bbbp+mutagenicity+plogp+qed",
        "ood",
        ("bbbp", "mutagenicity", "plogp", "qed"),
        "MuMO OOD",
    ),
    ExternalTaskSpec(
        "mumo",
        "HMPQ",
        "hia+mutagenicity+plogp+qed",
        "ood",
        ("hia", "mutagenicity", "plogp", "qed"),
        "MuMO OOD",
    ),
    ExternalTaskSpec("cmumo", "BPQ", "bbbp+plogp+qed", "ind", ("bbbp", "plogp", "qed"), "C-MuMO IND"),
    ExternalTaskSpec("cmumo", "ELQ", "erg+liver+qed", "ind", ("erg", "liver", "qed"), "C-MuMO IND"),
    ExternalTaskSpec(
        "cmumo",
        "ACEP",
        "ampa+carc+erg+plogp",
        "ind",
        ("ampa", "carc", "erg", "plogp"),
        "C-MuMO IND",
    ),
    ExternalTaskSpec(
        "cmumo",
        "BDPQ",
        "bbbp+drd2+plogp+qed",
        "ind",
        ("bbbp", "drd2", "plogp", "qed"),
        "C-MuMO IND",
    ),
    ExternalTaskSpec(
        "cmumo",
        "DHMQ",
        "drd2+hia+mutagenicity+qed",
        "ind",
        ("drd2", "hia", "mutagenicity", "qed"),
        "C-MuMO IND",
    ),
    ExternalTaskSpec("cmumo", "CDE", "carc+drd2+erg", "ood", ("carc", "drd2", "erg"), "C-MuMO OOD"),
    ExternalTaskSpec(
        "cmumo",
        "ABMP",
        "ampa+bbbp+mutagenicity+plogp",
        "ood",
        ("ampa", "bbbp", "mutagenicity", "plogp"),
        "C-MuMO OOD",
    ),
    ExternalTaskSpec(
        "cmumo",
        "BCMQ",
        "bbbp+carc+mutagenicity+qed",
        "ood",
        ("bbbp", "carc", "mutagenicity", "qed"),
        "C-MuMO OOD",
    ),
    ExternalTaskSpec(
        "cmumo",
        "BDEQ",
        "bbbp+drd2+erg+qed",
        "ood",
        ("bbbp", "drd2", "erg", "qed"),
        "C-MuMO OOD",
    ),
    ExternalTaskSpec(
        "cmumo",
        "HLMPQ",
        "hia+liver+mutagenicity+plogp+qed",
        "ood",
        ("hia", "liver", "mutagenicity", "plogp", "qed"),
        "C-MuMO OOD",
    ),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--task-spec-json", type=Path, default=None)
    parser.add_argument("--suite", choices=("mumo", "cmumo", "both"), default="both")
    parser.add_argument("--task-split", choices=("ind", "ood", "all"), default="all")
    parser.add_argument("--tasks", default="", help="Comma-separated task ids or task keys to keep.")
    parser.add_argument("--max-rows-per-task", type=int, default=0, help="0 keeps all rows.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--input-split",
        default="all",
        help="Comma-separated raw dataset splits to keep, e.g. train,test,eval. Default keeps all input rows.",
    )
    parser.add_argument("--source-smiles-column", default=None)
    parser.add_argument("--target-smiles-column", default=None)
    parser.add_argument("--id-column", default=None)
    parser.add_argument(
        "--respect-input-task",
        action="store_true",
        help="If input rows include a task field, do not replicate them across other selected tasks.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_rows = read_source_rows(args.source_file)
    source_rows = filter_source_rows_by_input_split(source_rows, args.input_split)
    if not source_rows:
        raise ValueError(f"No rows found in {args.source_file} for input split {args.input_split!r}")
    specs = select_specs(args.suite, args.task_split, args.tasks)
    if not specs:
        raise ValueError("No external task specs selected")

    rng = random.Random(int(args.seed))
    output_rows = build_rows(
        source_rows,
        specs=specs,
        rng=rng,
        max_rows_per_task=int(args.max_rows_per_task),
        source_smiles_column=args.source_smiles_column,
        target_smiles_column=args.target_smiles_column,
        id_column=args.id_column,
        respect_input_task=bool(args.respect_input_task),
    )
    if not output_rows:
        raise ValueError("No external benchmark rows were exported")

    write_rows(args.output_csv, output_rows)
    summary = summarize_rows(args, specs, output_rows, source_row_count=len(source_rows))
    summary_path = args.summary_json or args.output_csv.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    task_spec_path = args.task_spec_json or args.output_csv.with_suffix(".task_specs.json")
    task_spec_path.parent.mkdir(parents=True, exist_ok=True)
    task_spec_path.write_text(json.dumps([task_spec_dict(spec) for spec in specs], indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def select_specs(suite: str, task_split: str, tasks: str) -> list[ExternalTaskSpec]:
    task_filter = {normalize_task_name(item) for item in tasks.split(",") if item.strip()}
    specs = []
    for spec in TASK_SPECS:
        if suite != "both" and spec.suite != suite:
            continue
        if task_split != "all" and spec.split != task_split:
            continue
        if task_filter and not task_matches_filter(spec, task_filter):
            continue
        specs.append(spec)
    return specs


def task_matches_filter(spec: ExternalTaskSpec, task_filter: set[str]) -> bool:
    aliases = {spec.task_id, spec.task_key, f"{spec.suite}:{spec.task_id}", f"{spec.suite}:{spec.task_key}"}
    return bool({normalize_task_name(alias) for alias in aliases} & task_filter)


def normalize_task_name(value: object) -> str:
    text = str(value or "").strip().lower().replace(" ", "").replace("_", "+")
    for prefix in ("cmumo:", "c-mumo:", "c:"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text


def read_source_rows(path: Path) -> list[dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    rows.extend(flatten_json_payload(json.loads(text)))
        return rows
    if suffix == ".json":
        return flatten_json_payload(json.loads(path.read_text(encoding="utf-8")))
    raise ValueError(f"Unsupported source file suffix: {path.suffix}")


def filter_source_rows_by_input_split(rows: list[dict[str, object]], input_split: str) -> list[dict[str, object]]:
    filters = {normalize_input_split(item) for item in str(input_split or "all").split(",") if item.strip()}
    if not filters or "all" in filters:
        return rows
    return [row for row in rows if input_split_matches(row, filters)]


def input_split_matches(row: Mapping[str, object], filters: set[str]) -> bool:
    raw_split = normalize_input_split(
        first_value(row, ("split", "dataset_split", "data_split", "set", "subset", "partition"))
    )
    if not raw_split:
        return False
    aliases = {
        raw_split,
        "eval" if raw_split in {"test", "valid", "validation", "eval"} else raw_split,
        "test" if raw_split in {"test", "eval"} else raw_split,
        "valid" if raw_split in {"valid", "validation"} else raw_split,
    }
    return bool(aliases & filters)


def normalize_input_split(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def flatten_json_payload(value: object) -> list[dict[str, object]]:
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    if isinstance(value, Mapping):
        rows: list[dict[str, object]] = []
        for key in ("train", "validation", "valid", "test", "eval", "data"):
            nested = value.get(key)
            if isinstance(nested, list):
                for item in nested:
                    if isinstance(item, Mapping):
                        row = dict(item)
                        row.setdefault("split", "eval" if key in {"test", "eval", "validation", "valid"} else key)
                        rows.append(row)
        if rows:
            return rows
        return [dict(value)]
    return []


def build_rows(
    source_rows: list[dict[str, object]],
    *,
    specs: list[ExternalTaskSpec],
    rng: random.Random,
    max_rows_per_task: int,
    source_smiles_column: str | None,
    target_smiles_column: str | None,
    id_column: str | None,
    respect_input_task: bool,
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for raw_index, raw_row in enumerate(source_rows):
        row_task = normalize_task_name(first_value(raw_row, ("task", "task_key", "external_task_key", "external_task_id")))
        matching_specs = [
            spec
            for spec in specs
            if not row_task or task_matches_filter(spec, {row_task})
        ]
        if row_task and respect_input_task:
            matching_specs = [spec for spec in matching_specs if task_matches_filter(spec, {row_task})]
        if not matching_specs:
            continue
        for spec in matching_specs:
            groups.setdefault(spec_key(spec), []).append({"raw_index": raw_index, "raw_row": raw_row, "spec": spec})

    output_rows: list[dict[str, object]] = []
    for key in sorted(groups):
        items = list(groups[key])
        rng.shuffle(items)
        if max_rows_per_task > 0:
            items = items[:max_rows_per_task]
        for local_index, item in enumerate(items):
            output_rows.append(
                build_condition_row(
                    item["raw_row"],  # type: ignore[arg-type]
                    spec=item["spec"],  # type: ignore[arg-type]
                    raw_index=int(item["raw_index"]),
                    local_index=local_index,
                    source_smiles_column=source_smiles_column,
                    target_smiles_column=target_smiles_column,
                    id_column=id_column,
                )
            )
    return output_rows


def build_condition_row(
    raw_row: Mapping[str, object],
    *,
    spec: ExternalTaskSpec,
    raw_index: int,
    local_index: int,
    source_smiles_column: str | None,
    target_smiles_column: str | None,
    id_column: str | None,
) -> dict[str, object]:
    source_smiles = first_smiles_value(raw_row, source_smiles_candidates(source_smiles_column))
    if not source_smiles:
        raise ValueError(f"Missing source SMILES for input row {raw_index}")
    target_smiles = first_smiles_value(raw_row, target_smiles_candidates(target_smiles_column))
    target_placeholder = False
    if not target_smiles:
        target_smiles = source_smiles
        target_placeholder = True
    source_id = str(first_value(raw_row, id_candidates(id_column)) or f"row_{raw_index:08d}").strip()
    condition_id = f"external_{spec.suite}_{spec.task_id.lower()}_{local_index:06d}"
    split = str(raw_row.get("split") or raw_row.get("dataset_split") or "eval")
    official_source = {prop: read_property_value(raw_row, prop, prefix="source") for prop in spec.properties}
    official_target = {prop: read_property_value(raw_row, prop, prefix="target") for prop in spec.properties}
    objectives = property_objectives(raw_row, spec)
    local_source_props = local_properties_for_smiles(source_smiles)
    local_targets = local_proxy_targets(
        spec,
        local_source_props=local_source_props,
        official_source=official_source,
        official_target=official_target,
        objectives=objectives,
    )
    local_selected = [prop for prop in PROPERTY_COLUMNS if prop in local_targets]
    instruction = render_instruction(source_smiles, spec, objectives=objectives)
    unsupported = [prop for prop in spec.properties if prop not in LOCAL_PROXY_PROPERTY_MAP]
    row: dict[str, object] = {
        "sample_id": condition_id,
        "condition_id": condition_id,
        "variant_id": f"{condition_id}:full",
        "variant": "full",
        "pair_id": str(first_value(raw_row, ("pair_id", "pair", "example_id")) or source_id),
        "split": split,
        "task_type": "edit_generation",
        "benchmark_task": f"external_multiproperty_{spec.suite}",
        "source_smiles": source_smiles,
        "target_smiles": target_smiles,
        "source_scaffold": "",
        "target_scaffold": "",
        "condition_properties": ",".join(local_selected),
        "property_count": len(spec.properties),
        "prompt": instruction,
        "instruction": instruction,
        "image_path": "",
        "source_image": "",
        "target_image": "",
        "molecule_id": source_id,
        "source_condition_mode": "source",
        "external_suite": spec.suite,
        "external_task_id": spec.task_id,
        "external_task_key": spec.task_key,
        "external_task_split": spec.split,
        "external_task_properties": ",".join(spec.properties),
        "external_property_directions_json": json.dumps(spec.directions, sort_keys=True),
        "external_property_objectives_json": json.dumps(objectives, sort_keys=True),
        "external_property_thresholds_json": json.dumps(
            {prop: spec.thresholds[prop] for prop in spec.properties},
            sort_keys=True,
        ),
        "external_supported_proxy_properties": ",".join(local_selected),
        "external_unsupported_properties": ",".join(unsupported),
        "external_requires_oracle_properties": ",".join(unsupported),
        "external_target_placeholder": "True" if target_placeholder else "False",
        "external_source_row_index": raw_index,
    }
    for prop in spec.properties:
        source_value = official_source.get(prop)
        target_value = official_target.get(prop)
        if source_value is not None:
            row[f"external_source_{prop}"] = format_float(source_value)
        if target_value is not None:
            row[f"external_target_{prop}"] = format_float(target_value)
    for prop in PROPERTY_COLUMNS:
        source_value = local_source_props.get(prop)
        target_value = local_targets.get(prop, source_value)
        row[f"source_{prop}"] = "" if source_value is None else format_float(source_value)
        row[f"target_{prop}"] = "" if target_value is None else format_float(target_value)
        row[f"delta_{prop}"] = (
            ""
            if source_value is None or target_value is None
            else format_float(float(target_value) - float(source_value))
        )
        row[f"{prop}_active"] = "True" if prop in local_selected else "False"
        row[f"{prop}_direction"] = local_direction_for(spec, prop) if prop in local_selected else ""
    row["sketchmol_preset_str"] = render_local_preset(local_selected, local_targets)
    row["moledit_task_key"] = "+".join(f"{prop}:{direction}" for prop, direction in spec.directions.items())
    return row


def local_proxy_targets(
    spec: ExternalTaskSpec,
    *,
    local_source_props: Mapping[str, float],
    official_source: Mapping[str, float | None],
    official_target: Mapping[str, float | None],
    objectives: Mapping[str, str] | None = None,
) -> dict[str, float]:
    targets: dict[str, float] = {}
    objectives = objectives or spec.objectives
    for external_prop in spec.properties:
        local_prop = LOCAL_PROXY_PROPERTY_MAP.get(external_prop)
        if not local_prop:
            continue
        target = official_target.get(external_prop)
        if target is None:
            source_value = local_source_props.get(local_prop)
            if source_value is None:
                source_value = official_source.get(external_prop)
            if source_value is None:
                continue
            direction = DEFAULT_DIRECTION[external_prop]
            threshold = float(spec.thresholds[external_prop])
            if normalize_objective(objectives.get(external_prop, "improve")) == "maintain":
                target = float(source_value)
            else:
                target = float(source_value) + threshold if direction == "increase" else float(source_value) - threshold
        if local_prop == "QED":
            target = min(1.0, max(0.0, float(target)))
        targets[local_prop] = float(target)
    return targets


def local_properties_for_smiles(smiles: str) -> dict[str, float]:
    try:
        props = molecular_properties(smiles) or {}
    except RuntimeError:
        props = {}
    out = {}
    for prop, key in LOCAL_PROPERTY_VALUE_KEYS.items():
        value = props.get(key)
        if value is not None and is_finite(value):
            out[prop] = float(value)
    return out


def local_direction_for(spec: ExternalTaskSpec, local_prop: str) -> str:
    external_props = [prop for prop, mapped in LOCAL_PROXY_PROPERTY_MAP.items() if mapped == local_prop]
    for prop in external_props:
        if prop in spec.properties:
            return DEFAULT_DIRECTION[prop]
    return ""


def render_instruction(source_smiles: str, spec: ExternalTaskSpec, *, objectives: Mapping[str, str] | None = None) -> str:
    objectives = objectives or spec.objectives
    clauses = []
    for prop in spec.properties:
        direction = DEFAULT_DIRECTION[prop]
        if normalize_objective(objectives.get(prop, "improve")) == "maintain":
            verb = "maintain high" if direction == "increase" else "maintain low"
        else:
            verb = "increase" if direction == "increase" else "decrease"
        clauses.append(f"{verb} {PROPERTY_DISPLAY_NAMES.get(prop, prop)}")
    if len(clauses) == 1:
        objective = clauses[0]
    else:
        objective = ", ".join(clauses[:-1]) + f", and {clauses[-1]}"
    return (
        "Modify the given molecule while keeping structural changes minimal. "
        f"Input: <SMILES> {source_smiles} </SMILES>. "
        f"Adjust: {objective}. Return only one valid SMILES string."
    )


def render_local_preset(selected: Sequence[str], targets: Mapping[str, float]) -> str:
    return ",".join(f"{prop}:{format_float(targets[prop])}" for prop in selected if prop in targets)


def read_property_value(row: Mapping[str, object], prop: str, *, prefix: str) -> float | None:
    properties = parse_properties_payload(row.get("properties"))
    if isinstance(properties.get(prop), Mapping):
        nested = properties[prop]  # type: ignore[index]
        keys = ("target", "value", "output") if prefix == "target" else ("source", "input", "value")
        for key in keys:
            value = parse_float(nested.get(key))  # type: ignore[attr-defined]
            if value is not None:
                return value
    elif prop in properties:
        value = parse_float(properties.get(prop))
        if value is not None:
            return value

    keys = (
        f"{prefix}_{prop}",
        f"{prefix}_{prop.upper()}",
        f"{prefix}_{prop.lower()}",
        f"{prop}_{prefix}",
        prop,
        prop.upper(),
    )
    for key in keys:
        value = parse_float(row.get(key))
        if value is not None:
            return value
    return None


def property_objectives(row: Mapping[str, object], spec: ExternalTaskSpec) -> dict[str, str]:
    payload = parse_objective_payload(first_value(row, ("external_property_objectives_json", "property_objectives", "objectives", "objective")))
    improved = objective_property_set(row.get("improved"))
    stable = objective_property_set(row.get("stable"))
    out = {}
    for prop in spec.properties:
        if prop in stable:
            out[prop] = "maintain"
            continue
        if prop in improved:
            out[prop] = "improve"
            continue
        raw = first_value(
            row,
            (
                f"{prop}_objective",
                f"objective_{prop}",
                f"{prop}_mode",
                f"{prop}_goal",
            ),
        )
        if raw is None:
            raw = payload.get(prop)
        out[prop] = normalize_objective(raw or DEFAULT_OBJECTIVE[prop])
    return out


def objective_property_set(value: object) -> set[str]:
    if isinstance(value, str):
        items = [item.strip() for item in value.replace("|", ",").split(",") if item.strip()]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        return set()
    return {str(item).strip().lower() for item in items}


def parse_objective_payload(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {str(key).lower(): val for key, val in value.items()}
    if isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, Mapping):
            return {str(key).lower(): val for key, val in parsed.items()}
        out = {}
        for item in text.replace("|", ",").replace(";", ",").split(","):
            if ":" not in item:
                continue
            key, raw_value = item.split(":", 1)
            out[key.strip().lower()] = raw_value.strip()
        return out
    return {}


def normalize_objective(value: object) -> str:
    key = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key in {"maintain", "keep", "preserve", "retain", "near_optimal", "nearoptimal", "sustain"}:
        return "maintain"
    return "improve"


def parse_properties_payload(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {str(key).lower(): val for key, val in value.items()}
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, Mapping):
            return {str(key).lower(): val for key, val in parsed.items()}
    return {}


def first_value(row: Mapping[str, object], keys: Iterable[str]) -> object | None:
    for key in keys:
        if key in row and str(row.get(key) or "").strip():
            return row.get(key)
    return None


def first_smiles_value(row: Mapping[str, object], keys: Iterable[str]) -> str:
    for key in keys:
        if key not in row:
            continue
        canonical = canonical_smiles_or_blank(row.get(key))
        if canonical:
            return canonical
    return ""


def canonical_smiles_or_blank(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return canonical_smiles(text) or ""
    except RuntimeError:
        return text


def source_smiles_candidates(value: str | None) -> tuple[str, ...]:
    base = (
        "source_smiles",
        "input_smiles",
        "input",
        "input_mol",
        "input_molecule",
        "source",
        "src_smiles",
        "src",
        "original_smiles",
        "original",
        "mol",
        "molecule",
        "molecule_smiles",
        "smiles",
        "SMILES",
        "canonical_smiles",
    )
    return ((value,) + base) if value else base


def target_smiles_candidates(value: str | None) -> tuple[str, ...]:
    base = (
        "target_smiles",
        "output_smiles",
        "edited_smiles",
        "target_mol",
        "target_molecule",
        "output",
        "output_mol",
        "output_molecule",
        "tgt_smiles",
        "optimized_smiles",
        "response_smiles",
        "answer_smiles",
        "target",
    )
    return ((value,) + base) if value else base


def id_candidates(value: str | None) -> tuple[str, ...]:
    base = ("sample_id", "condition_id", "id", "mol_id", "row_id")
    return ((value,) + base) if value else base


def parse_float(value: object) -> float | None:
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def is_finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def format_float(value: float, *, digits: int = 4) -> str:
    if not math.isfinite(float(value)):
        return ""
    text = f"{float(value):.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def spec_key(spec: ExternalTaskSpec) -> str:
    return f"{spec.suite}:{spec.task_id}"


def task_spec_dict(spec: ExternalTaskSpec) -> dict[str, object]:
    unsupported = [prop for prop in spec.properties if prop not in LOCAL_PROXY_PROPERTY_MAP]
    return {
        "suite": spec.suite,
        "task_id": spec.task_id,
        "task_key": spec.task_key,
        "split": spec.split,
        "properties": list(spec.properties),
        "directions": spec.directions,
        "thresholds": {prop: spec.thresholds[prop] for prop in spec.properties},
        "local_proxy_properties": {
            prop: LOCAL_PROXY_PROPERTY_MAP[prop] for prop in spec.properties if prop in LOCAL_PROXY_PROPERTY_MAP
        },
        "requires_external_oracle_properties": unsupported,
        "description": spec.description,
    }


def summarize_rows(
    args: argparse.Namespace,
    specs: list[ExternalTaskSpec],
    rows: list[dict[str, object]],
    *,
    source_row_count: int,
) -> dict[str, object]:
    task_counts = Counter(f"{row['external_suite']}:{row['external_task_id']}" for row in rows)
    unsupported = sorted(
        {
            prop
            for spec in specs
            for prop in spec.properties
            if prop not in LOCAL_PROXY_PROPERTY_MAP
        }
    )
    return {
        "source_file": str(args.source_file),
        "output_csv": str(args.output_csv),
        "suite": args.suite,
        "task_split": args.task_split,
        "input_split": args.input_split,
        "tasks": args.tasks or "all",
        "source_rows": source_row_count,
        "exported_rows": len(rows),
        "task_counts": dict(task_counts),
        "selected_task_specs": [task_spec_dict(spec) for spec in specs],
        "local_proxy_property_map": LOCAL_PROXY_PROPERTY_MAP,
        "requires_external_oracle_properties": unsupported,
        "notes": [
            "This is a source-conditioned edit benchmark adapter, not a zero-source absolute-target benchmark.",
            "Unsupported properties are preserved in prompt metadata and require external generated-property CSVs for fair evaluation.",
            "plogp is mapped to LogP as a local numeric proxy for SUCC conditioning; official plogp reporting should use the external evaluator/oracle path.",
        ],
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
