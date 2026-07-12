#!/usr/bin/env python3
"""Prepare leakage-audited, task-capped rows for one joint Unified checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


PROPERTY_COLUMNS = ("MW", "LogP", "QED", "TPSA", "HBD", "HBA", "RB")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-train-csv", required=True, type=Path)
    parser.add_argument("--source-eval-csv", required=True, type=Path)
    parser.add_argument("--source-candidate-csv", required=True, type=Path)
    parser.add_argument("--ood-eval-csv", required=True, type=Path)
    parser.add_argument("--train-output-csv", required=True, type=Path)
    parser.add_argument("--validation-output-csv", required=True, type=Path)
    parser.add_argument("--denovo-validation-output-csv", required=True, type=Path)
    parser.add_argument("--table1-validation-output-csv", required=True, type=Path)
    parser.add_argument("--denovo-test-output-csv", required=True, type=Path)
    parser.add_argument("--table1-test-output-csv", required=True, type=Path)
    parser.add_argument("--ood-test-output-csv", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--denovo-train-per-count", type=int, default=2000)
    parser.add_argument("--denovo-validation-per-count", type=int, default=200)
    parser.add_argument("--denovo-test-per-count", type=int, default=1000)
    parser.add_argument("--edit-train-per-task", type=int, default=450)
    parser.add_argument("--edit-validation-per-task", type=int, default=50)
    parser.add_argument("--edit-test-per-task", type=int, default=100)
    parser.add_argument("--ood-test-per-spec", type=int, default=100)
    parser.add_argument("--overlap-policy", choices=("fail", "drop_train"), default="drop_train")
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args(argv)


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [
            {str(key): "" if value is None else str(value) for key, value in row.items()}
            for row in reader
        ]
        return rows, list(reader.fieldnames or [])


def task_mode(row: Mapping[str, str]) -> str:
    raw = str(row.get("task_mode", "") or row.get("unified_task_mode", "")).strip().lower()
    normalized = raw.replace("-", "_").replace(" ", "_")
    if normalized in {"de_novo", "denovo", "generation", "generate"}:
        return "de_novo"
    if normalized in {"edit", "source_edit", "conditional_edit", "edit_generation"}:
        return "edit"
    return "edit" if str(row.get("source_smiles", "")).strip() else "de_novo"


def include_joint_row(row: Mapping[str, str]) -> bool:
    mode = task_mode(row)
    benchmark = str(row.get("benchmark_task", "") or "").strip().lower()
    sample_id = str(row.get("sample_id", "") or row.get("condition_id", "")).strip().lower()
    if mode == "de_novo":
        if "ood" in benchmark or "ood" in sample_id:
            return False
        return "2p7p" in benchmark or sample_id.startswith("denovo_")
    if benchmark and ("external" in benchmark or "ood" in benchmark):
        return False
    return not benchmark or "table1" in benchmark or "moledit" in benchmark


def parse_instruction_tasks(row: Mapping[str, str]) -> list[tuple[str, str]]:
    raw = str(row.get("instruction_tasks", "") or "").strip()
    if raw:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, list):
            specs = []
            for item in value:
                if isinstance(item, dict):
                    prop = str(item.get("property", "") or item.get("name", "") or "").strip()
                    direction = str(item.get("direction", "") or item.get("operation", "") or "").strip().lower()
                    if prop:
                        specs.append((prop, direction))
            if specs:
                return sorted(specs)
    props = [part.strip() for part in str(row.get("condition_properties", "") or "").replace(";", ",").split(",") if part.strip()]
    specs = []
    for prop in props:
        direction = str(row.get(f"{prop}_direction", "") or row.get(f"{prop.lower()}_direction", "")).strip().lower()
        specs.append((prop, direction))
    return sorted(specs)


def group_key(row: Mapping[str, str]) -> str:
    mode = task_mode(row)
    if mode == "de_novo":
        ood_spec = str(row.get("ood_spec_id", "") or "").strip()
        if ood_spec:
            return f"ood:{ood_spec}"
        raw = str(row.get("property_count", "") or "0").strip()
        try:
            count = int(float(raw))
        except ValueError:
            count = len([part for part in str(row.get("condition_properties", "")).split(",") if part.strip()])
        return f"de_novo:{count}p"
    specs = parse_instruction_tasks(row)
    rendered = "+".join(f"{prop}:{direction or 'none'}" for prop, direction in specs)
    return f"edit:{rendered or 'unknown'}"


def cap_groups(rows: Sequence[dict[str, str]], *, per_group: int, seed: int) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)
    selected = []
    for offset, key in enumerate(sorted(grouped)):
        values = list(grouped[key])
        random.Random(seed + offset).shuffle(values)
        selected.extend(values[:per_group] if per_group > 0 else values)
    return selected


def split_groups(
    rows: Sequence[dict[str, str]],
    *,
    train_per_group: int,
    validation_per_group: int,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)
    train_rows = []
    validation_rows = []
    for offset, key in enumerate(sorted(grouped)):
        values = list(grouped[key])
        random.Random(seed + offset).shuffle(values)
        validation_rows.extend(values[: max(0, validation_per_group)])
        start = max(0, validation_per_group)
        train_rows.extend(values[start : start + max(0, train_per_group)])
    return train_rows, validation_rows


def build_denovo_validation_rows(
    candidates: Sequence[dict[str, str]],
    *,
    excluded_targets: set[str],
    per_count: int,
    seed: int,
) -> list[dict[str, str]]:
    pool = []
    seen_targets = set()
    for row in candidates:
        target = target_smiles(row)
        if not target or target in excluded_targets or target in seen_targets:
            continue
        seen_targets.add(target)
        pool.append(row)
    random.Random(seed).shuffle(pool)
    needed = max(0, int(per_count)) * 6
    if len(pool) < needed:
        raise ValueError(f"Need {needed} disjoint de novo validation molecules, found {len(pool)}")
    rows = []
    cursor = 0
    for property_count in range(2, 8):
        combinations = list(itertools.combinations(PROPERTY_COLUMNS, property_count))
        random.Random(seed + property_count).shuffle(combinations)
        for local_index in range(int(per_count)):
            source = dict(pool[cursor])
            cursor += 1
            selected = combinations[local_index % len(combinations)]
            condition_id = f"joint_validation_{property_count}p_{local_index:06d}"
            validation_target = target_smiles(source)
            source.update(
                {
                    "sample_id": condition_id,
                    "condition_id": condition_id,
                    "variant_id": f"{condition_id}:full",
                    "variant": "full",
                    "split": "validation",
                    "task_mode": "de_novo",
                    "task_type": "de_novo_design",
                    "benchmark_task": "denovo_2p7p_property_design",
                    "target_smiles": validation_target,
                    "source_smiles": "",
                    "source_image": "",
                    "condition_properties": ",".join(selected),
                    "property_count": str(property_count),
                    "prompt": render_denovo_prompt(source, selected),
                    "instruction": render_denovo_prompt(source, selected),
                }
            )
            for prop in PROPERTY_COLUMNS:
                source[f"target_{prop}"] = str(
                    source.get(f"target_{prop}", "") or source.get(prop, "") or ""
                )
                source[f"{prop}_active"] = "True" if prop in selected else "False"
                source[f"{prop}_direction"] = ""
            rows.append(source)
    return rows


def render_denovo_prompt(row: Mapping[str, str], properties: Sequence[str]) -> str:
    clauses = []
    for prop in properties:
        value = str(row.get(f"target_{prop}", "") or row.get(prop, "")).strip()
        clauses.append(f"{prop} around {value}" if value else prop)
    return "Generate a new molecule with " + ", ".join(clauses) + "."


def target_smiles(row: Mapping[str, str]) -> str:
    return str(
        row.get("target_smiles", "")
        or row.get("canonical_smiles", "")
        or row.get("smiles", "")
        or row.get("SMILES", "")
        or ""
    ).strip()


def target_set(rows: Sequence[Mapping[str, str]]) -> set[str]:
    return {target_smiles(row) for row in rows if target_smiles(row)}


def edit_pair_set(rows: Sequence[Mapping[str, str]]) -> set[str]:
    pairs = set()
    for row in rows:
        if task_mode(row) != "edit":
            continue
        source = str(row.get("source_smiles", "") or "").strip()
        target = target_smiles(row)
        if source or target:
            pairs.add(f"{source}|{target}")
    return pairs


def scaffold_set(rows: Sequence[Mapping[str, str]]) -> tuple[set[str], bool]:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception:
        return set(), False
    scaffolds = set()
    for smiles in target_set(rows):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
        if scaffold:
            scaffolds.add(scaffold)
    return scaffolds, True


def split_audit(train_rows: Sequence[Mapping[str, str]], test_rows: Sequence[Mapping[str, str]]) -> dict[str, object]:
    train_targets = target_set(train_rows)
    test_targets = target_set(test_rows)
    train_pairs = edit_pair_set(train_rows)
    test_pairs = edit_pair_set(test_rows)
    train_scaffolds, scaffold_available = scaffold_set(train_rows)
    test_scaffolds, _ = scaffold_set(test_rows) if scaffold_available else (set(), False)
    scaffold_overlap = train_scaffolds & test_scaffolds
    return {
        "train_test_target_overlap": len(train_targets & test_targets),
        "train_test_edit_pair_overlap": len(train_pairs & test_pairs),
        "scaffold_audit_available": scaffold_available,
        "train_scaffolds": len(train_scaffolds),
        "test_scaffolds": len(test_scaffolds),
        "overlap_scaffolds": len(scaffold_overlap),
        "test_scaffold_overlap_rate": len(scaffold_overlap) / max(len(test_scaffolds), 1),
    }


def overlap_keys(row: Mapping[str, str]) -> set[str]:
    mode = task_mode(row)
    target = target_smiles(row)
    if mode == "de_novo":
        molecule_id = str(row.get("molecule_id", "") or row.get("mol_id", "")).strip()
        keys = {f"denovo_target:{target}"} if target else set()
        if molecule_id:
            keys.add(f"denovo_molecule:{molecule_id}")
        return keys
    source = str(row.get("source_smiles", "") or "").strip()
    example_id = str(row.get("example_id", "") or "").strip()
    keys = {f"edit_pair:{source}|{target}"} if source or target else set()
    if example_id:
        keys.add(f"edit_example:{example_id}")
    return keys


def remove_train_eval_overlap(
    train_rows: Sequence[dict[str, str]],
    eval_rows: Sequence[dict[str, str]],
    *,
    policy: str,
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    eval_keys = set().union(*(overlap_keys(row) for row in eval_rows)) if eval_rows else set()
    kept = []
    dropped = []
    for row in train_rows:
        collisions = sorted(overlap_keys(row) & eval_keys)
        if not collisions:
            kept.append(row)
            continue
        dropped.append({"sample_id": row.get("sample_id", ""), "group": group_key(row), "keys": collisions})
    if dropped and policy == "fail":
        raise ValueError(f"Detected {len(dropped)} train rows overlapping eval rows; see overlap audit.")
    return kept, dropped


def write_rows(path: Path, rows: Sequence[Mapping[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def counts(rows: Sequence[Mapping[str, str]]) -> dict[str, object]:
    return {
        "rows": len(rows),
        "task_modes": dict(sorted(Counter(task_mode(row) for row in rows).items())),
        "groups": dict(sorted(Counter(group_key(row) for row in rows).items())),
    }


def require_exact_groups(
    rows: Sequence[Mapping[str, str]],
    *,
    expected_groups: int,
    expected_per_group: int,
    label: str,
) -> None:
    grouped = Counter(group_key(row) for row in rows)
    if len(grouped) != expected_groups or any(value != expected_per_group for value in grouped.values()):
        raise ValueError(
            f"{label} must have {expected_groups} groups x {expected_per_group} rows; "
            f"got {dict(sorted(grouped.items()))}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_train, train_fields = read_rows(args.source_train_csv)
    source_eval, eval_fields = read_rows(args.source_eval_csv)
    source_candidates, candidate_fields = read_rows(args.source_candidate_csv)
    ood_rows, ood_fields = read_rows(args.ood_eval_csv)
    train_rows = [row for row in source_train if include_joint_row(row)]
    eval_rows = [row for row in source_eval if include_joint_row(row)]

    denovo_test = cap_groups(
        [row for row in eval_rows if task_mode(row) == "de_novo"],
        per_group=int(args.denovo_test_per_count),
        seed=int(args.seed) + 200,
    )
    table1_test = cap_groups(
        [row for row in eval_rows if task_mode(row) == "edit"],
        per_group=int(args.edit_test_per_task),
        seed=int(args.seed) + 300,
    )
    ood_test = cap_groups(
        ood_rows,
        per_group=int(args.ood_test_per_spec),
        seed=int(args.seed) + 400,
    )
    require_exact_groups(
        denovo_test,
        expected_groups=6,
        expected_per_group=int(args.denovo_test_per_count),
        label="de novo test",
    )
    require_exact_groups(
        table1_test,
        expected_groups=10,
        expected_per_group=int(args.edit_test_per_task),
        label="Table1 test",
    )
    require_exact_groups(
        ood_test,
        expected_groups=10,
        expected_per_group=int(args.ood_test_per_spec),
        label="OOD test",
    )
    formal_tests = denovo_test + table1_test + ood_test

    source_train_ood_overlap = len(target_set(source_train) & target_set(ood_test))
    if source_train_ood_overlap:
        raise ValueError(
            f"OOD test contains {source_train_ood_overlap} target molecules exposed in source training rows; rebuild OOD with exclusions."
        )

    cleaned_train, dropped = remove_train_eval_overlap(
        train_rows,
        formal_tests,
        policy=str(args.overlap_policy),
    )
    train_denovo = cap_groups(
        [row for row in cleaned_train if task_mode(row) == "de_novo"],
        per_group=int(args.denovo_train_per_count),
        seed=int(args.seed),
    )
    train_edit, validation_edit = split_groups(
        [row for row in cleaned_train if task_mode(row) == "edit"],
        train_per_group=int(args.edit_train_per_task),
        validation_per_group=int(args.edit_validation_per_task),
        seed=int(args.seed) + 100,
    )
    require_exact_groups(
        train_denovo,
        expected_groups=6,
        expected_per_group=int(args.denovo_train_per_count),
        label="de novo train",
    )
    require_exact_groups(
        train_edit,
        expected_groups=10,
        expected_per_group=int(args.edit_train_per_task),
        label="Table1 train",
    )
    joint_train = train_denovo + train_edit
    excluded_validation_targets = target_set(source_train + formal_tests + validation_edit)
    validation_denovo = build_denovo_validation_rows(
        source_candidates,
        excluded_targets=excluded_validation_targets,
        per_count=int(args.denovo_validation_per_count),
        seed=int(args.seed) + 500,
    )
    joint_validation = validation_denovo + validation_edit
    require_exact_groups(
        validation_denovo,
        expected_groups=6,
        expected_per_group=int(args.denovo_validation_per_count),
        label="de novo validation",
    )
    require_exact_groups(
        validation_edit,
        expected_groups=10,
        expected_per_group=int(args.edit_validation_per_task),
        label="Table1 validation",
    )

    generated_fields = [
        "sample_id",
        "condition_id",
        "variant_id",
        "variant",
        "split",
        "task_mode",
        "task_type",
        "benchmark_task",
        "source_smiles",
        "source_image",
        "target_smiles",
        "condition_properties",
        "property_count",
        "prompt",
        "instruction",
    ]
    for prop in PROPERTY_COLUMNS:
        generated_fields.extend([f"target_{prop}", f"{prop}_active", f"{prop}_direction"])
    fields = list(dict.fromkeys(train_fields + eval_fields + candidate_fields + ood_fields + generated_fields))
    write_rows(args.train_output_csv, joint_train, fields)
    write_rows(args.validation_output_csv, joint_validation, fields)
    write_rows(args.denovo_validation_output_csv, validation_denovo, fields)
    write_rows(args.table1_validation_output_csv, validation_edit, fields)
    write_rows(args.denovo_test_output_csv, denovo_test, fields)
    write_rows(args.table1_test_output_csv, table1_test, fields)
    write_rows(args.ood_test_output_csv, ood_test, fields)
    audits = {
        "denovo_test": split_audit(joint_train, denovo_test),
        "table1_test": split_audit(joint_train, table1_test),
        "ood_test": split_audit(joint_train, ood_test),
        "validation": split_audit(joint_train, joint_validation),
    }
    for name, audit in audits.items():
        if int(audit["train_test_target_overlap"]) or int(audit["train_test_edit_pair_overlap"]):
            raise ValueError(f"Non-zero train overlap remains for {name}: {audit}")
    manifest = {
        "protocol": "unified_joint_fair_v2",
        "seed": int(args.seed),
        "overlap_policy": str(args.overlap_policy),
        "source_train_csv": str(args.source_train_csv),
        "source_train_sha256": sha256(args.source_train_csv),
        "source_eval_csv": str(args.source_eval_csv),
        "source_eval_sha256": sha256(args.source_eval_csv),
        "source_candidate_csv": str(args.source_candidate_csv),
        "source_candidate_sha256": sha256(args.source_candidate_csv),
        "ood_eval_csv": str(args.ood_eval_csv),
        "ood_eval_sha256": sha256(args.ood_eval_csv),
        "train_output_csv": str(args.train_output_csv),
        "train_output_sha256": sha256(args.train_output_csv),
        "validation_output_csv": str(args.validation_output_csv),
        "validation_output_sha256": sha256(args.validation_output_csv),
        "denovo_validation_output_csv": str(args.denovo_validation_output_csv),
        "denovo_validation_output_sha256": sha256(args.denovo_validation_output_csv),
        "table1_validation_output_csv": str(args.table1_validation_output_csv),
        "table1_validation_output_sha256": sha256(args.table1_validation_output_csv),
        "denovo_test_output_csv": str(args.denovo_test_output_csv),
        "denovo_test_output_sha256": sha256(args.denovo_test_output_csv),
        "table1_test_output_csv": str(args.table1_test_output_csv),
        "table1_test_output_sha256": sha256(args.table1_test_output_csv),
        "ood_test_output_csv": str(args.ood_test_output_csv),
        "ood_test_output_sha256": sha256(args.ood_test_output_csv),
        "train": counts(joint_train),
        "validation": counts(joint_validation),
        "denovo_test": counts(denovo_test),
        "table1_test": counts(table1_test),
        "ood_test": counts(ood_test),
        "source_train_ood_target_overlap": source_train_ood_overlap,
        "split_audits": audits,
        "dropped_train_eval_overlap_rows": len(dropped),
        "overlap_examples": dropped[:50],
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
