#!/usr/bin/env python3
"""Export de novo 2p-7p property-design rows aligned to SketchMol.

The output rows contain property-only targets, no source molecule, and the same
target/active columns consumed by the UniVideo materialized evaluator. A
separate candidate CSV can be emitted for property-nearest baselines so eval
targets are not silently reused as retrieval candidates.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence


REPO_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO_DIR / "SketchMol-Understanding-Condition"
MULTIPROPERTY_DIR = REPO_DIR / "SketchMol-MultiProperty-EditDataset"
for path in (PROJECT_DIR, MULTIPROPERTY_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sketchmol_multiproperty_dataset.common import (  # noqa: E402
    DISPLAY_NAMES,
    PROPERTY_COLUMNS,
    format_float,
    normalize_properties,
    sketchmol_condition_columns,
)


SMILES_COLUMNS = ("canonical_smiles", "smiles", "SMILES", "target_smiles")
ID_COLUMNS = ("mol_id", "mol_index", "id", "row_id")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule-db-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument(
        "--candidate-output-csv",
        type=Path,
        default=None,
        help="Optional property-nearest candidate rows CSV. Defaults to output sibling denovo_candidate_rows.csv.",
    )
    parser.add_argument("--rows-per-property-count", type=int, default=1000)
    parser.add_argument("--min-properties", type=int, default=2)
    parser.add_argument("--max-properties", type=int, default=7)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--smiles-column", default=None)
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--candidate-limit", type=int, default=0, help="0 keeps all candidate rows.")
    parser.add_argument(
        "--include-eval-in-candidates",
        action="store_true",
        help="Keep eval target molecules in the candidate CSV. Defaults to excluding them.",
    )
    parser.add_argument(
        "--train-rows-per-property-count",
        type=int,
        default=0,
        help="Optional additional zero-source train rows per property-count bucket.",
    )
    parser.add_argument(
        "--train-output-csv",
        type=Path,
        default=None,
        help="Optional train rows CSV. Defaults to output sibling denovo_2p7p_train_rows.csv.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.rows_per_property_count <= 0:
        raise ValueError("--rows-per-property-count must be positive")
    if args.min_properties < 1 or args.max_properties > len(PROPERTY_COLUMNS):
        raise ValueError(f"property count range must be within 1..{len(PROPERTY_COLUMNS)}")
    if args.min_properties > args.max_properties:
        raise ValueError("--min-properties cannot exceed --max-properties")

    molecules = read_molecules(args.molecule_db_csv, smiles_column=args.smiles_column, id_column=args.id_column)
    if not molecules:
        raise ValueError(f"No usable molecules with complete {PROPERTY_COLUMNS} properties in {args.molecule_db_csv}")

    rng = random.Random(args.seed)
    shuffled = list(molecules)
    rng.shuffle(shuffled)
    eval_rows, selected_keys = build_eval_rows(
        shuffled,
        rows_per_property_count=args.rows_per_property_count,
        min_properties=args.min_properties,
        max_properties=args.max_properties,
        rng=rng,
    )
    candidate_rows = build_candidate_rows(
        molecules,
        selected_keys=selected_keys,
        include_eval=bool(args.include_eval_in_candidates),
        candidate_limit=int(args.candidate_limit),
        rng=rng,
    )
    if not candidate_rows:
        raise ValueError(
            "Candidate CSV would be empty. Use --include-eval-in-candidates or provide a larger molecule database."
        )

    candidate_output = args.candidate_output_csv or args.output_csv.with_name("denovo_candidate_rows.csv")
    train_rows: list[dict[str, object]] = []
    train_output = args.train_output_csv
    if args.train_rows_per_property_count > 0:
        train_rows = build_train_rows(
            molecules,
            exclude_keys=selected_keys,
            rows_per_property_count=args.train_rows_per_property_count,
            min_properties=args.min_properties,
            max_properties=args.max_properties,
            rng=rng,
        )
        train_output = train_output or args.output_csv.with_name("denovo_2p7p_train_rows.csv")
        write_rows(train_output, train_rows)

    write_rows(args.output_csv, eval_rows)
    write_rows(candidate_output, candidate_rows)
    summary = {
        "molecule_db_csv": str(args.molecule_db_csv),
        "output_csv": str(args.output_csv),
        "candidate_output_csv": str(candidate_output),
        "input_molecules": len(molecules),
        "eval_rows": len(eval_rows),
        "candidate_rows": len(candidate_rows),
        "rows_per_property_count": int(args.rows_per_property_count),
        "min_properties": int(args.min_properties),
        "max_properties": int(args.max_properties),
        "include_eval_in_candidates": bool(args.include_eval_in_candidates),
        "property_count_distribution": dict(
            sorted(Counter(str(row["property_count"]) for row in eval_rows).items(), key=lambda item: int(item[0]))
        ),
        "train_rows": len(train_rows),
        "train_output_csv": str(train_output) if train_output else None,
        "train_rows_per_property_count": int(args.train_rows_per_property_count),
    }
    args.output_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def read_molecules(path: Path, *, smiles_column: str | None, id_column: str | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        smiles_key = smiles_column or first_present(fieldnames, SMILES_COLUMNS)
        if smiles_key is None:
            raise ValueError(f"No SMILES column found in {path}; tried {SMILES_COLUMNS}")
        id_key = id_column or first_present(fieldnames, ID_COLUMNS)
        for index, raw in enumerate(reader):
            smiles = str(raw.get(smiles_key, "") or "").strip()
            if not smiles:
                continue
            props = normalize_properties(raw)
            if any(prop not in props for prop in PROPERTY_COLUMNS):
                continue
            mol_id = str(raw.get(id_key, "") or "").strip() if id_key else ""
            if not mol_id:
                mol_id = f"mol_{index:08d}"
            rows.append(
                {
                    "mol_key": f"{mol_id}:{smiles}",
                    "mol_id": mol_id,
                    "smiles": smiles,
                    "scaffold": str(raw.get("scaffold", "") or raw.get("target_scaffold", "") or "").strip(),
                    "props": {prop: float(props[prop]) for prop in PROPERTY_COLUMNS},
                }
            )
    return rows


def build_eval_rows(
    molecules: list[dict[str, object]],
    *,
    rows_per_property_count: int,
    min_properties: int,
    max_properties: int,
    rng: random.Random,
) -> tuple[list[dict[str, object]], set[str]]:
    rows: list[dict[str, object]] = []
    selected_keys: set[str] = set()
    cursor = 0
    total_needed = rows_per_property_count * (max_properties - min_properties + 1)
    disjoint = len(molecules) >= total_needed
    for property_count in range(min_properties, max_properties + 1):
        combos = list(itertools.combinations(PROPERTY_COLUMNS, property_count))
        rng.shuffle(combos)
        if disjoint:
            selected_molecules = molecules[cursor : cursor + rows_per_property_count]
            cursor += rows_per_property_count
        else:
            selected_molecules = [molecules[(cursor + idx) % len(molecules)] for idx in range(rows_per_property_count)]
            cursor += rows_per_property_count
        for local_index, molecule in enumerate(selected_molecules):
            selected_props = list(combos[local_index % len(combos)])
            condition_id = f"denovo_{property_count}p_{local_index:06d}"
            rows.append(make_condition_row(molecule, selected_props, condition_id=condition_id, role="eval"))
            selected_keys.add(str(molecule["mol_key"]))
    return rows, selected_keys


def build_train_rows(
    molecules: list[dict[str, object]],
    *,
    exclude_keys: set[str],
    rows_per_property_count: int,
    min_properties: int,
    max_properties: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    pool = [molecule for molecule in molecules if str(molecule["mol_key"]) not in exclude_keys]
    if not pool:
        pool = list(molecules)
    rng.shuffle(pool)
    rows: list[dict[str, object]] = []
    cursor = 0
    for property_count in range(min_properties, max_properties + 1):
        combos = list(itertools.combinations(PROPERTY_COLUMNS, property_count))
        rng.shuffle(combos)
        for local_index in range(rows_per_property_count):
            molecule = pool[(cursor + local_index) % len(pool)]
            selected_props = list(combos[local_index % len(combos)])
            condition_id = f"denovo_train_{property_count}p_{local_index:06d}"
            row = make_condition_row(molecule, selected_props, condition_id=condition_id, role="train")
            row["split"] = "train"
            rows.append(row)
        cursor += rows_per_property_count
    return rows


def build_candidate_rows(
    molecules: list[dict[str, object]],
    *,
    selected_keys: set[str],
    include_eval: bool,
    candidate_limit: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    candidates = [
        molecule
        for molecule in molecules
        if include_eval or str(molecule["mol_key"]) not in selected_keys
    ]
    if candidate_limit > 0 and candidate_limit < len(candidates):
        candidates = rng.sample(candidates, candidate_limit)
    selected_props = list(PROPERTY_COLUMNS)
    return [
        make_condition_row(molecule, selected_props, condition_id=f"denovo_candidate_{idx:08d}", role="candidate")
        for idx, molecule in enumerate(candidates)
    ]


def make_condition_row(
    molecule: Mapping[str, object],
    selected_props: list[str],
    *,
    condition_id: str,
    role: str,
) -> dict[str, object]:
    props = dict(molecule["props"])  # type: ignore[arg-type]
    row: dict[str, object] = {
        "sample_id": condition_id,
        "condition_id": condition_id,
        "variant_id": f"{condition_id}:full",
        "variant": "full",
        "pair_id": "",
        "split": "train" if role == "train" else ("eval" if role == "eval" else "candidate"),
        "task_type": "de_novo_design",
        "benchmark_task": "denovo_2p7p_property_design",
        "source_smiles": "",
        "target_smiles": molecule["smiles"],
        "source_scaffold": "",
        "target_scaffold": molecule.get("scaffold", ""),
        "condition_properties": ",".join(selected_props),
        "property_count": len(selected_props),
        "prompt": render_denovo_instruction(selected_props, props),
        "instruction": render_denovo_instruction(selected_props, props),
        "sketchmol_preset_str": render_sketchmol_preset(selected_props, props),
        "image_path": "",
        "source_image": "",
        "target_image": "",
        "molecule_id": molecule["mol_id"],
    }
    for prop in PROPERTY_COLUMNS:
        row[f"target_{prop}"] = format_float(props[prop], digits=4)
        row[f"{prop}_active"] = "True" if prop in selected_props else "False"
        row[f"{prop}_direction"] = ""
    row.update(sketchmol_condition_columns(props, selected_props))
    return row


def render_denovo_instruction(selected_props: list[str], props: Mapping[str, float]) -> str:
    clauses = []
    for prop in selected_props:
        digits = 3 if prop == "QED" else 0 if prop in {"HBD", "HBA", "RB"} else 2
        clauses.append(f"{DISPLAY_NAMES[prop]} around {format_float(float(props[prop]), digits=digits)}")
    if len(clauses) == 1:
        target_text = clauses[0]
    elif len(clauses) == 2:
        target_text = f"{clauses[0]} and {clauses[1]}"
    else:
        target_text = f"{', '.join(clauses[:-1])}, and {clauses[-1]}"
    return f"Generate a new molecule with {target_text}."


def render_sketchmol_preset(selected_props: list[str], props: Mapping[str, float]) -> str:
    names = {"MW": "MW", "LogP": "LogP", "QED": "QED", "TPSA": "TPSA", "HBD": "HBD", "HBA": "HBA", "RB": "RB"}
    return ",".join(f"{names[prop]}:{format_float(float(props[prop]), digits=4)}" for prop in selected_props)


def first_present(fieldnames: Sequence[str], candidates: Sequence[str]) -> str | None:
    available = set(fieldnames)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


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
