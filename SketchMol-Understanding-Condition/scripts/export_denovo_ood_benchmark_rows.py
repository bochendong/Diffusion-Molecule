#!/usr/bin/env python3
"""Export de novo OOD benchmark rows for the SUCC UniVideo model."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
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
PROPERTY_NORMALIZERS = {
    "MW": 75.0,
    "LogP": 1.0,
    "QED": 0.15,
    "TPSA": 35.0,
    "HBD": 1.5,
    "HBA": 2.0,
    "RB": 2.0,
}
PROPERTY_ALIASES = {
    "mw": "MW",
    "molwt": "MW",
    "molecular_weight": "MW",
    "logp": "LogP",
    "qed": "QED",
    "tpsa": "TPSA",
    "hbd": "HBD",
    "hba": "HBA",
    "haccept": "HBA",
    "hacceptor": "HBA",
    "rb": "RB",
    "rotbonds": "RB",
    "rotbond": "RB",
    "rotatable": "RB",
}
DEFAULT_SPECS = [
    {"bucket": "forward_extreme", "name": "mw_high", "positive": "MW:650"},
    {"bucket": "forward_extreme", "name": "logp_high", "positive": "LogP:6"},
    {"bucket": "forward_extreme", "name": "tpsa_high", "positive": "TPSA:160"},
    {"bucket": "forward_extreme", "name": "rb_high", "positive": "RB:12"},
    {"bucket": "rare_combo", "name": "qed_high_mw_high", "positive": "QED:0.90 MW:520"},
    {"bucket": "rare_combo", "name": "mw_high_logp_low", "positive": "MW:560 LogP:0"},
    {"bucket": "rare_combo", "name": "tpsa_high_hbd_low", "positive": "TPSA:140 HBD:0"},
    {"bucket": "rare_combo", "name": "seven_property_edge", "positive": "MW:520 LogP:4 QED:0.80 TPSA:120 HBD:1 HBA:8 RB:8"},
    {
        "bucket": "reverse_stimulation",
        "name": "reverse_mw300",
        "positive": "MW:300",
        "negative": "MW:300 HBD:0 HBA:1 RB:1",
    },
    {
        "bucket": "reverse_stimulation",
        "name": "reverse_logp4",
        "positive": "LogP:4",
        "negative": "LogP:4 HBD:0 HBA:1 RB:1",
    },
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule-db-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--negative-output-csv", type=Path, default=None)
    parser.add_argument("--candidate-output-csv", type=Path, default=None)
    parser.add_argument("--spec-json", type=Path, default=None)
    parser.add_argument("--rows-per-spec", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--smiles-column", default=None)
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--candidate-limit", type=int, default=0, help="0 keeps all candidate rows.")
    parser.add_argument("--include-eval-in-candidates", action="store_true")
    parser.add_argument(
        "--train-rows-per-spec",
        type=int,
        default=0,
        help="Optional additional zero-source OOD train rows per spec.",
    )
    parser.add_argument(
        "--train-output-csv",
        type=Path,
        default=None,
        help="Optional train rows CSV. Defaults to output sibling denovo_ood_train_rows.csv.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.rows_per_spec <= 0:
        raise ValueError("--rows-per-spec must be positive")
    molecules = read_molecules(args.molecule_db_csv, smiles_column=args.smiles_column, id_column=args.id_column)
    if not molecules:
        raise ValueError(f"No usable molecules with complete {PROPERTY_COLUMNS} properties in {args.molecule_db_csv}")
    specs = load_specs(args.spec_json)
    rng = random.Random(args.seed)
    eval_rows, negative_rows, selected_keys = build_ood_rows(
        molecules,
        specs=specs,
        rows_per_spec=args.rows_per_spec,
        rng=rng,
        split="eval",
    )
    train_rows: list[dict[str, object]] = []
    train_output = args.train_output_csv
    if args.train_rows_per_spec > 0:
        train_rows, _, _ = build_ood_rows(
            molecules,
            specs=specs,
            rows_per_spec=args.train_rows_per_spec,
            rng=random.Random(args.seed + 1),
            split="train",
            exclude_keys=selected_keys,
        )
        train_output = train_output or args.output_csv.with_name("denovo_ood_train_rows.csv")
        write_rows(train_output, train_rows)
    candidate_rows = build_candidate_rows(
        molecules,
        selected_keys=selected_keys,
        include_eval=bool(args.include_eval_in_candidates),
        candidate_limit=int(args.candidate_limit),
        rng=rng,
    )
    if not candidate_rows:
        raise ValueError("Candidate CSV would be empty. Use --include-eval-in-candidates or a larger molecule DB.")

    negative_output = args.negative_output_csv or args.output_csv.with_name("denovo_ood_negative_rows.csv")
    candidate_output = args.candidate_output_csv or args.output_csv.with_name("denovo_ood_candidate_rows.csv")
    write_rows(args.output_csv, eval_rows)
    write_rows(negative_output, negative_rows)
    write_rows(candidate_output, candidate_rows)
    summary = {
        "molecule_db_csv": str(args.molecule_db_csv),
        "output_csv": str(args.output_csv),
        "negative_output_csv": str(negative_output),
        "candidate_output_csv": str(candidate_output),
        "input_molecules": len(molecules),
        "eval_rows": len(eval_rows),
        "negative_rows": len(negative_rows),
        "candidate_rows": len(candidate_rows),
        "rows_per_spec": int(args.rows_per_spec),
        "include_eval_in_candidates": bool(args.include_eval_in_candidates),
        "bucket_distribution": dict(Counter(str(row["ood_bucket"]) for row in eval_rows)),
        "spec_distribution": dict(Counter(str(row["ood_spec_id"]) for row in eval_rows)),
        "train_rows": len(train_rows),
        "train_output_csv": str(train_output) if train_output else None,
        "train_rows_per_spec": int(args.train_rows_per_spec),
        "train_bucket_distribution": dict(Counter(str(row["ood_bucket"]) for row in train_rows)),
    }
    args.output_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def load_specs(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return [dict(spec) for spec in DEFAULT_SPECS]
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("--spec-json must contain a list of OOD specs")
    specs = []
    for idx, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ValueError(f"spec #{idx} is not an object")
        spec = {str(key): str(val) for key, val in raw.items()}
        if not spec.get("positive"):
            raise ValueError(f"spec #{idx} is missing positive preset")
        spec.setdefault("name", f"spec_{idx:03d}")
        spec.setdefault("bucket", "custom_ood")
        specs.append(spec)
    return specs


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


def build_ood_rows(
    molecules: list[dict[str, object]],
    *,
    specs: list[dict[str, str]],
    rows_per_spec: int,
    rng: random.Random,
    split: str = "eval",
    exclude_keys: set[str] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], set[str]]:
    rows: list[dict[str, object]] = []
    negative_rows: list[dict[str, object]] = []
    selected_keys: set[str] = set()
    excluded = set(exclude_keys or ())
    pool = [molecule for molecule in molecules if str(molecule["mol_key"]) not in excluded] or list(molecules)
    for spec in specs:
        positive = parse_preset(spec["positive"])
        negative = parse_preset(spec.get("negative", ""))
        ranked = sorted(
            pool,
            key=lambda mol: (preset_distance(mol["props"], positive), rng.random()),  # type: ignore[arg-type]
        )
        picked = []
        for molecule in ranked:
            if len(picked) >= rows_per_spec:
                break
            if str(molecule["mol_key"]) in selected_keys and len(pool) >= rows_per_spec * len(specs):
                continue
            picked.append(molecule)
            selected_keys.add(str(molecule["mol_key"]))
        if len(picked) < rows_per_spec:
            for molecule in ranked:
                if len(picked) >= rows_per_spec:
                    break
                picked.append(molecule)
        for local_index, molecule in enumerate(picked):
            prefix = "denovo_ood_train" if split == "train" else "denovo_ood"
            condition_id = f"{prefix}_{sanitize(spec['name'])}_{local_index:06d}"
            rows.append(
                make_condition_row(
                    molecule,
                    selected_props=list(positive),
                    target_values=dict(molecule["props"]),  # type: ignore[arg-type]
                    condition_id=condition_id,
                    split=split,
                    instruction=render_instruction(list(positive), molecule["props"]),  # type: ignore[arg-type]
                    preset_values=positive,
                    negative_preset=negative,
                    spec=spec,
                    role=split,
                    negative_condition_id=f"{condition_id}:negative",
                )
            )
            if negative and split == "eval":
                negative_values = dict(molecule["props"])  # type: ignore[arg-type]
                negative_values.update(negative)
                negative_props = list(negative)
                negative_instruction = render_instruction(negative_props, negative_values)
            else:
                negative_values = dict(molecule["props"])  # type: ignore[arg-type]
                negative_props = []
                negative_instruction = "Generate a valid molecule."
            negative_rows.append(
                make_condition_row(
                    molecule,
                    selected_props=negative_props,
                    target_values=negative_values,
                    condition_id=f"{condition_id}:negative",
                    split="eval",
                    instruction=negative_instruction,
                    preset_values=negative,
                    negative_preset={},
                    spec=spec,
                    role="negative",
                    negative_condition_id="",
                )
            )
    return rows, negative_rows, selected_keys


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
        make_condition_row(
            molecule,
            selected_props=selected_props,
            target_values=molecule["props"],  # type: ignore[arg-type]
            condition_id=f"denovo_ood_candidate_{idx:08d}",
            split="candidate",
            instruction=render_instruction(selected_props, molecule["props"]),  # type: ignore[arg-type]
            preset_values={prop: float(molecule["props"][prop]) for prop in selected_props},  # type: ignore[index]
            negative_preset={},
            spec={"bucket": "candidate", "name": "candidate", "positive": ""},
            role="candidate",
            negative_condition_id="",
        )
        for idx, molecule in enumerate(candidates)
    ]


def make_condition_row(
    molecule: Mapping[str, object],
    *,
    selected_props: list[str],
    target_values: Mapping[str, float],
    condition_id: str,
    split: str,
    instruction: str,
    preset_values: Mapping[str, float],
    negative_preset: Mapping[str, float],
    spec: Mapping[str, str],
    role: str,
    negative_condition_id: str,
) -> dict[str, object]:
    props = dict(molecule["props"])  # type: ignore[arg-type]
    row: dict[str, object] = {
        "sample_id": condition_id,
        "condition_id": condition_id,
        "variant_id": f"{condition_id}:full",
        "variant": "full",
        "pair_id": "",
        "split": split,
        "task_type": "de_novo_design",
        "benchmark_task": "denovo_ood_property_design",
        "source_smiles": "",
        "target_smiles": molecule["smiles"],
        "source_scaffold": "",
        "target_scaffold": molecule.get("scaffold", ""),
        "condition_properties": ",".join(selected_props),
        "property_count": len(selected_props),
        "prompt": instruction,
        "instruction": instruction,
        "sketchmol_preset_str": render_preset(selected_props, target_values),
        "image_path": "",
        "source_image": "",
        "target_image": "",
        "molecule_id": molecule["mol_id"],
        "source_condition_mode": "zero",
        "ood_bucket": spec.get("bucket", "custom_ood"),
        "ood_spec_id": spec.get("name", ""),
        "ood_role": role,
        "ood_positive_preset_str": render_preset(list(preset_values), preset_values),
        "ood_negative_preset_str": render_preset(list(negative_preset), negative_preset),
        "negative_condition_id": negative_condition_id,
        "ood_preset_distance": format_float(preset_distance(props, preset_values), digits=6) if preset_values else "",
    }
    for prop in PROPERTY_COLUMNS:
        row[f"target_{prop}"] = format_float(float(target_values.get(prop, props[prop])), digits=4)
        row[f"{prop}_active"] = "True" if prop in selected_props else "False"
        row[f"{prop}_direction"] = ""
    row.update(sketchmol_condition_columns({prop: float(target_values.get(prop, props[prop])) for prop in PROPERTY_COLUMNS}, selected_props))
    return row


def parse_preset(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for token in re.split(r"[\s,]+", text.strip()):
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid preset token {token!r}; expected PROPERTY:value")
        key, raw_value = token.split(":", 1)
        prop = PROPERTY_ALIASES.get(key.strip().lower(), key.strip())
        if prop not in PROPERTY_COLUMNS:
            raise ValueError(f"Unsupported OOD property {key!r}; expected one of {PROPERTY_COLUMNS}")
        out[prop] = float(raw_value)
    return out


def preset_distance(props: Mapping[str, float], preset: Mapping[str, float]) -> float:
    if not preset:
        return 0.0
    total = 0.0
    for prop, target in preset.items():
        scale = PROPERTY_NORMALIZERS.get(prop, 1.0)
        total += ((float(props[prop]) - float(target)) / scale) ** 2
    return total / max(len(preset), 1)


def render_instruction(selected_props: list[str], props: Mapping[str, float]) -> str:
    if not selected_props:
        return "Generate a valid molecule."
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


def render_preset(selected_props: list[str], props: Mapping[str, float]) -> str:
    names = {"MW": "MW", "LogP": "LogP", "QED": "QED", "TPSA": "TPSA", "HBD": "HBD", "HBA": "HBA", "RB": "RB"}
    return ",".join(f"{names[prop]}:{format_float(float(props[prop]), digits=4)}" for prop in selected_props)


def sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "ood"


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
