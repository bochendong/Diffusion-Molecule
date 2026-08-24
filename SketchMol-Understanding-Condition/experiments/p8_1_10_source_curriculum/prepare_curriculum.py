#!/usr/bin/env python3
"""Prepare train-only reconstruction/edit curricula and leakage audits."""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
TOKEN_RE = re.compile(
    r"(\[[^\]]+\]|Br|Cl|Si|Se|Na|Li|Mg|Ca|Al|Fe|Zn|Cu|Mn|@@?|%\d{2}|\d|"
    r"\.|=|#|-|/|\\|\+|:|~|\(|\)|[BCNOFPSIHK]|[bcnops]|.)"
)


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def canonical(value: object) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True) if molecule is not None else ""


def group(row: dict[str, str]) -> str:
    props = [part.strip() for part in str(row.get("condition_properties", "")).split(",") if part.strip()]
    return "+".join(f"{prop}:{row.get(prop + '_direction', '')}" for prop in props) or "edit"


def balanced(rows: list[dict[str, str]], limit: int, seed: int) -> list[dict[str, str]]:
    pools: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        pools[group(row)].append(row)
    rng = random.Random(seed)
    for values in pools.values():
        rng.shuffle(values)
    output: list[dict[str, str]] = []
    names = sorted(pools)
    cursor = 0
    while names and len(output) < limit:
        name = names[cursor % len(names)]
        if pools[name]:
            output.append(pools[name].pop())
        names = [value for value in names if pools[value]]
        cursor += 1
    return output


def corrupt(smiles: str, seed: int) -> str:
    tokens = [token for token in TOKEN_RE.findall(smiles) if token]
    if len(tokens) < 6:
        return smiles
    rng = random.Random(seed)
    width = max(1, min(4, round(len(tokens) * 0.12)))
    start = rng.randrange(1, max(2, len(tokens) - width))
    return "".join(tokens[:start] + tokens[start + width :])


def write(path: Path, rows: list[dict[str, str]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--eval", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    train = read(args.train)
    eval_rows = read(args.eval)
    eval_targets = {canonical(row.get("target_smiles")) for row in eval_rows}
    eval_targets.discard("")
    eval_pairs = {(canonical(row.get("source_smiles")), canonical(row.get("target_smiles"))) for row in eval_rows}
    eval_ids = {str(row.get("condition_id", "")).strip() for row in eval_rows}
    eval_pair_ids = {str(row.get("pair_id", "")).strip() for row in eval_rows}
    eligible: list[dict[str, str]] = []
    removed = 0
    for source in train:
        row = dict(source)
        src, tgt = canonical(row.get("source_smiles")), canonical(row.get("target_smiles"))
        if not src or not tgt or src == tgt:
            continue
        if src in eval_targets or tgt in eval_targets:
            removed += 1
            continue
        eligible.append(row)
    selected = balanced(eligible, min(args.limit, len(eligible)), args.seed)
    clean: list[dict[str, str]] = []
    corrupted: list[dict[str, str]] = []
    edit: list[dict[str, str]] = []
    for idx, source in enumerate(selected):
        edit_row = dict(source)
        edit_row["task_mode"] = "edit"
        edit_row["curriculum_stage"] = "property_edit"
        edit.append(edit_row)
        for mode, output in (("clean", clean), ("span_corrupt", corrupted)):
            row = dict(source)
            original = str(source["source_smiles"])
            row["target_smiles"] = original
            row["source_smiles"] = original if mode == "clean" else corrupt(original, args.seed * 100000 + idx)
            row["condition_id"] = f"p810-recon-{mode}-{idx:06d}"
            row["variant_id"] = row["sample_id"] = row["pair_id"] = ""
            row["condition_properties"] = ""
            row["property_count"] = "0"
            row["task_mode"] = "edit"
            row["curriculum_stage"] = "source_reconstruction"
            row["corruption_mode"] = mode
            for key in list(row):
                if key.endswith("_active"):
                    row[key] = "False"
                elif key.endswith("_direction"):
                    row[key] = ""
            output.append(row)
    write(args.output / "reconstruction_clean.csv", clean)
    write(args.output / "reconstruction_span_corrupt.csv", corrupted)
    write(args.output / "property_edit.csv", edit)
    selected_pairs = {(canonical(row.get("source_smiles")), canonical(row.get("target_smiles"))) for row in edit}
    selected_ids = {str(row.get("condition_id", "")).strip() for row in edit}
    selected_pair_ids = {str(row.get("pair_id", "")).strip() for row in edit}
    with (args.features / "index.csv").open(newline="", encoding="utf-8") as handle:
        feature_rows = list(csv.DictReader(handle))
    feature_keys = {
        str(row.get(key, "")).strip()
        for row in feature_rows
        for key in ("variant_id", "condition_id", "sample_id", "pair_id")
        if str(row.get(key, "")).strip()
    }
    coverage = sum(
        any(str(row.get(key, "")).strip() in feature_keys for key in ("variant_id", "condition_id", "sample_id", "pair_id"))
        for row in edit
    )
    payload = {
        "protocol": "p8_1_10_train_only_overlap_audit_v1",
        "train_rows": len(train),
        "eligible_rows": len(eligible),
        "selected_rows": len(edit),
        "removed_due_eval_target_overlap": removed,
        "exact_eval_pair_overlap": len(selected_pairs & eval_pairs),
        "condition_id_overlap": len((selected_ids - {""}) & (eval_ids - {""})),
        "pair_id_overlap": len((selected_pair_ids - {""}) & (eval_pair_ids - {""})),
        "eval_target_used_as_edit_target": sum(canonical(row.get("target_smiles")) in eval_targets for row in edit),
        "eval_target_used_as_reconstruction_target": sum(canonical(row.get("target_smiles")) in eval_targets for row in clean),
        "edit_feature_coverage": coverage / max(len(edit), 1),
        "clean_reconstruction_rows": len(clean),
        "corrupted_reconstruction_rows": len(corrupted),
        "single_r2_factor": "delete one local contiguous source-token span during stage-1 reconstruction",
    }
    (args.output / "overlap_audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    forbidden = (
        "exact_eval_pair_overlap",
        "condition_id_overlap",
        "pair_id_overlap",
        "eval_target_used_as_edit_target",
        "eval_target_used_as_reconstruction_target",
    )
    if any(payload[key] for key in forbidden) or payload["edit_feature_coverage"] < 0.999:
        raise SystemExit("leakage/support audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
