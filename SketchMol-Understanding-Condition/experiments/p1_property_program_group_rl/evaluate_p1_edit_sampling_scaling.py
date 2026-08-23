#!/usr/bin/env python3
"""Evaluate ordered MolEdit candidate prefixes for the P1 MolProgram table.

The script deliberately ignores every property-aware candidate rank.  Prefixes
are defined by the original generation index, so k=1 is a true first draw and
pass@k only asks whether the first k draws contain a successful edit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parents[2]
MOLEDIT_METRIC_DIR = REPO_DIR / "SketchMol-Unified-3MDiffusion" / "scripts"
if str(MOLEDIT_METRIC_DIR) not in sys.path:
    sys.path.insert(0, str(MOLEDIT_METRIC_DIR))

from evaluate_moledit_table_metrics import (  # noqa: E402
    Chemistry,
    TASK_LABELS,
    evaluate_prediction,
    normalize_id,
    task_key,
    task_specs_for_reference,
)


DEFAULT_BUDGETS = (1, 4, 8, 20)
THRESHOLDS = (0.15, 0.65)
MODELS = ("sft", "group_rl")
ID_COLUMNS = ("example_id", "condition_id", "sample_id", "pair_hash")
SMILES_COLUMNS = (
    "direct_candidate_canonical_smiles",
    "generated_smiles",
    "direct_candidate_raw_smiles",
    "candidate_smiles",
    "predicted_smiles",
    "smiles",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--sft-candidates", required=True, type=Path)
    parser.add_argument("--group-rl-candidates", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--budgets", default=",".join(str(value) for value in DEFAULT_BUDGETS))
    return parser.parse_args(argv)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def first_value(row: Mapping[str, object], columns: Iterable[str]) -> str:
    for column in columns:
        value = str(row.get(column, "") or "").strip()
        if value:
            return value
    return ""


def row_id(row: Mapping[str, object]) -> str:
    return normalize_id(first_value(row, ID_COLUMNS))


def generation_index(row: Mapping[str, object]) -> int:
    # Native Direct-SMILES pools use direct_candidate_index.  Unified pools
    # retain the actual draw order in generation_rank; candidate_rank is the
    # property-aware finalizer rank and must never define a prefix.
    for column in ("direct_candidate_index", "generation_rank"):
        raw = str(row.get(column, "") or "").strip()
        if raw:
            return int(float(raw))
    raise ValueError("candidate row has neither direct_candidate_index nor generation_rank")


def parse_budgets(text: str) -> tuple[int, ...]:
    budgets = tuple(sorted({int(item) for item in text.split(",") if item.strip()}))
    if not budgets or budgets[0] < 1:
        raise ValueError("budgets must contain positive integers")
    return budgets


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_references(path: Path) -> dict[str, dict[str, str]]:
    references: dict[str, dict[str, str]] = {}
    for row in read_csv(path):
        key = row_id(row)
        if key:
            references[key] = row
    if not references:
        raise RuntimeError(f"no reference rows in {path}")
    return references


def load_candidate_groups(path: Path, *, required_count: int) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        key = row_id(row)
        if key:
            grouped[key].append(row)
    for key, rows in grouped.items():
        rows.sort(key=generation_index)
        prefix = rows[:required_count]
        indices = [generation_index(row) for row in prefix]
        if len(prefix) != required_count:
            raise RuntimeError(f"{path}: {key} has {len(prefix)} candidates; expected {required_count}")
        start = indices[0]
        if start not in (0, 1) or indices != list(range(start, start + required_count)):
            raise RuntimeError(f"{path}: {key} has non-contiguous generation indices {indices[:8]}")
        grouped[key] = prefix
    return dict(grouped)


def candidate_metrics(
    reference: Mapping[str, str],
    candidates: Sequence[Mapping[str, str]],
    *,
    chem: Chemistry,
) -> list[dict[str, object]]:
    specs = task_specs_for_reference(reference)
    if not specs:
        raise RuntimeError(f"reference {row_id(reference)} has no recognized MolEdit task")
    missing = chem.missing_oracles(specs)
    if missing:
        raise RuntimeError(f"reference {row_id(reference)} is missing property oracles: {sorted(missing)}")

    evaluated: list[dict[str, object]] = []
    for candidate in candidates:
        smiles = first_value(candidate, SMILES_COLUMNS)
        result = evaluate_prediction(reference, smiles, specs, chem=chem, thresholds=list(THRESHOLDS))
        canonical = ""
        if result["valid"]:
            molecule = chem.mol(smiles)
            canonical = chem.Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else ""
        evaluated.append({**result, "canonical_smiles": canonical})
    return evaluated


def condition_rows(
    model: str,
    references: Mapping[str, Mapping[str, str]],
    candidates: Mapping[str, Sequence[Mapping[str, str]]],
    budgets: Sequence[int],
    *,
    chem: Chemistry,
) -> list[dict[str, object]]:
    missing = sorted(set(references) - set(candidates))
    extra = sorted(set(candidates) - set(references))
    if missing or extra:
        raise RuntimeError(
            f"{model} condition mismatch: missing={missing[:3]} ({len(missing)}), "
            f"extra={extra[:3]} ({len(extra)})"
        )

    output: list[dict[str, object]] = []
    for key, reference in references.items():
        evaluated = candidate_metrics(reference, candidates[key], chem=chem)
        specs = task_specs_for_reference(reference)
        canonical_task = task_key(specs)
        for budget in budgets:
            prefix = evaluated[:budget]
            valid = [row for row in prefix if row["valid"]]
            row: dict[str, object] = {
                "model": model,
                "condition_id": key,
                "task": canonical_task,
                "task_label": TASK_LABELS.get(canonical_task, canonical_task),
                "candidate_budget": budget,
                "validity_fraction": sum(bool(item["valid"]) for item in prefix) / budget,
                "selected_validity_at_k": float(bool(valid)),
                "unique_valid_fraction": len({str(item["canonical_smiles"]) for item in valid}) / budget,
            }
            for threshold in THRESHOLDS:
                metric = f"success_t{threshold:g}"
                successes = sum(bool(item[metric]) for item in prefix)
                row[f"raw_success_fraction_t{threshold:g}"] = successes / budget
                row[f"empirical_pass_at_k_t{threshold:g}"] = float(successes > 0)
            output.append(row)
    return output


METRICS = (
    "validity_fraction",
    "selected_validity_at_k",
    "unique_valid_fraction",
    "raw_success_fraction_t0.15",
    "empirical_pass_at_k_t0.15",
    "raw_success_fraction_t0.65",
    "empirical_pass_at_k_t0.65",
)


def summarize(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        model = str(row["model"])
        budget = int(row["candidate_budget"])
        grouped[(model, "overall", "all", budget)].append(row)
        grouped[(model, "task", str(row["task_label"]), budget)].append(row)

    output: list[dict[str, object]] = []
    for (model, group_type, group, budget), values in sorted(grouped.items()):
        out: dict[str, object] = {
            "model": model,
            "group_type": group_type,
            "group": group,
            "candidate_budget": budget,
            "conditions": len(values),
        }
        for metric in METRICS:
            out[metric] = sum(float(row[metric]) for row in values) / len(values)
        output.append(out)
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def percent(value: object) -> str:
    return f"{100.0 * float(value):.2f}"


def write_paper_table(path: Path, summary: Sequence[Mapping[str, object]]) -> None:
    overall = [row for row in summary if row["group_type"] == "overall"]
    overall.sort(key=lambda row: (int(row["candidate_budget"]), MODELS.index(str(row["model"]))))
    lines = [
        "# MolProgram editing: ordered-prefix sampling table",
        "",
        "All values are percentages over 1,000 MolEdit Table1 conditions. Candidate prefixes follow",
        "the original generation order; no property-aware reranking is used. The primary success",
        "threshold is source Tanimoto >= 0.15 together with satisfaction of every requested edit.",
        "",
        "| method | k | raw validity | unique valid | raw strict success | pass@k | pass@k (Tanimoto >= 0.65) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = {"sft": "SFT", "group_rl": "Group-RL"}
    for row in overall:
        lines.append(
            "| {method} | {budget} | {validity}% | {unique}% | {raw}% | {passed}% | {passed065}% |".format(
                method=labels[str(row["model"])],
                budget=row["candidate_budget"],
                validity=percent(row["validity_fraction"]),
                unique=percent(row["unique_valid_fraction"]),
                raw=percent(row["raw_success_fraction_t0.15"]),
                passed=percent(row["empirical_pass_at_k_t0.15"]),
                passed065=percent(row["empirical_pass_at_k_t0.65"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    budgets = parse_budgets(args.budgets)
    max_budget = max(budgets)
    references = load_references(args.reference)
    candidate_paths = {
        "sft": args.sft_candidates,
        "group_rl": args.group_rl_candidates,
    }
    candidate_groups = {
        model: load_candidate_groups(path, required_count=max_budget)
        for model, path in candidate_paths.items()
    }

    chem = Chemistry()
    per_condition: list[dict[str, object]] = []
    for model in MODELS:
        per_condition.extend(
            condition_rows(model, references, candidate_groups[model], budgets, chem=chem)
        )
    summary = summarize(per_condition)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "edit_sampling_per_condition.csv", per_condition)
    write_csv(args.output_dir / "edit_sampling_summary.csv", summary)
    write_paper_table(args.output_dir / "edit_sampling_paper_table.md", summary)
    provenance = {
        "reference": {"path": str(args.reference), "sha256": file_sha256(args.reference)},
        "candidate_pools": {
            model: {"path": str(path), "sha256": file_sha256(path)}
            for model, path in candidate_paths.items()
        },
        "budgets": list(budgets),
        "thresholds": list(THRESHOLDS),
        "conditions": len(references),
        "prefix_order": "direct_candidate_index else generation_rank",
        "property_reranking": False,
    }
    (args.output_dir / "edit_sampling_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
