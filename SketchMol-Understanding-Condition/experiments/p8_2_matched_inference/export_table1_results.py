#!/usr/bin/env python3
"""Export the completed matched Table1 evaluation without reranking."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
METRICS = ("Validity", "Acc_all(0.65)", "Acc_valid(0.65)", "Acc_all(0.15)", "Acc_valid(0.15)")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate(rows: list[dict[str, str]]) -> dict[str, float]:
    n = sum(float(row["n"]) for row in rows)
    valid_n = sum(float(row["valid_n"]) for row in rows)
    result = {"Validity": valid_n / max(n, 1.0)}
    for metric in METRICS[1:]:
        denominator_key = "valid_n" if metric.startswith("Acc_valid") else "n"
        denominator = valid_n if denominator_key == "valid_n" else n
        result[metric] = sum(float(row[metric]) * float(row[denominator_key]) for row in rows) / max(denominator, 1.0)
    return result


def canon(value: object) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True) if molecule is not None else ""


def format_percent(value: float) -> str:
    return f"{100.0 * value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-dir", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--sampling-summary", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    selections = {name: read_csv(args.table_dir / name / "moledit_table_summary.csv") for name in ("any1", "any8", "any20", "candidate20")}
    tasks = [row["task"] for row in selections["candidate20"]]
    indexed = {name: {row["task"]: row for row in rows} for name, rows in selections.items()}
    candidate_rows = read_csv(args.candidates)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    valid = identity = 0
    for row in candidate_rows:
        grouped[str(row.get("condition_id") or row.get("sample_id") or "")].append(row)
        candidate = canon(row.get("generated_smiles"))
        source = canon(row.get("source_smiles"))
        valid += int(bool(candidate))
        identity += int(bool(candidate and source and candidate == source))
    unique = sum(len({canon(row.get("generated_smiles")) for row in rows if canon(row.get("generated_smiles"))}) / 20 for rows in grouped.values()) / max(len(grouped), 1)
    summary = json.loads(args.sampling_summary.read_text(encoding="utf-8"))
    task_rows = []
    for task in tasks:
        task_rows.append({
            "task": task,
            "task_key": indexed["candidate20"][task]["task_key"],
            "candidate20": {metric: float(indexed["candidate20"][task][metric]) for metric in METRICS},
            "any1": {metric: float(indexed["any1"][task][metric]) for metric in METRICS},
            "any8": {metric: float(indexed["any8"][task][metric]) for metric in METRICS},
            "any20": {metric: float(indexed["any20"][task][metric]) for metric in METRICS},
        })
    payload = {
        "protocol": "p8_2_matched_table1_raw20_v1",
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "conditions": len(grouped),
        "candidate_rows": len(candidate_rows),
        "candidates_per_condition": 20,
        "property_reranking": bool(summary.get("property_reranking")),
        "target_molecule_used_at_inference": bool(summary.get("target_molecule_used_at_inference")),
        "candidate_identity_fraction": identity / max(len(candidate_rows), 1),
        "candidate_unique_fraction": unique,
        "aggregates": {name: aggregate(rows) for name, rows in selections.items()},
        "tasks": task_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# P8.2 matched-inference Table1 results",
        "",
        "All numbers below are percentages from the original generation order. Each of the 200 conditions has 20 raw candidates; no property reranking is used.",
        "",
        "## Aggregate",
        "",
        "| Selection | Validity | Acc_all (0.65) | Acc_valid (0.65) | Acc_all (0.15) | Acc_valid (0.15) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {"any1": "Any@1", "any8": "Any@8", "any20": "Any@20", "candidate20": "Candidate-level"}
    for name in ("any1", "any8", "any20", "candidate20"):
        values = payload["aggregates"][name]
        lines.append("| " + labels[name] + " | " + " | ".join(format_percent(values[metric]) for metric in METRICS) + " |")
    lines.extend([
        "",
        f"Candidate identity: {format_percent(payload['candidate_identity_fraction'])}%. Candidate uniqueness: {format_percent(payload['candidate_unique_fraction'])}%.",
        "",
        "## Per-task",
        "",
        "| Task | Cand. Validity | Cand. Acc_all .65 | Cand. Acc_valid .65 | Cand. Acc_all .15 | Cand. Acc_valid .15 | Strict Any@1 | Strict Any@8 | Strict Any@20 | Relaxed Any@1 | Relaxed Any@8 | Relaxed Any@20 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in task_rows:
        candidate = row["candidate20"]
        values = [
            row["task"],
            *[format_percent(candidate[metric]) for metric in METRICS],
            *[format_percent(row[name]["Acc_all(0.65)"]) for name in ("any1", "any8", "any20")],
            *[format_percent(row[name]["Acc_all(0.15)"]) for name in ("any1", "any8", "any20")],
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend([
        "",
        f"Checkpoint SHA-256: `{payload['checkpoint_sha256']}`.",
        "",
        "The JSON companion preserves all five metrics for candidate-level and Any@1/8/20 for every task.",
    ])
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md), "tasks": len(task_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
