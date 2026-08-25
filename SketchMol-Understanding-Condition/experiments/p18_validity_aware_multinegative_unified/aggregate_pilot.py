#!/usr/bin/env python3
"""Create paired P17/P18 Table1 and hard de-novo pilot tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text())


def table_macro(rows: list[dict]) -> dict[str, float]:
    fields = {
        "validity": "Validity", "property_anyk": "property_anyk",
        "acc_0.15": "Acc_all(0.15)", "strict_acc_0.65": "Acc_all(0.65)",
        "mean_best_source_similarity": "mean_best_source_tanimoto",
    }
    return {out: sum(float(row[src]) for row in rows) / len(rows) for out, src in fields.items()}


def table_block(root: Path) -> dict:
    out = {}
    for k in (1, 4, 8):
        rows = load(root / f"any{k}" / "moledit_table_summary.json")
        out[str(k)] = {
            "macro": table_macro(rows),
            "strict_by_task": {row["task_key"]: float(row["Acc_all(0.65)"]) for row in rows},
        }
    return out


def denovo_fraction(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped = defaultdict(list)
    for row in rows:
        key = str(row.get("condition_id") or row.get("sample_id"))
        grouped[key].append(row)
    result = {}
    for stratum in ("all", "6p", "7p"):
        selected = {
            key: items for key, items in grouped.items()
            if stratum == "all" or str(items[0].get("property_count")) == stratum[:-1]
        }
        result[stratum] = {}
        for k in (1, 4, 8):
            best = []
            for items in selected.values():
                prefix = sorted(items, key=lambda row: int(row["candidate_rank"]))[:k]
                best.append(max(float(row["direct_candidate_strict_fraction"]) for row in prefix))
            result[stratum][str(k)] = sum(best) / len(best)
    return result


def denovo_block(metrics_path: Path, annotated_path: Path) -> dict:
    metrics = load(metrics_path)
    records = {f"{row['stratum']}@{row['k']}": row for row in metrics["records"]}
    return {"records": records, "mean_best_property_fraction": denovo_fraction(annotated_path)}


def numeric_delta(right: dict, left: dict) -> dict:
    return {key: right[key] - left[key] for key in right if isinstance(right[key], (int, float)) and key in left}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--p17-table-root", required=True, type=Path)
    parser.add_argument("--p18-table-root", required=True, type=Path)
    parser.add_argument("--p17-denovo-metrics", required=True, type=Path)
    parser.add_argument("--p18-denovo-metrics", required=True, type=Path)
    parser.add_argument("--p17-denovo-annotated", required=True, type=Path)
    parser.add_argument("--p18-denovo-annotated", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    table17, table18 = table_block(args.p17_table_root), table_block(args.p18_table_root)
    for k in ("1", "4", "8"):
        table18[k]["delta_p18_minus_p17"] = numeric_delta(table18[k]["macro"], table17[k]["macro"])
    d17 = denovo_block(args.p17_denovo_metrics, args.p17_denovo_annotated)
    d18 = denovo_block(args.p18_denovo_metrics, args.p18_denovo_annotated)
    ddelta = {
        key: numeric_delta(d18["records"][key], d17["records"][key]) for key in d18["records"]
    }
    fraction_delta = {
        stratum: {k: d18["mean_best_property_fraction"][stratum][k] - d17["mean_best_property_fraction"][stratum][k] for k in ("1", "4", "8")}
        for stratum in ("all", "6p", "7p")
    }
    payload = {
        "protocol": "p18_paired_frozen_benchmark_pilot_v1",
        "status_label": "paired pilot estimate; not full Table1 or full de-novo benchmark",
        "validation_gate": load(args.gate),
        "table1": {"p17": table17, "p18": table18},
        "hard_denovo": {
            "p17": d17, "p18": d18,
            "record_delta_p18_minus_p17": ddelta,
            "mean_best_property_fraction_delta_p18_minus_p17": fraction_delta,
        },
        "tuned_on_benchmark": False,
        "static_candidate_pool": False,
        "property_reranking": False,
        "candidate_order": "greedy first then seven raw samples",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "protocol": payload["protocol"], "gate_passed": payload["validation_gate"]["gate_passed"],
        "table1_budgets": [1, 4, 8], "denovo_records": len(d18["records"]),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
