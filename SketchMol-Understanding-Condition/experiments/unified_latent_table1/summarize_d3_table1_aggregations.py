#!/usr/bin/env python3
"""Combine candidate-level and Any@k MolEdit summaries for one D3 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("Validity", "Acc_all(0.65)", "Acc_valid(0.65)", "Acc_all(0.15)", "Acc_valid(0.15)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-json", required=True, type=Path)
    parser.add_argument("--any-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--moleditrl-strict", type=float, default=0.450)
    parser.add_argument("--moleditrl-relaxed", type=float, default=0.727)
    return parser.parse_args()


def numeric(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    measured = [row for row in rows if numeric(row.get("n")) and numeric(row.get("n")) > 0]
    result: dict[str, object] = {
        "task_count": len(measured),
        "row_count": int(sum(float(row["n"]) for row in measured)),
        "selection": sorted({str(row.get("selection", "")) for row in measured}),
    }
    macro: dict[str, float] = {}
    micro: dict[str, float] = {}
    for metric in METRICS:
        weight_key = "valid_n" if metric.startswith("Acc_valid") else "n"
        pairs = []
        for row in measured:
            value = numeric(row.get(metric))
            weight = numeric(row.get(weight_key))
            if weight is None and weight_key == "valid_n":
                weight = numeric(row.get("n"))
            pairs.append((value, weight))
        pairs = [(value, weight) for value, weight in pairs if value is not None and weight is not None and weight > 0]
        if not pairs:
            continue
        macro[metric] = sum(value for value, _ in pairs) / len(pairs)
        micro[metric] = sum(value * weight for value, weight in pairs) / sum(weight for _, weight in pairs)
    result["macro"] = macro
    result["micro"] = micro
    result["per_task"] = rows
    return result


def percentage(value: float) -> str:
    return f"{100.0 * value:.3f}"


def main() -> int:
    args = parse_args()
    candidate = aggregate(json.loads(args.candidate_json.read_text(encoding="utf-8")))
    anyk = aggregate(json.loads(args.any_json.read_text(encoding="utf-8")))
    strict = float(candidate["macro"]["Acc_all(0.65)"])
    relaxed = float(candidate["macro"]["Acc_all(0.15)"])
    result = {
        "protocol": "d3_table1_candidate_and_anyk_v1",
        "model": args.model_name,
        "candidate_limit": int(args.candidate_limit),
        "candidate_level": candidate,
        "any_at_k": anyk,
        "external_reference": {
            "method": "MolEditRL",
            "aggregation": "candidate-level macro mean over Table1 tasks",
            "Acc_all(0.65)": float(args.moleditrl_strict),
            "Acc_all(0.15)": float(args.moleditrl_relaxed),
        },
        "candidate_level_delta_vs_moleditrl": {
            "Acc_all(0.65)": strict - float(args.moleditrl_strict),
            "Acc_all(0.15)": relaxed - float(args.moleditrl_relaxed),
        },
        "fair_comparison_gate": {
            "strict_beats_moleditrl": strict > float(args.moleditrl_strict),
            "relaxed_beats_moleditrl": relaxed > float(args.moleditrl_relaxed),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        f"# {args.model_name}: MolEdit Table1 aggregation audit",
        "",
        "The candidate-level row is the fair comparison to MolEditRL. Any@20 is reported separately and is not substituted into the external baseline table.",
        "",
        "| Aggregation | Validity | Acc_all(0.65) | Acc_valid(0.65) | Acc_all(0.15) | Acc_valid(0.15) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, block in (("Candidate-level", candidate), (f"Any@{args.candidate_limit}", anyk)):
        values = block["macro"]
        lines.append(
            "| " + label + " | " + " | ".join(percentage(float(values[metric])) for metric in METRICS) + " |"
        )
    lines.extend(
        [
            f"| MolEditRL candidate-level | -- | {percentage(args.moleditrl_strict)} | -- | {percentage(args.moleditrl_relaxed)} | -- |",
            "",
            f"Candidate-level strict delta versus MolEditRL: **{100.0 * (strict - args.moleditrl_strict):+.3f} pp**.",
            f"Candidate-level relaxed delta versus MolEditRL: **{100.0 * (relaxed - args.moleditrl_relaxed):+.3f} pp**.",
            "",
        ]
    )
    args.output_markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "strict": strict, "relaxed": relaxed}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
