#!/usr/bin/env python3
"""Collect the preregistered P4 raw and honest any@k result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--teacher-manifest", required=True, type=Path)
    parser.add_argument("--sft-audit", required=True, type=Path)
    parser.add_argument("--grpo-audit", required=True, type=Path)
    return parser.parse_args()


def mean_metric(path: Path, key: str) -> float:
    rows = json.loads(path.read_text(encoding="utf-8"))
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return sum(values) / max(len(values), 1)


def main() -> int:
    args = parse_args()
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    teacher = json.loads(args.teacher_manifest.read_text(encoding="utf-8"))
    sft_audit = json.loads(args.sft_audit.read_text(encoding="utf-8"))
    grpo_audit = json.loads(args.grpo_audit.read_text(encoding="utf-8"))
    records = []
    candidate_records = []
    for variant in ("base", "sft", "grpo"):
        for budget in (1, 8, 20):
            path = args.output_root / "eval" / variant / f"any{budget}" / "moledit_table_summary.json"
            records.append(
                {
                    "variant": variant,
                    "budget": budget,
                    "acc_all_0_65": mean_metric(path, "Acc_all(0.65)"),
                    "acc_all_0_15": mean_metric(path, "Acc_all(0.15)"),
                    "validity": mean_metric(path, "Validity"),
                    "metrics_json": str(path),
                }
            )
        candidate_path = args.output_root / "eval" / variant / "candidate20" / "moledit_table_summary.json"
        candidate_records.append(
            {
                "variant": variant,
                "candidates_per_input": 20,
                "aggregation": "candidate-level",
                "acc_all_0_65": mean_metric(candidate_path, "Acc_all(0.65)"),
                "acc_all_0_15": mean_metric(candidate_path, "Acc_all(0.15)"),
                "validity": mean_metric(candidate_path, "Validity"),
                "metrics_json": str(candidate_path),
            }
        )
    by_key = {(row["variant"], row["budget"]): row for row in records}
    final = by_key[("grpo", 20)]
    raw = by_key[("grpo", 1)]
    gates = prereg["gates"]
    checks = {
        "teacher_strict_coverage": float(teacher["covered_fraction"]) >= float(gates["min_teacher_strict_coverage"]),
        "raw_acc_0_65": float(raw["acc_all_0_65"]) >= float(gates["minimum_raw_acc_0_65"]),
        "any20_acc_0_65": float(final["acc_all_0_65"]) >= float(gates["minimum_any20_acc_0_65"]),
        "validity": float(final["validity"]) >= float(gates["minimum_validity"]),
        "sft_de_novo_path_bit_identical": bool(sft_audit["de_novo_path_bit_identical"]),
        "grpo_de_novo_path_bit_identical": bool(grpo_audit["de_novo_path_bit_identical"]),
    }
    decision = "go" if all(checks.values()) else "stop"
    strong = float(raw["acc_all_0_65"]) >= float(gates["strong_raw_acc_0_65"])
    payload = {
        "protocol": prereg["protocol"],
        "decision": decision,
        "strong_raw_gate": strong,
        "checks": checks,
        "teacher": teacher,
        "records": records,
        "candidate_level_n20": candidate_records,
        "sft_audit": sft_audit,
        "grpo_audit": grpo_audit,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "p4_summary.json"
    report_path = args.output_root / "p4_report.md"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# P4 Event-to-SMILES Distillation",
        "",
        f"Decision: **{decision}**. Strong raw gate: **{'pass' if strong else 'fail'}**.",
        "",
        "| Variant | k | Validity | Acc@0.65 | Acc@0.15 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in records:
        lines.append(
            f"| {row['variant']} | {row['budget']} | {row['validity']:.1%} | "
            f"{row['acc_all_0_65']:.1%} | {row['acc_all_0_15']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Matched MolEdit Table 1 protocol",
            "",
            "All values below are candidate-level means over the complete unranked pool of 20 generations per input.",
            "",
            "| Variant | n | Validity | Acc@0.65 | Acc@0.15 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in candidate_records:
        lines.append(
            f"| {row['variant']} | {row['candidates_per_input']} | {row['validity']:.1%} | "
            f"{row['acc_all_0_65']:.1%} | {row['acc_all_0_15']:.1%} |"
        )
    lines.extend(
        [
            "",
            f"- Teacher strict coverage: {float(teacher['covered_fraction']):.1%}",
            f"- SFT de-novo path bit-identical: {sft_audit['de_novo_path_bit_identical']}",
            f"- GRPO de-novo path bit-identical: {grpo_audit['de_novo_path_bit_identical']}",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
