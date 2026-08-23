#!/usr/bin/env python3
"""Collect honest raw, any@k, and candidate-level P5 metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--base-eval-root", required=True, type=Path)
    parser.add_argument("--sft-audit", required=True, type=Path)
    parser.add_argument("--grpo-audit", required=True, type=Path)
    return parser.parse_args()


def mean_metric(path: Path, key: str) -> float:
    rows = json.loads(path.read_text(encoding="utf-8"))
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return sum(values) / max(len(values), 1)


def unique_fraction(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values = [str(row.get("generated_smiles", "") or "").strip() for row in rows]
    nonempty = [value for value in values if value]
    return len(set(nonempty)) / max(len(values), 1)


def main() -> int:
    args = parse_args()
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    audits = {
        "sft": json.loads(args.sft_audit.read_text(encoding="utf-8")),
        "grpo": json.loads(args.grpo_audit.read_text(encoding="utf-8")),
    }
    records = []
    candidates = []
    for variant in ("base", "sft", "grpo"):
        root = args.base_eval_root / variant if variant == "base" else args.output_root / "eval" / variant
        for budget in (1, 8, 20):
            path = root / f"any{budget}" / "moledit_table_summary.json"
            records.append({
                "variant": variant,
                "budget": budget,
                "validity": mean_metric(path, "Validity"),
                "acc_all_0_65": mean_metric(path, "Acc_all(0.65)"),
                "acc_all_0_15": mean_metric(path, "Acc_all(0.15)"),
            })
        candidate_path = root / "candidate20" / "moledit_table_summary.json"
        candidate_csv = root / "candidates.csv"
        candidates.append({
            "variant": variant,
            "validity": mean_metric(candidate_path, "Validity"),
            "acc_all_0_65": mean_metric(candidate_path, "Acc_all(0.65)"),
            "acc_all_0_15": mean_metric(candidate_path, "Acc_all(0.15)"),
            "unique_fraction": unique_fraction(candidate_csv),
        })
    by_key = {(row["variant"], row["budget"]): row for row in records}
    candidate_by_variant = {row["variant"]: row for row in candidates}
    final = by_key[("grpo", 20)]
    raw = by_key[("grpo", 1)]
    final_candidate = candidate_by_variant["grpo"]
    gates = prereg["gates"]
    checks = {
        "raw_acc_0_65": raw["acc_all_0_65"] >= float(gates["minimum_raw_acc_0_65"]),
        "any20_acc_0_65": final["acc_all_0_65"] >= float(gates["minimum_any20_acc_0_65"]),
        "any20_validity": final["validity"] >= float(gates["minimum_any20_validity"]),
        "candidate_validity": final_candidate["validity"] >= float(gates["minimum_candidate_validity"]),
        "sft_de_novo_path_bit_identical": bool(audits["sft"]["de_novo_path_bit_identical"]),
        "grpo_de_novo_path_bit_identical": bool(audits["grpo"]["de_novo_path_bit_identical"]),
    }
    decision = "go" if all(checks.values()) else "stop"
    payload = {
        "protocol": prereg["protocol"],
        "decision": decision,
        "strong_raw_gate": raw["acc_all_0_65"] >= float(gates["strong_raw_acc_0_65"]),
        "strong_any20_gate": final["acc_all_0_65"] >= float(gates["strong_any20_acc_0_65"]),
        "checks": checks,
        "records": records,
        "candidate_level_n20": candidates,
        "audits": audits,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "p5_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# P5 Source-Anchored MolProgram",
        "",
        f"Decision: **{decision}**.",
        "",
        "| Variant | k | Validity | Acc@0.65 | Acc@0.15 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in records:
        lines.append(
            f"| {row['variant']} | {row['budget']} | {row['validity']:.1%} | "
            f"{row['acc_all_0_65']:.1%} | {row['acc_all_0_15']:.1%} |"
        )
    lines.extend([
        "", "## Candidate-level n=20", "",
        "| Variant | Validity | Unique | Acc@0.65 | Acc@0.15 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for row in candidates:
        lines.append(
            f"| {row['variant']} | {row['validity']:.1%} | {row['unique_fraction']:.1%} | "
            f"{row['acc_all_0_65']:.1%} | {row['acc_all_0_15']:.1%} |"
        )
    (args.output_root / "p5_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

