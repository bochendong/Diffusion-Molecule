#!/usr/bin/env python3
"""Collect the validation-only source-consistency and SMILES-validity pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
UNIFIED_DIR = SCRIPT_DIR.parent / "unified_smiles_generator"
if str(UNIFIED_DIR) not in sys.path:
    sys.path.insert(0, str(UNIFIED_DIR))

import collect_umtp_v1_rl_pilot as paired  # noqa: E402


EDIT_VARIANTS = ("policy", "consistent", "strong_consistent")
DENOVO_VARIANTS = ("baseline", "grammar_valid")
SELECTIONS = ("raw", "finalizer")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--budgets", default="1,8")
    return parser.parse_args(argv)


def parse_budgets(raw: str) -> tuple[int, ...]:
    budgets = tuple(dict.fromkeys(int(value.strip()) for value in raw.split(",") if value.strip()))
    if not budgets or 1 not in budgets:
        raise ValueError("Pilot budgets must include n=1")
    return budgets


def collect(root: Path, budgets: Sequence[int]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for variant in EDIT_VARIANTS:
        for budget in budgets:
            for selection in SELECTIONS:
                path = paired.summary_path(root, variant, "table1", int(budget), selection)
                metrics = paired.table1_summary(path)
                records.append(
                    {
                        "variant": variant,
                        "task": "edit",
                        "budget": int(budget),
                        "selection": selection,
                        **metrics,
                    }
                )
    for variant in DENOVO_VARIANTS:
        for budget in budgets:
            for selection in SELECTIONS:
                path = paired.summary_path(root, variant, "retention", int(budget), selection)
                metrics = paired.denovo_summary(path)
                records.append(
                    {
                        "variant": variant,
                        "task": "de_novo",
                        "budget": int(budget),
                        "selection": selection,
                        **metrics,
                    }
                )
    return records


def finite(value: object, fallback: float = -1e9) -> float:
    parsed = paired.parse_float(value)
    return parsed if math.isfinite(parsed) else fallback


def index_records(
    records: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str, int, str], Mapping[str, object]]:
    return {
        (str(row["variant"]), str(row["task"]), int(row["budget"]), str(row["selection"])): row
        for row in records
    }


def choose_edit_variant(
    index: Mapping[tuple[str, str, int, str], Mapping[str, object]],
    largest_budget: int,
) -> str:
    def key(variant: str) -> tuple[float, ...]:
        raw = index[(variant, "edit", 1, "raw")]
        final = index[(variant, "edit", largest_budget, "finalizer")]
        return (
            finite(raw.get("acc_all_0_65")),
            finite(raw.get("acc_all_0_15")),
            finite(final.get("acc_all_0_65")),
            finite(final.get("acc_all_0_15")),
        )

    return max(EDIT_VARIANTS, key=key)


def decision(records: Sequence[Mapping[str, object]], budgets: Sequence[int]) -> dict[str, object]:
    index = index_records(records)
    best_edit = choose_edit_variant(index, max(budgets))
    policy = index[("policy", "edit", 1, "raw")]
    best = index[(best_edit, "edit", 1, "raw")]
    edit_strict_delta = finite(best.get("acc_all_0_65")) - finite(policy.get("acc_all_0_65"))
    edit_relaxed_delta = finite(best.get("acc_all_0_15")) - finite(policy.get("acc_all_0_15"))

    baseline = index[("baseline", "de_novo", 1, "raw")]
    grammar = index[("grammar_valid", "de_novo", 1, "raw")]
    validity_delta = finite(grammar.get("validity")) - finite(baseline.get("validity"))
    strict_delta = finite(grammar.get("strict_success_rate")) - finite(baseline.get("strict_success_rate"))

    edit_go = edit_strict_delta >= 0.02 or edit_relaxed_delta >= 0.03
    validity_go = validity_delta >= 0.10 and strict_delta >= -0.02
    return {
        "decision": "go" if edit_go and validity_go else "stop",
        "best_edit_variant": best_edit,
        "edit_go": edit_go,
        "validity_go": validity_go,
        "criteria": {
            "edit": "raw n=1 Acc@0.65 +2pp or Acc@0.15 +3pp on validation",
            "validity": "raw n=1 validity +10pp with de novo strict drop no worse than 2pp",
        },
        "observed": {
            "edit_raw_n1_acc_0_65_delta": edit_strict_delta,
            "edit_raw_n1_acc_0_15_delta": edit_relaxed_delta,
            "denovo_raw_n1_validity_delta": validity_delta,
            "denovo_raw_n1_strict_delta": strict_delta,
        },
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pct(value: object) -> str:
    parsed = paired.parse_float(value)
    return "" if not math.isfinite(parsed) else f"{100.0 * parsed:.1f}%"


def write_report(
    path: Path,
    records: Sequence[Mapping[str, object]],
    verdict: Mapping[str, object],
) -> None:
    lines = [
        "# P1 Source-Consistency + Validity Pilot",
        "",
        f"Decision: **{verdict['decision']}**",
        "",
        "Validation-only pilot. Edit ranking uses source fingerprint, scaffold retention, and local edit magnitude; it never reads target molecules or output property oracles. De novo generation uses the same common decoder with grammar-constrained SMILES sampling and reduced repetition suppression.",
        "",
        "| Variant | Task | n | Selection | Validity | Strict / Acc@0.65 | Acc@0.15 |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in records:
        strict = row.get("acc_all_0_65", row.get("strict_success_rate", ""))
        lines.append(
            f"| {row['variant']} | {row['task']} | {row['budget']} | {row['selection']} | "
            f"{pct(row.get('validity'))} | {pct(strict)} | {pct(row.get('acc_all_0_15'))} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- Best consistency variant: `{verdict['best_edit_variant']}`",
            f"- Edit raw n=1 Acc@0.65 delta: {pct(verdict['observed']['edit_raw_n1_acc_0_65_delta'])}",
            f"- Edit raw n=1 Acc@0.15 delta: {pct(verdict['observed']['edit_raw_n1_acc_0_15_delta'])}",
            f"- De novo raw n=1 validity delta: {pct(verdict['observed']['denovo_raw_n1_validity_delta'])}",
            f"- De novo raw n=1 strict delta: {pct(verdict['observed']['denovo_raw_n1_strict_delta'])}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    budgets = parse_budgets(str(args.budgets))
    records = collect(args.pilot_root, budgets)
    verdict = decision(records, budgets)
    metrics_csv = args.output_prefix.with_name(args.output_prefix.name + "_metrics.csv")
    summary_json = args.output_prefix.with_name(args.output_prefix.name + "_summary.json")
    report_md = args.output_prefix.with_name(args.output_prefix.name + "_report.md")
    write_csv(metrics_csv, records)
    payload = {
        "protocol": "p1_source_consistency_validity_validation_pilot_v1",
        "pilot_root": str(args.pilot_root),
        "budgets": list(budgets),
        "records": records,
        "decision": verdict,
        "metrics_csv": str(metrics_csv),
        "report": str(report_md),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(report_md, records, verdict)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
