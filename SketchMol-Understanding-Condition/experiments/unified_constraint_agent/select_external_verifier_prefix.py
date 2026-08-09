#!/usr/bin/env python3
"""Select one MuMO candidate with an official-oracle vector verifier."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--selection-mode", choices=("raw", "verifier"), default="verifier")
    parser.add_argument("--group-column", default="condition_id")
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def number(value: object, default: float) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def candidate_rank(row: Mapping[str, object]) -> int:
    return int(number(row.get("candidate_rank") or row.get("generation_rank"), 1e9))


def evaluated_success_fraction(row: Mapping[str, object]) -> float:
    raw = str(row.get("external_property_success_json", "") or "")
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0.0
    if not isinstance(payload, Mapping) or not payload:
        return 0.0
    values = [value for value in payload.values() if value is not None]
    return sum(bool(value) for value in values) / max(len(payload), 1)


def verifier_key(row: Mapping[str, object]) -> tuple[float, ...]:
    """No target molecule fields participate in this bounded selection."""
    return (
        float(truthy(row.get("external_strict_success"))),
        float(truthy(row.get("external_all_property_success"))),
        float(truthy(row.get("external_source_similarity_success"))),
        evaluated_success_fraction(row),
        number(row.get("external_evaluated_property_fraction"), 0.0),
        number(row.get("external_mean_relative_improvement"), -math.inf),
        number(row.get("external_source_tanimoto"), -1.0),
        number(row.get("llm_mean_log_probability"), -math.inf),
        -candidate_rank(row),
    )


def group_key(row: Mapping[str, object], column: str, index: int) -> str:
    for key in (column, "condition_id", "sample_id", "variant_id", "pair_id"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return f"row_{index:08d}"


def select_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    budget: int,
    selection_mode: str,
    group_column: str,
) -> list[dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    order: list[str] = []
    for index, row in enumerate(rows):
        key = group_key(row, group_column, index)
        if key not in groups:
            order.append(key)
        groups[key].append(dict(row))
    selected = []
    for key in order:
        pool = sorted(groups[key], key=candidate_rank)[: max(1, int(budget))]
        if not pool:
            continue
        row = pool[0] if selection_mode == "raw" else max(pool, key=verifier_key)
        row["selection_mode"] = selection_mode
        row["candidate_budget"] = str(int(budget))
        row["oracle_assisted"] = "True" if selection_mode == "verifier" else "False"
        row["oracle_call_type"] = "admet_ai_tdc_vector" if selection_mode == "verifier" else "none"
        selected.append(row)
    return selected


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(str(key) for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 1 <= int(args.budget) <= 20:
        raise ValueError("MuMO paper comparison uses a prefix budget between 1 and 20")
    rows = read_rows(args.detail_csv)
    selected = select_rows(
        rows,
        budget=int(args.budget),
        selection_mode=str(args.selection_mode),
        group_column=str(args.group_column),
    )
    write_rows(args.output_csv, selected)
    summary = {
        "detail_csv": str(args.detail_csv),
        "output_csv": str(args.output_csv),
        "input_candidate_rows": len(rows),
        "selected_rows": len(selected),
        "candidate_budget": int(args.budget),
        "selection_mode": str(args.selection_mode),
        "target_information_used_for_selection": False,
    }
    args.output_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
