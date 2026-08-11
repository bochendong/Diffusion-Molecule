#!/usr/bin/env python3
"""Select a task-balanced audit set with no proposer-train source overlap."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_hierarchical_action_support as support_audit  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposer-train-csv", required=True, type=Path)
    parser.add_argument("--audit-candidate-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--rows-per-task", required=True, type=int)
    return parser.parse_args(argv)


def task_id(row: Mapping[str, object]) -> str:
    value = str(
        row.get("external_task_id", "")
        or row.get("external_task_key", "")
        or row.get("benchmark_task", "")
        or ""
    ).strip()
    if not value:
        raise ValueError("Audit candidate is missing a task id")
    return value


def select_disjoint_rows(
    proposer_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    *,
    rows_per_task: int,
) -> tuple[list[Mapping[str, object]], dict[str, object]]:
    limit = max(1, int(rows_per_task))
    proposer_sources = {support_audit.source_key(row) for row in proposer_rows}
    selected = []
    selected_sources = set()
    selected_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    overlap_counts: Counter[str] = Counter()
    duplicate_counts: Counter[str] = Counter()
    for row in candidate_rows:
        task = task_id(row)
        candidate_counts[task] += 1
        if selected_counts[task] >= limit:
            continue
        key = support_audit.source_key(row)
        if key in proposer_sources:
            overlap_counts[task] += 1
            continue
        if key in selected_sources:
            duplicate_counts[task] += 1
            continue
        selected.append(row)
        selected_sources.add(key)
        selected_counts[task] += 1
    tasks = sorted(candidate_counts)
    incomplete = {task: selected_counts[task] for task in tasks if selected_counts[task] != limit}
    if incomplete:
        raise ValueError(
            f"Could not build {limit} disjoint audit rows per task: {incomplete}; "
            f"candidate_counts={dict(candidate_counts)} overlap_counts={dict(overlap_counts)}"
        )
    manifest = {
        "protocol": "hierarchical_common_agent_disjoint_support_split_v1",
        "rows_per_task": limit,
        "proposer_train_rows": len(proposer_rows),
        "audit_candidate_rows": len(candidate_rows),
        "selected_rows": len(selected),
        "candidate_counts_by_task": dict(sorted(candidate_counts.items())),
        "selected_counts_by_task": dict(sorted(selected_counts.items())),
        "excluded_overlap_counts_by_task": dict(sorted(overlap_counts.items())),
        "excluded_duplicate_counts_by_task": dict(sorted(duplicate_counts.items())),
        "source_overlap": 0,
    }
    return selected, manifest


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    selected, manifest = select_disjoint_rows(
        support_audit.read_rows(args.proposer_train_csv),
        support_audit.read_rows(args.audit_candidate_csv),
        rows_per_task=int(args.rows_per_task),
    )
    write_rows(args.output_csv, selected)
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
