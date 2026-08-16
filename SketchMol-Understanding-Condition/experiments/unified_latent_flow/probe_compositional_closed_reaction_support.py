#!/usr/bin/env python3
"""Measure source-disjoint support of complete reaction-component tuples.

The probe uses the locked B37/B43 source split.  Only exact-self-replaying,
single-product fit transactions enter the action grammar.  Each complete tuple
is then applied to a source-disjoint development molecule of the same task.
No property oracle is used and no generated molecule is ranked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
for path in (SCRIPT_DIR, PROJECT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import probe_compositional_closed_reaction_templates as probe  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-atoms", type=int, default=96)
    parser.add_argument("--fingerprint-bits", type=int, default=512)
    parser.add_argument("--mcs-timeout", type=int, default=1)
    parser.add_argument("--min-common-fraction", type=float, default=0.45)
    parser.add_argument("--context-radius", type=int, default=1)
    parser.add_argument("--max-frontier", type=int, default=128)
    parser.add_argument("--split-seed", type=int, default=1982)
    parser.add_argument("--development-source-limit", type=int, default=160)
    parser.add_argument("--development-pair-limit", type=int, default=48)
    return parser.parse_args(argv)


def source_order(source: str, seed: int) -> str:
    return hashlib.sha256(f"{int(seed)}\0{source}".encode("utf-8")).hexdigest()


def read_strict_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if bool(record.get("strict")):
            records.append(record)
    return records


def template_key(templates: Sequence[probe.ComponentTemplate]) -> tuple[str, ...]:
    return tuple(value.reaction_smarts for value in templates)


def extract_exact_fit_transactions(
    records: Sequence[Mapping[str, object]], args: argparse.Namespace
) -> tuple[dict[str, list[tuple[probe.ComponentTemplate, ...]]], dict[str, object]]:
    grammar: defaultdict[str, dict[tuple[str, ...], tuple[probe.ComponentTemplate, ...]]] = (
        defaultdict(dict)
    )
    counts: Counter[str] = Counter()
    by_task: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for index, record in enumerate(records, start=1):
        counts["fit_records"] += 1
        task = str(record["task"])
        by_task[task]["records"] += 1
        target = probe.canonical(str(record["target_smiles"]))
        if not target or "." in target:
            counts["fit_disconnected_target"] += 1
            by_task[task]["disconnected_target"] += 1
            continue
        pair = probe.aligned_pair(record, args)
        if pair is None:
            counts["fit_alignment_failed"] += 1
            continue
        source, aligned_target = pair
        try:
            templates = probe.extract_templates(
                source, aligned_target, int(args.context_radius)
            )
        except Exception:
            counts["fit_extraction_failed"] += 1
            continue
        if not templates:
            counts["fit_template_empty"] += 1
            continue
        products, _raw = probe.apply_component_tuple(
            str(record["source_smiles"]),
            templates,
            max_frontier=int(args.max_frontier),
        )
        if target not in products:
            counts["fit_nonexact_replay"] += 1
            continue
        key = template_key(templates)
        grammar[task].setdefault(key, tuple(templates))
        counts["fit_exact_transactions"] += 1
        by_task[task]["exact_transactions"] += 1
        if index % 128 == 0 or index == len(records):
            print(
                json.dumps(
                    {
                        "stage": "extract_fit_closed_transactions",
                        "fit_records": index,
                        "exact_transactions": counts["fit_exact_transactions"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    frozen = {
        task: [values[key] for key in sorted(values)]
        for task, values in grammar.items()
    }
    manifest = {
        "counts": dict(counts),
        "by_task": {task: dict(value) for task, value in sorted(by_task.items())},
        "unique_transaction_tuples": sum(len(value) for value in frozen.values()),
        "unique_by_task": {task: len(value) for task, value in sorted(frozen.items())},
    }
    return frozen, manifest


def evaluate_cross_support(
    records: Sequence[Mapping[str, object]],
    grammar: Mapping[str, Sequence[Sequence[probe.ComponentTemplate]]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    action_counts: list[int] = []
    unique_counts: list[int] = []
    by_task: defaultdict[str, Counter[str]] = defaultdict(Counter)
    eligible = [
        record
        for record in records
        if probe.canonical(str(record["target_smiles"]))
        and "." not in probe.canonical(str(record["target_smiles"]))
    ][: int(args.development_pair_limit)]
    for index, record in enumerate(eligible, start=1):
        task = str(record["task"])
        source = probe.canonical(str(record["source_smiles"]))
        target = probe.canonical(str(record["target_smiles"]))
        products: set[str] = set()
        applicable = 0
        for templates in grammar.get(task, []):
            outputs, _raw = probe.apply_component_tuple(
                source,
                templates,
                max_frontier=int(args.max_frontier),
            )
            if outputs:
                applicable += 1
                products.update(outputs)
        target_supported = target in products
        counts["conditions"] += 1
        counts["conditions_with_support"] += int(bool(products))
        counts["target_supported"] += int(target_supported)
        by_task[task]["conditions"] += 1
        by_task[task]["conditions_with_support"] += int(bool(products))
        by_task[task]["target_supported"] += int(target_supported)
        action_counts.append(applicable)
        unique_counts.append(len(products))
        rows.append(
            {
                "condition_index": index - 1,
                "task": task,
                "source_smiles": source,
                "target_smiles": target,
                "fit_transaction_tuples": len(grammar.get(task, [])),
                "applicable_transaction_tuples": applicable,
                "unique_valid_products": len(products),
                "target_supported": target_supported,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "cross_apply_closed_transactions",
                    "conditions": index,
                    "target_supported": counts["target_supported"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    metrics = {
        "conditions": counts["conditions"],
        "condition_support_rate": counts["conditions_with_support"]
        / max(1, counts["conditions"]),
        "target_support_rate": counts["target_supported"] / max(1, counts["conditions"]),
        "mean_applicable_transaction_tuples": float(np.mean(action_counts))
        if action_counts
        else 0.0,
        "mean_unique_valid_products": float(np.mean(unique_counts)) if unique_counts else 0.0,
        "by_task": {
            task: {
                **dict(value),
                "condition_support_rate": value["conditions_with_support"]
                / max(1, value["conditions"]),
                "target_support_rate": value["target_supported"]
                / max(1, value["conditions"]),
            }
            for task, value in sorted(by_task.items())
        },
    }
    return rows, metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = read_strict_records(args.records)
    sources = sorted(
        {str(record["source_smiles"]) for record in records},
        key=lambda value: (source_order(value, int(args.split_seed)), value),
    )
    development_sources = set(sources[: int(args.development_source_limit)])
    fit_records = [
        record for record in records if str(record["source_smiles"]) not in development_sources
    ]
    development_records = [
        record for record in records if str(record["source_smiles"]) in development_sources
    ]
    grammar, fit_manifest = extract_exact_fit_transactions(fit_records, args)
    rows, cross_support = evaluate_cross_support(development_records, grammar, args)
    summary = {
        "protocol": "source_disjoint_compositional_closed_reaction_support_v1",
        "split": {
            "strict_records": len(records),
            "strict_sources": len(sources),
            "fit_records": len(fit_records),
            "development_records": len(development_records),
            "development_sources": len(development_sources),
            "fit_development_source_overlap": 0,
            "split_seed": int(args.split_seed),
        },
        "fit_grammar": fit_manifest,
        "cross_support": cross_support,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
