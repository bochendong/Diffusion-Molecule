#!/usr/bin/env python3
"""Collect paired P37 overlap effects and uncertainty intervals."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Mapping, Sequence


SHARED_PROPERTIES = frozenset({"MW", "LogP", "QED", "HBA", "RB"})


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def candidate_map(path: Path) -> dict[str, dict[str, object]]:
    rows = read_jsonl(path)
    result = {str(row["condition_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate candidate IDs in {path}")
    return result


def group_for(row: Mapping[str, object]) -> str:
    properties = {
        part.split(":", 1)[0] for part in str(row["task_key"]).split("+")
    }
    return "shared_only" if properties <= SHARED_PROPERTIES else "contains_denovo_only"


def exact_mcnemar(joint_only: int, specialist_only: int) -> float:
    discordant = joint_only + specialist_only
    if discordant == 0:
        return 1.0
    lower = min(joint_only, specialist_only)
    tail = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def paired_summary(
    joint: Mapping[str, Mapping[str, object]],
    specialist: Mapping[str, Mapping[str, object]],
    *,
    group: str,
    arities: tuple[int, ...],
    bootstrap_seed: int,
    replicates: int = 100000,
) -> dict[str, object]:
    cells = {
        arity: [
            identity
            for identity, row in joint.items()
            if int(row["property_count"]) == arity and group_for(row) == group
        ]
        for arity in arities
    }
    if any(not values for values in cells.values()):
        raise ValueError(f"missing {group} cell in arities {arities}")
    ids = [identity for arity in arities for identity in cells[arity]]
    if set(joint) != set(specialist):
        raise ValueError("joint and specialist candidate IDs differ")

    def strict(arm, identity: str) -> int:
        return int(bool(arm[identity]["strict"]))

    joint_macro = sum(
        sum(strict(joint, identity) for identity in cells[arity]) / len(cells[arity])
        for arity in arities
    ) / len(arities)
    specialist_macro = sum(
        sum(strict(specialist, identity) for identity in cells[arity]) / len(cells[arity])
        for arity in arities
    ) / len(arities)
    joint_only = sum(strict(joint, identity) and not strict(specialist, identity) for identity in ids)
    specialist_only = sum(strict(specialist, identity) and not strict(joint, identity) for identity in ids)
    both = sum(strict(joint, identity) and strict(specialist, identity) for identity in ids)
    neither = len(ids) - joint_only - specialist_only - both

    rng = random.Random(bootstrap_seed)
    draws = []
    for _ in range(replicates):
        arity_deltas = []
        for arity in arities:
            sample = [rng.choice(cells[arity]) for _ in cells[arity]]
            arity_deltas.append(
                sum(
                    strict(joint, identity) - strict(specialist, identity)
                    for identity in sample
                )
                / len(sample)
            )
        draws.append(sum(arity_deltas) / len(arity_deltas))
    draws.sort()
    lower = draws[int(0.025 * replicates)]
    upper = draws[int(0.975 * replicates)]
    return {
        "arities": list(arities),
        "rows": len(ids),
        "joint_strict_arity_macro": joint_macro,
        "specialist_strict_arity_macro": specialist_macro,
        "delta": joint_macro - specialist_macro,
        "paired_bootstrap_95ci": [lower, upper],
        "bootstrap_probability_delta_positive": sum(value > 0 for value in draws)
        / replicates,
        "paired_outcomes": {
            "both_success": both,
            "joint_only_success": joint_only,
            "specialist_only_success": specialist_only,
            "neither_success": neither,
        },
        "mcnemar_exact_two_sided_p": exact_mcnemar(joint_only, specialist_only),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--bootstrap-seed", type=int, default=37201)
    parser.add_argument("--bootstrap-replicates", type=int, default=100000)
    args = parser.parse_args(argv)
    result: dict[str, object] = {
        "protocol": "p37_denovo_overlap_expanded_raw1_v1",
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_replicates": args.bootstrap_replicates,
        "scales": {},
    }
    csv_rows = []
    for scale in (10000, 100000):
        base = args.output_root / f"scale_{scale}" / "eval"
        joint = candidate_map(base / "joint" / "candidates.jsonl")
        specialist = candidate_map(base / "denovo" / "candidates.jsonl")
        scale_result = {}
        for scope, arities in (("2p4p", (2, 3, 4)), ("2p5p", (2, 3, 4, 5))):
            scope_result = {}
            for index, group in enumerate(("shared_only", "contains_denovo_only")):
                summary = paired_summary(
                    joint,
                    specialist,
                    group=group,
                    arities=arities,
                    bootstrap_seed=args.bootstrap_seed + scale + 10 * len(arities) + index,
                    replicates=args.bootstrap_replicates,
                )
                scope_result[group] = summary
                csv_rows.append({"scale": scale, "scope": scope, "group": group, **summary})
            scale_result[scope] = scope_result
        result["scales"][str(scale)] = scale_result

    output_dir = args.output_root / "result"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    fields = [
        "scale", "scope", "group", "arities", "rows",
        "joint_strict_arity_macro", "specialist_strict_arity_macro", "delta",
        "paired_bootstrap_95ci", "bootstrap_probability_delta_positive",
        "paired_outcomes", "mcnemar_exact_two_sided_p",
    ]
    with (output_dir / "result.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({
                key: json.dumps(row[key], sort_keys=True)
                if isinstance(row[key], (list, dict)) else row[key]
                for key in fields
            })
    (output_dir / "COLLECT_COMPLETE").touch()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
