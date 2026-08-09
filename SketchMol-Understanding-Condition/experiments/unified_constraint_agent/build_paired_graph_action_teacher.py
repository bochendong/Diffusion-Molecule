#!/usr/bin/env python3
"""Project train-only source/target pairs into target-aligned GraphEditDSL labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence


POLICY_ROOT = Path(__file__).resolve().parents[1] / "unified_smiles_generator"


def load_policy_module():
    if str(POLICY_ROOT) not in sys.path:
        sys.path.insert(0, str(POLICY_ROOT))
    import umtp_graph_action_policy

    return umtp_graph_action_policy


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--site-limit", type=int, default=32)
    parser.add_argument("--max-actions-per-row", type=int, default=512)
    parser.add_argument("--min-target-similarity", type=float, default=0.20)
    parser.add_argument("--min-source-similarity", type=float, default=0.10)
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def finite(value: float, fallback: float = -1.0) -> float:
    return float(value) if math.isfinite(float(value)) else fallback


def candidate_key(
    policy_module: object,
    target_smiles: str,
    source_smiles: str,
    candidate: tuple[object, str, list[str]],
) -> tuple[float, ...]:
    action, smiles, _ = candidate
    canonical_target = policy_module.unified.safe_canonical_smiles(target_smiles)
    target_similarity = (
        policy_module.unified.morgan_tanimoto(canonical_target, smiles) if canonical_target else math.nan
    )
    source_similarity = policy_module.unified.morgan_tanimoto(source_smiles, smiles)
    action_score = float(getattr(action, "policy_score", 0.0) or 0.0)
    return (
        float(bool(canonical_target and canonical_target == smiles)),
        finite(target_similarity),
        finite(source_similarity),
        action_score,
    )


def project_row(
    row: Mapping[str, str],
    *,
    policy_module: object,
    site_limit: int,
    max_actions_per_row: int,
    min_target_similarity: float,
    min_source_similarity: float,
) -> tuple[dict[str, object] | None, str]:
    source = str(row.get("source_smiles", "") or "").strip()
    target = str(row.get("target_smiles", "") or "").strip()
    if not source or not target:
        return None, "missing_source_or_target"
    candidates = policy_module.enumerate_action_candidates(
        row,
        site_limit=int(site_limit),
        max_actions_per_row=int(max_actions_per_row),
    )
    if not candidates:
        return None, "no_executable_action"
    best = max(candidates, key=lambda item: candidate_key(policy_module, target, source, item))
    action, smiles, program = best
    canonical_target = policy_module.unified.safe_canonical_smiles(target)
    target_similarity = (
        policy_module.unified.morgan_tanimoto(canonical_target, smiles) if canonical_target else math.nan
    )
    source_similarity = policy_module.unified.morgan_tanimoto(source, smiles)
    if finite(target_similarity) < float(min_target_similarity):
        return None, "low_target_similarity"
    if finite(source_similarity) < float(min_source_similarity):
        return None, "low_source_similarity"
    output = dict(row)
    output.update(
        {
            "task_mode": "edit",
            "policy_target_tokens_json": json.dumps(program),
            "policy_target_action_json": json.dumps(asdict(action), sort_keys=True),
            "policy_target_smiles": smiles,
            "policy_target_exact": str(bool(canonical_target and canonical_target == smiles)),
            "policy_target_similarity": target_similarity,
            "policy_target_source_tanimoto": source_similarity,
            "policy_target_paired_teacher": "True",
            "policy_action_candidate_count": len(candidates),
        }
    )
    return output, "selected"


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    policy_module = load_policy_module()
    rows = read_rows(args.input_csv)
    output = []
    outcomes = Counter()
    for index, row in enumerate(rows):
        projected, outcome = project_row(
            row,
            policy_module=policy_module,
            site_limit=args.site_limit,
            max_actions_per_row=args.max_actions_per_row,
            min_target_similarity=args.min_target_similarity,
            min_source_similarity=args.min_source_similarity,
        )
        outcomes[outcome] += 1
        if projected is not None:
            output.append(projected)
        if (index + 1) % 50 == 0:
            print(f"[paired-action-teacher] {index + 1}/{len(rows)} rows", flush=True)
    if not output:
        raise SystemExit("No paired GraphEditDSL labels passed the teacher gates")
    write_rows(args.output_csv, output)
    manifest = {
        "protocol": "paired_graph_action_teacher_v1",
        "data_role": "train_only",
        "input_csv": str(args.input_csv),
        "output_csv": str(args.output_csv),
        "input_rows": len(rows),
        "output_rows": len(output),
        "coverage": len(output) / max(len(rows), 1),
        "outcomes": dict(sorted(outcomes.items())),
        "site_limit": args.site_limit,
        "max_actions_per_row": args.max_actions_per_row,
        "min_target_similarity": args.min_target_similarity,
        "min_source_similarity": args.min_source_similarity,
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
