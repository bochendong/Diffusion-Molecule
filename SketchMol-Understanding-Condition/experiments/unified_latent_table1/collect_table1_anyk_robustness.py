#!/usr/bin/env python3
"""Compare any@k curves from already-frozen Table1 candidate files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b41-curve", required=True, type=Path)
    parser.add_argument("--canonical-curve", required=True, type=Path)
    parser.add_argument("--d3-curve", required=True, type=Path)
    parser.add_argument("--b41-candidates", required=True, type=Path)
    parser.add_argument("--canonical-candidates", required=True, type=Path)
    parser.add_argument("--d3-candidates", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_curve(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("ks") != [1, 2, 5, 10, 20]:
        raise SystemExit(f"Unexpected any@k grid in {path}: {payload.get('ks')}")
    if int(payload.get("candidate_conditions") or 0) <= 0:
        raise SystemExit(f"No candidate conditions in {path}")
    evaluated = int(payload.get("evaluated_conditions") or 0)
    candidates = int(payload.get("candidate_conditions") or 0)
    if evaluated <= 0 or evaluated > candidates:
        raise SystemExit(
            f"Invalid candidate/evaluation counts in {path}: {candidates}/{evaluated}"
        )
    return payload


def numeric_series(curve: dict[str, object], key: str) -> dict[str, float]:
    raw = curve.get(key)
    if not isinstance(raw, dict):
        raise SystemExit(f"Missing curve {key}")
    output: dict[str, float] = {}
    for k in ("1", "2", "5", "10", "20"):
        value = raw.get(k)
        if value in (None, ""):
            raise SystemExit(f"Missing {key}@{k}")
        output[k] = float(value)
    return output


def pack(curve: dict[str, object]) -> dict[str, object]:
    candidate_conditions = int(curve["candidate_conditions"])
    evaluated_conditions = int(curve["evaluated_conditions"])
    return {
        "model": curve.get("model"),
        "real5_anyk_t0_65": numeric_series(curve, "real5_anyk_t0_65"),
        "gsk3b_anyk_t0_65": numeric_series(curve, "gsk3b_anyk_t0_65"),
        "auc_real5_t0_65": float(curve["auc_real5_t0_65"]),
        "auc_gsk3b_t0_65": float(curve["auc_gsk3b_t0_65"]),
        "mean_unique_smiles": float(curve["mean_unique_smiles"]),
        "candidate_conditions": candidate_conditions,
        "evaluated_conditions": evaluated_conditions,
        "candidate_ids_outside_matched_table1_reference": (
            candidate_conditions - evaluated_conditions
        ),
    }


def subtract(
    left: dict[str, float], right: dict[str, float]
) -> dict[str, float]:
    return {k: float(left[k]) - float(right[k]) for k in ("1", "2", "5", "10", "20")}


def main() -> int:
    args = parse_args()
    curves = {
        "b41": load_curve(args.b41_curve),
        "canonical": load_curve(args.canonical_curve),
        "d3_grpo": load_curve(args.d3_curve),
    }
    packed = {name: pack(curve) for name, curve in curves.items()}
    evaluated_counts = {
        name: int(arm["evaluated_conditions"]) for name, arm in packed.items()
    }
    max_evaluated = max(evaluated_counts.values())
    relative_coverage = {
        name: float(count) / float(max_evaluated)
        for name, count in evaluated_counts.items()
    }
    minimum_relative_coverage = min(relative_coverage.values())
    if minimum_relative_coverage < 0.995:
        raise SystemExit(
            "Table1 arm coverage is not aligned: "
            f"counts={evaluated_counts}, relative={relative_coverage}"
        )
    b41_real = packed["b41"]["real5_anyk_t0_65"]
    canonical_real = packed["canonical"]["real5_anyk_t0_65"]
    d3_real = packed["d3_grpo"]["real5_anyk_t0_65"]
    assert isinstance(b41_real, dict)
    assert isinstance(canonical_real, dict)
    assert isinstance(d3_real, dict)
    candidate_paths = {
        "b41": args.b41_candidates,
        "canonical": args.canonical_candidates,
        "d3_grpo": args.d3_candidates,
    }
    payload = {
        "protocol": "table1_frozen_candidate_anyk_robustness_v1",
        "scope": "posthoc_paper_diagnostic_on_previously_opened_table1",
        "contract": {
            "existing_frozen_candidates": True,
            "new_generation": False,
            "model_training": False,
            "molecular_candidate_ranking": False,
            "oracle_selection": False,
            "exact_max_attempts_per_condition": 20,
            "ks": [1, 2, 5, 10, 20],
            "minimum_relative_reference_coverage": 0.995,
        },
        "coverage_alignment": {
            "evaluated_conditions": evaluated_counts,
            "relative_to_largest_arm": relative_coverage,
            "minimum_relative_coverage": minimum_relative_coverage,
            "passed": True,
            "note": (
                "candidate_conditions counts every ID in the frozen files; "
                "evaluated_conditions counts IDs matched to the Table1 reference"
            ),
        },
        "candidate_sha256": {
            name: sha256(path) for name, path in candidate_paths.items()
        },
        "arms": packed,
        "real5_delta_vs_b41": {
            "canonical": subtract(canonical_real, b41_real),
            "d3_grpo": subtract(d3_real, b41_real),
        },
        "saturation_gain_real5_any20_minus_any1": {
            name: float(arm["real5_anyk_t0_65"]["20"])
            - float(arm["real5_anyk_t0_65"]["1"])
            for name, arm in packed.items()
        },
        "claim_policy": "diagnostic_only_no_method_selection_or_retuning",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
