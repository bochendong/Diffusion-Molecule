#!/usr/bin/env python3
"""Exhaustive target-free assay support audit for the frozen B24 grammar.

This is a diagnostic ceiling, not a generator or selector.  For one fixed B29
GSK3B/DRD2 condition, it enumerates every source MMPA site times every frozen
train-only B24 fragment token.  Pinned, fail-closed property oracles are used
only to measure whether the action grammar contains support; no molecule is
returned as a selected prediction and no evaluation target is read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
UCA_DIR = PROJECT_DIR / "experiments" / "unified_constraint_agent"
for path in (SCRIPT_DIR, PROJECT_DIR, UCA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import table1_energy_tilted_latent_transfer as b29  # noqa: E402
import pinned_table1_assay_oracles as pinned_oracles  # noqa: E402


kernel = b29.kernel
belief = b29.belief
graph = b29.graph
unified = b29.unified

PROTOCOL = "target_free_table1_assay_latent_action_support_v30_r1"
TASK_SPECS = {
    "GSK3B:increase": (("GSK3B", 1),),
    "DRD2:decrease+MW:decrease+SA:decrease": (
        ("DRD2", -1),
        ("MW", -1),
        ("SA", -1),
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragment-checkpoint", type=Path, required=True)
    parser.add_argument("--b29-summary", type=Path, required=True)
    parser.add_argument("--b29-candidates", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--gsk3b-oracle", type=Path, required=True)
    parser.add_argument("--drd2-oracle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    return parser.parse_args(argv)


def read_preregistration(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_repair_run",
        "frozen_fragment_protocol": kernel.PROTOCOL,
        "b29_protocol": b29.PROTOCOL,
        "b26_heldout_access": False,
        "official_test_access": False,
        "moledit_target_access": False,
        "diagnostic_only": True,
        "exhaustive_train_only_vocabulary": True,
        "molecular_candidate_ranking": False,
        "similarity_prefilter": 0.15,
        "oracle_failure_policy": "raise",
        "oracle_preflight_required": True,
        "oracle_batch_size": 256,
        "shards": 8,
    }
    drift = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in required.items()
        if payload.get(key) != value
    }
    if drift:
        raise ValueError(f"B30-r1 preregistration drift: {drift}")
    if payload.get("tasks") != list(TASK_SPECS):
        raise ValueError("B30-r1 task order drift")
    if set(dict(payload.get("oracles", {}))) != {"GSK3B", "DRD2"}:
        raise ValueError("B30-r1 pinned oracle contract drift")
    return payload


def load_b29_contract(summary_path: Path) -> dict[str, object]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("protocol") != b29.PROTOCOL:
        raise ValueError("B30 refuses a non-B29 transfer summary")
    manifest = dict(payload.get("manifest", {}))
    contract = {
        "generation_target_access": False,
        "moledit_target_access": False,
        "molecular_candidate_ranking": False,
        "exact_raw_attempts_per_condition": 20,
    }
    drift = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in contract.items()
        if manifest.get(key) != value
    }
    if drift:
        raise ValueError(f"B29 input contract drift: {drift}")
    selection = dict(manifest.get("selection", {}))
    if selection.get("b24_train_source_overlap_after_filter") != 0:
        raise ValueError("B30 refuses B29 sources overlapping B24 train")
    if selection.get("target_columns_used") != 0:
        raise ValueError("B30 refuses B29 selection that used target columns")
    return payload


def selected_conditions(path: Path) -> list[dict[str, str]]:
    by_id: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task = str(row.get("task", ""))
            if task not in TASK_SPECS:
                continue
            condition_id = str(row.get("condition_id", ""))
            source = graph.canonical_smiles(str(row.get("source_smiles", "")))
            if not condition_id or not source:
                continue
            current = {"condition_id": condition_id, "task": task, "source_smiles": source}
            prior = by_id.setdefault(condition_id, current)
            if prior != current:
                raise ValueError(f"B30 condition drift within B29 candidates: {condition_id}")
    ordered = sorted(by_id.values(), key=lambda row: (list(TASK_SPECS).index(row["task"]), row["condition_id"]))
    counts = {
        task: sum(row["task"] == task for row in ordered) for task in TASK_SPECS
    }
    if counts != {task: 4 for task in TASK_SPECS}:
        raise ValueError(f"B30 expected four conditions per assay task: {counts}")
    if len({row["source_smiles"] for row in ordered}) != 8:
        raise ValueError("B30 assay sources are not unique")
    return ordered


def load_vocabulary(
    path: Path, preregistration: Mapping[str, object]
) -> tuple[list[str], dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("stage") != preregistration["frozen_fragment_protocol"]:
        raise ValueError("B30 fragment checkpoint protocol drift")
    manifest = dict(payload.get("manifest", {}))
    contract = {
        "generation_target_access": False,
        "property_oracle_generation_access": False,
        "molecular_candidate_ranking": False,
        "failed_attachment_retry": False,
        "exact_raw_attempts_per_condition": 20,
    }
    drift = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in contract.items()
        if manifest.get(key) != value
    }
    if drift:
        raise ValueError(f"B30 frozen B24 contract drift: {drift}")
    vocabulary = [str(value) for value in payload.get("target_fragments", [])]
    if len(vocabulary) < int(preregistration["minimum_fragment_vocabulary"]):
        raise ValueError(f"B30 fragment vocabulary is unexpectedly small: {len(vocabulary)}")
    if len(vocabulary) != len(set(vocabulary)):
        raise ValueError("B30 fragment vocabulary contains duplicates")
    digest = hashlib.sha256("\n".join(vocabulary).encode("utf-8")).hexdigest()
    return vocabulary, {"checkpoint_manifest": manifest, "vocabulary_sha256": digest}


def finite_descriptor_score(smiles: str, prop: str) -> float | None:
    if prop in {"GSK3B", "DRD2"}:
        raise ValueError(f"B30-r1 refuses unpinned assay scoring for {prop}")
    value = unified.score_property(smiles, prop)
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def audit_condition(
    condition: Mapping[str, str],
    vocabulary: Sequence[str],
    preregistration: Mapping[str, object],
    assay_oracles: Mapping[str, pinned_oracles.PinnedAssayOracle],
) -> dict[str, object]:
    source = condition["source_smiles"]
    task = condition["task"]
    specs = TASK_SPECS[task]
    assay_prop, assay_direction = specs[0]
    source_values = {
        assay_prop: assay_oracles[assay_prop].score_many([source], batch_size=1)[0]
    }
    source_values.update(
        {
            prop: finite_descriptor_score(source, prop)
            for prop, _direction in specs[1:]
        }
    )
    if any(value is None for value in source_values.values()):
        raise ValueError(f"B30 source oracle coverage failed: {source_values}")
    config = SimpleNamespace(
        min_core_heavy_atoms=int(preregistration["min_core_heavy_atoms"]),
        max_variable_heavy_atoms=int(preregistration["max_variable_heavy_atoms"]),
        fingerprint_bits=int(preregistration["fingerprint_bits"]),
    )
    sites = kernel.source_sites(source, config)
    if not sites:
        raise ValueError(f"B30 source has no B24 attachment sites: {condition['condition_id']}")

    seen: set[str] = set()
    attempts = 0
    invalid = 0
    identity = 0
    duplicate = 0
    similar15 = 0
    similar65 = 0
    assay_evaluated = 0
    assay_improved15 = 0
    assay_improved65 = 0
    full_success15 = 0
    full_success65 = 0
    max_assay_margin15 = -float("inf")
    max_assay_similarity = 0.0
    top_support: list[dict[str, object]] = []
    similar_candidates: list[tuple[str, float, int, str, str]] = []
    for site_index, site in enumerate(sites):
        for token in vocabulary:
            if token == site.variable:
                continue
            attempts += 1
            product = graph.canonical_smiles(
                kernel.fragments.join_fragments(site.core, token)
            )
            if not product:
                invalid += 1
                continue
            if product == source:
                identity += 1
                continue
            if product in seen:
                duplicate += 1
                continue
            seen.add(product)
            similarity = graph.morgan_tanimoto(source, product)
            if similarity is None or similarity < float(preregistration["similarity_prefilter"]):
                continue
            similar15 += 1
            if similarity >= 0.65:
                similar65 += 1
            similar_candidates.append(
                (product, float(similarity), site_index, site.variable, token)
            )
        print(
            json.dumps(
                {
                    "condition_id": condition["condition_id"],
                    "stage": "enumerate",
                    "site": site_index + 1,
                    "sites": len(sites),
                    "unique_valid": len(seen),
                    "similar15": similar15,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    assay_values = assay_oracles[assay_prop].score_many(
        [row[0] for row in similar_candidates],
        batch_size=int(preregistration["oracle_batch_size"]),
    )
    assay_evaluated = len(assay_values)
    if assay_evaluated != similar15:
        raise RuntimeError(
            f"B30-r1 {assay_prop} coverage mismatch: {assay_evaluated}/{similar15}"
        )
    for candidate_index, (candidate, assay_value) in enumerate(
        zip(similar_candidates, assay_values)
    ):
        product, similarity, site_index, source_fragment, token = candidate
        assay_margin = float(assay_direction) * (
            assay_value - float(source_values[assay_prop])
        )
        if assay_margin <= 0.0:
            continue
        assay_improved15 += 1
        if similarity >= 0.65:
            assay_improved65 += 1
        max_assay_margin15 = max(max_assay_margin15, assay_margin)
        max_assay_similarity = max(max_assay_similarity, float(similarity))
        full_success = True
        property_margins = {assay_prop: assay_margin}
        for prop, direction in specs[1:]:
            value = finite_descriptor_score(product, prop)
            if value is None:
                full_success = False
                property_margins[prop] = None
                continue
            margin = float(direction) * (value - float(source_values[prop]))
            property_margins[prop] = margin
            full_success = full_success and margin > 0.0
        if full_success:
            full_success15 += 1
            if similarity >= 0.65:
                full_success65 += 1
        top_support.append(
            {
                "generated_smiles": product,
                "site_index": site_index,
                "source_fragment": source_fragment,
                "target_fragment_token": token,
                "source_tanimoto": float(similarity),
                "assay_margin": assay_margin,
                "property_margins": property_margins,
                "full_property_success": full_success,
            }
        )
        if candidate_index % 4096 == 0:
            print(
                json.dumps(
                    {
                        "condition_id": condition["condition_id"],
                        "stage": "assay",
                        "evaluated": candidate_index + 1,
                        "total": assay_evaluated,
                        "assay_improved15": assay_improved15,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if assay_values and max(assay_values) - min(assay_values) <= 0.0:
        raise ValueError(
            f"B30-r1 {assay_prop} candidate scores are unexpectedly constant: "
            f"{assay_values[0]} across {len(assay_values)} candidates"
        )
    top_support.sort(
        key=lambda row: (
            bool(row["full_property_success"]),
            float(row["source_tanimoto"]),
            float(row["assay_margin"]),
        ),
        reverse=True,
    )
    return {
        "condition_id": condition["condition_id"],
        "task": task,
        "source_smiles": source,
        "source_property_values": source_values,
        "sites": len(sites),
        "fragment_vocabulary": len(vocabulary),
        "assembly_attempts": attempts,
        "unique_valid_nonidentity": len(seen),
        "invalid_assemblies": invalid,
        "identity_assemblies": identity,
        "duplicate_assemblies": duplicate,
        "similar_t0_15": similar15,
        "similar_t0_65": similar65,
        "assay_oracle_coverage": assay_evaluated / max(1, similar15),
        "assay_score_min": min(assay_values) if assay_values else None,
        "assay_score_max": max(assay_values) if assay_values else None,
        "assay_score_distinct_rounded_12": len(
            {round(value, 12) for value in assay_values}
        ),
        "assay_improved_t0_15": assay_improved15,
        "assay_improved_t0_65": assay_improved65,
        "full_property_success_t0_15": full_success15,
        "full_property_success_t0_65": full_success65,
        "has_assay_support_t0_15": assay_improved15 > 0,
        "has_full_support_t0_15": full_success15 > 0,
        "max_assay_margin_t0_15": (
            max_assay_margin15 if math.isfinite(max_assay_margin15) else 0.0
        ),
        "max_similarity_with_assay_improvement": max_assay_similarity,
        "top_support_examples_diagnostic_only": top_support[:10],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    preregistration = read_preregistration(args.protocol_manifest)
    if not 0 <= int(args.shard_index) < int(preregistration["shards"]):
        raise ValueError("B30 shard index is outside the preregistered range")
    b29_summary = load_b29_contract(args.b29_summary)
    conditions = selected_conditions(args.b29_candidates)
    condition = conditions[int(args.shard_index)]
    vocabulary, vocabulary_manifest = load_vocabulary(
        args.fragment_checkpoint, preregistration
    )
    assay_oracles, oracle_provenance = pinned_oracles.load_pinned_oracles(
        gsk3b_path=args.gsk3b_oracle,
        drd2_path=args.drd2_oracle,
        specifications=dict(preregistration["oracles"]),
    )
    result = audit_condition(
        condition, vocabulary, preregistration, assay_oracles
    )
    manifest = {
        "protocol": PROTOCOL,
        "preregistration_sha256": belief.file_sha256(args.protocol_manifest),
        "fragment_checkpoint_sha256": belief.file_sha256(args.fragment_checkpoint),
        "b29_summary_sha256": belief.file_sha256(args.b29_summary),
        "b29_candidates_sha256": belief.file_sha256(args.b29_candidates),
        "b29_decision": b29_summary.get("decision"),
        "shard_index": int(args.shard_index),
        "diagnostic_only": True,
        "model_training": False,
        "generation_target_access": False,
        "moledit_target_access": False,
        "official_test_access": False,
        "property_oracle_generation_access": False,
        "property_oracle_support_audit_access": True,
        "oracle_failure_policy": "raise",
        "oracle_preflight_passed": all(
            bool(dict(value)["preflight"]["passed"])
            for value in oracle_provenance.values()
        ),
        "pinned_oracles": oracle_provenance,
        "exhaustive_train_only_vocabulary": True,
        "molecular_candidate_ranking": False,
        "selected_prediction_output": False,
        **vocabulary_manifest,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"protocol": PROTOCOL, "manifest": manifest, "support": result}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
