#!/usr/bin/env python3
"""Build target-hidden fit-only preferences for the MuMO residual ranker."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
PROJECT_SCRIPTS = PROJECT_DIR / "scripts"
for path in (SCRIPT_DIR, PROJECT_DIR, PROJECT_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_mumo_closed_loop_dev as closed_loop  # noqa: E402
import export_external_multiproperty_benchmark_rows as export  # noqa: E402
import mumo_parallel_protocol as protocol  # noqa: E402


SYSTEM_PROMPT = (
    "You are the residual ranking component of a unified molecular constraint agent. "
    "Prefer candidates whose explicit verifier margins jointly satisfy every constraint, "
    "while preserving source similarity. Return exactly the supplied JSON action."
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-conditions-per-task", type=int, default=100)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1712)
    return parser.parse_args(argv)


def stable_value(value: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{value}".encode()).digest()[:8], "big")


def prompt_messages(row: Mapping[str, object], properties: Sequence[str]) -> list[dict[str, str]]:
    payload = {
        "source_smiles": str(row["source_smiles"]),
        "task_id": str(row["_uca_task_id"]),
        "constraints": [
            {
                "property": prop,
                "direction": export.DEFAULT_DIRECTION[prop],
                "threshold": export.MUMO_THRESHOLDS[prop],
            }
            for prop in properties
        ],
        "candidate_observations": "verifier_margins,source_tanimoto,retrieval_similarity,frequency",
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, sort_keys=True, separators=(",", ":"))},
    ]


def action_payload(
    smiles: str,
    margins: Mapping[str, float],
    *,
    source_tanimoto: float,
    retrieval_similarity: float,
    frequency: int,
    candidate_source: str = "residual_candidate",
) -> dict[str, object]:
    return {
        "action_type": "rank_candidate",
        "generated_smiles": smiles,
        "verifier_margins": {key: round(float(value), 6) for key, value in sorted(margins.items())},
        "source_tanimoto": round(float(source_tanimoto), 6),
        "retrieval_similarity": round(float(retrieval_similarity), 6),
        "frequency": int(frequency),
        "candidate_source": candidate_source,
    }


def actual_success(row: Mapping[str, object], properties: Sequence[str]) -> bool:
    for prop in properties:
        source = export.read_property_value(row, prop, prefix="source")
        target = export.read_property_value(row, prop, prefix="target")
        if source is None or target is None:
            return False
        signed = float(target) - float(source)
        if export.DEFAULT_DIRECTION[prop] == "decrease":
            signed = -signed
        if signed + 1e-12 < float(export.MUMO_THRESHOLDS[prop]):
            return False
    return True


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.0 < float(args.validation_fraction) < 0.5:
        raise ValueError("validation_fraction must be between zero and 0.5")
    models = closed_loop.load_models(args.evidence_root)
    by_task: dict[str, list[dict[str, object]]] = defaultdict(list)
    for path in sorted(args.data_dir.glob("train_shard_*.jsonl")):
        for row in protocol.read_jsonl(path):
            if row.get("_uca_partition") == "fit":
                by_task[str(row["_uca_task_id"])].append(row)
    selected = []
    for task, rows in sorted(by_task.items()):
        rows.sort(key=lambda row: stable_value(str(row["_uca_source_group"]), int(args.seed)))
        selected.extend(rows[: int(args.max_conditions_per_task)])

    pairs = []
    outcomes: Counter[str] = Counter()
    for row in selected:
        spec = next(
            item for item in export.TASK_SPECS
            if item.suite == "mumo" and item.task_id == str(row["_uca_task_id"])
        )
        properties = tuple(spec.properties)
        if not actual_success(row, properties):
            outcomes["not_actual_joint_success"] += 1
            continue
        source_smiles = str(row["source_smiles"])
        target_smiles = str(row["target_smiles"])
        source_feature = closed_loop.candidate_feature(source_smiles)
        target_feature = closed_loop.candidate_feature(target_smiles)
        if source_feature is None or target_feature is None:
            outcomes["invalid_feature"] += 1
            continue
        tanimoto = float(closed_loop.delta.graph.revise.morgan_tanimoto(source_smiles, target_smiles))
        if tanimoto < 0.4:
            outcomes["source_similarity_below_0p4"] += 1
            continue
        scores, margin_rows = closed_loop.score_candidates_batch(
            source_feature,
            [target_feature, source_feature],
            properties=properties,
            models=models,
            source_tanimotos=[tanimoto, 1.0],
            retrieval_similarities=[tanimoto, 1.0],
            frequencies=[1, 0],
        )
        if min(margin_rows[0].values()) <= min(margin_rows[1].values()):
            outcomes["verifier_not_preferred"] += 1
            continue
        group = str(row["_uca_source_group"])
        pair = {
            "pair_id": f"{row['_uca_pair_digest']}:residual",
            "condition_id": str(row["_uca_pair_digest"]),
            "source_group": group,
            "origin": "mumo_residual_ranker",
            "data_role": "fit_only_joint_success_preference",
            "prompt_target_access": False,
            "training_target_role": "positive_label_only",
            "prompt_messages": prompt_messages(row, properties),
            "chosen": action_payload(
                target_smiles,
                margin_rows[0],
                source_tanimoto=tanimoto,
                retrieval_similarity=tanimoto,
                frequency=1,
                candidate_source="residual_candidate",
            ),
            "rejected": action_payload(
                source_smiles,
                margin_rows[1],
                source_tanimoto=1.0,
                retrieval_similarity=1.0,
                frequency=0,
                candidate_source="residual_candidate",
            ),
        }
        if "target_smiles" in json.dumps(pair["prompt_messages"]):
            raise AssertionError("Training target leaked into residual prompt")
        pairs.append(pair)
        outcomes["preference_pairs"] += 1
    if not pairs:
        raise ValueError("No residual preferences were built")
    validation_groups = {
        str(row["source_group"])
        for row in pairs
        if stable_value(str(row["source_group"]), int(args.seed) + 1) / float(2**64)
        < float(args.validation_fraction)
    }
    train = [row for row in pairs if str(row["source_group"]) not in validation_groups]
    validation = [row for row in pairs if str(row["source_group"]) in validation_groups]
    if not train or not validation:
        raise ValueError(f"Empty residual preference split: train={len(train)} validation={len(validation)}")
    random.Random(int(args.seed)).shuffle(train)
    validation.sort(key=lambda row: str(row["pair_id"]))
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    source_group_overlap = len(
        {str(row["source_group"]) for row in train}
        & {str(row["source_group"]) for row in validation}
    )
    manifest = {
        "protocol": "common_llm_mumo_residual_preference_v1",
        "data_role": "fit_only_joint_success_preference",
        "prompt_target_access": False,
        "training_target_role": "positive_label_only",
        "evaluation_target_access": False,
        "source_group_overlap": source_group_overlap,
        "selected_fit_conditions": len(selected),
        "train_pairs": len(train),
        "validation_pairs": len(validation),
        "seed": int(args.seed),
        "outcomes": dict(sorted(outcomes.items())),
    }
    protocol.write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
