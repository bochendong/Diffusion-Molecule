#!/usr/bin/env python3
"""Build train-only pseudo-pairs by selecting inside the verified-success set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
UNIFIED_DIR = PROJECT_DIR / "experiments" / "unified_smiles_generator"
P811_DIR = PROJECT_DIR / "experiments" / "p8_1_1_short_transaction"
P819_DIR = PROJECT_DIR / "experiments" / "p8_1_9_transaction_outcome_distill"
for path in (PROJECT_DIR, UNIFIED_DIR, P811_DIR, P819_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_teacher_pseudopairs as common  # noqa: E402
import sample_raw_transactions as teacher_sampler  # noqa: E402
import unified_smiles_generator as unified  # noqa: E402
import umtp_graph_action_policy as policy  # noqa: E402


def group_name(row: dict[str, str]) -> str:
    return str(row.get("benchmark_task", "") or row.get("task_name", "") or row.get("instruction", "edit"))


def property_count(row: dict[str, str]) -> int:
    return len(unified.instruction_task_specs(row))


def coverage_table(selected: list[dict[str, str]], success_keys: set[str], key_fn) -> dict[str, dict[str, object]]:
    totals: Counter[str] = Counter()
    successes: Counter[str] = Counter()
    for row in selected:
        name = str(key_fn(row))
        totals[name] += 1
        if common.stable_key(row) in success_keys:
            successes[name] += 1
    return {
        name: {
            "eligible_rows": totals[name],
            "verified_rows": successes[name],
            "coverage": successes[name] / max(totals[name], 1),
        }
        for name in sorted(totals)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument("--student-checkpoint", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--train-features-dir", required=True, type=Path)
    parser.add_argument("--eval-csv", required=True, type=Path)
    parser.add_argument("--r1-output", required=True, type=Path)
    parser.add_argument("--r2-output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=768)
    parser.add_argument("--site-limit", type=int, default=32)
    parser.add_argument("--max-actions", type=int, default=512)
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    eval_rows = common.read_rows(args.eval_csv)
    eval_ids = {
        str(row.get(key, "") or "").strip()
        for row in eval_rows for key in common.ID_FIELDS
        if str(row.get(key, "") or "").strip()
    }
    eval_molecules = {
        value for row in eval_rows
        for value in (common.canonical(row.get("source_smiles", "")), common.canonical(row.get("target_smiles", "")))
        if value
    }
    raw_train = [row for row in common.read_rows(args.train_csv) if unified.task_mode_for_row(row) == unified.EDIT_MODE]
    rejected_id = rejected_molecule = 0
    eligible = []
    for row in raw_train:
        ids = {str(row.get(key, "") or "").strip() for key in common.ID_FIELDS if str(row.get(key, "") or "").strip()}
        if ids & eval_ids:
            rejected_id += 1
            continue
        source = common.canonical(row.get("source_smiles", ""))
        original_target = common.canonical(row.get("target_smiles", ""))
        if not source or source in eval_molecules or (original_target and original_target in eval_molecules):
            rejected_molecule += 1
            continue
        eligible.append(row)
    chosen = common.select_balanced(eligible, min(int(args.limit), len(eligible)), int(args.seed))

    device = unified.resolve_device(str(args.device))
    teacher_ckpt = unified.load_checkpoint(args.teacher_checkpoint)
    student_ckpt = unified.load_checkpoint(args.student_checkpoint)
    if teacher_ckpt is None or student_ckpt is None:
        raise FileNotFoundError("teacher or student checkpoint missing")
    teacher_vocab = unified.SmilesVocabulary.from_dict(teacher_ckpt["vocab"])
    teacher_config = dict(teacher_ckpt["model_config"])
    teacher = unified.ConditionedSmilesDecoder(**teacher_config).to(device)
    teacher.load_state_dict(teacher_ckpt["model_state"])
    teacher.eval()
    student_vocab = unified.SmilesVocabulary.from_dict(student_ckpt["vocab"])
    store = unified.FeatureStore(args.train_features_dir, array_name="query_tokens", variant="full")

    outcomes: list[dict[str, object]] = []
    success_keys: set[str] = set()
    no_support = no_verified = oov = pseudo_overlap = 0
    valid_candidates = strict_candidates = identity_strict_candidates = 0
    verified_pool_sizes: list[int] = []
    for row_index, row in enumerate(chosen):
        candidates = teacher_sampler.source_only_candidates(row, site_limit=int(args.site_limit), limit=int(args.max_actions))
        if not candidates:
            no_support += 1
            continue
        canonical_source = common.canonical(row.get("source_smiles", ""))
        verified = []
        for action, outcome, program in candidates:
            outcome = common.canonical(outcome)
            if not outcome:
                continue
            metrics = unified.candidate_metrics(row, outcome, source_similarity_threshold=0.65)
            valid_candidates += int(metrics.get("valid_smiles") == "True")
            strict = metrics.get("table1_strict_success") == "True"
            strict_candidates += int(strict)
            identity = outcome == canonical_source
            identity_strict_candidates += int(strict and identity)
            if strict and not identity:
                verified.append((action, outcome, program, metrics))
        verified_pool_sizes.append(len(verified))
        if not verified:
            no_verified += 1
            continue
        condition = unified.condition_array_for_row(
            row, store, int(teacher_config["condition_dim"]), max_source_tokens=96,
            condition_layout="direct_compat",
        ).astype(np.float32)
        scores = policy.score_programs(
            teacher, teacher_vocab, condition, [item[2] for item in verified],
            batch_size=int(args.score_batch_size), device=device,
        )
        best = int(np.argmax(np.asarray(scores, dtype=np.float64)))
        action, outcome, program, metrics = verified[best]
        if outcome in eval_molecules:
            pseudo_overlap += 1
            continue
        if any(token not in student_vocab.token_to_id for token in unified.tokenize_smiles(outcome)):
            oov += 1
            continue
        probabilities = torch.softmax(torch.tensor(scores, dtype=torch.float64), dim=0)
        confidence = float(probabilities[best])
        entropy = float(-(probabilities * probabilities.clamp_min(1e-300).log()).sum())
        normalized_entropy = entropy / max(math.log(max(len(verified), 2)), 1e-12)
        item: dict[str, object] = dict(row)
        item.update(metrics)
        item.update({
            "target_smiles": outcome,
            "task_mode": "edit",
            "pseudo_label_origin": "p8_1_1_train_only_verified_success_set",
            "teacher_action_json": json.dumps(asdict(action), sort_keys=True),
            "teacher_program_tokens_json": json.dumps(program),
            "teacher_logprob": policy.format_float(float(scores[best])),
            "teacher_success_set_confidence": policy.format_float(confidence),
            "teacher_success_set_normalized_entropy": policy.format_float(normalized_entropy),
            "verified_success_pool_size": len(verified),
            "verified_before_teacher_selection": "True",
            "distill_weight_group": "uniform",
        })
        outcomes.append(item)
        success_keys.add(common.stable_key(row))
        if (row_index + 1) % 50 == 0 or row_index + 1 == len(chosen):
            print(f"[p8.1.12-verified] {row_index + 1}/{len(chosen)} verified={len(outcomes)}", flush=True)

    if not outcomes:
        audit = {
            "protocol": "p8_1_12_verified_success_preflight_v1",
            "decision": "fail_closed_zero_verified_coverage",
            "raw_train_rows": len(raw_train), "eligible_rows": len(chosen),
            "no_transaction_support": no_support, "no_verified_success": no_verified,
            "coverage_by_task": coverage_table(chosen, success_keys, group_name),
            "coverage_by_property_count": coverage_table(chosen, success_keys, property_count),
            "eval_candidates_used_for_training": False,
        }
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit("No train-only verified-success outcomes; fail closed")

    confidences = sorted(float(row["teacher_success_set_confidence"]) for row in outcomes)
    median = confidences[len(confidences) // 2]
    r1_rows: list[dict[str, object]] = []
    r2_rows: list[dict[str, object]] = []
    for row in outcomes:
        r1_item = dict(row); r1_item["distill_replica"] = 0; r1_rows.append(r1_item)
        repeats = 2 if float(row["teacher_success_set_confidence"]) >= median else 1
        for replica in range(repeats):
            r2_item = dict(row)
            r2_item["distill_weight_group"] = "success_set_confidence_2x_above_median"
            r2_item["distill_replica"] = replica
            r2_rows.append(r2_item)
    common.write_rows(args.r1_output, r1_rows)
    common.write_rows(args.r2_output, r2_rows)

    output_ids = {str(row.get(key, "") or "").strip() for row in outcomes for key in common.ID_FIELDS if str(row.get(key, "") or "").strip()}
    output_sources = {common.canonical(row.get("source_smiles", "")) for row in outcomes}
    output_targets = {common.canonical(str(row.get("target_smiles", ""))) for row in outcomes}
    overlap = {
        "id": sorted(output_ids & eval_ids),
        "source": sorted(output_sources & eval_molecules),
        "pseudo_target": sorted(output_targets & eval_molecules),
    }
    audit = {
        "protocol": "p8_1_12_verified_success_preflight_v1",
        "decision": "pass",
        "teacher_checkpoint_sha256": policy.checkpoint_sha256(args.teacher_checkpoint),
        "student_checkpoint_sha256": policy.checkpoint_sha256(args.student_checkpoint),
        "raw_train_rows": len(raw_train), "eval_rows": len(eval_rows),
        "rejected_identifier_overlap": rejected_id,
        "rejected_source_or_original_target_overlap": rejected_molecule,
        "selected_train_rows": len(chosen), "verified_outcome_rows": len(outcomes),
        "row_coverage": len(outcomes) / max(len(chosen), 1),
        "no_transaction_support": no_support, "no_verified_success": no_verified,
        "student_vocab_oov": oov, "rejected_pseudo_target_eval_overlap": pseudo_overlap,
        "candidate_counts": {
            "valid": valid_candidates, "strict": strict_candidates,
            "identity_strict_rejected": identity_strict_candidates,
            "mean_verified_pool": sum(verified_pool_sizes) / max(len(verified_pool_sizes), 1),
            "max_verified_pool": max(verified_pool_sizes, default=0),
        },
        "coverage_by_task": coverage_table(chosen, success_keys, group_name),
        "coverage_by_property_count": coverage_table(chosen, success_keys, property_count),
        "r1_rows": len(r1_rows), "r2_weighted_rows": len(r2_rows),
        "success_set_confidence_median": median,
        "remaining_eval_overlap": overlap,
        "official_predicate": "table1_strict_success at source_similarity_threshold=0.65",
        "verified_filter_precedes_teacher_selection": True,
        "eval_candidates_used_for_training": False,
        "property_reranking_at_inference": False,
        "teacher_used_at_student_inference": False,
        "r2_single_factor": "teacher likelihood confidence weighting inside success set",
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if any(overlap.values()):
        raise SystemExit("Fail-closed overlap audit found evaluation leakage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
