#!/usr/bin/env python3
"""Build leak-free deterministic full-SMILES outcomes from the P8.1.1 teacher."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
UNIFIED_DIR = PROJECT_DIR / "experiments" / "unified_smiles_generator"
P811_DIR = PROJECT_DIR / "experiments" / "p8_1_1_short_transaction"
for path in (PROJECT_DIR, UNIFIED_DIR, P811_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import sample_raw_transactions as teacher_sampler  # noqa: E402
import unified_smiles_generator as unified  # noqa: E402
import umtp_graph_action_policy as policy  # noqa: E402


ID_FIELDS = ("variant_id", "condition_id", "sample_id", "pair_id")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def canonical(value: str) -> str:
    return unified.safe_canonical_smiles(str(value or "").strip()) or ""


def stable_key(row: dict[str, str]) -> str:
    values = [str(row.get(key, "") or "").strip() for key in ID_FIELDS]
    values += [str(row.get("source_smiles", "") or ""), str(row.get("instruction", "") or "")]
    return "|".join(values)


def select_balanced(rows: list[dict[str, str]], limit: int, seed: int) -> list[dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        name = str(row.get("benchmark_task", "") or row.get("task_name", "") or row.get("instruction", "edit"))
        groups[name].append(row)
    for name, values in groups.items():
        values.sort(key=lambda row: hashlib.sha256(f"{seed}|{name}|{stable_key(row)}".encode()).hexdigest())
    names = sorted(groups)
    output: list[dict[str, str]] = []
    cursor = 0
    while names and len(output) < limit:
        next_names = []
        for name in names:
            if cursor < len(groups[name]) and len(output) < limit:
                output.append(groups[name][cursor])
            if cursor + 1 < len(groups[name]):
                next_names.append(name)
        names = next_names
        cursor += 1
    return output


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

    eval_rows = read_rows(args.eval_csv)
    eval_ids = {str(row.get(key, "") or "").strip() for row in eval_rows for key in ID_FIELDS if str(row.get(key, "") or "").strip()}
    eval_molecules = {
        value
        for row in eval_rows
        for value in (canonical(row.get("source_smiles", "")), canonical(row.get("target_smiles", "")))
        if value
    }
    raw_train = [row for row in read_rows(args.train_csv) if unified.task_mode_for_row(row) == unified.EDIT_MODE]
    rejected_id = rejected_molecule = 0
    eligible: list[dict[str, str]] = []
    for row in raw_train:
        ids = {str(row.get(key, "") or "").strip() for key in ID_FIELDS if str(row.get(key, "") or "").strip()}
        if ids & eval_ids:
            rejected_id += 1
            continue
        source = canonical(row.get("source_smiles", ""))
        original_target = canonical(row.get("target_smiles", ""))
        if not source or source in eval_molecules or (original_target and original_target in eval_molecules):
            rejected_molecule += 1
            continue
        eligible.append(row)
    chosen = select_balanced(eligible, min(int(args.limit), len(eligible)), int(args.seed))

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
    no_support = oov = pseudo_eval_overlap = 0
    for row_index, row in enumerate(chosen):
        candidates = teacher_sampler.source_only_candidates(
            row, site_limit=int(args.site_limit), limit=int(args.max_actions)
        )
        if not candidates:
            no_support += 1
            continue
        condition = unified.condition_array_for_row(
            row, store, int(teacher_config["condition_dim"]), max_source_tokens=96,
            condition_layout="direct_compat",
        ).astype(np.float32)
        scores = policy.score_programs(
            teacher, teacher_vocab, condition, [item[2] for item in candidates],
            batch_size=int(args.score_batch_size), device=device,
        )
        best = int(np.argmax(np.asarray(scores, dtype=np.float64)))
        action, outcome, program = candidates[best]
        outcome = canonical(outcome)
        if not outcome:
            no_support += 1
            continue
        if outcome in eval_molecules:
            pseudo_eval_overlap += 1
            continue
        tokens = unified.tokenize_smiles(outcome)
        if any(token not in student_vocab.token_to_id for token in tokens):
            oov += 1
            continue
        logits = torch.tensor(scores, dtype=torch.float64)
        probs = torch.softmax(logits, dim=0)
        confidence = float(probs[best])
        entropy = float(-(probs * probs.clamp_min(1e-300).log()).sum())
        normalized_entropy = entropy / max(math.log(max(len(candidates), 2)), 1e-12)
        item: dict[str, object] = dict(row)
        item["target_smiles"] = outcome
        item["task_mode"] = "edit"
        item["pseudo_label_origin"] = "p8_1_1_train_only_teacher"
        item["teacher_action_json"] = json.dumps(asdict(action), sort_keys=True)
        item["teacher_program_tokens_json"] = json.dumps(program)
        item["teacher_logprob"] = policy.format_float(float(scores[best]))
        item["teacher_confidence"] = policy.format_float(confidence)
        item["teacher_normalized_entropy"] = policy.format_float(normalized_entropy)
        item["distill_weight_group"] = "uniform"
        outcomes.append(item)
        if (row_index + 1) % 50 == 0 or row_index + 1 == len(chosen):
            print(f"[p8.1.9-teacher] {row_index + 1}/{len(chosen)}", flush=True)

    if not outcomes:
        raise SystemExit("No leak-free teacher outcomes were produced")
    confidence_values = sorted(float(row["teacher_confidence"]) for row in outcomes)
    median = confidence_values[len(confidence_values) // 2]
    r1_rows = []
    r2_rows = []
    for row in outcomes:
        r1_item = dict(row)
        r1_item["distill_replica"] = 0
        r1_rows.append(r1_item)
        repeats = 2 if float(row["teacher_confidence"]) >= median else 1
        for replica in range(repeats):
            r2_item = dict(row)
            r2_item["distill_weight_group"] = "teacher_confidence_2x_above_median"
            r2_item["distill_replica"] = replica
            r2_rows.append(r2_item)
    write_rows(args.r1_output, r1_rows)
    write_rows(args.r2_output, r2_rows)

    train_sources = {canonical(row.get("source_smiles", "")) for row in outcomes}
    train_targets = {canonical(str(row.get("target_smiles", ""))) for row in outcomes}
    output_ids = {str(row.get(key, "") or "").strip() for row in outcomes for key in ID_FIELDS if str(row.get(key, "") or "").strip()}
    remaining_overlap = {
        "id": sorted(output_ids & eval_ids),
        "source": sorted(train_sources & eval_molecules),
        "pseudo_target": sorted(train_targets & eval_molecules),
    }
    audit = {
        "protocol": "p8_1_9_train_only_teacher_outcome_audit_v1",
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "teacher_checkpoint_sha256": policy.checkpoint_sha256(args.teacher_checkpoint),
        "student_checkpoint": str(args.student_checkpoint),
        "student_checkpoint_sha256": policy.checkpoint_sha256(args.student_checkpoint),
        "raw_train_rows": len(raw_train),
        "eval_rows": len(eval_rows),
        "rejected_identifier_overlap": rejected_id,
        "rejected_source_or_original_target_overlap": rejected_molecule,
        "selected_train_rows": len(chosen),
        "teacher_outcomes": len(outcomes),
        "no_transaction_support": no_support,
        "student_vocab_oov": oov,
        "rejected_pseudo_target_eval_overlap": pseudo_eval_overlap,
        "r1_rows": len(r1_rows),
        "r2_weighted_rows": len(r2_rows),
        "confidence_median": median,
        "remaining_eval_overlap": remaining_overlap,
        "eval_candidates_used_for_training": False,
        "property_reranking": False,
        "teacher_used_at_student_inference": False,
        "r2_single_factor": "teacher confidence weighting",
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if any(remaining_overlap.values()):
        raise SystemExit("Fail-closed overlap audit found evaluation leakage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
