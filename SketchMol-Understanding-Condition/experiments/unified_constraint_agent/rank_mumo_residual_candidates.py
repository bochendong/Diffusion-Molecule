#!/usr/bin/env python3
"""Use a common 1.5B LLM only to fill five residual slots at fixed n=20."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_mumo_residual_preferences as preference  # noqa: E402
import evaluate_common_llm_constrained_actions as constrained  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", required=True, type=Path)
    parser.add_argument("--enumerated-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--reference-adapter-dir", type=Path, default=None)
    parser.add_argument("--preference-manifest", required=True, type=Path)
    parser.add_argument("--baseline-prefix", type=int, default=15)
    parser.add_argument("--residual-slots", type=int, default=5)
    parser.add_argument("--max-llm-rank-shift", type=float, default=12.0)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--method-name", default="common_llm_1p5b_residual_v9")
    return parser.parse_args(argv)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def candidate_payload(row: Mapping[str, object]) -> dict[str, object]:
    margins = json.loads(str(row.get("verifier_margins_json") or "{}"))
    return preference.action_payload(
        str(row["generated_smiles"]),
        margins,
        source_tanimoto=float(row["source_tanimoto"]),
        retrieval_similarity=float(row["retrieval_similarity"]),
        frequency=int(row["transform_frequency"]),
        candidate_source="residual_candidate",
    )


def condition_prompt(row: Mapping[str, object]) -> list[dict[str, str]]:
    properties = tuple(str(row["external_task_properties"]).split(","))
    proxy = {
        "source_smiles": row["source_smiles"],
        "_uca_task_id": row["external_task_id"],
    }
    return preference.prompt_messages(proxy, properties)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bounded_residual_ranking(
    rows: Sequence[Mapping[str, object]],
    llm_scores: Sequence[float],
    *,
    max_llm_rank_shift: float,
) -> list[tuple[Mapping[str, object], float, float]]:
    """Add a bounded LLM rank correction to the deterministic verifier rank.

    The common LLM is deliberately a residual policy: it can move a candidate
    by at most ``max_llm_rank_shift`` rank units, but it cannot replace the
    train-only verifier score with an uncalibrated sequence likelihood.
    """

    if len(rows) != len(llm_scores):
        raise ValueError("Residual rows/scores length mismatch")
    order = sorted(range(len(rows)), key=lambda index: (float(llm_scores[index]), -index))
    percentiles = [0.5] * len(rows)
    if len(rows) > 1:
        for rank, index in enumerate(order):
            percentiles[index] = rank / float(len(rows) - 1)
    ranked = []
    for row, llm_score, percentile in zip(rows, llm_scores, percentiles):
        base_rank = int(row["internal_candidate_rank"])
        correction = float(max_llm_rank_shift) * (float(percentile) - 0.5)
        combined = -float(base_rank) + correction
        ranked.append((row, float(llm_score), combined))
    return sorted(
        ranked,
        key=lambda item: (item[2], item[1], -int(item[0]["internal_candidate_rank"])),
        reverse=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.baseline_prefix) + int(args.residual_slots) != 20:
        raise ValueError("Residual planner must emit exactly n=20")
    import peft
    import torch
    import transformers
    if not torch.cuda.is_available():
        raise SystemExit("MuMO residual ranking requires CUDA")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float32, low_cpu_mem_usage=True
    )
    if args.reference_adapter_dir is not None:
        model = peft.PeftModel.from_pretrained(
            base, args.adapter_dir, adapter_name="candidate"
        )
        model.load_adapter(args.reference_adapter_dir, adapter_name="reference")
        model.set_adapter("candidate")
    else:
        model = peft.PeftModel.from_pretrained(base, args.adapter_dir)
    model = model.cuda().eval()
    model.config.use_cache = False

    baseline_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    internal_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(args.baseline_csv):
        baseline_groups[row["condition_id"]].append(row)
    for row in read_csv(args.enumerated_csv):
        internal_groups[row["condition_id"]].append(row)
    if set(baseline_groups) != set(internal_groups):
        raise ValueError("Baseline/internal condition sets differ")
    output = []
    source_counts: Counter[str] = Counter()
    changed_conditions = 0
    unique_counts = []
    noop_attempt_rows = 0
    for condition in sorted(baseline_groups):
        baseline = sorted(baseline_groups[condition], key=lambda row: int(row["candidate_rank"]))
        internal = sorted(internal_groups[condition], key=lambda row: int(row["internal_candidate_rank"]))
        if len(baseline) != 20:
            raise ValueError(f"{condition} baseline is not exact n=20")
        prefix = [dict(row) for row in baseline[: int(args.baseline_prefix)]]
        prefix_smiles = {row["generated_smiles"] for row in prefix}
        residual_pool = [
            row for row in internal
            if int(row["internal_candidate_rank"]) > int(args.baseline_prefix)
            and row["generated_smiles"] not in prefix_smiles
        ]
        encoded = [
            constrained.encoded_action(
                tokenizer,
                condition_prompt(row),
                candidate_payload(row),
                max_length=int(args.max_length),
            )
            for row in residual_pool
        ]
        reference_scores = None
        if args.reference_adapter_dir is not None:
            model.set_adapter("reference")
            reference_scores = constrained.score_encoded_actions(
                model, tokenizer, encoded, batch_size=int(args.score_batch_size)
            ) if encoded else []
            model.set_adapter("candidate")
        scores = constrained.score_encoded_actions(
            model, tokenizer, encoded, batch_size=int(args.score_batch_size)
        ) if encoded else []
        ranking_scores = (
            [candidate - reference for candidate, reference in zip(scores, reference_scores)]
            if reference_scores is not None
            else scores
        )
        scored = bounded_residual_ranking(
            residual_pool,
            ranking_scores,
            max_llm_rank_shift=float(args.max_llm_rank_shift),
        )
        selected_tail = []
        score_by_smiles = {
            str(row["generated_smiles"]): float(score)
            for row, score in zip(residual_pool, scores)
        }
        reference_by_smiles = {
            str(row["generated_smiles"]): float(score)
            for row, score in zip(residual_pool, reference_scores or [])
        }
        for row, residual_score, combined_score in scored[: int(args.residual_slots)]:
            item = dict(row)
            smiles = str(row["generated_smiles"])
            item["residual_llm_log_probability"] = score_by_smiles[smiles]
            item["stable_reference_log_probability"] = reference_by_smiles.get(smiles, "")
            item["residual_log_probability_delta"] = float(residual_score)
            item["residual_combined_rank_score"] = float(combined_score)
            item["residual_selected"] = True
            selected_tail.append(item)
            source_counts[str(item["method"])] += 1
        if len(selected_tail) < int(args.residual_slots):
            existing = {row["generated_smiles"] for row in prefix + selected_tail}
            for row in baseline[int(args.baseline_prefix):]:
                if row["generated_smiles"] in existing and len(internal) >= 20:
                    continue
                item = dict(row)
                item["residual_llm_log_probability"] = ""
                item["residual_selected"] = False
                selected_tail.append(item)
                existing.add(row["generated_smiles"])
                if len(selected_tail) == int(args.residual_slots):
                    break
        selected = prefix + selected_tail
        if len(selected) != 20:
            raise ValueError(f"{condition} residual output has {len(selected)} attempts")
        baseline_tail = [
            row["generated_smiles"]
            for row in baseline[int(args.baseline_prefix) :]
        ]
        changed_conditions += int([row["generated_smiles"] for row in selected_tail] != baseline_tail)
        unique_counts.append(len({row["generated_smiles"] for row in selected}))
        unique_ranks: dict[str, int] = {}
        for rank, row in enumerate(selected, start=1):
            generated_smiles = row["generated_smiles"]
            original_method = row.get("method", "")
            is_repeat = generated_smiles in unique_ranks
            if not is_repeat:
                unique_ranks[generated_smiles] = len(unique_ranks) + 1
            row["candidate_rank"] = rank
            row["candidate_selected"] = rank == 1
            row["candidate_attempt_is_repeat"] = is_repeat
            row["candidate_unique_rank"] = unique_ranks[generated_smiles]
            row["residual_candidate_source"] = original_method
            row["method"] = str(args.method_name)
            output.append(row)
            noop_attempt_rows += int(str(row.get("candidate_is_noop", "")).lower() == "true")
    write_csv(args.output_csv, output)
    pref = json.loads(args.preference_manifest.read_text(encoding="utf-8"))
    manifest = {
        "protocol": "common_llm_mumo_residual_planner_v1",
        "evaluation_target_access": False,
        "evaluation_oracle_access": False,
        "official_test_content_access": False,
        "candidate_budget": 20,
        "baseline_prefix": int(args.baseline_prefix),
        "residual_slots": int(args.residual_slots),
        "max_llm_rank_shift": float(args.max_llm_rank_shift),
        "selection_policy": "deterministic_prefix_plus_bounded_common_llm_residual",
        "stable_reference_adapter": (
            str(args.reference_adapter_dir) if args.reference_adapter_dir is not None else None
        ),
        "stable_reference_residual_scoring": args.reference_adapter_dir is not None,
        "conditions": len(baseline_groups),
        "output_rows": len(output),
        "attempted_candidates_total": len(output),
        "unique_candidates_total": sum(unique_counts),
        "unique_valid_candidates_total": sum(unique_counts),
        "mean_unique_candidates_per_condition": sum(unique_counts) / max(len(unique_counts), 1),
        "min_unique_candidates_per_condition": min(unique_counts, default=0),
        "repeated_attempt_rows": len(output) - sum(unique_counts),
        "noop_attempt_rows": noop_attempt_rows,
        "changed_conditions": changed_conditions,
        "candidate_source_counts": dict(sorted(source_counts.items())),
        "preference_prompt_target_access": pref.get("prompt_target_access"),
        "preference_source_group_overlap": pref.get("source_group_overlap"),
        "enumerated_sha256": sha256(args.enumerated_csv),
        "baseline_sha256": sha256(args.baseline_csv),
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
