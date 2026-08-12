#!/usr/bin/env python3
"""Rank official Table1 or MuMO GraphEditDSL pools with the frozen common LLM.

The evaluation contract deliberately separates three components:

1. GraphEditDSL enumerates and executes valid source-conditioned edits.
2. The frozen common LLM ranks exactly ``candidate_budget`` candidates.
3. A target-free vector verifier may select only within the first ``verifier_k``.

Every completed input is appended to a JSONL checkpoint, so a pre-empted Slurm
job resumes without recomputing finished rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
POLICY_DIR = SCRIPT_DIR.parent / "unified_smiles_generator"
for import_dir in (SCRIPT_DIR, POLICY_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from build_common_llm_sft_dataset import SYSTEM_PROMPT  # noqa: E402
from molecular_constraint_ir import build_constraint_ir  # noqa: E402

import evaluate_common_llm_constrained_actions as constrained  # noqa: E402
import anchor_residual_ranking as anchor_ranking  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--suite", required=True, choices=("table1", "mumo"))
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--variant", default="verifier_preference_seed_1705")
    parser.add_argument("--candidate-budget", type=int, default=20)
    parser.add_argument("--verifier-k", type=int, default=5)
    parser.add_argument("--enumeration-attempt-budget", type=int, default=64)
    parser.add_argument("--max-enumeration-attempt-budget", type=int, default=512)
    parser.add_argument("--site-limit", type=int, default=32)
    parser.add_argument("--score-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--checkpoint-name", default="progress.jsonl")
    parser.add_argument("--reference-adapter-dir", type=Path, default=None)
    parser.add_argument("--anchor-top-k", type=int, default=5)
    parser.add_argument("--max-residual-rank-shift", type=float, default=4.0)
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def row_id(row: Mapping[str, object], index: int | None = None) -> str:
    for key in ("condition_id", "sample_id", "example_id", "variant_id", "pair_id"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value.split(":")[-1] if value.startswith("edit:") else value
    if index is not None:
        return f"row_{index:08d}"
    payload = json.dumps(dict(row), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def prompt_messages(row: Mapping[str, object]) -> list[dict[str, str]]:
    ir = build_constraint_ir(row)
    if ir.task_mode != "edit":
        raise ValueError(f"Official edit evaluation received task_mode={ir.task_mode!r}")
    user_payload = {
        "constraint_ir": ir.to_dict(),
        "response_schema": {"action_type": ir.action_space, "value": "one action"},
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(user_payload, sort_keys=True, separators=(",", ":")),
        },
    ]


def candidate_pool(
    row: Mapping[str, str],
    *,
    candidate_budget: int,
    initial_attempt_budget: int,
    max_attempt_budget: int,
    site_limit: int,
) -> tuple[list[tuple[object, str, list[str]]], int]:
    """Return a stable valid prefix, widening enumeration only when underfilled."""
    policy = constrained.policy_module()
    attempt_budget = max(int(candidate_budget), int(initial_attempt_budget))
    max_attempt_budget = max(attempt_budget, int(max_attempt_budget))
    candidates: list[tuple[object, str, list[str]]] = []
    while True:
        candidates = policy.enumerate_action_candidates(
            row,
            site_limit=int(site_limit),
            max_actions_per_row=int(attempt_budget),
        )
        if len(candidates) >= int(candidate_budget) or attempt_budget >= max_attempt_budget:
            break
        attempt_budget = min(max_attempt_budget, attempt_budget * 2)
    return candidates[: int(candidate_budget)], attempt_budget


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def finite_float(value: object, default: float = -math.inf) -> float:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def verifier_key(row: Mapping[str, object]) -> tuple[float, ...]:
    """Target-free vector verifier ordering used only inside a bounded prefix."""
    return (
        float(truthy(row.get("table1_strict_success")) or truthy(row.get("diagnostic_strict_success"))),
        float(truthy(row.get("table1_instruction_success"))),
        float(truthy(row.get("source_similarity_success"))),
        finite_float(row.get("unified_property_success_fraction"), 0.0),
        -finite_float(row.get("unified_property_distance"), 1e6),
        finite_float(row.get("source_tanimoto"), -1.0),
        finite_float(row.get("llm_mean_log_probability"), -math.inf),
        -finite_float(row.get("candidate_rank"), 1e9),
    )


def evaluate_row(
    row: Mapping[str, str],
    *,
    index: int,
    model: object,
    tokenizer: object,
    args: argparse.Namespace,
) -> dict[str, object]:
    policy = constrained.policy_module()
    planner_row = dict(row)
    candidates, attempt_budget = candidate_pool(
        planner_row,
        candidate_budget=int(args.candidate_budget),
        initial_attempt_budget=int(args.enumeration_attempt_budget),
        max_attempt_budget=int(args.max_enumeration_attempt_budget),
        site_limit=int(args.site_limit),
    )
    messages = prompt_messages(row)
    encoded = [
        constrained.encoded_action(
            tokenizer,
            messages,
            constrained.action_payload(action),
            max_length=int(args.max_length),
        )
        for action, _smiles, _program in candidates
    ]
    reference_scores: list[float] | None = None
    if args.reference_adapter_dir is not None:
        model.set_adapter("reference")
        reference_scores = (
            constrained.score_encoded_actions(
                model,
                tokenizer,
                encoded,
                batch_size=int(args.score_batch_size),
            )
            if encoded
            else []
        )
        model.set_adapter("candidate")
    scores = (
        constrained.score_encoded_actions(
            model,
            tokenizer,
            encoded,
            batch_size=int(args.score_batch_size),
        )
        if encoded
        else []
    )
    if reference_scores is None:
        order = sorted(range(len(scores)), key=lambda item: scores[item], reverse=True)
    else:
        order = anchor_ranking.anchored_order(
            reference_scores,
            scores,
            anchor_top_k=int(args.anchor_top_k),
            max_residual_rank_shift=float(args.max_residual_rank_shift),
        )
    ranked = [(*candidates[item], scores[item], item) for item in order]
    identity = row_id(row, index)
    pool_hash = hashlib.sha256(
        "\n".join(item[1] for item in ranked).encode("utf-8")
    ).hexdigest()
    threshold = 0.65 if args.suite == "table1" else 0.40
    candidate_rows: list[dict[str, object]] = []
    for rank, (action, smiles, program, score, original_index) in enumerate(ranked, start=1):
        if args.suite == "table1":
            oracle = policy.action_oracle_record(
                planner_row,
                (action, smiles, program),
                source_similarity_threshold=threshold,
            )
            metrics = policy.unified.candidate_metrics(
                planner_row,
                smiles,
                source_similarity_threshold=threshold,
            )
        else:
            # MuMO's paper oracle is ADMET-AI + TDC and runs once over all
            # unique candidates after ranking. Do not substitute local proxy
            # properties here or spend thousands of repeated oracle calls.
            similarity = policy.unified.morgan_tanimoto(
                str(planner_row.get("source_smiles", "") or ""),
                smiles,
            )
            source_success = bool(math.isfinite(similarity) and similarity >= threshold)
            oracle = {
                "strict_success": False,
                "instruction_success_fraction": math.nan,
                "source_similarity": similarity,
            }
            metrics = {
                "valid_smiles": "True",
                "unified_finalizer_score": "" if not math.isfinite(similarity) else similarity,
                "unified_property_success_fraction": "",
                "unified_property_distance": "",
                "source_tanimoto": "" if not math.isfinite(similarity) else similarity,
                "source_similarity_success": "True" if source_success else "False",
            }
        out: dict[str, object] = dict(row)
        out.update(
            {
                "generated_smiles": smiles,
                "method": f"common_llm_graph_edit_{args.variant}",
                "generation_rank": rank,
                "candidate_rank": rank,
                "candidate_budget": int(args.candidate_budget),
                "candidate_pool_id": f"common-llm:{args.variant}:{args.suite}:{identity}",
                "candidate_pool_hash": pool_hash,
                "graph_action_json": json.dumps(asdict(action), sort_keys=True),
                "graph_action_program_tokens_json": json.dumps(program),
                "llm_mean_log_probability": score,
                "reference_mean_log_probability": (
                    reference_scores[original_index] if reference_scores is not None else ""
                ),
                "residual_log_probability_delta": (
                    score - reference_scores[original_index]
                    if reference_scores is not None
                    else ""
                ),
                "diagnostic_strict_success": "True" if oracle["strict_success"] else "False",
                "diagnostic_instruction_success_fraction": oracle["instruction_success_fraction"],
                "diagnostic_source_similarity": oracle["source_similarity"],
            }
        )
        out.update(metrics)
        candidate_rows.append(out)

    selected_llm = candidate_rows[0] if candidate_rows else {}
    verifier_prefix = candidate_rows[: max(1, int(args.verifier_k))]
    selected_verifier = max(verifier_prefix, key=verifier_key) if verifier_prefix else {}
    strict_ranks = [
        int(item["candidate_rank"])
        for item in candidate_rows
        if truthy(item.get("diagnostic_strict_success"))
    ]
    reference_top_k_preserved = True
    if reference_scores is not None:
        reference_top_k_preserved = set(order[: int(args.anchor_top_k)]) == anchor_ranking.top_k_set(
            reference_scores, int(args.anchor_top_k)
        )
    detail = {
        "row_id": identity,
        "suite": args.suite,
        "candidate_count": len(candidate_rows),
        "candidate_pool_full": len(candidate_rows) == int(args.candidate_budget),
        "enumeration_attempt_budget_used": attempt_budget,
        "llm_at_1_strict": bool(strict_ranks and strict_ranks[0] == 1),
        "verifier_at_k_strict": truthy(selected_verifier.get("diagnostic_strict_success")),
        "any_strict_at_k": any(rank <= int(args.verifier_k) for rank in strict_ranks),
        "any_strict_at_20": bool(strict_ranks),
        "reference_top_k_preserved": reference_top_k_preserved,
        "first_strict_rank": strict_ranks[0] if strict_ranks else None,
        "llm_selected_rank": selected_llm.get("candidate_rank"),
        "verifier_selected_rank": selected_verifier.get("candidate_rank"),
        "llm_selected_smiles": selected_llm.get("generated_smiles", ""),
        "verifier_selected_smiles": selected_verifier.get("generated_smiles", ""),
        "llm_selected_property_success_fraction": selected_llm.get("unified_property_success_fraction", ""),
        "verifier_selected_property_success_fraction": selected_verifier.get(
            "unified_property_success_fraction", ""
        ),
        "llm_selected_source_similarity": selected_llm.get("source_tanimoto", ""),
        "verifier_selected_source_similarity": selected_verifier.get("source_tanimoto", ""),
    }
    return {"row_id": identity, "detail": detail, "candidates": candidate_rows}


def read_progress(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    records = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # A scheduler kill can interrupt only the final append. Earlier
            # corruption is not safe to ignore.
            if line_number == len(lines):
                break
            raise
        if isinstance(record, Mapping) and record.get("row_id"):
            records.append(dict(record))
    return records


def append_progress(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(str(key) for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def bool_rate(rows: Sequence[Mapping[str, object]], key: str) -> float:
    return sum(bool(row.get(key)) for row in rows) / max(len(rows), 1)


def finite_mean(rows: Sequence[Mapping[str, object]], key: str) -> float | None:
    values = [finite_float(row.get(key)) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return mean(values) if values else None


def summarize(details: Sequence[Mapping[str, object]], args: argparse.Namespace) -> dict[str, object]:
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    groups["all"] = list(details)
    if args.suite == "mumo":
        # The split is copied into candidate rows, but the compact detail keeps
        # only protocol metrics. Recover it from row id prefix-independent input
        # order in main before calling this function.
        for row in details:
            split = str(row.get("external_task_split", "") or "")
            if split:
                groups[split].append(row)
    return {
        "protocol": "common_llm_official_graph_edit_v1",
        "suite": args.suite,
        "variant": args.variant,
        "candidate_budget": int(args.candidate_budget),
        "verifier_k": int(args.verifier_k),
        "target_information_used_for_ranking": False,
        "groups": {
            name: {
                "rows": len(items),
                "full_pool_rate": bool_rate(items, "candidate_pool_full"),
                "mean_candidate_count": finite_mean(items, "candidate_count"),
                "llm_at_1_strict": bool_rate(items, "llm_at_1_strict"),
                "verifier_at_k_strict": bool_rate(items, "verifier_at_k_strict"),
                "any_strict_at_k": bool_rate(items, "any_strict_at_k"),
                "any_strict_at_20": bool_rate(items, "any_strict_at_20"),
                "reference_top_k_preserved_rate": bool_rate(
                    items, "reference_top_k_preserved"
                ),
                "mean_llm_property_success_fraction": finite_mean(
                    items, "llm_selected_property_success_fraction"
                ),
                "mean_verifier_property_success_fraction": finite_mean(
                    items, "verifier_selected_property_success_fraction"
                ),
                "mean_llm_source_similarity": finite_mean(items, "llm_selected_source_similarity"),
                "mean_verifier_source_similarity": finite_mean(
                    items, "verifier_selected_source_similarity"
                ),
            }
            for name, items in groups.items()
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.candidate_budget) != 20:
        raise ValueError("Official comparison fixes candidate_budget=20")
    if not 1 <= int(args.verifier_k) <= int(args.candidate_budget):
        raise ValueError("verifier_k must be inside the fixed candidate pool")
    if not args.adapter_dir.joinpath("adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Missing frozen adapter: {args.adapter_dir}")
    if args.reference_adapter_dir is not None:
        if not args.reference_adapter_dir.joinpath("adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"Missing reference adapter: {args.reference_adapter_dir}")
        if not 1 <= int(args.anchor_top_k) <= int(args.candidate_budget):
            raise ValueError("anchor_top_k must be inside the fixed candidate pool")
    try:
        import peft
        import torch
        import transformers
    except ImportError as exc:
        raise SystemExit(f"Missing common-LLM evaluation dependency: {exc}") from exc
    if not torch.cuda.is_available():
        raise SystemExit("Official common-LLM action evaluation requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = True

    rows = read_rows(args.input_csv)
    if args.max_rows > 0:
        rows = rows[: int(args.max_rows)]
    identities = [row_id(row, index) for index, row in enumerate(rows)]
    if len(set(identities)) != len(identities):
        raise ValueError("Official input contains duplicate row identifiers")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / args.checkpoint_name
    progress = read_progress(progress_path)
    completed = {str(record["row_id"]): record for record in progress}

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    if args.reference_adapter_dir is not None:
        model = peft.PeftModel.from_pretrained(
            model, args.adapter_dir, adapter_name="candidate"
        )
        model.load_adapter(args.reference_adapter_dir, adapter_name="reference")
        model.set_adapter("candidate")
    else:
        model = peft.PeftModel.from_pretrained(model, args.adapter_dir)
    model.config.use_cache = False
    model = model.cuda().eval()

    for index, row in enumerate(rows):
        identity = row_id(row, index)
        if identity in completed:
            print(f"[official-action] {index + 1}/{len(rows)} resume {identity}", flush=True)
            continue
        record = evaluate_row(
            row,
            index=index,
            model=model,
            tokenizer=tokenizer,
            args=args,
        )
        # Preserve split/task metadata in compact per-input summaries.
        detail = record["detail"]
        assert isinstance(detail, dict)
        for key in ("external_suite", "external_task_split", "external_task_id", "external_task_key"):
            if str(row.get(key, "") or ""):
                detail[key] = row[key]
        append_progress(progress_path, record)
        completed[identity] = record
        print(f"[official-action] {index + 1}/{len(rows)} done {identity}", flush=True)

    ordered_records = [completed[row_id(row, index)] for index, row in enumerate(rows)]
    details = [record["detail"] for record in ordered_records]
    candidates = [candidate for record in ordered_records for candidate in record["candidates"]]
    write_csv(args.output_dir / "details.csv", details)
    write_csv(args.output_dir / "candidates.csv", candidates)
    write_csv(
        args.output_dir / "llm_top1.csv",
        [record["candidates"][0] for record in ordered_records if record["candidates"]],
    )
    if args.suite == "table1":
        verifier_selected = []
        for record in ordered_records:
            pool = record["candidates"][: int(args.verifier_k)]
            if pool:
                verifier_selected.append(max(pool, key=verifier_key))
        write_csv(args.output_dir / f"verifier_top{int(args.verifier_k)}.csv", verifier_selected)
    summary = summarize(details, args)
    summary.update(
        {
            "input_csv": str(args.input_csv),
            "adapter_dir": str(args.adapter_dir),
            "output_dir": str(args.output_dir),
            "enumeration_attempt_budget": int(args.enumeration_attempt_budget),
            "max_enumeration_attempt_budget": int(args.max_enumeration_attempt_budget),
            "reference_adapter_dir": (
                str(args.reference_adapter_dir) if args.reference_adapter_dir is not None else None
            ),
            "anchor_top_k": int(args.anchor_top_k) if args.reference_adapter_dir else None,
            "max_residual_rank_shift": (
                float(args.max_residual_rank_shift) if args.reference_adapter_dir else None
            ),
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
