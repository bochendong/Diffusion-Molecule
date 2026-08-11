#!/usr/bin/env python3
"""Let a common LLM select the fixed final n=20 RetrievedDeltaEdit pool.

The model sees only ConstraintIR plus typed train-retrieved delta actions.  It
never sees evaluation targets or official oracle values.  Internal candidates
are bounded before scoring; only the selected 20 are emitted for the oracle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_common_llm_constrained_actions as constrained  # noqa: E402
import retrieved_delta_plan_protocol as protocol  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enumerated-candidates-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--preference-manifest-json", required=True, type=Path)
    parser.add_argument("--candidate-budget", type=int, default=20)
    parser.add_argument("--planner-candidate-limit", type=int, default=96)
    parser.add_argument("--min-source-tanimoto", type=float, default=0.4)
    parser.add_argument("--score-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(str(key) for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite(value: object, fallback: float = -math.inf) -> float:
    try:
        number = float(str(value or "").strip())
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def candidate_rank(row: Mapping[str, object]) -> int:
    return int(finite(row.get("candidate_rank"), 10**9))


def condition_id(row: Mapping[str, object]) -> str:
    value = str(row.get("condition_id", "") or "").strip()
    if not value:
        raise ValueError("Enumerated candidate is missing condition_id")
    return value


def is_retrieved(row: Mapping[str, object]) -> bool:
    return str(row.get("graph_edit_candidate_source", "") or "") == "retrieved_delta_edit"


def action_record(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "delta_query_variable": row.get("delta_query_variable", ""),
        "delta_source_variable": row.get("delta_source_variable", ""),
        "delta_target_variable": row.get("delta_target_variable", ""),
    }


def read_progress(path: Path) -> dict[str, list[dict[str, object]]]:
    if not path.is_file():
        return {}
    output: dict[str, list[dict[str, object]]] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
        output[str(record["condition_id"])] = [dict(row) for row in record["rows"]]
    return output


def append_progress(path: Path, key: str, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"condition_id": key, "rows": list(rows)}, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def selected_key(row: Mapping[str, object]) -> tuple[float, ...]:
    score = finite(row.get("delta_llm_log_probability"), -math.inf)
    safe = finite(row.get("delta_source_tanimoto"), -math.inf) >= finite(
        row.get("delta_planner_min_source_tanimoto"), 0.4
    )
    return (
        float(safe),
        float(math.isfinite(score)),
        score,
        finite(row.get("delta_retrieval_similarity"), 0.0),
        finite(row.get("delta_source_tanimoto"), -1.0),
        finite(row.get("delta_selection_score"), -math.inf),
        -candidate_rank(row),
    )


def rank_condition(
    rows: Sequence[Mapping[str, str]],
    *,
    model: object,
    tokenizer: object,
    candidate_budget: int,
    planner_candidate_limit: int,
    min_source_tanimoto: float,
    score_batch_size: int,
    max_length: int,
) -> list[dict[str, object]]:
    if not rows:
        return []
    bounded = sorted(rows, key=candidate_rank)[: max(int(candidate_budget), int(planner_candidate_limit))]
    messages = protocol.prompt_messages(bounded[0])
    retrieved_rows = [row for row in bounded if is_retrieved(row)]
    encoded = [
        constrained.encoded_action(
            tokenizer,
            messages,
            protocol.action_payload(action_record(row)),
            max_length=int(max_length),
        )
        for row in retrieved_rows
    ]
    scores = constrained.score_encoded_actions(
        model,
        tokenizer,
        encoded,
        batch_size=int(score_batch_size),
    ) if encoded else []
    score_by_smiles = {
        str(row.get("generated_smiles", "") or ""): float(score)
        for row, score in zip(retrieved_rows, scores)
    }
    scored: list[dict[str, object]] = []
    for row in bounded:
        output: dict[str, object] = dict(row)
        smiles = str(row.get("generated_smiles", "") or "")
        score = score_by_smiles.get(smiles, -math.inf)
        output["delta_heuristic_rank"] = candidate_rank(row)
        output["delta_llm_log_probability"] = score if math.isfinite(score) else ""
        output["delta_llm_scored"] = str(math.isfinite(score))
        output["delta_planner_min_source_tanimoto"] = float(min_source_tanimoto)
        scored.append(output)
    scored.sort(key=selected_key, reverse=True)
    selected = scored[: int(candidate_budget)]
    if len(selected) != int(candidate_budget):
        raise ValueError(
            f"Condition {condition_id(rows[0])} has {len(selected)} planner candidates; "
            f"fixed n={candidate_budget} is required"
        )
    for rank, row in enumerate(selected, start=1):
        row["candidate_rank"] = rank
        row["generation_rank"] = rank
        row["candidate_selected"] = "True" if rank == 1 else "False"
        row["method"] = "common_llm_retrieved_delta_planner_v6"
    return selected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.candidate_budget) != 20:
        raise ValueError("The paper-facing planner protocol fixes candidate_budget=20")
    if int(args.planner_candidate_limit) < int(args.candidate_budget):
        raise ValueError("planner_candidate_limit cannot be below candidate_budget")
    if not args.adapter_dir.joinpath("adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Missing common-LLM adapter: {args.adapter_dir}")
    try:
        import peft
        import torch
        import transformers
    except ImportError as exc:
        raise SystemExit(f"Missing common-LLM planner dependency: {exc}") from exc
    if not torch.cuda.is_available():
        raise SystemExit("Common-LLM RetrievedDelta planner requires CUDA")

    all_rows = read_rows(args.enumerated_candidates_csv)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    order: list[str] = []
    for row in all_rows:
        key = condition_id(row)
        if key not in grouped:
            order.append(key)
        grouped[key].append(row)
    if not order:
        raise ValueError("Enumerated candidate pool is empty")
    if any(len(grouped[key]) < int(args.candidate_budget) for key in order):
        raise ValueError("Enumerated pool contains a condition with fewer than n=20 candidates")

    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model = peft.PeftModel.from_pretrained(model, args.adapter_dir)
    model.config.use_cache = False
    model = model.cuda().eval()

    progress_path = args.output_dir / "ranking_progress.jsonl"
    completed = read_progress(progress_path)
    for index, key in enumerate(order, start=1):
        if key in completed:
            print(f"[delta-planner] {index}/{len(order)} resume {key}", flush=True)
            continue
        selected = rank_condition(
            grouped[key],
            model=model,
            tokenizer=tokenizer,
            candidate_budget=int(args.candidate_budget),
            planner_candidate_limit=int(args.planner_candidate_limit),
            min_source_tanimoto=float(args.min_source_tanimoto),
            score_batch_size=int(args.score_batch_size),
            max_length=int(args.max_length),
        )
        append_progress(progress_path, key, selected)
        completed[key] = selected
        print(f"[delta-planner] {index}/{len(order)} done {key}", flush=True)

    output = [row for key in order for row in completed[key]]
    write_rows(args.output_csv, output)
    source_counts = Counter(str(row.get("graph_edit_candidate_source", "") or "unknown") for row in output)
    preference_manifest = json.loads(args.preference_manifest_json.read_text(encoding="utf-8"))
    manifest = {
        "protocol": "common_llm_retrieved_delta_planner_v1",
        "data_role": "train_only_to_disjoint_train_audit",
        "evaluation_target_access": False,
        "candidate_budget": int(args.candidate_budget),
        "planner_candidate_limit": int(args.planner_candidate_limit),
        "min_source_tanimoto": float(args.min_source_tanimoto),
        "evaluation_conditions": len(order),
        "enumerated_input_rows": len(all_rows),
        "planner_considered_rows": sum(
            min(len(grouped[key]), int(args.planner_candidate_limit)) for key in order
        ),
        "output_rows": len(output),
        "candidate_source_counts": dict(sorted(source_counts.items())),
        "base_model": str(args.base_model),
        "adapter_dir": str(args.adapter_dir),
        "preference_protocol": preference_manifest.get("protocol"),
        "preference_prompt_target_access": preference_manifest.get("prompt_target_access"),
        "preference_source_group_overlap": preference_manifest.get("source_group_overlap"),
        "preference_train_pairs": preference_manifest.get("train_pairs"),
        "enumerated_candidates_sha256": sha256(args.enumerated_candidates_csv),
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
