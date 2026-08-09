#!/usr/bin/env python3
"""Rerank an existing MuMO top-20 pool with the frozen common LLM.

The 2-step GraphEditDSL builder stores only the last action on candidate rows,
but keeps every executed action in a condition-grouped plan JSONL. This script
streams that large file once, reconstructs the one- or two-step plan for each
candidate, scores each action with the frozen common LLM, and averages action
log probabilities to obtain a length-normalized plan score.

Official ADMET-AI/TDC outcomes are read from an already evaluated detail CSV.
They never participate in LLM ranking and are used only by the bounded top-k
verifier and final metric aggregation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_common_llm_constrained_actions as constrained  # noqa: E402
import evaluate_common_llm_official_actions as official  # noqa: E402
import select_external_verifier_prefix as external_select  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-detail-csv", required=True, type=Path)
    parser.add_argument("--plan-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--candidate-budget", type=int, default=20)
    parser.add_argument("--verifier-k", type=int, default=5)
    parser.add_argument("--score-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-rows", type=int, default=0)
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


def candidate_key(row: Mapping[str, object]) -> tuple[str, str]:
    return (
        str(row.get("condition_id", "") or "").strip(),
        str(row.get("generated_smiles", "") or "").strip(),
    )


def clean_action(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    op = str(value.get("op", "") or "").strip()
    if not op:
        return None
    # Keep the training-time GraphEditAction schema, including semantic fields
    # such as prop/direction/reason. Convert JSON lists back to tuple-like lists
    # only through serialization; the model sees the same JSON representation.
    return {
        "op": op,
        "site": value.get("site"),
        "bond": value.get("bond"),
        "atom": str(value.get("atom", "") or ""),
        "fragment": str(value.get("fragment", "") or ""),
        "bond_order": str(value.get("bond_order", "") or ""),
        "prop": str(value.get("prop", "") or ""),
        "direction": str(value.get("direction", "") or ""),
        "reason": str(value.get("reason", "") or ""),
        "policy_score": float(value.get("policy_score", 0.0) or 0.0),
    }


def trace_action(row: Mapping[str, object]) -> dict[str, object] | None:
    raw = str(row.get("graph_edit_action_trace", "") or "")
    marker = ":dsl:"
    if marker not in raw:
        return None
    try:
        return clean_action(json.loads(raw.split(marker, 1)[1]))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def reconstruct_condition_plans(
    desired_smiles: Iterable[str],
    plan_records: Mapping[str, Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    """Recover the step-1 parent and step-2 action without target access."""
    output: dict[str, list[dict[str, object]]] = {}
    for smiles in desired_smiles:
        record = plan_records.get(smiles)
        if not record:
            continue
        last_action = clean_action(record.get("action"))
        if last_action is None:
            continue
        step = int(record.get("step", 1) or 1)
        if step <= 1:
            output[smiles] = [last_action]
            continue
        parent = str(record.get("parent_smiles", "") or "").strip()
        parent_record = plan_records.get(parent)
        first_action = clean_action(parent_record.get("action")) if parent_record else None
        if first_action is not None and int(parent_record.get("step", 0) or 0) == 1:
            output[smiles] = [first_action, last_action]
        else:
            # A last-action-only fallback is explicit in the output coverage;
            # this can occur if canonical duplicates were pruned upstream.
            output[smiles] = [last_action]
    return output


def write_plan_checkpoint(path: Path, plans: Mapping[tuple[str, str], Sequence[Mapping[str, object]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for (condition_id, smiles), actions in sorted(plans.items()):
            handle.write(
                json.dumps(
                    {"condition_id": condition_id, "generated_smiles": smiles, "actions": list(actions)},
                    sort_keys=True,
                )
                + "\n"
            )
    os.replace(temporary, path)


def read_plan_checkpoint(path: Path) -> dict[tuple[str, str], list[dict[str, object]]]:
    output = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = (str(record["condition_id"]), str(record["generated_smiles"]))
            output[key] = [dict(action) for action in record.get("actions", [])]
    return output


def reconstruct_candidate_plans(
    candidate_rows: Sequence[Mapping[str, str]],
    plan_jsonl: Path,
    checkpoint_path: Path,
) -> dict[tuple[str, str], list[dict[str, object]]]:
    desired: dict[str, set[str]] = defaultdict(set)
    fallback: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in candidate_rows:
        condition_id, smiles = candidate_key(row)
        if condition_id and smiles:
            desired[condition_id].add(smiles)
            action = trace_action(row)
            if action is not None:
                fallback[(condition_id, smiles)] = [action]

    output: dict[tuple[str, str], list[dict[str, object]]] = {}
    current_condition = ""
    current_records: dict[str, Mapping[str, object]] = {}

    def flush() -> None:
        if not current_condition or current_condition not in desired:
            return
        reconstructed = reconstruct_condition_plans(desired[current_condition], current_records)
        for smiles, actions in reconstructed.items():
            output[(current_condition, smiles)] = actions

    with plan_jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            condition_id = str(record.get("condition_id", "") or "")
            if current_condition and condition_id != current_condition:
                flush()
                current_records = {}
            current_condition = condition_id
            generated = str(record.get("generated_smiles", "") or "").strip()
            if generated and bool(record.get("valid", False)):
                current_records.setdefault(generated, record)
            if line_number % 1_000_000 == 0:
                print(f"[plan-reconstruct] scanned {line_number} records", flush=True)
    flush()
    for key, actions in fallback.items():
        output.setdefault(key, actions)
    write_plan_checkpoint(checkpoint_path, output)
    return output


def action_payload(action: Mapping[str, object]) -> dict[str, object]:
    return {"action_type": "graph_edit_dsl", "value": dict(action)}


def read_progress(path: Path) -> dict[str, list[dict[str, object]]]:
    if not path.is_file():
        return {}
    output = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if line_number == len(lines):
                break
            raise
        output[str(record["condition_id"])] = [dict(row) for row in record["rows"]]
    return output


def append_progress(path: Path, condition_id: str, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"condition_id": condition_id, "rows": list(rows)}, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def score_condition(
    rows: Sequence[Mapping[str, str]],
    plans: Mapping[tuple[str, str], Sequence[Mapping[str, object]]],
    *,
    model: object,
    tokenizer: object,
    batch_size: int,
    max_length: int,
) -> list[dict[str, object]]:
    if not rows:
        return []
    messages = official.prompt_messages(rows[0])
    encoded = []
    owners: list[int] = []
    action_counts = [0] * len(rows)
    for row_index, row in enumerate(rows):
        actions = list(plans.get(candidate_key(row), []))
        action_counts[row_index] = len(actions)
        for action in actions:
            encoded.append(
                constrained.encoded_action(
                    tokenizer,
                    messages,
                    action_payload(action),
                    max_length=int(max_length),
                )
            )
            owners.append(row_index)
    scores = (
        constrained.score_encoded_actions(model, tokenizer, encoded, batch_size=int(batch_size))
        if encoded
        else []
    )
    by_owner: dict[int, list[float]] = defaultdict(list)
    for owner, score in zip(owners, scores):
        by_owner[owner].append(score)
    scored = []
    for row_index, row in enumerate(rows):
        values = by_owner.get(row_index, [])
        out: dict[str, object] = dict(row)
        out["original_candidate_rank"] = row.get("candidate_rank", "")
        out["llm_plan_action_count"] = action_counts[row_index]
        out["llm_plan_reconstructed"] = "True" if action_counts[row_index] else "False"
        out["llm_plan_mean_log_probability"] = mean(values) if values else -math.inf
        out["llm_plan_min_log_probability"] = min(values) if values else -math.inf
        scored.append(out)
    scored.sort(
        key=lambda row: (
            official.finite_float(row.get("llm_plan_mean_log_probability"), -math.inf),
            -external_select.candidate_rank(row),
        ),
        reverse=True,
    )
    for rank, row in enumerate(scored, start=1):
        row["candidate_rank"] = rank
        row["generation_rank"] = rank
        row["candidate_selected"] = "True" if rank == 1 else "False"
        row["method"] = "common_llm_seed1705_rerank_existing_2step_top20"
    return scored


def metric_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    def rate(key: str) -> float:
        return sum(external_select.truthy(row.get(key)) for row in rows) / max(len(rows), 1)

    success_sims = [
        official.finite_float(row.get("external_source_tanimoto"))
        for row in rows
        if external_select.truthy(row.get("external_official_success"))
    ]
    success_sims = [value for value in success_sims if math.isfinite(value)]
    return {
        "rows": len(rows),
        "success_rate": rate("external_official_success"),
        "strict_success_rate": rate("external_strict_success"),
        "source_similarity_success_rate": rate("external_source_similarity_success"),
        "mean_success_similarity": mean(success_sims) if success_sims else None,
    }


def select_candidate_rows(
    rows: Sequence[Mapping[str, object]], *, budget: int, verifier: bool
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("condition_id", "") or "")].append(dict(row))
    output = []
    for group in groups.values():
        pool = sorted(group, key=external_select.candidate_rank)[: int(budget)]
        if not pool:
            continue
        output.append(max(pool, key=external_select.verifier_key) if verifier else pool[0])
    return output


def summarize(
    reranked: Sequence[Mapping[str, object]],
    original: Sequence[Mapping[str, object]],
    *,
    verifier_k: int,
    candidate_budget: int,
) -> dict[str, object]:
    selections = {
        "original_heuristic_at_1": select_candidate_rows(original, budget=1, verifier=False),
        "llm_at_1": select_candidate_rows(reranked, budget=1, verifier=False),
        f"llm_verifier_at_{verifier_k}": select_candidate_rows(
            reranked, budget=verifier_k, verifier=True
        ),
        f"any_at_{candidate_budget}": select_candidate_rows(
            reranked, budget=candidate_budget, verifier=True
        ),
    }
    result = {}
    for name, rows in selections.items():
        groups = {"all": list(rows)}
        for split in ("ind", "ood"):
            groups[split] = [row for row in rows if str(row.get("external_task_split", "")) == split]
        result[name] = {split: metric_summary(items) for split, items in groups.items()}
    reachable = sum(
        external_select.truthy(row.get("external_official_success"))
        for row in selections[f"any_at_{candidate_budget}"]
    )
    verifier_hits = sum(
        external_select.truthy(row.get("external_official_success"))
        for row in selections[f"llm_verifier_at_{verifier_k}"]
    )
    return {
        "protocol": "common_llm_existing_2step_plan_rerank_v1",
        "candidate_budget": int(candidate_budget),
        "verifier_k": int(verifier_k),
        "target_information_used_for_llm_ranking": False,
        "official_verifier_used_only_after_llm_ranking": True,
        "plan_score": "mean per-action assistant log probability",
        "selections": result,
        "verifier_recovery_of_reachable": verifier_hits / max(reachable, 1),
    }


def render_report(summary: Mapping[str, object]) -> str:
    selections = summary["selections"]
    assert isinstance(selections, Mapping)
    lines = [
        "# Common LLM rerank of existing MuMO 2-step top-20",
        "",
        f"- candidate_budget: `{summary['candidate_budget']}`",
        f"- verifier_k: `{summary['verifier_k']}`",
        "- LLM plan score: mean per-action assistant log probability",
        "- target information used for LLM ranking: `False`",
        "",
        "| Selection | Split | SR | Strict | Sim>=0.4 | Sim(success) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, split_rows in selections.items():
        assert isinstance(split_rows, Mapping)
        for split in ("ind", "ood", "all"):
            row = split_rows[split]
            assert isinstance(row, Mapping)
            similarity = row.get("mean_success_similarity")
            sim_text = "" if similarity is None else f"{float(similarity):.4f}"
            lines.append(
                f"| {name} | {split} | {float(row['success_rate']):.4f} | "
                f"{float(row['strict_success_rate']):.4f} | "
                f"{float(row['source_similarity_success_rate']):.4f} | {sim_text} |"
            )
    lines.extend(
        [
            "",
            f"Verifier recovery of reachable top-20 successes: "
            f"`{float(summary['verifier_recovery_of_reachable']):.4f}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.candidate_budget) != 20:
        raise ValueError("The comparable MuMO experiment fixes candidate_budget=20")
    if not 1 <= int(args.verifier_k) <= int(args.candidate_budget):
        raise ValueError("verifier_k must be within the candidate pool")
    if not args.adapter_dir.joinpath("adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Missing frozen adapter: {args.adapter_dir}")
    try:
        import peft
        import torch
        import transformers
    except ImportError as exc:
        raise SystemExit(f"Missing common-LLM dependency: {exc}") from exc
    if not torch.cuda.is_available():
        raise SystemExit("Common-LLM plan reranking requires CUDA")

    original_rows = read_rows(args.official_detail_csv)
    grouped_original: dict[str, list[dict[str, str]]] = defaultdict(list)
    condition_order = []
    for row in original_rows:
        condition_id = str(row.get("condition_id", "") or "")
        if condition_id not in grouped_original:
            condition_order.append(condition_id)
        grouped_original[condition_id].append(row)
    for rows in grouped_original.values():
        rows.sort(key=external_select.candidate_rank)
        del rows[int(args.candidate_budget) :]
    if args.max_rows > 0:
        condition_order = condition_order[: int(args.max_rows)]
        grouped_original = {key: grouped_original[key] for key in condition_order}
    candidate_rows = [row for key in condition_order for row in grouped_original[key]]
    if len(candidate_rows) != len(condition_order) * int(args.candidate_budget):
        raise ValueError("Existing pool is not a complete fixed n=20 candidate set")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_checkpoint = args.output_dir / "reconstructed_candidate_plans.jsonl"
    if plan_checkpoint.is_file():
        plans = read_plan_checkpoint(plan_checkpoint)
        if len(plans) >= int(0.95 * len(candidate_rows)):
            print(f"[plan-reconstruct] reused {len(plans)} reconstructed candidates", flush=True)
        else:
            print(
                f"[plan-reconstruct] incomplete checkpoint {len(plans)}/{len(candidate_rows)}; rebuilding",
                flush=True,
            )
            plans = reconstruct_candidate_plans(candidate_rows, args.plan_jsonl, plan_checkpoint)
    else:
        plans = reconstruct_candidate_plans(candidate_rows, args.plan_jsonl, plan_checkpoint)
        print(f"[plan-reconstruct] reconstructed {len(plans)}/{len(candidate_rows)} candidates", flush=True)

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
    for index, condition_id in enumerate(condition_order, start=1):
        if condition_id in completed:
            print(f"[plan-rerank] {index}/{len(condition_order)} resume {condition_id}", flush=True)
            continue
        rows = score_condition(
            grouped_original[condition_id],
            plans,
            model=model,
            tokenizer=tokenizer,
            batch_size=int(args.score_batch_size),
            max_length=int(args.max_length),
        )
        append_progress(progress_path, condition_id, rows)
        completed[condition_id] = rows
        print(f"[plan-rerank] {index}/{len(condition_order)} done {condition_id}", flush=True)

    reranked = [row for condition_id in condition_order for row in completed[condition_id]]
    original = [row for condition_id in condition_order for row in grouped_original[condition_id]]
    write_rows(args.output_dir / "reranked_candidates.csv", reranked)
    write_rows(args.output_dir / "llm_top1.csv", select_candidate_rows(reranked, budget=1, verifier=False))
    write_rows(
        args.output_dir / f"verifier_top{int(args.verifier_k)}.csv",
        select_candidate_rows(reranked, budget=int(args.verifier_k), verifier=True),
    )
    summary = summarize(
        reranked,
        original,
        verifier_k=int(args.verifier_k),
        candidate_budget=int(args.candidate_budget),
    )
    summary.update(
        {
            "official_detail_csv": str(args.official_detail_csv),
            "plan_jsonl": str(args.plan_jsonl),
            "adapter_dir": str(args.adapter_dir),
            "conditions": len(condition_order),
            "candidate_rows": len(reranked),
            "reconstructed_candidate_rows": sum(
                external_select.truthy(row.get("llm_plan_reconstructed")) for row in reranked
            ),
            "one_action_candidate_rows": sum(
                int(row.get("llm_plan_action_count", 0) or 0) == 1 for row in reranked
            ),
            "two_action_candidate_rows": sum(
                int(row.get("llm_plan_action_count", 0) or 0) == 2 for row in reranked
            ),
            "missing_action_candidate_rows": sum(
                int(row.get("llm_plan_action_count", 0) or 0) == 0 for row in reranked
            ),
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(render_report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
