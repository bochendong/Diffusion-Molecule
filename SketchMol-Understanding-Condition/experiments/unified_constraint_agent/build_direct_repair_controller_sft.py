#!/usr/bin/env python3
"""Build train-only common-LLM supervision for direct constraint-repair plans."""

from __future__ import annotations

import argparse
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

import build_composed_retrieved_delta_candidates as composed  # noqa: E402
import build_mumo_closed_loop_dev as closed_loop  # noqa: E402
import build_mumo_delta_shard as delta_shard  # noqa: E402
import direct_repair_agent_protocol as repair  # noqa: E402
import export_external_multiproperty_benchmark_rows as export  # noqa: E402
import mumo_parallel_protocol as protocol  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--stable-sft-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-plan-rows-per-task", type=int, default=256)
    parser.add_argument("--replay-per-origin", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1715)
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return protocol.read_jsonl(path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    protocol.write_jsonl(path, rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.0 < float(args.validation_fraction) < 0.5:
        raise ValueError("validation_fraction must be between zero and 0.5")
    models = closed_loop.load_models(args.evidence_root)
    specs = {item.task_id: item for item in export.TASK_SPECS if item.suite == "mumo"}
    by_task: dict[str, list[dict[str, object]]] = defaultdict(list)
    for path in sorted(args.data_dir.glob("train_shard_*.jsonl")):
        for raw in read_jsonl(path):
            if raw.get("_uca_partition") == "fit":
                by_task[str(raw["_uca_task_id"])].append(raw)

    plan_rows = []
    outcomes: Counter[str] = Counter()
    for task_id, rows in sorted(by_task.items()):
        rows.sort(key=lambda row: repair.stable_value(str(row["_uca_source_group"]), int(args.seed)))
        accepted = 0
        for raw in rows:
            if accepted >= int(args.max_plan_rows_per_task):
                break
            spec = specs[task_id]
            normalized = delta_shard.normalized_row(raw, spec)
            effects = composed.observed_property_effects(normalized)
            if set(effects) != set(spec.properties) or min(effects.values()) < 1.0:
                outcomes["non_joint_success_pair"] += 1
                continue
            source_feature = closed_loop.candidate_feature(str(raw["source_smiles"]))
            if source_feature is None:
                outcomes["invalid_source_feature"] += 1
                continue
            _scores, margin_rows = closed_loop.score_candidates_batch(
                source_feature,
                [source_feature],
                properties=spec.properties,
                models=models,
                source_tanimotos=[1.0],
                retrieval_similarities=[1.0],
                frequencies=[0],
            )
            margins = margin_rows[0]
            # A successful paired edit teaches which constraint was the
            # bottleneck, without exposing its target molecule in the prompt.
            order = sorted(
                spec.properties,
                key=lambda prop: (float(margins[prop]), float(effects[prop]), prop),
            )
            messages = repair.prompt_messages(raw, spec.properties, margins, max_steps=int(args.max_steps))
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        repair.plan_payload(order, max_steps=int(args.max_steps)),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )
            serialized_prompt = json.dumps(messages[:-1], sort_keys=True)
            if "target_smiles" in serialized_prompt:
                raise AssertionError("Target leaked into direct-repair controller prompt")
            plan_rows.append(
                {
                    "example_id": f"repair-plan:{raw['_uca_pair_digest']}",
                    "origin": "mumo_repair_plan",
                    "task_mode": "edit",
                    "source_group": str(raw["_uca_source_group"]),
                    "messages": messages,
                }
            )
            accepted += 1
            outcomes["plan_rows"] += 1

    validation_groups = {
        str(row["source_group"])
        for row in plan_rows
        if repair.stable_value(str(row["source_group"]), int(args.seed) + 1) / float(2**64)
        < float(args.validation_fraction)
    }
    plan_train = [row for row in plan_rows if str(row["source_group"]) not in validation_groups]
    plan_validation = [row for row in plan_rows if str(row["source_group"]) in validation_groups]
    if not plan_train or not plan_validation:
        raise ValueError("Direct-repair plan split is empty")

    replay_by_origin: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in read_jsonl(args.stable_sft_dir / "train.jsonl"):
        replay_by_origin[str(row.get("origin", "unknown"))].append(row)
    replay = []
    for origin, rows in sorted(replay_by_origin.items()):
        rows.sort(key=lambda row: repair.stable_value(str(row.get("example_id", "")), int(args.seed)))
        replay.extend(rows[: int(args.replay_per_origin)])
    train = [*plan_train, *replay]
    random.Random(int(args.seed)).shuffle(train)
    plan_validation.sort(key=lambda row: str(row["example_id"]))
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", plan_validation)
    overlap = len(
        {str(row["source_group"]) for row in plan_train}
        & {str(row["source_group"]) for row in plan_validation}
    )
    manifest = {
        "protocol": "direct_constraint_repair_controller_sft_v1",
        "data_role": "fit_only_success_plan_plus_balanced_replay",
        "prompt_target_access": False,
        "evaluation_target_access": False,
        "evaluation_oracle_access": False,
        "official_test_content_access": False,
        "max_steps": int(args.max_steps),
        "plan_train_rows": len(plan_train),
        "plan_validation_rows": len(plan_validation),
        "replay_rows": len(replay),
        "replay_origin_counts": dict(sorted(Counter(str(row.get("origin", "")) for row in replay).items())),
        "source_group_overlap": overlap,
        "outcomes": dict(sorted(outcomes.items())),
        "seed": int(args.seed),
    }
    protocol.write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
