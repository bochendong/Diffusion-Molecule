#!/usr/bin/env python3
"""Build balanced, target-blind P23 SFT and contrastive Stage-1 data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import p23_protocol as protocol


TABLE1_KEYS = {
    "GSK3B:increase",
    "RB:decrease",
    "MW:increase",
    "SA:decrease",
    "HBA:decrease+SA:decrease",
    "QED:increase+SA:decrease",
    "HBA:decrease+LogP:increase",
    "HBA:decrease+MW:decrease",
    "DRD2:decrease+MW:decrease+SA:decrease",
    "HBA:increase+MW:increase+QED:decrease",
}
TABLE1_SIGNATURES = {signature: key for signature, key in protocol.TABLE1_TASK_KEYS.items()}
TABLE1_PROGRAMS = {
    key: [
        {"property": prop, "goal": goal}
        for prop, goal in sorted(signature, key=lambda item: protocol.PROPERTY_ORDER[item[0]])
    ]
    for signature, key in protocol.TABLE1_TASK_KEYS.items()
}


def rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def raw_id(row: Mapping[str, object]) -> str:
    for key in ("example_id", "condition_id", "sample_id", "pair_id"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    payload = json.dumps(dict(row), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def raw_smiles(row: Mapping[str, object], kind: str) -> str:
    return str(
        row.get(f"{kind}_canonical_smiles", "")
        or row.get(f"{kind}_smiles", "")
        or (row.get("policy_target_smiles", "") if kind == "target" else "")
        or ""
    ).strip()


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def stable_rank(row: Mapping[str, object], seed: int) -> int:
    material = f"{seed}:{raw_id(row)}:{raw_smiles(row, 'source')}:{raw_smiles(row, 'target')}"
    return int(hashlib.sha256(material.encode()).hexdigest(), 16)


def similarity_ok(row: Mapping[str, object], minimum: float) -> bool:
    raw = str(row.get("source_target_tanimoto", "") or row.get("source_tanimoto", "")).strip()
    if not raw:
        return True
    try:
        return float(raw) >= minimum
    except ValueError:
        return False


def pair_key(row: Mapping[str, object]) -> tuple[str, str]:
    return raw_smiles(row, "source"), raw_smiles(row, "target")


def pair_eligible_for_projection(
    row: Mapping[str, object], held_sources: set[str], held_targets: set[str],
    minimum_similarity: float, audit: Counter[str],
) -> bool:
    source, target = pair_key(row)
    if not source or not target or source == target:
        audit["missing_or_identity_edit"] += 1
        return False
    if str(row.get("source_valid", "")).strip() and not truthy(row.get("source_valid")):
        audit["invalid_source_flag"] += 1
        return False
    if str(row.get("target_valid", "")).strip() and not truthy(row.get("target_valid")):
        audit["invalid_target_flag"] += 1
        return False
    if not similarity_ok(row, minimum_similarity):
        audit["below_similarity_floor"] += 1
        return False
    if protocol.smiles_hash(source) in held_sources or protocol.smiles_hash(target) in held_targets:
        audit["heldout_smiles_overlap"] += 1
        return False
    return True


def cheap_clause_success(row: Mapping[str, object], prop: str, goal: str) -> bool:
    """Use cached deterministic deltas before invoking the exact paper oracle."""
    if prop in {"SA", "GSK3B", "DRD2"}:
        return True
    raw = str(row.get(f"delta_{prop}", "") or "").strip()
    if not raw:
        return True
    try:
        delta = float(raw)
    except ValueError:
        return False
    return delta > 0.0 if goal == "increase" else delta < 0.0


def projected_row(row: Mapping[str, object], task_key: str) -> dict[str, object]:
    program = TABLE1_PROGRAMS[task_key]
    output = dict(row)
    tasks = [
        {"property": item["property"], "direction": item["goal"]}
        for item in program
    ]
    output["instruction_tasks"] = json.dumps(tasks, separators=(",", ":"))
    output["instruction_task_properties"] = "|".join(str(item["property"]) for item in program)
    output["instruction_task_directions"] = json.dumps(
        {str(item["property"]): str(item["goal"]) for item in program},
        separators=(",", ":"), sort_keys=True,
    )
    output["p23_alignment_source"] = "train_only_source_target_oracle_projection"
    return output


def paper_chemistry():
    repo = Path(__file__).resolve().parents[3]
    scripts = repo / "SketchMol-Unified-3MDiffusion" / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from rdkit import RDLogger  # type: ignore
    from evaluate_moledit_table_metrics import Chemistry  # type: ignore

    # MorganGenerator emits one warning per fingerprint through the legacy
    # wrapper.  Large candidate pools otherwise spend substantial time and
    # log space printing a warning that does not affect the oracle scores.
    RDLogger.DisableLog("rdApp.warning")
    chemistry = Chemistry()
    missing = chemistry.missing_oracles(
        item for program in TABLE1_PROGRAMS.values() for item in program
    )
    if missing:
        raise ValueError(f"missing required Table2 training oracles: {sorted(missing)}")
    return chemistry


def select_oracle_aligned_paper_edits(
    path: Path, rows_per_task: int, pool_per_task: int,
    held_sources: set[str], held_targets: set[str], minimum_similarity: float, seed: int,
) -> tuple[list[dict[str, object]], dict[str, object], set[tuple[str, str]]]:
    if rows_per_task <= 0 or pool_per_task < rows_per_task:
        raise ValueError("oracle-aligned paper pool must be at least the per-task quota")
    heaps: dict[str, list[tuple[int, int, dict[str, str]]]] = {
        key: [] for key in TABLE1_PROGRAMS
    }
    audit: Counter[str] = Counter()
    sequence = 0
    for row in rows(path):
        audit["input"] += 1
        if not pair_eligible_for_projection(
            row, held_sources, held_targets, minimum_similarity, audit
        ):
            continue
        audit["eligible"] += 1
        for task_index, (task_key, program) in enumerate(TABLE1_PROGRAMS.items()):
            if not all(
                cheap_clause_success(row, str(item["property"]), str(item["goal"]))
                for item in program
            ):
                continue
            rank = stable_rank(row, seed + 1009 * (task_index + 1))
            heap = heaps[task_key]
            entry = (-rank, sequence, dict(row))
            sequence += 1
            if len(heap) < pool_per_task:
                heapq.heappush(heap, entry)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, entry)

    chemistry = paper_chemistry()
    score_cache: dict[tuple[str, str], float | None] = {}

    def score(smiles: str, prop: str) -> float | None:
        key = (smiles, prop)
        if key not in score_cache:
            score_cache[key] = chemistry.score(smiles, prop)
        return score_cache[key]

    passing: dict[str, list[dict[str, str]]] = {}
    for task_key, heap in heaps.items():
        accepted: list[dict[str, str]] = []
        for _negative_rank, _sequence, row in sorted(heap, key=lambda item: -item[0]):
            source, target = pair_key(row)
            success = True
            for item in TABLE1_PROGRAMS[task_key]:
                prop, goal = str(item["property"]), str(item["goal"])
                before, after = score(source, prop), score(target, prop)
                if before is None or after is None:
                    success = False
                    break
                if goal == "increase" and not after > before:
                    success = False
                    break
                if goal == "decrease" and not after < before:
                    success = False
                    break
            if success:
                accepted.append(row)
        passing[task_key] = accepted

    selected_raw: dict[str, list[dict[str, str]]] = {key: [] for key in TABLE1_PROGRAMS}
    used: set[tuple[str, str]] = set()
    for task_key in sorted(TABLE1_PROGRAMS, key=lambda key: (len(passing[key]), key)):
        for row in passing[task_key]:
            key = pair_key(row)
            if key in used:
                continue
            used.add(key)
            selected_raw[task_key].append(row)
            if len(selected_raw[task_key]) == rows_per_task:
                break
        if len(selected_raw[task_key]) != rows_per_task:
            raise ValueError(
                f"oracle-aligned Table2 quota unavailable for {task_key}: "
                f"wanted {rows_per_task}, found {len(selected_raw[task_key])} "
                f"from {len(passing[task_key])} verified candidates"
            )

    selected = [
        convert(projected_row(row, task_key), "edit")
        for task_key in TABLE1_PROGRAMS for row in selected_raw[task_key]
    ]
    report: dict[str, object] = {
        "alignment": "train_only_source_target_oracle_projection",
        "requested_rows_per_task": rows_per_task,
        "candidate_pool_per_task": pool_per_task,
        "scan": dict(audit),
        "verified_candidates": {key: len(passing[key]) for key in TABLE1_PROGRAMS},
        "selected_exact_task_rows": {
            key: len(selected_raw[key]) for key in TABLE1_PROGRAMS
        },
        "unique_scored_smiles_properties": len(score_cache),
    }
    return selected, report, used


def quick_program(row: Mapping[str, object], mode: str) -> list[dict[str, object]]:
    return protocol.condition_program(row, mode)


def table1_bucket(program: Sequence[Mapping[str, object]]) -> str:
    signature = frozenset(
        (str(item["property"]), str(item["goal"]))
        for item in program if isinstance(item["goal"], str)
    )
    matches = [
        (len(required), key) for required, key in TABLE1_SIGNATURES.items()
        if required <= signature
    ]
    return max(matches, default=(0, ""), key=lambda item: (item[0], item[1]))[1]


def quick_eligible(
    row: Mapping[str, object], mode: str, held_sources: set[str], held_targets: set[str],
    minimum_similarity: float, audit: Counter[str],
) -> tuple[str, list[dict[str, object]]] | None:
    source, target = raw_smiles(row, "source"), raw_smiles(row, "target")
    if mode == "edit" and (not source or not target or source == target):
        audit["missing_or_identity_edit"] += 1
        return None
    if mode == "de_novo" and not target:
        audit["missing_denovo_target"] += 1
        return None
    if str(row.get("source_valid", "")).strip() and not truthy(row.get("source_valid")):
        audit["invalid_source_flag"] += 1
        return None
    if str(row.get("target_valid", "")).strip() and not truthy(row.get("target_valid")):
        audit["invalid_target_flag"] += 1
        return None
    if mode == "edit" and not similarity_ok(row, minimum_similarity):
        audit["below_similarity_floor"] += 1
        return None
    if protocol.smiles_hash(source) in held_sources or protocol.smiles_hash(target) in held_targets:
        audit["heldout_smiles_overlap"] += 1
        return None
    try:
        program = quick_program(row, mode)
    except ValueError:
        audit["missing_or_unsupported_explicit_program"] += 1
        return None
    if mode == "edit":
        paper_bucket = table1_bucket(program)
        key = f"table1::{paper_bucket}" if paper_bucket else f"broad::{protocol.task_key(program)}"
    else:
        key = f"{len(program)}p"
    return key, program


def load_heldout(paths: Sequence[Path]) -> tuple[set[str], set[str], dict[str, int]]:
    sources: set[str] = set()
    targets: set[str] = set()
    counts: dict[str, int] = {}
    for path in paths:
        count = 0
        for row in rows(path):
            source = protocol.canonical_smiles(raw_smiles(row, "source"))
            target = protocol.canonical_smiles(raw_smiles(row, "target"))
            if source:
                sources.add(protocol.smiles_hash(source))
            if target:
                targets.add(protocol.smiles_hash(target))
            count += 1
        counts[str(path)] = count
    return sources, targets, counts


def count_strata(
    path: Path, mode: str, held_sources: set[str], held_targets: set[str],
    minimum_similarity: float,
    excluded_pairs: set[tuple[str, str]] | None = None,
) -> tuple[Counter[str], Counter[str]]:
    counts: Counter[str] = Counter()
    audit: Counter[str] = Counter()
    for row in rows(path):
        if excluded_pairs and pair_key(row) in excluded_pairs:
            audit["excluded_projected_pair"] += 1
            audit["input"] += 1
            continue
        item = quick_eligible(row, mode, held_sources, held_targets, minimum_similarity, audit)
        if item is not None:
            counts[item[0]] += 1
            audit["eligible"] += 1
        audit["input"] += 1
    return counts, audit


def distribute(
    quotas: dict[str, int], keys: Sequence[str], counts: Mapping[str, int], budget: int,
) -> int:
    remaining = budget
    while remaining:
        progressed = False
        for key in keys:
            if quotas[key] < int(counts[key]):
                quotas[key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            break
    return remaining


def allocate(
    counts: Mapping[str, int], total: int, minimum_rows: int, mode: str,
    table1_fraction: float,
) -> dict[str, int]:
    keys = [key for key, count in counts.items() if count >= minimum_rows]
    if mode == "edit":
        paper_keys = sorted(key for key in counts if key.startswith("table1::"))
        keys = sorted(set(keys) | set(paper_keys), key=lambda key: (not key.startswith("table1::"), key))
    else:
        keys = sorted(keys, key=lambda key: int(key[:-1]))
    if not keys:
        raise ValueError(f"no eligible {mode} strata")
    quotas = {key: 0 for key in keys}
    capacity = sum(int(counts[key]) for key in keys)
    remaining = min(total, capacity)
    if mode == "edit":
        paper_keys = [key for key in keys if key.startswith("table1::")]
        reserve = min(remaining, round(total * table1_fraction))
        unused = distribute(quotas, paper_keys, counts, reserve) if paper_keys else reserve
        remaining -= reserve - unused
    distribute(quotas, keys, counts, remaining)
    if sum(quotas.values()) < total:
        raise ValueError(f"requested {total} {mode} rows but only {sum(quotas.values())} pass filters")
    return {key: value for key, value in quotas.items() if value}


def select(
    path: Path, mode: str, quotas: Mapping[str, int], held_sources: set[str],
    held_targets: set[str], minimum_similarity: float, seed: int,
    excluded_pairs: set[tuple[str, str]] | None = None,
) -> list[dict[str, object]]:
    heaps: dict[str, list[tuple[int, int, dict[str, str]]]] = {key: [] for key in quotas}
    sequence = 0
    ignored: Counter[str] = Counter()
    for row in rows(path):
        if excluded_pairs and pair_key(row) in excluded_pairs:
            ignored["excluded_projected_pair"] += 1
            continue
        item = quick_eligible(row, mode, held_sources, held_targets, minimum_similarity, ignored)
        if item is None or item[0] not in quotas:
            continue
        key = item[0]
        rank = stable_rank(row, seed)
        heap = heaps[key]
        entry = (-rank, sequence, dict(row))
        sequence += 1
        if len(heap) < quotas[key]:
            heapq.heappush(heap, entry)
        elif rank < -heap[0][0]:
            heapq.heapreplace(heap, entry)
    selected: list[dict[str, object]] = []
    for key, heap in heaps.items():
        if len(heap) != quotas[key]:
            raise ValueError(f"selection underflow for {mode}/{key}: {len(heap)} != {quotas[key]}")
        selected.extend(convert(entry[2], mode) for entry in sorted(heap, key=lambda item: -item[0]))
    return sorted(selected, key=lambda row: str(row["selection_rank"]))


def convert(row: Mapping[str, object], declared_mode: str) -> dict[str, object]:
    messages, source, mode = protocol.build_prompt(row)
    if mode != declared_mode:
        raise ValueError(f"mode mismatch: expected={declared_mode} actual={mode}")
    target = protocol.canonical_smiles(raw_smiles(row, "target"))
    if not target or (mode == "edit" and target == source):
        raise ValueError("missing, invalid, or identity target")
    program = protocol.condition_program(row, mode)
    key = protocol.task_key(program)
    return {
        "example_id": f"p23:{mode}:{raw_id(row)}",
        "task_mode": mode,
        "task_key": key,
        "condition_program": program,
        "condition_hash": protocol.condition_hash_from_program(program),
        "source_hash": protocol.smiles_hash(source),
        "target_hash": protocol.smiles_hash(target),
        "source_smiles": source,
        "target_smiles": target,
        "source_tanimoto": str(row.get("source_target_tanimoto", "") or row.get("source_tanimoto", "")),
        "messages": [*messages, {"role": "assistant", "content": protocol.response(target, mode)}],
        "selection_rank": f"{stable_rank(row, 2323):064x}",
    }


def write_jsonl(path: Path, items: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(dict(item), sort_keys=True) + "\n")


def interleave(left: Sequence[dict[str, object]], right: Sequence[dict[str, object]], seed: int) -> list[dict[str, object]]:
    if len(left) != len(right):
        raise ValueError("P23 requires exactly 1:1 de-novo/edit mode weighting")
    left_order, right_order = list(left), list(right)
    random.Random(seed).shuffle(left_order)
    random.Random(seed + 1).shuffle(right_order)
    mixed = [item for pair in zip(left_order, right_order) for item in pair]
    return mixed


def program_pairs(row: Mapping[str, object]) -> frozenset[tuple[str, str]]:
    return frozenset(
        (str(item["property"]), str(item["goal"]))
        for item in row["condition_program"]
        if isinstance(item["goal"], str)
    )


def choose_donor(row: Mapping[str, object], pool: Sequence[Mapping[str, object]]) -> Mapping[str, object] | None:
    candidates = [
        donor for donor in pool
        if donor["task_key"] != row["task_key"] and donor["target_hash"] != row["target_hash"]
        and donor["target_smiles"] != row.get("source_smiles", "")
    ]
    if not candidates:
        return None
    index = int(hashlib.sha256(str(row["example_id"]).encode()).hexdigest(), 16) % len(candidates)
    return candidates[index]


def negative_instance(
    row: Mapping[str, object], kind: str, rejected: str, margin: float, weight: float,
    ce_weight: float,
) -> dict[str, object]:
    return {
        "pair_id": f"{row['example_id']}:{kind}", "example_id": row["example_id"],
        "task_mode": row["task_mode"], "task_key": row["task_key"],
        "condition_hash": row["condition_hash"], "source_hash": row["source_hash"],
        "target_hash": row["target_hash"], "messages": row["messages"],
        "negative_type": kind, "rejected_assistant": rejected,
        "margin": margin, "negative_weight": weight, "chosen_ce_weight": ce_weight,
    }


def build_negatives(items: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    by_mode = {
        mode: [row for row in items if row["task_mode"] == mode]
        for mode in ("de_novo", "edit")
    }
    by_signature: dict[tuple[str, frozenset[tuple[str, str]]], list[dict[str, object]]] = defaultdict(list)
    for row in items:
        by_signature[(str(row["task_mode"]), program_pairs(row))].append(row)
    output: list[dict[str, object]] = []
    for row in items:
        mode, target = str(row["task_mode"]), str(row["target_smiles"])
        candidates: list[tuple[str, str, float, float]] = []
        if mode == "edit":
            candidates.append(("source_copy", protocol.response(str(row["source_smiles"]), mode), 0.10, 0.10))
        plan = "BUILD" if mode == "de_novo" else "MODIFY"
        invalid = json.dumps({"plan": plan, "smiles": target + "("}, separators=(",", ":"))
        candidates.append(("invalid_corruption", invalid, 0.15, 0.20))
        signature = program_pairs(row)
        if signature:
            opposite = frozenset(
                (prop, "decrease" if goal == "increase" else "increase" if goal == "decrease" else goal)
                for prop, goal in signature
            )
            opposite_pool = by_signature.get((mode, opposite), [])
            opposite_donor = choose_donor(row, opposite_pool) if opposite_pool else None
            if opposite_donor is not None:
                candidates.append(("opposite_program_target", protocol.response(str(opposite_donor["target_smiles"]), mode), 0.16, 0.18))
            if len(signature) > 1:
                subset_pools = [
                    pool for (candidate_mode, candidate_signature), pool in by_signature.items()
                    if candidate_mode == mode and candidate_signature and candidate_signature < signature
                ]
                partial_pool = [donor for pool in subset_pools for donor in pool]
                partial_donor = choose_donor(row, partial_pool) if partial_pool else None
                if partial_donor is not None:
                    candidates.append(("partial_program_target", protocol.response(str(partial_donor["target_smiles"]), mode), 0.14, 0.16))
        # Add the generic mismatch last so a duplicate response keeps its more
        # informative opposite/partial-program label and stronger margin.
        mismatch = choose_donor(row, by_mode[mode])
        if mismatch is not None:
            candidates.append(("condition_mismatch", protocol.response(str(mismatch["target_smiles"]), mode), 0.12, 0.14))
        unique: dict[str, tuple[str, str, float, float]] = {}
        for candidate in candidates:
            unique.setdefault(candidate[1], candidate)
        candidates = list(unique.values())
        ce_weight = 1.0 / len(candidates)
        output.extend(negative_instance(row, *candidate, ce_weight) for candidate in candidates)
    return output


def overlap_audit(items: Sequence[Mapping[str, object]], held_sources: set[str], held_targets: set[str]) -> dict[str, int]:
    return {
        "source": sum(bool(row["source_hash"]) and row["source_hash"] in held_sources for row in items),
        "target": sum(row["target_hash"] in held_targets for row in items),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--denovo-csv", required=True, type=Path)
    parser.add_argument("--edit-csv", required=True, type=Path)
    parser.add_argument("--heldout-csv", action="append", default=[], type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rows-per-mode", type=int, default=12000)
    parser.add_argument("--minimum-edit-task-rows", type=int, default=32)
    parser.add_argument("--minimum-source-similarity", type=float, default=0.65)
    parser.add_argument("--table1-edit-fraction", type=float, default=0.60)
    parser.add_argument("--oracle-aligned-paper-edits", action="store_true")
    parser.add_argument("--paper-rows-per-task", type=int, default=720)
    parser.add_argument("--paper-candidate-pool-per-task", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2323)
    args = parser.parse_args(argv)

    held_sources, held_targets, heldout_counts = load_heldout(args.heldout_csv)
    denovo_counts, denovo_scan = count_strata(
        args.denovo_csv, "de_novo", held_sources, held_targets, args.minimum_source_similarity
    )
    edit_counts, edit_scan = count_strata(
        args.edit_csv, "edit", held_sources, held_targets, args.minimum_source_similarity
    )
    if not 0.0 <= args.table1_edit_fraction <= 1.0:
        raise ValueError("--table1-edit-fraction must be in [0, 1]")
    denovo_quotas = allocate(denovo_counts, args.rows_per_mode, 1, "de_novo", 0.0)
    denovo = select(
        args.denovo_csv, "de_novo", denovo_quotas, held_sources, held_targets,
        args.minimum_source_similarity, args.seed,
    )
    alignment_report: dict[str, object] | None = None
    if args.oracle_aligned_paper_edits:
        paper, alignment_report, projected_pairs = select_oracle_aligned_paper_edits(
            args.edit_csv, args.paper_rows_per_task, args.paper_candidate_pool_per_task,
            held_sources, held_targets, args.minimum_source_similarity, args.seed + 1,
        )
        broad_budget = args.rows_per_mode - len(paper)
        if broad_budget < 0:
            raise ValueError(
                f"paper quota {len(paper)} exceeds edit budget {args.rows_per_mode}"
            )
        # Recount after reserving the exact paper-task pairs.  Allocating from
        # the pre-reservation counts can overbook a rare broad stratum even
        # when total broad capacity is ample.
        remaining_edit_counts, remaining_edit_scan = count_strata(
            args.edit_csv, "edit", held_sources, held_targets,
            args.minimum_source_similarity, projected_pairs,
        )
        alignment_report["remaining_edit_scan"] = dict(remaining_edit_scan)
        broad_counts = {
            key: count for key, count in remaining_edit_counts.items()
            if key.startswith("broad::") and count >= args.minimum_edit_task_rows
        }
        broad_quotas = {key: 0 for key in sorted(broad_counts)}
        if broad_budget and (
            not broad_quotas
            or distribute(broad_quotas, list(broad_quotas), broad_counts, broad_budget)
        ):
            raise ValueError(f"insufficient broad edit capacity for {broad_budget} rows")
        broad = select(
            args.edit_csv, "edit", broad_quotas, held_sources, held_targets,
            args.minimum_source_similarity, args.seed + 2, projected_pairs,
        ) if broad_budget else []
        edit = paper + broad
        edit_quotas = {
            **{f"table1::{key}": args.paper_rows_per_task for key in TABLE1_PROGRAMS},
            **{key: value for key, value in broad_quotas.items() if value},
        }
    else:
        edit_quotas = allocate(
            edit_counts, args.rows_per_mode, args.minimum_edit_task_rows,
            "edit", args.table1_edit_fraction,
        )
        edit = select(
            args.edit_csv, "edit", edit_quotas, held_sources, held_targets,
            args.minimum_source_similarity, args.seed + 1,
        )
    mixed = interleave(denovo, edit, args.seed)
    negatives = build_negatives(mixed)
    overlap = overlap_audit(mixed, held_sources, held_targets)
    if overlap["source"] or overlap["target"]:
        raise SystemExit(f"held-out leakage audit failed: {overlap}")

    write_jsonl(args.output_dir / "train.sft.jsonl", mixed)
    write_jsonl(args.output_dir / "train.contrastive.jsonl", negatives)
    manifest = {
        "protocol": protocol.PROTOCOL, "seed": args.seed,
        "inputs": {"denovo": str(args.denovo_csv), "edit": str(args.edit_csv)},
        "heldout_csvs": heldout_counts,
        "filters": {
            "minimum_source_similarity": args.minimum_source_similarity,
            "minimum_edit_task_rows": args.minimum_edit_task_rows,
            "table1_edit_fraction": args.table1_edit_fraction,
            "edit_condition_source": "instruction_tasks or instruction_task_* only",
            "target_derived_condition_fallback": False,
        },
        "scan": {"de_novo": dict(denovo_scan), "edit": dict(edit_scan)},
        "selected": {
            "total": len(mixed), "de_novo": len(denovo), "edit": len(edit),
            "denovo_strata": denovo_quotas, "edit_task_quotas": edit_quotas,
            "table1_task_rows": dict(sorted(Counter(row["task_key"] for row in edit if row["task_key"] in TABLE1_KEYS).items())),
        },
        "contrastive": {
            "pair_instances": len(negatives),
            "negative_types": dict(sorted(Counter(row["negative_type"] for row in negatives).items())),
            "chosen_ce_total_weight_by_mode": {
                mode: sum(float(row["chosen_ce_weight"]) for row in negatives if row["task_mode"] == mode)
                for mode in ("de_novo", "edit")
            },
        },
        "paper_edit_alignment": alignment_report,
        "heldout_overlap": overlap, "mode_weighting": "1:1 unique rows",
        "prompt_target_fields": False, "mode_router": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
