#!/usr/bin/env python3
"""D2: source-ball short rectified flow in frozen graph-AE latent.

Train only on MCS-aligned edit pairs with Tanimoto >= 0.65. Keep the source
atom canvas. Decode once per attempt. Table1 n=20, no ranking, no DSL.
Condition vectors are source-only: no target SMILES and no target properties.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
WORKTREE_LATENT = PROJECT_DIR / "experiments" / "unified_latent_flow"
for path in (WORKTREE_LATENT, SCRIPT_DIR, PROJECT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import categorical_graph_latent_flow as cflow  # noqa: E402

graph = cflow.graph
TARGET_FIELDS = {
    "target_smiles",
    "target_canonical_smiles",
    "target_scaffold_smiles",
    "source_target_tanimoto",
}


@dataclass
class Table1Condition:
    row: dict[str, str]
    source_smiles: str
    source: object
    condition: np.ndarray
    task: str
    condition_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-csv", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--validation-csv", required=True, type=Path)
    parser.add_argument("--representation-checkpoint", required=True, type=Path)
    parser.add_argument("--representation-summary", required=True, type=Path)
    parser.add_argument("--d2-protocol-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    d2 = json.loads(args.d2_protocol_manifest.read_text(encoding="utf-8"))
    started = time.perf_counter()
    device = cflow.resolve_device(str(args.device))
    cflow.seed_everything(int(d2["seed"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    representation, config, representation_summary = cflow.load_representation(
        args.representation_checkpoint, args.representation_summary, device
    )
    max_atoms = int(config["max_atoms"])
    fingerprint_bits = int(config.get("fingerprint_bits") or 512)
    condition_dim = 64
    allowed_counts = {int(value) for value in d2["property_counts"]}
    min_tanimoto = float(d2["min_source_tanimoto"])
    train_args = SimpleNamespace(
        learning_rate=3e-4,
        weight_decay=1e-4,
        epochs=int(d2["epochs"]),
        batch_size=int(d2["batch_size"]),
        seed=int(d2["seed"]),
        source_noise=float(d2["source_noise"]),
        size_adaptive=bool(d2["size_adaptive"]),
        endpoint_weight=0.10,
        count_loss_weight=0.30,
        occupancy_loss_weight=0.20,
        grad_clip=1.0,
    )

    eval_rows = cflow.read_rows(args.eval_csv)
    forbidden_sources = {
        smiles
        for smiles in (graph.canonical_smiles(row.get("source_smiles", "")) for row in eval_rows)
        if smiles
    }
    train_rows = [
        row
        for row in cflow.read_rows(args.train_csv)
        if keep_train_row(row, min_tanimoto, forbidden_sources)
    ]
    print(
        json.dumps(
            {
                "stage": "build_pairs",
                "prefiltered_train_rows": len(train_rows),
                "forbidden_eval_sources": len(forbidden_sources),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    pair_started = time.perf_counter()
    raw_pairs, pair_counts = cflow.build_pairs(
        train_rows,
        max_atoms=max_atoms,
        fingerprint_bits=fingerprint_bits,
        condition_dim=condition_dim,
        allowed_counts=allowed_counts,
        timeout=1,
        min_common_fraction=0.45,
        limit=int(d2["train_limit"]),
        seed=int(d2["seed"]),
        forbidden_sources=forbidden_sources,
    )
    train_pairs = []
    for pair in raw_pairs:
        if not in_source_ball(pair, min_tanimoto):
            continue
        specs = cflow.task_specs(pair.row)
        pair.condition = cflow.condition_vector(source_only_row(pair.row, specs), condition_dim)
        train_pairs.append(pair)
    pair_stats = {
        "stage": "pairs",
        "raw_aligned": len(raw_pairs),
        "source_ball": len(train_pairs),
        "min_source_tanimoto": min_tanimoto,
        "pair_filter_counts": pair_counts,
        "elapsed_sec": round(time.perf_counter() - pair_started, 1),
    }
    print(json.dumps(pair_stats, sort_keys=True), flush=True)
    if len(train_pairs) < 32:
        raise ValueError(f"Need at least 32 source-ball pairs, found {len(train_pairs)}")

    flow = cflow.EquivariantGraphVelocity(
        node_dim=int(config["node_dim"]),
        edge_dim=int(config["edge_dim"]),
        condition_dim=condition_dim,
        hidden_dim=256,
        max_atoms=max_atoms,
    ).to(device)
    print(json.dumps({"stage": "train_start", "epochs": train_args.epochs}, sort_keys=True), flush=True)
    train_started = time.perf_counter()
    history = cflow.train_flow(flow, representation, train_pairs, train_args, device)
    train_sec = time.perf_counter() - train_started
    print(
        json.dumps(
            {
                "stage": "train_done",
                "elapsed_sec": round(train_sec, 1),
                "sec_per_epoch": round(train_sec / max(1, int(d2["epochs"])), 1),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    checkpoint_path = args.output_dir / "source_ball_latent_flow.pt"
    torch.save(
        {
            "model_state": flow.state_dict(),
            "model_config": {
                "node_dim": int(config["node_dim"]),
                "edge_dim": int(config["edge_dim"]),
                "condition_dim": condition_dim,
                "hidden_dim": 256,
                "max_atoms": max_atoms,
            },
            "protocol": d2["protocol"],
            "pair_stats": pair_stats,
            "history": history,
        },
        checkpoint_path,
    )

    conditions = load_table1_flow_conditions(
        args.eval_csv,
        limit=int(args.eval_limit),
        max_atoms=max_atoms,
        fingerprint_bits=fingerprint_bits,
        condition_dim=condition_dim,
    )
    attempts = int(d2["exact_raw_attempts_per_condition"])
    rows: list[dict[str, object]] = []
    skipped = 0
    sample_started = time.perf_counter()
    for index, condition in enumerate(conditions):
        try:
            generated = cflow.sample_from_source(
                flow,
                representation,
                condition.source,
                np.asarray(condition.condition),
                attempts=attempts,
                batch_size=5,
                flow_steps=int(d2["flow_steps"]),
                source_noise=float(d2["source_noise"]),
                size_adaptive=bool(d2["size_adaptive"]),
                device=device,
                seed=int(d2["seed"]) * 100000 + index,
            )
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "stage": "sample_failed",
                        "condition_id": condition.condition_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            generated = [(None, 0)] * attempts
            skipped += 1
        if len(generated) != attempts:
            generated = list(generated) + [(None, 0)] * (attempts - len(generated))
            generated = generated[:attempts]
            skipped += 1
        for attempt, (smiles, _atom_count) in enumerate(generated, start=1):
            rows.append(
                {
                    "condition_id": condition.condition_id,
                    "task": condition.task,
                    "source_smiles": condition.source_smiles,
                    "generated_smiles": graph.canonical_smiles(smiles or "") or "",
                    "sample_index": attempt,
                    "candidate_index": attempt,
                    "method": d2["protocol"],
                    "family": "source_ball_latent_flow",
                    "op": "graph_ae_rectified_flow",
                }
            )
        if (index + 1) % 20 == 0 or index + 1 == len(conditions):
            elapsed = time.perf_counter() - sample_started
            done = index + 1
            sec_per = elapsed / done
            print(
                json.dumps(
                    {
                        "stage": "sampled",
                        "done": done,
                        "total": len(conditions),
                        "elapsed_sec": round(elapsed, 1),
                        "sec_per_condition": round(sec_per, 3),
                        "eta_sec": round(sec_per * (len(conditions) - done), 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    candidate_path = args.output_dir / "d2_source_ball_table1_n20_candidates.csv"
    write_rows(candidate_path, rows)
    sampling = {
        "protocol": d2["protocol"],
        "device": str(device),
        "eval_csv": str(args.eval_csv),
        "loaded_conditions": len(conditions),
        "candidate_rows": len(rows),
        "attempts_per_condition": attempts,
        "skipped_count": skipped,
        "candidate_csv": str(candidate_path),
        "checkpoint": str(checkpoint_path),
        "molecular_candidate_ranking": False,
        "task_router": False,
        "oracle_in_environment": False,
        "family_mixer": False,
        "size_adaptive": bool(d2["size_adaptive"]),
        "flow_steps": int(d2["flow_steps"]),
        "min_source_tanimoto": min_tanimoto,
        "train_pairs_raw": len(raw_pairs),
        "train_pairs_source_ball": len(train_pairs),
        "train_sec": round(train_sec, 1),
        "sample_sec": round(time.perf_counter() - sample_started, 1),
        "elapsed_sec": round(time.perf_counter() - started, 1),
        "representation_protocol": representation_summary.get("protocol"),
        "pair_stats": pair_stats,
        "history": history,
        "condition_encoding": "source_only_property_program",
    }
    (args.output_dir / "sampling_summary.json").write_text(
        json.dumps(sampling, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(sampling, indent=2, sort_keys=True))
    return 0


def keep_train_row(
    row: Mapping[str, str],
    min_tanimoto: float,
    forbidden_sources: set[str],
) -> bool:
    source = graph.canonical_smiles(str(row.get("source_smiles", "") or ""))
    target = graph.canonical_smiles(str(row.get("target_smiles", "") or ""))
    if not source or not target or source == target:
        return False
    if source in forbidden_sources:
        return False
    raw = str(row.get("source_tanimoto", "") or "").strip()
    if raw:
        try:
            if float(raw) < float(min_tanimoto):
                return False
        except ValueError:
            return False
    return True


def in_source_ball(pair: object, threshold: float) -> bool:
    similarity = graph.morgan_tanimoto(pair.source_smiles, pair.target_smiles)
    return similarity is not None and float(similarity) >= float(threshold)


def table1_task_key(specs: Sequence[tuple[str, int]]) -> str:
    direction_name = {1: "increase", -1: "decrease"}
    if not specs or any(direction not in direction_name for _prop, direction in specs):
        return "unknown"
    return "+".join(
        f"{prop}:{direction_name[direction]}" for prop, direction in sorted(specs)
    )


def source_only_row(raw: Mapping[str, str], specs: Sequence[tuple[str, int]]) -> dict[str, str]:
    condition_id = str(raw.get("example_id", "") or raw.get("condition_id", "")).strip()
    source = str(raw.get("source_smiles", "") or "").strip()
    instruction = str(raw.get("instruction", "") or "").strip()
    direction_name = {1: "increase", -1: "decrease"}
    tasks = [
        {"property": prop, "direction": direction_name[direction]}
        for prop, direction in specs
    ]
    row = {
        "condition_id": condition_id,
        "sample_id": condition_id,
        "task_type": "edit_generation",
        "source_smiles": source,
        "instruction": instruction,
        "condition_properties": ",".join(prop for prop, _direction in specs),
        "instruction_tasks": json.dumps(
            tasks, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ),
    }
    for prop, direction in specs:
        row[f"{prop}_active"] = "True"
        row[f"{prop}_direction"] = direction_name[direction]
    if any(key in row for key in TARGET_FIELDS) or any(
        key.startswith("target_") or key.startswith("delta_") for key in row
    ):
        raise AssertionError("source-only projection retained a target field")
    return row


def load_table1_flow_conditions(
    path: Path,
    *,
    limit: int,
    max_atoms: int,
    fingerprint_bits: int,
    condition_dim: int,
) -> list[Table1Condition]:
    out: list[Table1Condition] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            specs = cflow.task_specs(raw)
            task = table1_task_key(specs)
            if task == "unknown":
                continue
            row = source_only_row(raw, specs)
            source = graph.canonical_smiles(row["source_smiles"])
            if not source or not row["condition_id"]:
                continue
            source_graph = graph.molecule_example(
                source, max_atoms=int(max_atoms), fingerprint_bits=int(fingerprint_bits)
            )
            if source_graph is None:
                continue
            out.append(
                Table1Condition(
                    row=row,
                    source_smiles=source,
                    source=source_graph,
                    condition=cflow.condition_vector(row, int(condition_dim)),
                    task=task,
                    condition_id=row["condition_id"],
                )
            )
            if limit > 0 and len(out) >= int(limit):
                break
    return out


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
