#!/usr/bin/env python3
"""D0b: frozen B41 + valid-terminal STOP, Table1 n=20, no ranking.

Same eval rows and Acc@0.65 any@20 contract as C5. No DSL. Decision stays in
the latent graph-event jump process. Molecule materialization is STOP support,
not a selector.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
REPO_DIR = PROJECT_DIR.parent
LATENT_DIR = PROJECT_DIR / "experiments" / "unified_latent_flow"
C_DIR = PROJECT_DIR / "experiments" / "unified_action_categorical"
for path in (LATENT_DIR, C_DIR, PROJECT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import valid_terminal_molecule_latent_jump as valid_terminal  # noqa: E402
import viability_preserving_interacting_particle_transport as b41  # noqa: E402
import table1_energy_tilted_latent_transfer as b29  # noqa: E402

b40 = b41.b40
b39 = b41.b39
b37 = b41.b37
b36 = b41.b36
base = b41.base
delta = b41.delta
graph = b41.graph
hierarchical = b41.hierarchical
unified = b41.unified


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-csv", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--validation-csv", required=True, type=Path)
    parser.add_argument("--representation-checkpoint", required=True, type=Path)
    parser.add_argument("--representation-summary", required=True, type=Path)
    parser.add_argument("--b22-checkpoint", required=True, type=Path)
    parser.add_argument("--b22-summary", required=True, type=Path)
    parser.add_argument("--b36-summary", required=True, type=Path)
    parser.add_argument("--b37-summary", required=True, type=Path)
    parser.add_argument("--b38-checkpoint", required=True, type=Path)
    parser.add_argument("--b38-summary", required=True, type=Path)
    parser.add_argument("--b39-checkpoint", required=True, type=Path)
    parser.add_argument("--b39-summary", required=True, type=Path)
    parser.add_argument("--b39-evaluated-candidates", required=True, type=Path)
    parser.add_argument("--b40-summary", required=True, type=Path)
    parser.add_argument("--b40-evaluated-candidates", required=True, type=Path)
    parser.add_argument("--b41-checkpoint", required=True, type=Path)
    parser.add_argument("--b41-summary", required=True, type=Path)
    parser.add_argument("--b41-evaluated-candidates", required=True, type=Path)
    parser.add_argument("--b41-protocol-manifest", required=True, type=Path)
    parser.add_argument("--d0-protocol-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    d0 = json.loads(args.d0_protocol_manifest.read_text(encoding="utf-8"))
    b41_prereg = b41.read_preregistration(args.b41_protocol_manifest)
    device = base.resolve_device(str(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    (
        _b22_summary,
        b22_checkpoint,
        _b36_summary,
        _b37_summary,
        _b39_checkpoint,
        _b40_summary,
    ) = b41.check_locked_inputs(args, b41_prereg)
    b41_checkpoint = torch.load(args.b41_checkpoint, map_location="cpu", weights_only=False)

    selected_pairs = reconstruct_support_pairs(args, b41_prereg, b22_checkpoint)
    fit_pairs, _development_pairs, _split = b37.strict_source_group_split(
        selected_pairs,
        seed=int(b41_prereg["development_split_seed"]),
        development_source_limit=int(b41_prereg["development_source_limit"]),
    )
    representation, representation_config, _summary = base.load_representation(
        args.representation_checkpoint, args.representation_summary, device
    )
    vocabulary = b37.checkpoint_vocabulary(b22_checkpoint)
    support = b40.build_support(fit_pairs, vocabulary)
    support_tensors = b40._device_support(support, device)
    node_action_count, edge_action_count = delta.action_space_sizes(vocabulary)
    model = b39.LatentCardinalityGraphJumpBridge(
        node_dim=int(representation_config["node_dim"]),
        edge_dim=int(representation_config["edge_dim"]),
        condition_dim=int(b41_prereg["condition_dim"]),
        transport_dim=int(b41_prereg["transport_dim"]),
        hidden_dim=int(b41_prereg["hidden_dim"]),
        max_atoms=int(representation_config["max_atoms"]),
        max_jumps=int(b41_prereg["max_jumps"]),
        property_count=len(unified.PROPERTY_COLUMNS),
        node_state_count=node_action_count,
        edge_state_count=edge_action_count,
        message_layers=int(b41_prereg["message_layers"]),
    ).to(device)
    model.load_state_dict(dict(b41_checkpoint["model_state"]), strict=True)
    model.eval().requires_grad_(False)

    conditions = load_table1_conditions(
        args.eval_csv,
        limit=int(args.eval_limit),
        condition_dim=int(b41_prereg["condition_dim"]),
        graph_fingerprint_bits=int(b41_prereg["fingerprint_bits"]),
        max_atoms=int(representation_config["max_atoms"]),
    )
    exact_support = valid_terminal.ExactMoleculeStopSupport(vocabulary)
    original_support = b41.viability_event_mask
    b41.viability_event_mask = exact_support
    rows: list[dict[str, object]] = []
    skipped = 0
    try:
        for index, condition in enumerate(conditions):
            generated = b41.sample_from_source(
                model,
                representation,
                vocabulary,
                support,
                support_tensors,
                condition.source,
                np.asarray(condition.condition),
                b41_prereg,
                device,
                int(d0["seed"]) * 100000 + index,
            )
            if len(generated) != int(d0["exact_raw_attempts_per_condition"]):
                skipped += 1
                continue
            for attempt, candidate in enumerate(generated, start=1):
                rows.append(
                    {
                        "condition_id": condition.condition_id,
                        "task": condition.task,
                        "source_smiles": condition.source_smiles,
                        "generated_smiles": candidate.get("generated_smiles", ""),
                        "sample_index": attempt,
                        "candidate_index": attempt,
                        "method": d0["protocol"],
                        "family": "b41_graph_event",
                        "op": "latent_graph_jump",
                    }
                )
            if (index + 1) % 20 == 0 or index + 1 == len(conditions):
                print(
                    json.dumps(
                        {
                            "stage": "sampled",
                            "done": index + 1,
                            "total": len(conditions),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        b41.viability_event_mask = original_support

    candidate_path = args.output_dir / "d0_b41_table1_n20_candidates.csv"
    write_rows(candidate_path, rows)
    sampling = {
        "protocol": d0["protocol"],
        "device": str(device),
        "eval_csv": str(args.eval_csv),
        "loaded_conditions": len(conditions),
        "candidate_rows": len(rows),
        "attempts_per_condition": int(d0["exact_raw_attempts_per_condition"]),
        "skipped_count": skipped,
        "candidate_csv": str(candidate_path),
        "molecular_candidate_ranking": False,
        "task_router": False,
        "oracle_in_environment": False,
        "exact_stop_support": exact_support.manifest(),
    }
    (args.output_dir / "sampling_summary.json").write_text(
        json.dumps(sampling, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(sampling, indent=2, sort_keys=True))
    return 0


def reconstruct_support_pairs(
    args: argparse.Namespace,
    preregistration: dict,
    checkpoint: dict,
) -> list[object]:
    """Rebuild B22 train pairs for event-support only. Skip B36 trajectory lock.

    Table1 eval does not replay B36 internal metrics. RDKit/MCS drift currently
    selects more early-stop pairs than the locked B36 summary; using that
    broader fit set only widens grammar support.
    """
    config = dict(checkpoint["model_config"])
    allowed_counts = {int(value) for value in preregistration["property_counts"]}
    validation_rows = base.read_rows(args.validation_csv)
    common = {
        "max_atoms": int(config["max_atoms"]),
        "fingerprint_bits": int(preregistration["fingerprint_bits"]),
        "condition_dim": int(preregistration["condition_dim"]),
        "allowed_counts": allowed_counts,
        "timeout": int(preregistration["mcs_timeout"]),
        "min_common_fraction": float(preregistration["min_common_fraction"]),
        "limit": int(preregistration["historical_validation_limit"]),
    }
    excluded_pairs, _ = base.build_pairs(
        validation_rows,
        seed=int(preregistration["validation_exclusion_seed"]),
        **common,
    )
    excluded_sources = {pair.source_smiles for pair in excluded_pairs}
    excluded_keys = {(pair.source_smiles, pair.target_smiles) for pair in excluded_pairs}
    validation_pairs, _ = base.build_pairs(
        validation_rows,
        seed=int(preregistration["validation_selection_seed"]),
        forbidden_sources=excluded_sources,
        forbidden_pairs=excluded_keys,
        **common,
    )
    full_train_pairs, _ = base.build_pairs(
        base.read_rows(args.train_csv),
        max_atoms=int(config["max_atoms"]),
        fingerprint_bits=int(preregistration["fingerprint_bits"]),
        condition_dim=int(preregistration["condition_dim"]),
        allowed_counts=allowed_counts,
        timeout=int(preregistration["mcs_timeout"]),
        min_common_fraction=float(preregistration["min_common_fraction"]),
        limit=int(preregistration["train_limit"]),
        seed=int(preregistration["train_selection_seed"]),
        forbidden_sources={pair.source_smiles for pair in validation_pairs},
        forbidden_pairs={
            (pair.source_smiles, pair.target_smiles) for pair in validation_pairs
        },
    )
    for pair in full_train_pairs:
        pair.condition = hierarchical.property_latent_slot_tokens(
            pair.row, int(preregistration["condition_dim"])
        )
    trajectory_args = SimpleNamespace(
        fingerprint_bits=int(preregistration["fingerprint_bits"]),
        trajectory_fractions=",".join(
            str(value) for value in preregistration["trajectory_fractions"]
        ),
        trajectory_max_orders=int(preregistration["trajectory_max_orders"]),
    )
    selected_pairs, _trajectory = b36.b22.select_early_stop_pairs(
        full_train_pairs, checkpoint["vocabulary"], trajectory_args
    )
    return selected_pairs


def load_table1_conditions(
    path: Path,
    *,
    limit: int,
    condition_dim: int,
    graph_fingerprint_bits: int,
    max_atoms: int,
) -> list[b29.TransferCondition]:
    out: list[b29.TransferCondition] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            specs = base.task_specs(raw)
            task = b29.table1_task_key(specs)
            if task == "unknown":
                continue
            row = b29.source_only_row(raw, specs)
            source = graph.canonical_smiles(row["source_smiles"])
            if not source or not row["condition_id"]:
                continue
            source_graph = graph.molecule_example(
                source, max_atoms=int(max_atoms), fingerprint_bits=int(graph_fingerprint_bits)
            )
            if source_graph is None:
                continue
            out.append(
                b29.TransferCondition(
                    row=row,
                    source_smiles=source,
                    source=source_graph,
                    condition=hierarchical.property_latent_slot_tokens(
                        row, int(condition_dim)
                    ),
                    property_count=len(specs),
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
