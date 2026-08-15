#!/usr/bin/env python3
"""Evaluate a target-blind second fragment step for 3-property requests.

The frozen B24 kernel produces the first raw action.  For 3-property requests,
the assembled intermediate is encoded by the same frozen graph encoder and the
same property program conditions a second fragment latent.  The second step is
always part of the generative trajectory; no property oracle, validity signal,
candidate comparison, or early-stop decision chooses whether to run it.
Two-property requests retain the matched one-step B24 path.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import latent_fragment_attachment_kernel as kernel


base = kernel.base
belief = kernel.belief
graph = kernel.graph
hierarchical = kernel.hierarchical
unified = kernel.unified

PROTOCOL = "two_step_residual_fragment_latent_rollout_pilot_v25"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--representation-checkpoint", type=Path, required=True)
    parser.add_argument("--representation-summary", type=Path, required=True)
    parser.add_argument("--fragment-checkpoint", type=Path, required=True)
    parser.add_argument("--fragment-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-limit", type=int, default=20)
    parser.add_argument("--property-counts", default="2,3")
    parser.add_argument("--fingerprint-bits", type=int, default=256)
    parser.add_argument("--graph-fingerprint-bits", type=int, default=512)
    parser.add_argument("--condition-dim", type=int, default=64)
    parser.add_argument("--flow-steps", type=int, default=12)
    parser.add_argument("--site-temperature", type=float, default=0.80)
    parser.add_argument("--num-attempts", type=int, default=20)
    parser.add_argument("--mcs-timeout", type=int, default=1)
    parser.add_argument("--min-common-fraction", type=float, default=0.45)
    parser.add_argument("--min-core-heavy-atoms", type=int, default=5)
    parser.add_argument("--max-variable-heavy-atoms", type=int, default=30)
    parser.add_argument("--validation-selection-seed", type=int, default=2719)
    parser.add_argument("--validation-exclusion-seed", type=int, default=1742)
    parser.add_argument("--gate-validity", type=float, default=0.95)
    parser.add_argument("--gate-overall-strict-delta", type=float, default=-0.05)
    parser.add_argument("--gate-3p-strict-delta", type=float, default=0.14)
    parser.add_argument("--gate-mean-unique-valid", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=1761)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def load_fragment_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[kernel.FragmentAttachmentKernel, list[str], np.ndarray, dict[str, object]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = dict(payload["model_config"])
    model = kernel.FragmentAttachmentKernel(
        source_dim=int(config["source_dim"]),
        condition_dim=int(config["condition_dim"]),
        site_dim=int(config["site_dim"]),
        endpoint_dim=int(config["endpoint_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return (
        model,
        list(payload["target_fragments"]),
        np.asarray(payload["target_endpoints"], dtype=np.float32),
        payload,
    )


def validation_pairs(args: argparse.Namespace) -> tuple[list[object], dict[str, object]]:
    allowed_counts = base.parse_property_counts(str(args.property_counts))
    rows = base.read_rows(args.validation_csv)
    historical, historical_counts = base.build_pairs(
        rows,
        max_atoms=64,
        fingerprint_bits=int(args.graph_fingerprint_bits),
        condition_dim=int(args.condition_dim),
        allowed_counts=allowed_counts,
        timeout=int(args.mcs_timeout),
        min_common_fraction=float(args.min_common_fraction),
        limit=int(args.validation_limit),
        seed=int(args.validation_exclusion_seed),
    )
    historical_sources = {pair.source_smiles for pair in historical}
    historical_keys = {
        (pair.source_smiles, pair.target_smiles) for pair in historical
    }
    pairs, counts = base.build_pairs(
        rows,
        max_atoms=64,
        fingerprint_bits=int(args.graph_fingerprint_bits),
        condition_dim=int(args.condition_dim),
        allowed_counts=allowed_counts,
        timeout=int(args.mcs_timeout),
        min_common_fraction=float(args.min_common_fraction),
        limit=int(args.validation_limit),
        seed=int(args.validation_selection_seed),
        forbidden_sources=historical_sources,
        forbidden_pairs=historical_keys,
    )
    for pair in pairs:
        pair.condition = hierarchical.property_latent_slot_tokens(
            pair.row, int(args.condition_dim)
        )
    return pairs, {
        "validation_filter_counts": counts,
        "historical_validation_filter_counts": historical_counts,
        "historical_sources": historical_sources,
        "historical_keys": historical_keys,
    }


@torch.no_grad()
def encode_one_source(
    representation, example, device: torch.device
) -> np.ndarray:
    batch = base.move_graph_batch(graph.collate([example]), device)
    node, _edge = representation.encode(batch)
    mask = batch["node_mask"].to(node.dtype)
    pooled = (node * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
        dim=1, keepdim=True
    ).clamp_min(1.0)
    return pooled[0].float().cpu().numpy().astype(np.float32)


def second_step(
    model: kernel.FragmentAttachmentKernel,
    representation,
    pair: object,
    intermediate: str,
    target_fragments: Sequence[str],
    target_endpoints: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    canonical = graph.canonical_smiles(intermediate)
    example = graph.molecule_example(
        canonical, max_atoms=64, fingerprint_bits=int(args.graph_fingerprint_bits)
    )
    if example is None:
        return {
            "smiles": "",
            "site_core": "",
            "source_fragment": "",
            "target_fragment_token": "",
            "latent_norm": 0.0,
            "quantization_distance": float("inf"),
            "site_entropy": 0.0,
        }
    residual_pair = copy.copy(pair)
    residual_pair.source = example
    residual_pair.source_smiles = canonical
    local_args = copy.copy(args)
    local_args.num_attempts = 1
    generated = kernel.generate_actions(
        model,
        residual_pair,
        encode_one_source(representation, example, device),
        target_fragments,
        target_endpoints,
        local_args,
        device,
        seed=int(seed),
    )
    if len(generated) != 1:
        raise RuntimeError("B25 residual step must emit one action")
    return generated[0]


def freeze_rollouts(
    model: kernel.FragmentAttachmentKernel,
    representation,
    pairs: Sequence[object],
    source_latents: np.ndarray,
    target_fragments: Sequence[str],
    target_endpoints: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> list[tuple[object, list[dict[str, object]]]]:
    frozen: list[tuple[object, list[dict[str, object]]]] = []
    for pair_index, pair in enumerate(pairs):
        first = kernel.generate_actions(
            model,
            pair,
            source_latents[pair_index],
            target_fragments,
            target_endpoints,
            args,
            device,
            seed=int(args.seed) * 100000 + pair_index,
        )
        final: list[dict[str, object]] = []
        for attempt, raw in enumerate(first):
            stage_one = graph.canonical_smiles(str(raw["smiles"] or ""))
            if int(pair.property_count) < 3:
                final.append({**raw, "stage_one_smiles": stage_one, "steps": 1})
                continue
            residual = second_step(
                model,
                representation,
                pair,
                stage_one,
                target_fragments,
                target_endpoints,
                args,
                device,
                seed=int(args.seed) * 10000000 + pair_index * 100 + attempt,
            )
            final.append(
                {
                    **residual,
                    "stage_one_smiles": stage_one,
                    "stage_one_site_core": raw["site_core"],
                    "stage_one_source_fragment": raw["source_fragment"],
                    "stage_one_target_fragment_token": raw[
                        "target_fragment_token"
                    ],
                    "steps": 2,
                }
            )
        if len(final) != int(args.num_attempts):
            raise RuntimeError("B25 must freeze exactly 20 final attempts")
        frozen.append((pair, final))
    return frozen


def evaluate(
    frozen: Sequence[tuple[object, list[dict[str, object]]]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    from rdkit import Chem

    rows: list[dict[str, object]] = []
    for pair_index, (pair, generated) in enumerate(frozen):
        specs = base.task_specs(pair.row)
        condition_id = str(
            pair.row.get("condition_id", "")
            or pair.row.get("sample_id", "")
            or f"validation_{pair_index:04d}"
        )
        source_copy_target = (
            graph.morgan_tanimoto(pair.source_smiles, pair.target_smiles) or 0.0
        )
        for rank, raw in enumerate(generated, start=1):
            canonical = graph.canonical_smiles(str(raw["smiles"] or ""))
            valid = bool(canonical)
            molecule = Chem.MolFromSmiles(canonical) if valid else None
            source_tanimoto = (
                graph.morgan_tanimoto(pair.source_smiles, canonical) if valid else None
            )
            target_tanimoto = (
                graph.morgan_tanimoto(pair.target_smiles, canonical) if valid else None
            )
            fraction, _, evaluated, all_success = (
                unified.instruction_success_and_distance(
                    pair.row, canonical or "", task_specs=specs
                )
            )
            similarity_success = bool(
                source_tanimoto is not None and source_tanimoto >= 0.4
            )
            rows.append(
                {
                    "condition_id": condition_id,
                    "attempt": rank,
                    "property_count": pair.property_count,
                    "task": pair.task,
                    "steps": raw["steps"],
                    "source_smiles": pair.source_smiles,
                    "target_smiles": pair.target_smiles,
                    "stage_one_smiles": raw["stage_one_smiles"],
                    "generated_smiles": canonical or "",
                    "site_core": raw["site_core"],
                    "source_fragment": raw["source_fragment"],
                    "target_fragment_token": raw["target_fragment_token"],
                    "source_atom_count": int(pair.source.node_mask.sum()),
                    "target_atom_count": int(pair.target.node_mask.sum()),
                    "predicted_atom_count": (
                        int(molecule.GetNumAtoms()) if molecule is not None else 0
                    ),
                    "valid": valid,
                    "source_tanimoto": float(source_tanimoto or 0.0),
                    "target_tanimoto": float(target_tanimoto or 0.0),
                    "source_copy_target_tanimoto": float(source_copy_target),
                    "property_fraction": float(fraction),
                    "evaluated_properties": int(evaluated),
                    "property_success": bool(all_success),
                    "strict_success": bool(all_success and similarity_success),
                    "source_similarity_success": similarity_success,
                }
            )
    metrics = base.summarize_candidates(rows, int(args.num_attempts))
    metrics["source_copy_rate"] = sum(
        str(row["generated_smiles"])
        == graph.canonical_smiles(str(row["source_smiles"]))
        for row in rows
    ) / max(1, len(rows))
    return rows, metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.num_attempts) != 20:
        raise ValueError("B25 requires exactly 20 raw attempts per condition")
    base.seed_everything(int(args.seed))
    device = base.resolve_device(str(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(args.fragment_summary.read_text(encoding="utf-8"))
    baseline_metrics = baseline["evaluation"]
    representation, _config, representation_summary = base.load_representation(
        args.representation_checkpoint, args.representation_summary, device
    )
    model, target_fragments, target_endpoints, checkpoint = load_fragment_model(
        args.fragment_checkpoint, device
    )
    pairs, split = validation_pairs(args)
    if not pairs:
        raise ValueError("No B25 development pairs")
    source_latents = kernel.encode_sources(representation, pairs, device)
    frozen = freeze_rollouts(
        model,
        representation,
        pairs,
        source_latents,
        target_fragments,
        target_endpoints,
        args,
        device,
    )
    rows, metrics = evaluate(frozen, args)
    baseline_three = float(
        baseline_metrics["by_property_count"].get("3", {}).get("strict_any20", 0.0)
    )
    three_strict = float(
        metrics["by_property_count"].get("3", {}).get("strict_any20", 0.0)
    )
    checks = {
        "exact_attempts": {"value": metrics["attempted_per_condition"], "threshold": 20},
        "validity": {"value": metrics["validity"], "threshold": float(args.gate_validity)},
        "overall_strict_delta": {
            "value": float(metrics["strict_any20"])
            - float(baseline_metrics["strict_any20"]),
            "threshold": float(args.gate_overall_strict_delta),
        },
        "three_property_strict_delta": {
            "value": three_strict - baseline_three,
            "threshold": float(args.gate_3p_strict_delta),
        },
        "mean_unique_valid": {
            "value": metrics["mean_unique_valid"],
            "threshold": float(args.gate_mean_unique_valid),
        },
    }
    failures = [
        name
        for name, item in checks.items()
        if (
            item["value"] != item["threshold"]
            if name == "exact_attempts"
            else item["value"] < item["threshold"]
        )
    ]
    checkpoint_manifest = dict(checkpoint.get("manifest", {}))
    pair_sources = {pair.source_smiles for pair in pairs}
    pair_keys = {(pair.source_smiles, pair.target_smiles) for pair in pairs}
    manifest = {
        "protocol": PROTOCOL,
        "seed": int(args.seed),
        "heldout_role": "development_not_final_audit",
        "device": str(device),
        "representation_protocol": representation_summary.get("protocol"),
        "fragment_training_protocol": checkpoint.get("stage"),
        "fragment_checkpoint_sha256": belief.file_sha256(args.fragment_checkpoint),
        "fragment_train_validation_source_overlap": checkpoint_manifest.get(
            "train_validation_source_overlap"
        ),
        "fragment_train_validation_pair_overlap": checkpoint_manifest.get(
            "train_validation_pair_overlap"
        ),
        "selected_validation_pairs": len(pairs),
        "validation_filter_counts": split["validation_filter_counts"],
        "historical_validation_filter_counts": split[
            "historical_validation_filter_counts"
        ],
        "historical_validation_source_overlap": len(
            split["historical_sources"] & pair_sources
        ),
        "historical_validation_pair_overlap": len(
            split["historical_keys"] & pair_keys
        ),
        "two_property_fragment_steps": 1,
        "three_property_fragment_steps": 2,
        "intermediate_graph_reencoding": True,
        "oracle_free_residual_property_slots": True,
        "generation_target_access": False,
        "property_oracle_generation_access": False,
        "generation_rdkit_validity_feedback": False,
        "candidate_library": False,
        "molecular_candidate_ranking": False,
        "selector": False,
        "finalizer": False,
        "oracle_reranking": False,
        "posthoc_molecule_repair": False,
        "failed_attachment_retry": False,
        "property_based_early_stop": False,
        "exact_raw_attempts_per_condition": 20,
    }
    base.write_candidate_rows(args.output_dir / "validation_candidates.csv", rows)
    summary = {
        "protocol": PROTOCOL,
        "manifest": manifest,
        "baseline": {
            "protocol": baseline.get("protocol"),
            "validity": baseline_metrics["validity"],
            "strict_any20": baseline_metrics["strict_any20"],
            "three_property_strict_any20": baseline_three,
            "mean_unique_valid": baseline_metrics["mean_unique_valid"],
        },
        "evaluation": metrics,
        "gate": {"passed": not failures, "checks": checks, "failures": failures},
        "next_stage": (
            "train_residual_fragment_transition_on_two_step_train_trajectories"
            if not failures
            else "retain_one_step_b24_and_learn_train_only_property_residual_energy"
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
