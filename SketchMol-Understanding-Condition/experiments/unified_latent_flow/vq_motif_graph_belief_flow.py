#!/usr/bin/env python3
"""Single-token VQ motif graph-belief flow pilot.

A train-only posterior compresses each aligned source-to-target graph delta into
one discrete motif token.  A source-and-condition prior predicts that token at
generation time, and a shared graph-latent decoder maps source plus token to a
complete endpoint in one pass.  The 20 raw attempts differ only by sampled
latent motif token: atom/bond categories are deterministic decoder argmaxes,
not independent stochastic edit gates.  Targets and property oracles are never
available to generation, and there is no candidate selector or chemistry
repair pass.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
BELIEF_PATH = SCRIPT_DIR / "categorical_graph_belief_flow.py"
PROTOCOL = "contrastive_single_token_vq_motif_graph_belief_flow_pilot_v5b"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


belief = load_module("vq_motif_graph_belief_base", BELIEF_PATH)
base = belief.base
graph = belief.graph
unified = belief.unified
NODE_FIELDS = belief.NODE_FIELDS
EDGE_FIELDS = belief.EDGE_FIELDS


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--representation-checkpoint", type=Path, required=True)
    parser.add_argument("--representation-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=1500)
    parser.add_argument("--validation-limit", type=int, default=20)
    parser.add_argument("--property-counts", default="2,3")
    parser.add_argument("--fingerprint-bits", type=int, default=512)
    parser.add_argument("--condition-dim", type=int, default=64)
    parser.add_argument("--code-dim", type=int, default=64)
    parser.add_argument("--codebook-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--prior-loss-weight", type=float, default=0.50)
    parser.add_argument("--vq-loss-weight", type=float, default=0.25)
    parser.add_argument("--commitment-weight", type=float, default=0.25)
    parser.add_argument("--contrastive-loss-weight", type=float, default=0.25)
    parser.add_argument("--contrastive-margin", type=float, default=0.20)
    parser.add_argument("--sampling-temperature", type=float, default=0.80)
    parser.add_argument("--num-attempts", type=int, default=20)
    parser.add_argument("--sample-batch-size", type=int, default=5)
    parser.add_argument("--mcs-timeout", type=int, default=1)
    parser.add_argument("--min-common-fraction", type=float, default=0.45)
    parser.add_argument("--gate-validity", type=float, default=0.80)
    parser.add_argument("--gate-source-tanimoto", type=float, default=0.40)
    parser.add_argument("--gate-target-improvement-rate", type=float, default=0.25)
    parser.add_argument("--gate-strict-any20", type=float, default=0.20)
    parser.add_argument("--gate-min-active-codes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1737)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def change_masks(
    source: Mapping[str, object], target: Mapping[str, object]
) -> tuple[torch.Tensor, torch.Tensor]:
    source_atomic = source["atomic_number"]
    if not isinstance(source_atomic, torch.Tensor):
        raise TypeError("source graph tensors are required")
    node_changed = torch.zeros_like(source_atomic, dtype=torch.bool)
    for key in NODE_FIELDS:
        node_changed |= source[key].ne(target[key])
    edge_changed = torch.zeros_like(source["bond"], dtype=torch.bool)
    for key in EDGE_FIELDS:
        edge_changed |= source[key].ne(target[key])
    nodes = source_atomic.shape[1]
    upper = torch.triu(
        torch.ones(nodes, nodes, device=source_atomic.device, dtype=torch.bool), diagonal=1
    )
    return node_changed, edge_changed & upper.unsqueeze(0)


def masked_pool(values: torch.Tensor, mask: torch.Tensor, dimensions: tuple[int, ...]) -> torch.Tensor:
    weight = mask.to(values.dtype).unsqueeze(-1)
    total = (values * weight).sum(dim=dimensions)
    denominator = weight.sum(dim=dimensions).clamp_min(1.0)
    return total / denominator


class VQMotifGraphFlow(nn.Module):
    """Posterior VQ motif encoder, source-condition prior, and endpoint decoder."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        condition_dim: int,
        code_dim: int,
        codebook_size: int,
        hidden_dim: int,
        max_atoms: int,
    ) -> None:
        super().__init__()
        self.code_dim = int(code_dim)
        self.codebook_size = int(codebook_size)
        self.posterior = nn.Sequential(
            nn.LayerNorm(node_dim + edge_dim + condition_dim),
            nn.Linear(node_dim + edge_dim + condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, code_dim),
        )
        self.codebook = nn.Embedding(codebook_size, code_dim)
        nn.init.normal_(self.codebook.weight, std=0.10)
        self.prior = nn.Sequential(
            nn.LayerNorm(node_dim + condition_dim),
            nn.Linear(node_dim + condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, codebook_size),
        )
        self.decoder = belief.CategoricalEndpointField(
            node_dim,
            edge_dim,
            condition_dim + code_dim,
            hidden_dim,
            max_atoms,
        )

    @staticmethod
    def source_pool(source_node: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        total = (source_node * source_mask.unsqueeze(-1)).sum(dim=1)
        return total / source_mask.sum(dim=1, keepdim=True).clamp_min(1.0)

    def posterior_vector(
        self,
        source_node: torch.Tensor,
        source_edge: torch.Tensor,
        target_node: torch.Tensor,
        target_edge: torch.Tensor,
        node_changed: torch.Tensor,
        edge_changed: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        node_delta = masked_pool(target_node - source_node, node_changed, (1,))
        edge_delta = masked_pool(target_edge - source_edge, edge_changed, (1, 2))
        return self.posterior(torch.cat([node_delta, edge_delta, condition], dim=-1))

    def quantize(
        self, posterior: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distance = (
            posterior.square().sum(dim=1, keepdim=True)
            - 2.0 * posterior @ self.codebook.weight.transpose(0, 1)
            + self.codebook.weight.square().sum(dim=1).unsqueeze(0)
        )
        index = distance.argmin(dim=1)
        quantized = self.codebook(index)
        straight_through = posterior + (quantized - posterior).detach()
        return straight_through, quantized, index

    def prior_logits(
        self, source_node: torch.Tensor, source_mask: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        pooled = self.source_pool(source_node, source_mask)
        return self.prior(torch.cat([pooled, condition], dim=-1))

    def decode_endpoint(
        self,
        source_node: torch.Tensor,
        source_edge: torch.Tensor,
        source_mask: torch.Tensor,
        birth_rank: torch.Tensor,
        condition: torch.Tensor,
        code: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([condition, code], dim=-1)
        time = torch.zeros(source_node.shape[0], device=source_node.device, dtype=source_node.dtype)
        return self.decoder(
            source_node,
            source_edge,
            source_node,
            source_edge,
            source_mask,
            source_mask,
            birth_rank,
            time,
            combined,
        )


def train_flow(
    flow: VQMotifGraphFlow,
    representation,
    pairs: Sequence[object],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, float]], Counter[int]]:
    optimizer = torch.optim.AdamW(
        flow.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    history: list[dict[str, float]] = []
    final_usage: Counter[int] = Counter()
    for epoch in range(1, int(args.epochs) + 1):
        order = list(range(len(pairs)))
        random.Random(int(args.seed) + epoch).shuffle(order)
        totals: defaultdict[str, float] = defaultdict(float)
        epoch_usage: Counter[int] = Counter()
        batches = 0
        flow.train()
        for start in range(0, len(order), int(args.batch_size)):
            items = [pairs[index] for index in order[start : start + int(args.batch_size)]]
            collated = base.pair_collate(items)
            source = base.move_graph_batch(collated["source"], device)
            target = base.move_graph_batch(collated["target"], device)
            condition = collated["condition"].to(device)
            node_changed, edge_changed = change_masks(source, target)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                source_node, source_edge = representation.encode(source)
                target_node, target_edge = representation.encode(target)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                posterior = flow.posterior_vector(
                    source_node,
                    source_edge,
                    target_node,
                    target_edge,
                    node_changed,
                    edge_changed,
                    condition,
                )
                code, quantized, code_index = flow.quantize(posterior)
                prior_logits = flow.prior_logits(source_node, source["node_mask"], condition)
                endpoint_node, endpoint_edge = flow.decode_endpoint(
                    source_node,
                    source_edge,
                    source["node_mask"],
                    belief.source_birth_ranks(source["node_mask"]),
                    condition,
                    code,
                )
                logits = representation.decode(endpoint_node, endpoint_edge)
                endpoint_loss, parts = graph.reconstruction_loss(
                    logits, target, endpoint_node, geometry_weight=0.0
                )
                wrong_index = torch.roll(code_index, shifts=1, dims=0)
                wrong_index = torch.where(
                    wrong_index.eq(code_index),
                    (code_index + 1) % flow.codebook_size,
                    wrong_index,
                )
                wrong_code = flow.codebook(wrong_index).detach()
                wrong_node, wrong_edge = flow.decode_endpoint(
                    source_node,
                    source_edge,
                    source["node_mask"],
                    belief.source_birth_ranks(source["node_mask"]),
                    condition,
                    wrong_code,
                )
                wrong_logits = representation.decode(wrong_node, wrong_edge)
                wrong_endpoint_loss, _ = graph.reconstruction_loss(
                    wrong_logits, target, wrong_node, geometry_weight=0.0
                )
                contrastive_loss = F.relu(
                    float(args.contrastive_margin) + endpoint_loss - wrong_endpoint_loss
                )
                prior_loss = F.cross_entropy(prior_logits, code_index.detach())
                codebook_loss = F.mse_loss(quantized, posterior.detach())
                commitment_loss = F.mse_loss(posterior, quantized.detach())
                vq_loss = codebook_loss + float(args.commitment_weight) * commitment_loss
                loss = (
                    endpoint_loss
                    + float(args.prior_loss_weight) * prior_loss
                    + float(args.vq_loss_weight) * vq_loss
                    + float(args.contrastive_loss_weight) * contrastive_loss
                )
            loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), float(args.grad_clip))
            optimizer.step()
            for name, value in parts.items():
                totals[name] += float(value)
            totals["total_loss"] += float(loss.detach())
            totals["prior_loss"] += float(prior_loss.detach())
            totals["vq_loss"] += float(vq_loss.detach())
            totals["wrong_endpoint_loss"] += float(wrong_endpoint_loss.detach())
            totals["contrastive_loss"] += float(contrastive_loss.detach())
            epoch_usage.update(int(value) for value in code_index.detach().cpu().tolist())
            batches += 1
        final_usage = epoch_usage
        probabilities = np.asarray(list(epoch_usage.values()), dtype=np.float64)
        probabilities = probabilities / max(1.0, probabilities.sum())
        perplexity = float(np.exp(-(probabilities * np.log(probabilities + 1e-12)).sum()))
        row = {
            "epoch": epoch,
            **{name: value / max(1, batches) for name, value in totals.items()},
            "active_codes": len(epoch_usage),
            "code_perplexity": perplexity,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return history, final_usage


@torch.no_grad()
def sample_from_source(
    flow: VQMotifGraphFlow,
    representation,
    source_example,
    condition: np.ndarray,
    active_codes: Sequence[int],
    *,
    attempts: int,
    batch_size: int,
    temperature: float,
    device: torch.device,
    seed: int,
) -> list[tuple[str | None, int, int]]:
    """Sample one motif token per attempt without a target or property oracle."""
    if not active_codes:
        raise ValueError("At least one train-used motif code is required")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    outputs: list[tuple[str | None, int, int]] = []
    flow.eval()
    active = torch.as_tensor(active_codes, device=device, dtype=torch.long)
    for start in range(0, int(attempts), int(batch_size)):
        count = min(int(batch_size), int(attempts) - start)
        source = base.move_graph_batch(graph.collate([source_example] * count), device)
        condition_batch = torch.from_numpy(
            np.repeat(condition[None, :], count, axis=0)
        ).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            source_node, source_edge = representation.encode(source)
            prior_logits = flow.prior_logits(source_node, source["node_mask"], condition_batch)
        active_logits = prior_logits[:, active].float() / max(1e-4, float(temperature))
        probability = torch.softmax(active_logits, dim=-1)
        sampled_active = torch.multinomial(
            probability, 1, replacement=True, generator=generator
        ).squeeze(1)
        code_index = active[sampled_active]
        code = flow.codebook(code_index)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            endpoint_node, endpoint_edge = flow.decode_endpoint(
                source_node,
                source_edge,
                source["node_mask"],
                belief.source_birth_ranks(source["node_mask"]),
                condition_batch,
                code,
            )
            logits = representation.decode(endpoint_node, endpoint_edge)
        prediction = graph.predictions_from_logits(logits)
        atomic = prediction["atomic_number"] > 0
        active_pair = atomic[:, :, None] & atomic[:, None, :]
        prediction["bond"][~active_pair] = graph.BOND_NONE
        prediction["bond_stereo"][~active_pair] = 0
        for index in range(count):
            smiles, _ = graph.graph_to_smiles(prediction, index)
            outputs.append(
                (smiles, int(atomic[index].sum()), int(code_index[index].item()))
            )
    if len(outputs) != int(attempts):
        raise RuntimeError(f"Expected {attempts} attempts, produced {len(outputs)}")
    return outputs


def evaluate(
    flow: VQMotifGraphFlow,
    representation,
    pairs: Sequence[object],
    active_codes: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    candidate_rows: list[dict[str, object]] = []
    for pair_index, pair in enumerate(pairs):
        generated = sample_from_source(
            flow,
            representation,
            pair.source,
            pair.condition,
            active_codes,
            attempts=int(args.num_attempts),
            batch_size=int(args.sample_batch_size),
            temperature=float(args.sampling_temperature),
            device=device,
            seed=int(args.seed) * 100000 + pair_index,
        )
        source_copy_target = graph.morgan_tanimoto(pair.source_smiles, pair.target_smiles) or 0.0
        specs = base.task_specs(pair.row)
        condition_id = str(
            pair.row.get("condition_id", "")
            or pair.row.get("sample_id", "")
            or f"validation_{pair_index:04d}"
        )
        for rank, (smiles, predicted_atom_count, code_index) in enumerate(
            generated, start=1
        ):
            canonical = graph.canonical_smiles(smiles or "")
            valid = bool(canonical)
            source_tanimoto = graph.morgan_tanimoto(pair.source_smiles, canonical) if valid else None
            target_tanimoto = graph.morgan_tanimoto(pair.target_smiles, canonical) if valid else None
            fraction, _, evaluated, all_success = unified.instruction_success_and_distance(
                pair.row, canonical or "", task_specs=specs
            )
            source_similarity_success = bool(
                source_tanimoto is not None and source_tanimoto >= 0.4
            )
            candidate_rows.append(
                {
                    "condition_id": condition_id,
                    "attempt": rank,
                    "property_count": pair.property_count,
                    "task": pair.task,
                    "motif_code": int(code_index),
                    "source_smiles": pair.source_smiles,
                    "target_smiles": pair.target_smiles,
                    "generated_smiles": canonical or "",
                    "source_atom_count": int(pair.source.node_mask.sum()),
                    "target_atom_count": int(pair.target.node_mask.sum()),
                    "predicted_atom_count": int(predicted_atom_count),
                    "valid": valid,
                    "source_tanimoto": float(source_tanimoto or 0.0),
                    "target_tanimoto": float(target_tanimoto or 0.0),
                    "source_copy_target_tanimoto": float(source_copy_target),
                    "property_fraction": float(fraction),
                    "evaluated_properties": int(evaluated),
                    "property_success": bool(all_success),
                    "strict_success": bool(all_success and source_similarity_success),
                    "source_similarity_success": source_similarity_success,
                }
            )
    metrics = base.summarize_candidates(candidate_rows, int(args.num_attempts))
    metrics["sampled_active_codes"] = len(
        {int(row["motif_code"]) for row in candidate_rows}
    )
    return candidate_rows, metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.num_attempts) != 20:
        raise ValueError("The protocol requires exactly 20 raw attempts per condition")
    base.seed_everything(int(args.seed))
    device = base.resolve_device(str(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    representation, config, representation_summary = base.load_representation(
        args.representation_checkpoint, args.representation_summary, device
    )
    allowed_counts = base.parse_property_counts(str(args.property_counts))
    validation_pairs, validation_counts = base.build_pairs(
        base.read_rows(args.validation_csv),
        max_atoms=int(config["max_atoms"]),
        fingerprint_bits=int(args.fingerprint_bits),
        condition_dim=int(args.condition_dim),
        allowed_counts=allowed_counts,
        timeout=int(args.mcs_timeout),
        min_common_fraction=float(args.min_common_fraction),
        limit=int(args.validation_limit),
        seed=int(args.seed) + 1,
    )
    if not validation_pairs:
        raise ValueError("No validation edit pairs survived the fixed filters")
    validation_sources = {pair.source_smiles for pair in validation_pairs}
    validation_pair_keys = {(pair.source_smiles, pair.target_smiles) for pair in validation_pairs}
    train_pairs, train_counts = base.build_pairs(
        base.read_rows(args.train_csv),
        max_atoms=int(config["max_atoms"]),
        fingerprint_bits=int(args.fingerprint_bits),
        condition_dim=int(args.condition_dim),
        allowed_counts=allowed_counts,
        timeout=int(args.mcs_timeout),
        min_common_fraction=float(args.min_common_fraction),
        limit=int(args.train_limit),
        seed=int(args.seed),
        forbidden_sources=validation_sources,
        forbidden_pairs=validation_pair_keys,
    )
    if len(train_pairs) < 32:
        raise ValueError(f"Need at least 32 train pairs, found {len(train_pairs)}")
    flow = VQMotifGraphFlow(
        node_dim=int(config["node_dim"]),
        edge_dim=int(config["edge_dim"]),
        condition_dim=int(args.condition_dim),
        code_dim=int(args.code_dim),
        codebook_size=int(args.codebook_size),
        hidden_dim=int(args.hidden_dim),
        max_atoms=int(config["max_atoms"]),
    ).to(device)
    history, code_usage = train_flow(flow, representation, train_pairs, args, device)
    active_codes = sorted(code_usage)
    candidate_rows, metrics = evaluate(
        flow, representation, validation_pairs, active_codes, args, device
    )
    checks = {
        "exact_attempts": {"value": metrics["attempted_per_condition"], "threshold": 20},
        "train_active_codes": {
            "value": len(active_codes),
            "threshold": int(args.gate_min_active_codes),
        },
        "validity": {"value": metrics["validity"], "threshold": float(args.gate_validity)},
        "mean_source_tanimoto": {
            "value": metrics["mean_source_tanimoto"],
            "threshold": float(args.gate_source_tanimoto),
        },
        "target_improvement_any20": {
            "value": metrics["target_improvement_any20"],
            "threshold": float(args.gate_target_improvement_rate),
        },
        "strict_any20": {
            "value": metrics["strict_any20"],
            "threshold": float(args.gate_strict_any20),
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
    train_sources = {pair.source_smiles for pair in train_pairs}
    train_pair_keys = {(pair.source_smiles, pair.target_smiles) for pair in train_pairs}
    manifest = {
        "protocol": PROTOCOL,
        "seed": int(args.seed),
        "device": str(device),
        "representation_protocol": representation_summary.get("protocol"),
        "representation_gate_passed": bool(
            representation_summary.get("gate", {}).get("passed")
        ),
        "representation_checkpoint": str(args.representation_checkpoint),
        "representation_checkpoint_sha256": belief.file_sha256(
            args.representation_checkpoint
        ),
        "train_csv": str(args.train_csv),
        "train_csv_sha256": belief.file_sha256(args.train_csv),
        "validation_csv": str(args.validation_csv),
        "validation_csv_sha256": belief.file_sha256(args.validation_csv),
        "selected_train_pairs": len(train_pairs),
        "selected_validation_pairs": len(validation_pairs),
        "train_filter_counts": train_counts,
        "validation_filter_counts": validation_counts,
        "train_validation_source_overlap": len(train_sources & validation_sources),
        "train_validation_pair_overlap": len(train_pair_keys & validation_pair_keys),
        "property_counts": sorted(allowed_counts),
        "codebook_size": int(args.codebook_size),
        "train_active_codes": len(active_codes),
        "train_code_usage": {str(key): int(value) for key, value in sorted(code_usage.items())},
        "generation_target_access": False,
        "evaluation_target_access": True,
        "property_oracle_generation_access": False,
        "single_discrete_motif_token_per_attempt": True,
        "posterior_train_only": True,
        "source_condition_prior": True,
        "deterministic_category_decode_given_token": True,
        "token_contrastive_reconstruction": True,
        "contrastive_margin": float(args.contrastive_margin),
        "independent_atom_or_bond_sampling": False,
        "candidate_library": False,
        "selector": False,
        "finalizer": False,
        "oracle_reranking": False,
        "valence_projection_or_repair": False,
        "exact_raw_attempts_per_condition": 20,
        "source_target_mcs_alignment_training_only": True,
    }
    checkpoint_path = args.output_dir / "vq_motif_graph_belief_flow.pt"
    torch.save(
        {
            "stage": PROTOCOL,
            "model_state": flow.state_dict(),
            "model_config": {
                "node_dim": int(config["node_dim"]),
                "edge_dim": int(config["edge_dim"]),
                "condition_dim": int(args.condition_dim),
                "code_dim": int(args.code_dim),
                "codebook_size": int(args.codebook_size),
                "hidden_dim": int(args.hidden_dim),
                "max_atoms": int(config["max_atoms"]),
            },
            "active_codes": active_codes,
            "history": history,
            "manifest": manifest,
        },
        checkpoint_path,
    )
    base.write_candidate_rows(args.output_dir / "validation_candidates.csv", candidate_rows)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "protocol": PROTOCOL,
        "checkpoint": str(checkpoint_path),
        "manifest": manifest,
        "training": history,
        "evaluation": metrics,
        "gate": {"passed": not failures, "checks": checks, "failures": failures},
        "next_stage": (
            "expand_vq_motif_codebook_signal"
            if not failures
            else "diagnose_vq_motif_validity_or_code_collapse"
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
