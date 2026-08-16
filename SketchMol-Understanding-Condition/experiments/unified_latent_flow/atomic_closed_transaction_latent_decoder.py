#!/usr/bin/env python3
"""Learn a latent distribution over complete, chemically closed graph rewrites.

This stage replaces event-by-event molecular decoding with one atomic action.
The fit split supplies a vocabulary of one-cut graph rewrite transactions.  At
generation time the source is used to construct the complete set of applicable,
RDKit-sanitized transactions.  Each of exactly twenty orthogonal latent
particles samples one transaction directly from the learned conditional action
distribution.  No larger set of molecules is generated, ranked, repaired, or
selected with a property oracle.

The finite applicable set is the decoder action space, analogous to the legal
token mask of a constrained language model.  It is not a post-hoc candidate
pool: only the twenty sampled actions are committed and frozen for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
UCA_DIR = PROJECT_DIR / "experiments" / "unified_constraint_agent"
for path in (SCRIPT_DIR, PROJECT_DIR, UCA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_retrieved_delta_edit_candidates as retrieved  # noqa: E402
import set_closed_graph_transport as b43  # noqa: E402


base = b43.base
belief = b43.belief
evidence = b43.evidence
graph = b43.graph

PROTOCOL = "train_only_atomic_closed_transaction_latent_decoder_v1"


@dataclass(frozen=True)
class ClosedTransaction:
    smiles: str
    query_core: str
    query_variable: str
    source_variable: str
    target_variable: str
    retrieval_similarity: float
    transform_frequency: int
    train_condition_id: str
    source_tanimoto: float
    identity: bool = False


class TransactionEnergyDecoder(nn.Module):
    """Map one complete rewrite to a base energy and latent direction."""

    def __init__(self, feature_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(int(feature_dim)),
            nn.Linear(int(feature_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), 1 + int(latent_dim)),
        )

    def forward(
        self, features: torch.Tensor, particles: torch.Tensor, latent_scale: float
    ) -> torch.Tensor:
        output = self.network(features.float())
        base_energy = output[:, 0]
        direction = F.normalize(output[:, 1:], dim=1)
        return base_energy[None, :] + float(latent_scale) * particles @ direction.T


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--representation-checkpoint", type=Path, required=True)
    parser.add_argument("--representation-summary", type=Path, required=True)
    parser.add_argument("--b22-checkpoint", type=Path, required=True)
    parser.add_argument("--b22-summary", type=Path, required=True)
    parser.add_argument("--b36-records", type=Path, required=True)
    parser.add_argument("--b41-checkpoint", type=Path, required=True)
    parser.add_argument("--b41-summary", type=Path, required=True)
    parser.add_argument("--set-evidence-summary", type=Path, required=True)
    parser.add_argument("--set-evidence-records", type=Path, required=True)
    parser.add_argument("--b43-checkpoint", type=Path, required=True)
    parser.add_argument("--b43-summary", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def read_preregistration(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_first_run",
        "complete_transaction_decoding": True,
        "fit_only_transaction_grammar": True,
        "source_applicable_transaction_support": True,
        "support_is_action_space_not_posthoc_candidate_pool": True,
        "orthogonal_latent_particles": True,
        "particle_pool_size": 20,
        "exact_raw_attempts_per_condition": 20,
        "only_sampled_transactions_committed": True,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "retry_or_resampling": False,
        "posthoc_molecule_repair": False,
        "generation_target_access": False,
        "generation_property_oracle_access": False,
        "b26_heldout_access": False,
        "b33_fresh_source_access": False,
        "moledit_table1_benchmark_access": False,
        "official_test_access": False,
        "development_source_limit": 160,
        "fit_pair_limit": 512,
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"Atomic transaction preregistration drift: {drift}")
    if payload.get("property_counts") != [2, 3]:
        raise ValueError("Atomic transaction property-count contract drift")
    actual = belief.file_sha256(Path(__file__).resolve())
    if payload.get("implementation_sha256") != actual:
        raise ValueError(
            "Atomic transaction implementation drift: "
            f"expected {payload.get('implementation_sha256')}, found {actual}"
        )
    expected_inputs = {
        "b22_checkpoint_sha256",
        "b22_summary_sha256",
        "b36_records_sha256",
        "b41_checkpoint_sha256",
        "b41_summary_sha256",
        "b43_checkpoint_sha256",
        "b43_summary_sha256",
        "representation_checkpoint_sha256",
        "representation_summary_sha256",
        "set_evidence_records_sha256",
        "set_evidence_summary_sha256",
        "train_csv_sha256",
        "validation_csv_sha256",
    }
    if set(dict(payload.get("locked_inputs", {}))) != expected_inputs:
        raise ValueError("Atomic transaction locked-input manifest is incomplete")
    return payload


def check_locked_inputs(
    args: argparse.Namespace, preregistration: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    locked = dict(preregistration["locked_inputs"])
    paths = {
        "b22_checkpoint_sha256": args.b22_checkpoint,
        "b22_summary_sha256": args.b22_summary,
        "b36_records_sha256": args.b36_records,
        "b41_checkpoint_sha256": args.b41_checkpoint,
        "b41_summary_sha256": args.b41_summary,
        "b43_checkpoint_sha256": args.b43_checkpoint,
        "b43_summary_sha256": args.b43_summary,
        "representation_checkpoint_sha256": args.representation_checkpoint,
        "representation_summary_sha256": args.representation_summary,
        "set_evidence_records_sha256": args.set_evidence_records,
        "set_evidence_summary_sha256": args.set_evidence_summary,
        "train_csv_sha256": args.train_csv,
        "validation_csv_sha256": args.validation_csv,
    }
    drift = {
        name: {"expected": locked[name], "actual": belief.file_sha256(path)}
        for name, path in paths.items()
        if belief.file_sha256(path) != locked[name]
    }
    if drift:
        raise ValueError(f"Atomic transaction locked input drift: {drift}")

    b22_summary, b22_checkpoint, _b41_summary, _b41_checkpoint = (
        b43.check_locked_inputs(args, preregistration)
    )
    b43_summary = json.loads(args.b43_summary.read_text(encoding="utf-8"))
    if b43_summary.get("protocol") != b43.PROTOCOL:
        raise ValueError("Atomic transaction decoder requires the locked B43 protocol")
    if b43_summary.get("decision") != (
        "stop_and_diagnose_set_coverage_or_closed_support_without_gate_changes"
    ):
        raise ValueError("Atomic transaction decoder refuses B43 decision drift")
    baseline_drift = {
        key: {"expected": expected, "actual": b43_summary.get("metrics", {}).get(key)}
        for key, expected in dict(preregistration["b43_baseline"]).items()
        if not math.isclose(
            float(b43_summary.get("metrics", {}).get(key, math.nan)),
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    }
    if baseline_drift:
        raise ValueError(f"Atomic transaction B43 baseline drift: {baseline_drift}")
    b43_checkpoint = torch.load(
        args.b43_checkpoint, map_location="cpu", weights_only=False
    )
    if b43_checkpoint.get("stage") != b43.PROTOCOL:
        raise ValueError("Atomic transaction decoder refuses a non-B43 checkpoint")
    if belief.file_sha256(args.b43_checkpoint) != locked["b43_checkpoint_sha256"]:
        raise ValueError("Atomic transaction B43 checkpoint hash mismatch")
    return b22_summary, b22_checkpoint, b43_summary


def pair_row(pair: object, index: int) -> dict[str, object]:
    return {
        "condition_id": f"fit_{index:05d}",
        "external_task_key": base.task_key(pair.row),
        "source_smiles": pair.source_smiles,
        "target_smiles": pair.target_smiles,
    }


def build_fit_transform_index(
    fit_pairs: Sequence[object], preregistration: Mapping[str, object]
) -> tuple[dict[str, list[retrieved.DeltaTransform]], dict[str, object]]:
    rows = [pair_row(pair, index) for index, pair in enumerate(fit_pairs)]
    return retrieved.build_transform_index(
        rows,
        min_core_heavy_atoms=int(preregistration["min_core_heavy_atoms"]),
        max_variable_heavy_atoms=int(preregistration["max_variable_heavy_atoms"]),
    )


def _transaction_key(value: ClosedTransaction) -> tuple[object, ...]:
    return (
        value.query_core,
        value.query_variable,
        value.source_variable,
        value.target_variable,
        value.train_condition_id,
    )


def applicable_transactions(
    source_smiles: str,
    task: str,
    transforms: Mapping[str, Sequence[retrieved.DeltaTransform]],
    preregistration: Mapping[str, object],
) -> list[ClosedTransaction]:
    source = retrieved.canonical_smiles(source_smiles)
    by_product: dict[str, ClosedTransaction] = {}
    for split in retrieved.fragment_splits(
        source,
        int(preregistration["min_core_heavy_atoms"]),
        int(preregistration["max_variable_heavy_atoms"]),
    ):
        for transform in transforms.get(task, []):
            similarity = retrieved.variable_similarity(
                split.variable, transform.source_variable
            )
            if similarity < float(preregistration["min_retrieval_similarity"]):
                continue
            product = retrieved.join_fragments(split.core, transform.target_variable)
            if not product or product == source:
                continue
            transaction = ClosedTransaction(
                smiles=product,
                query_core=split.core,
                query_variable=split.variable,
                source_variable=transform.source_variable,
                target_variable=transform.target_variable,
                retrieval_similarity=float(similarity),
                transform_frequency=int(transform.frequency),
                train_condition_id=str(transform.train_condition_id),
                source_tanimoto=float(graph.morgan_tanimoto(source, product) or 0.0),
            )
            previous = by_product.get(product)
            if previous is None or _transaction_key(transaction) < _transaction_key(
                previous
            ):
                by_product[product] = transaction
    identity = ClosedTransaction(
        smiles=source,
        query_core="",
        query_variable="",
        source_variable="",
        target_variable="",
        retrieval_similarity=1.0,
        transform_frequency=0,
        train_condition_id="identity",
        source_tanimoto=1.0,
        identity=True,
    )
    return [identity, *[by_product[key] for key in sorted(by_product)]]


@lru_cache(maxsize=250000)
def fingerprint(smiles: str, bits: int) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return np.zeros(int(bits), dtype=np.float32)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=int(bits)
    )
    return np.asarray(generator.GetFingerprintAsNumPy(mol), dtype=np.float32)


def transaction_feature(
    source_smiles: str, value: ClosedTransaction, bits: int
) -> np.ndarray:
    source = fingerprint(source_smiles, bits)
    product = fingerprint(value.smiles, bits)
    query = fingerprint(value.query_variable, bits)
    transform_source = fingerprint(value.source_variable, bits)
    transform_target = fingerprint(value.target_variable, bits)
    scalars = np.asarray(
        [
            float(value.retrieval_similarity),
            min(1.0, math.log1p(value.transform_frequency) / 6.0),
            float(value.source_tanimoto),
            float(value.identity),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [source, product, np.abs(source - product), query, transform_source, transform_target, scalars]
    )


def transaction_features(
    source_smiles: str, actions: Sequence[ClosedTransaction], bits: int
) -> torch.Tensor:
    return torch.from_numpy(
        np.stack([transaction_feature(source_smiles, action, bits) for action in actions])
    )


def stable_training_subset(
    actions: Sequence[ClosedTransaction], target_smiles: str, limit: int, seed: int
) -> tuple[list[ClosedTransaction], torch.Tensor]:
    target = retrieved.canonical_smiles(target_smiles)
    positives = [action for action in actions if action.smiles == target]
    if not positives:
        return [], torch.zeros(0, dtype=torch.bool)
    negatives = [action for action in actions if action.smiles != target]
    negatives.sort(
        key=lambda action: hashlib.sha256(
            f"{seed}:{action.smiles}:{_transaction_key(action)}".encode("utf-8")
        ).hexdigest()
    )
    selected = [*positives, *negatives[: max(0, int(limit) - len(positives))]]
    selected.sort(key=lambda action: (action.smiles, _transaction_key(action)))
    mask = torch.as_tensor([action.smiles == target for action in selected])
    return selected, mask


def orthogonal_particles(count: int, dimension: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(int(dimension), int(count), generator=generator)
    return torch.linalg.qr(matrix, mode="reduced").Q.transpose(0, 1).float()


def set_transaction_loss(
    logits: torch.Tensor,
    target_mask: torch.Tensor,
    preregistration: Mapping[str, object],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    probabilities = torch.softmax(logits, dim=1)
    target_mass = probabilities[:, target_mask].sum(dim=1).clamp(1e-8, 1.0 - 1e-8)
    any_probability = 1.0 - torch.prod(1.0 - target_mass)
    any_nll = -torch.log(any_probability.clamp_min(1e-8))
    participation = -torch.log(target_mass).mean()
    normalized = F.normalize(probabilities, dim=1)
    cosine = normalized @ normalized.T
    off_diagonal = ~torch.eye(
        probabilities.shape[0], dtype=torch.bool, device=probabilities.device
    )
    diversity = cosine[off_diagonal].mean()
    entropy = -(
        probabilities * torch.log(probabilities.clamp_min(1e-8))
    ).sum(dim=1).mean()
    loss = (
        any_nll
        + float(preregistration["participation_weight"]) * participation
        + float(preregistration["diversity_weight"]) * diversity
        - float(preregistration["entropy_weight"]) * entropy
    )
    return loss, {
        "any_transaction_nll": any_nll,
        "participation_nll": participation,
        "particle_distribution_cosine": diversity,
        "distribution_entropy": entropy,
        "mean_target_mass": target_mass.mean(),
    }


def train_decoder(
    model: TransactionEnergyDecoder,
    fit_pairs: Sequence[object],
    transforms: Mapping[str, Sequence[retrieved.DeltaTransform]],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> tuple[list[dict[str, float]], dict[str, object]]:
    particles = orthogonal_particles(
        20, int(preregistration["latent_dim"]), int(preregistration["latent_seed"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(preregistration["learning_rate"]),
        weight_decay=float(preregistration["weight_decay"]),
    )
    selected_pairs = list(fit_pairs[: int(preregistration["fit_pair_limit"])])
    history: list[dict[str, float]] = []
    coverage = Counter()
    for epoch in range(1, int(preregistration["epochs"]) + 1):
        order = list(range(len(selected_pairs)))
        random.Random(int(preregistration["seed"]) + epoch).shuffle(order)
        totals: defaultdict[str, float] = defaultdict(float)
        trained = 0
        model.train()
        for pair_index in order:
            pair = selected_pairs[pair_index]
            actions = applicable_transactions(
                pair.source_smiles,
                base.task_key(pair.row),
                transforms,
                preregistration,
            )
            if epoch == 1:
                coverage["pairs"] += 1
                coverage["actions"] += len(actions) - 1
            selected, target_mask = stable_training_subset(
                actions,
                pair.target_smiles,
                int(preregistration["max_train_actions"]),
                int(preregistration["seed"]) * 100000 + pair_index,
            )
            if not selected:
                if epoch == 1:
                    coverage["target_unsupported"] += 1
                continue
            if epoch == 1:
                coverage["target_supported"] += 1
            features = transaction_features(
                pair.source_smiles, selected, int(preregistration["transaction_fingerprint_bits"])
            ).to(device)
            target_mask = target_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                features,
                particles,
                float(preregistration["latent_energy_scale"]),
            ) / float(preregistration["training_temperature"])
            loss, metrics = set_transaction_loss(logits, target_mask, preregistration)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite atomic transaction loss: {metrics}")
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), float(preregistration["grad_clip"])
            )
            optimizer.step()
            totals["loss"] += float(loss.detach())
            for name, value in metrics.items():
                totals[name] += float(value.detach())
            trained += 1
        row = {
            "epoch": epoch,
            "fit_pairs_requested": len(selected_pairs),
            "fit_pairs_trained": trained,
            **{name: value / max(1, trained) for name, value in totals.items()},
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    manifest = {
        "fit_pairs_requested": int(coverage["pairs"]),
        "fit_pairs_target_supported": int(coverage["target_supported"]),
        "fit_pairs_target_unsupported": int(coverage["target_unsupported"]),
        "fit_target_support_rate": float(
            coverage["target_supported"] / max(1, coverage["pairs"])
        ),
        "mean_applicable_fit_transactions": float(
            coverage["actions"] / max(1, coverage["pairs"])
        ),
    }
    model.eval().requires_grad_(False)
    return history, manifest


@torch.no_grad()
def freeze_candidates(
    model: TransactionEnergyDecoder,
    development_pairs: Sequence[object],
    transforms: Mapping[str, Sequence[retrieved.DeltaTransform]],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    particles = orthogonal_particles(
        20, int(preregistration["latent_dim"]), int(preregistration["latent_seed"])
    ).to(device)
    rows: list[dict[str, object]] = []
    support_counts = []
    identity_attempts = 0
    for pair_index, pair in enumerate(development_pairs):
        actions = applicable_transactions(
            pair.source_smiles,
            base.task_key(pair.row),
            transforms,
            preregistration,
        )
        support_counts.append(len(actions) - 1)
        features = transaction_features(
            pair.source_smiles, actions, int(preregistration["transaction_fingerprint_bits"])
        ).to(device)
        logits = model(
            features,
            particles,
            float(preregistration["latent_energy_scale"]),
        ) / float(preregistration["generation_temperature"])
        probabilities = torch.softmax(logits, dim=1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(preregistration["seed"]) * 100000 + pair_index)
        indices = [
            int(torch.multinomial(row.cpu(), 1, generator=generator).item())
            for row in probabilities
        ]
        condition_id = f"train_only_dev_{pair_index:04d}"
        for attempt, (particle, action_index) in enumerate(
            zip(particles, indices), start=1
        ):
            action = actions[action_index]
            identity_attempts += int(action.identity)
            rows.append(
                {
                    "condition_id": condition_id,
                    "pair_index": pair_index,
                    "attempt": attempt,
                    "property_count": int(pair.property_count),
                    "task": base.task_key(pair.row),
                    "source_smiles": pair.source_smiles,
                    "particle_index": attempt - 1,
                    "generated_smiles": action.smiles,
                    "transaction_support_size": len(actions) - 1,
                    "transaction_probability": float(
                        probabilities[attempt - 1, action_index].cpu()
                    ),
                    "query_core": action.query_core,
                    "query_variable": action.query_variable,
                    "source_variable": action.source_variable,
                    "target_variable": action.target_variable,
                    "retrieval_similarity": action.retrieval_similarity,
                    "transform_frequency": action.transform_frequency,
                    "train_condition_id": action.train_condition_id,
                    "identity_transaction": action.identity,
                    "latent_norm": float(particle.norm().cpu()),
                    "event_count": int(not action.identity),
                    "node_delete_events": 0,
                    "node_write_events": int(not action.identity),
                    "edge_delete_events": int(not action.identity),
                    "edge_set_events": int(not action.identity),
                    "affected_node_count": int(not action.identity),
                    "affected_components": int(not action.identity),
                    "outside_source_invariant": True,
                    "stopped_by_model": True,
                    "max_horizon_hit": False,
                }
            )
        if (pair_index + 1) % 16 == 0 or pair_index + 1 == len(development_pairs):
            print(
                json.dumps(
                    {
                        "stage": "freeze_atomic_closed_transactions",
                        "conditions": pair_index + 1,
                        "raw_rows": len(rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    expected = len(development_pairs) * 20
    if len(rows) != expected:
        raise RuntimeError(f"Atomic freeze expected {expected} rows, found {len(rows)}")
    support_manifest = {
        "conditions": len(development_pairs),
        "conditions_with_nonidentity_support": sum(count > 0 for count in support_counts),
        "condition_support_rate": sum(count > 0 for count in support_counts)
        / max(1, len(support_counts)),
        "mean_applicable_transactions": float(np.mean(support_counts)),
        "median_applicable_transactions": float(np.median(support_counts)),
        "min_applicable_transactions": min(support_counts, default=0),
        "max_applicable_transactions": max(support_counts, default=0),
        "identity_attempt_rate": identity_attempts / max(1, len(rows)),
    }
    return rows, support_manifest


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def gate_with_baselines(
    metrics: Mapping[str, object],
    b41_baseline: Mapping[str, object],
    b43_baseline: Mapping[str, object],
    preregistration: Mapping[str, object],
) -> dict[str, object]:
    thresholds = dict(preregistration["gates"])
    by_count = dict(metrics["by_property_count"])
    checks = {
        "exact_attempts": {"value": metrics["attempted_per_condition"], "threshold": 20},
        "validity": {"value": metrics["validity"], "threshold": thresholds["validity"]},
        "mean_unique_valid": {"value": metrics["mean_unique_valid"], "threshold": thresholds["mean_unique_valid"]},
        "mean_source_tanimoto": {"value": metrics["mean_source_tanimoto"], "threshold": thresholds["mean_source_tanimoto"]},
        "property_any20": {"value": metrics["property_any20"], "threshold": thresholds["property_any20"]},
        "strict_any20": {"value": metrics["strict_any20"], "threshold": thresholds["strict_any20"]},
        "two_property_strict_any20": {"value": by_count["2"]["strict_any20"], "threshold": thresholds["two_property_strict_any20"]},
        "three_property_strict_any20": {"value": by_count["3"]["strict_any20"], "threshold": thresholds["three_property_strict_any20"]},
        "target_improvement_any20": {"value": metrics["target_improvement_any20"], "threshold": thresholds["target_improvement_any20"]},
        "validity_delta_vs_b43": {"value": float(metrics["validity"]) - float(b43_baseline["validity"]), "threshold": thresholds["validity_delta_vs_b43"]},
        "unique_delta_vs_b43": {"value": float(metrics["mean_unique_valid"]) - float(b43_baseline["mean_unique_valid"]), "threshold": thresholds["unique_delta_vs_b43"]},
        "strict_delta_vs_b41": {"value": float(metrics["strict_any20"]) - float(b41_baseline["strict_any20"]), "threshold": thresholds["strict_delta_vs_b41"]},
    }
    failures = [
        name
        for name, check in checks.items()
        if float(check["value"]) < float(check["threshold"])
    ]
    return {"checks": checks, "failures": failures, "passed": not failures}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed atomic transaction result exists: {summary_path}")
    preregistration = read_preregistration(args.protocol_manifest)
    base.seed_everything(int(preregistration["seed"]))
    device = torch.device(str(args.device))
    if device.type != "cpu":
        raise ValueError("The preregistered atomic transaction signal is CPU-only")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    b22_summary, b22_checkpoint, b43_summary = check_locked_inputs(
        args, preregistration
    )
    selected_pairs, reconstruction = evidence.reconstruct_locked_b36_pairs(
        args, preregistration, b22_checkpoint, b22_summary
    )
    fit_pairs, development_pairs, split = b43.b41.b37.strict_source_group_split(
        selected_pairs,
        seed=int(preregistration["development_split_seed"]),
        development_source_limit=int(preregistration["development_source_limit"]),
    )
    transforms, transform_manifest = build_fit_transform_index(
        fit_pairs, preregistration
    )
    feature_dim = int(preregistration["transaction_fingerprint_bits"]) * 6 + 4
    model = TransactionEnergyDecoder(
        feature_dim,
        int(preregistration["hidden_dim"]),
        int(preregistration["latent_dim"]),
    ).to(device)
    history, training_support = train_decoder(
        model, fit_pairs, transforms, preregistration, device
    )
    checkpoint_path = args.output_dir / "atomic_closed_transaction_decoder.pt"
    torch.save(
        {
            "stage": PROTOCOL,
            "model_state": model.state_dict(),
            "model_config": {
                "feature_dim": feature_dim,
                "hidden_dim": int(preregistration["hidden_dim"]),
                "latent_dim": int(preregistration["latent_dim"]),
            },
            "transform_manifest": transform_manifest,
            "training_support": training_support,
        },
        checkpoint_path,
    )

    frozen, development_support = freeze_candidates(
        model, development_pairs, transforms, preregistration, device
    )
    frozen_path = args.output_dir / "frozen_train_only_dev_transactions.csv"
    write_rows(frozen_path, frozen)
    # The target/property evaluator is invoked only after exact-20 transactions
    # have been sampled, committed, written, and hashed.
    frozen_sha256 = belief.file_sha256(frozen_path)
    evaluated, metrics = b43.b41.b38.evaluate_frozen_candidates(
        frozen, development_pairs
    )
    evaluated_path = args.output_dir / "evaluated_train_only_dev_transactions.csv"
    write_rows(evaluated_path, evaluated)
    gate = gate_with_baselines(
        metrics,
        dict(preregistration["b41_baseline"]),
        dict(preregistration["b43_baseline"]),
        preregistration,
    )
    manifest = {
        "protocol": PROTOCOL,
        "seed": int(preregistration["seed"]),
        "device": str(device),
        "preregistration_sha256": belief.file_sha256(args.protocol_manifest),
        "implementation_sha256": belief.file_sha256(Path(__file__).resolve()),
        "locked_inputs": dict(preregistration["locked_inputs"]),
        "reconstruction": reconstruction,
        "split": split,
        "transform_manifest": transform_manifest,
        "training_support": training_support,
        "development_support": development_support,
        "complete_transaction_decoding": True,
        "fit_only_transaction_grammar": True,
        "source_applicable_transaction_support": True,
        "support_is_action_space_not_posthoc_candidate_pool": True,
        "orthogonal_latent_particles": True,
        "particle_pool_size": 20,
        "exact_raw_attempts_per_condition": 20,
        "only_sampled_transactions_committed": True,
        "frozen_before_target_or_property_evaluation": True,
        "frozen_candidates_sha256": frozen_sha256,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "retry_or_resampling": False,
        "posthoc_molecule_repair": False,
        "generation_target_access": False,
        "generation_property_oracle_access": False,
        "b26_heldout_access": False,
        "b33_fresh_source_access": False,
        "moledit_table1_benchmark_access": False,
        "moledit_table1_training_lineage": True,
        "official_test_access": False,
        "checkpoint_sha256": belief.file_sha256(checkpoint_path),
        "evaluated_candidates_sha256": belief.file_sha256(evaluated_path),
    }
    summary = {
        "protocol": PROTOCOL,
        "decision": (
            "advance_atomic_transaction_decoder_to_once_only_fresh_confirmation"
            if gate["passed"]
            else "stop_and_diagnose_transaction_support_or_latent_energy_without_gate_changes"
        ),
        "training": history,
        "b41_baseline": dict(preregistration["b41_baseline"]),
        "b43_baseline": dict(preregistration["b43_baseline"]),
        "internal_gate": gate,
        "metrics": metrics,
        "manifest": manifest,
        "checkpoint": str(checkpoint_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
