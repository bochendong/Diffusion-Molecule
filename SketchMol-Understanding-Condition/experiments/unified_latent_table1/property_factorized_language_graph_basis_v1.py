#!/usr/bin/env python3
"""Probe a property-factorized Common-LLM control basis for frozen graph jump.

The Common LLM never sees a molecule.  Its cached constraint-only hidden state
is mapped to one signed coefficient per molecular property.  Those coefficients
compose a closed-form property-token basis fitted on train-only canonical
condition tokens.  The frozen graph representation, event kernel, and transport
then test whether the composed language control improves held-out graph flow.

This is a representation kill test only.  It cannot generate or rank molecules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
LATENT_DIR = PROJECT_DIR / "experiments" / "unified_latent_flow"
for module_path in (SCRIPT_DIR, PROJECT_DIR, LATENT_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import semantic_energy_graph_jump_v1 as previous  # noqa: E402


PROTOCOL = "train_only_property_factorized_language_graph_basis_v1"
PREDECESSOR_PROTOCOL = previous.PROTOCOL
VARIANTS = ("matched", "reversed", "property_swap", "scrambled")
base = previous.base
unified = previous.unified


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", required=True, type=Path)
    parser.add_argument("--prepare-summary", required=True, type=Path)
    parser.add_argument("--fit-probe-bundle", required=True, type=Path)
    parser.add_argument("--embedding-cache", required=True, type=Path)
    parser.add_argument("--representation-checkpoint", required=True, type=Path)
    parser.add_argument("--representation-summary", required=True, type=Path)
    parser.add_argument("--canonical-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    return parser


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_preregistration(path: Path) -> dict[str, object]:
    payload = read_json(path)
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_first_run",
        "single_mechanism_change": "property_factorized_language_coefficients",
        "common_llm_prompt_contains_source": False,
        "molecule_generation": False,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "generation_target_access": False,
        "official_test_access": False,
        "fit_probe_split": "canonical_source_group_exact_condition_budget",
        "single_seed": True,
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"Property-basis preregistration drift: {drift}")
    actual = file_sha256(Path(__file__).resolve())
    if payload.get("implementation_sha256") != actual:
        raise ValueError(
            "Property-basis implementation drift: "
            f"expected {payload.get('implementation_sha256')}, found {actual}"
        )
    return payload


def check_locked_inputs(
    preregistration: Mapping[str, object], paths: Mapping[str, Path]
) -> dict[str, str]:
    locks = dict(preregistration["locked_inputs"])
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing property-basis inputs: {missing}")
    actual = {name: file_sha256(path) for name, path in paths.items()}
    drift = {
        name: {"expected": locks.get(name), "actual": digest}
        for name, digest in actual.items()
        if locks.get(name) != digest
    }
    if drift:
        raise ValueError(f"Property-basis locked-input drift: {drift}")
    return actual


def property_vector(
    specs: Sequence[tuple[str, int]], property_columns: Sequence[str]
) -> torch.Tensor:
    lookup = {str(name): index for index, name in enumerate(property_columns)}
    vector = torch.zeros(len(property_columns), dtype=torch.float32)
    for name, direction in specs:
        if str(name) not in lookup:
            raise ValueError(f"Unknown property in factorized basis: {name}")
        vector[lookup[str(name)]] = float(int(direction))
    return vector


def coefficient_targets(
    pairs: Sequence[object], property_columns: Sequence[str]
) -> dict[str, torch.Tensor]:
    rows: dict[str, list[torch.Tensor]] = {name: [] for name in VARIANTS}
    for pair in pairs:
        specs = previous.specs_for_row(pair.row)
        rows["matched"].append(property_vector(specs, property_columns))
        rows["reversed"].append(
            property_vector([(name, -direction) for name, direction in specs], property_columns)
        )
        rows["property_swap"].append(
            property_vector(previous.property_swap_specs(specs), property_columns)
        )
        rows["scrambled"].append(torch.zeros(len(property_columns), dtype=torch.float32))
    return {name: torch.stack(values) for name, values in rows.items()}


class LanguagePropertyComposer(nn.Module):
    """Map a molecule-free Common-LLM state to signed property coefficients."""

    def __init__(self, embedding_dim: int, hidden_dim: int, property_count: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(int(embedding_dim)),
            nn.Linear(int(embedding_dim), int(hidden_dim), bias=False),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(property_count)),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.network(embedding.float()))


def train_language_composer(
    composer: LanguagePropertyComposer,
    embeddings: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    train_indices: Sequence[int],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        composer.parameters(),
        lr=float(preregistration["composer_learning_rate"]),
        weight_decay=float(preregistration["composer_weight_decay"]),
    )
    history = []
    batch_size = int(preregistration["composer_batch_size"])
    examples = [(variant, index) for variant in VARIANTS for index in train_indices]
    for epoch in range(1, int(preregistration["composer_epochs"]) + 1):
        order = list(examples)
        random.Random(int(preregistration["training_seed"]) + epoch).shuffle(order)
        totals: defaultdict[str, float] = defaultdict(float)
        batches = 0
        composer.train()
        for start in range(0, len(order), batch_size):
            chosen = order[start : start + batch_size]
            batch_embedding = torch.stack(
                [embeddings[variant][index] for variant, index in chosen]
            ).to(device)
            batch_target = torch.stack(
                [targets[variant][index] for variant, index in chosen]
            ).to(device)
            prediction = composer(batch_embedding)
            active = batch_target.abs()
            active_loss = ((prediction - batch_target).square() * active).sum() / active.sum().clamp_min(1.0)
            inactive = 1.0 - active
            inactive_loss = (prediction.square() * inactive).sum() / inactive.sum().clamp_min(1.0)
            loss = active_loss + float(preregistration["inactive_loss_weight"]) * inactive_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                composer.parameters(), float(preregistration["composer_grad_clip"])
            )
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["active_mse"] += float(active_loss.detach())
            totals["inactive_mse"] += float(inactive_loss.detach())
            batches += 1
        row = {
            "epoch": epoch,
            **{name: value / max(1, batches) for name, value in totals.items()},
        }
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise FloatingPointError(f"Non-finite property-composer metrics: {row}")
        history.append(row)
        print(json.dumps({"stage": "property_composer_epoch", **row}, sort_keys=True), flush=True)
    composer.eval()
    return history


@torch.no_grad()
def coefficient_metrics(
    composer: LanguagePropertyComposer,
    embeddings: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    indices: Sequence[int],
    device: torch.device,
) -> dict[str, object]:
    selected = list(indices)
    output: dict[str, object] = {}
    predictions: dict[str, torch.Tensor] = {}
    for variant in VARIANTS:
        prediction = composer(embeddings[variant][selected].to(device)).cpu()
        target = targets[variant][selected]
        predictions[variant] = prediction
        active = target.abs() > 0
        inactive = ~active
        active_sign = (
            (torch.sign(prediction[active]) == torch.sign(target[active])).float().mean()
            if bool(active.any())
            else torch.tensor(1.0)
        )
        output[variant] = {
            "mse": float(F.mse_loss(prediction, target)),
            "active_sign_accuracy": float(active_sign),
            "inactive_abs_mean": float(prediction[inactive].abs().mean())
            if bool(inactive.any())
            else 0.0,
            "mean_abs_coefficient": float(prediction.abs().mean()),
        }
    shuffled_order = list(range(len(selected)))
    random.Random(7719).shuffle(shuffled_order)
    shuffled_prediction = composer(
        embeddings["matched"][selected][shuffled_order].to(device)
    ).cpu()
    matched_target = targets["matched"][selected]
    matched_mse = float(F.mse_loss(predictions["matched"], matched_target))
    shuffled_mse = float(F.mse_loss(shuffled_prediction, matched_target))
    output["aligned_vs_shuffled"] = {
        "aligned_mse": matched_mse,
        "shuffled_mse": shuffled_mse,
        "mse_gain": shuffled_mse - matched_mse,
    }
    return output


def fit_property_token_basis(
    pairs: Sequence[object],
    train_indices: Sequence[int],
    matched_targets: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    coefficients = matched_targets[list(train_indices)].double()
    design = torch.cat(
        [torch.ones(len(train_indices), 1, dtype=torch.float64), coefficients], dim=1
    )
    token_rows = torch.from_numpy(
        np.stack(
            [np.asarray(pairs[index].condition, dtype=np.float64) for index in train_indices]
        )
    ).reshape(len(train_indices), -1)
    gram = design.T @ design
    penalty = torch.eye(gram.shape[0], dtype=torch.float64) * float(ridge)
    penalty[0, 0] = 0.0
    weights = torch.linalg.solve(gram + penalty, design.T @ token_rows)
    return weights.float()


def compose_tokens(coefficients: torch.Tensor, basis: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    design = torch.cat(
        [torch.ones(len(coefficients), 1, device=coefficients.device), coefficients.float()],
        dim=1,
    )
    return (design @ basis.to(coefficients.device)).view(len(coefficients), *shape)


@torch.no_grad()
def token_metrics(
    composer: LanguagePropertyComposer,
    basis: torch.Tensor,
    pairs: Sequence[object],
    embeddings: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    indices: Sequence[int],
    token_shape: tuple[int, int],
    device: torch.device,
) -> dict[str, float]:
    selected = list(indices)
    canonical = torch.from_numpy(
        np.stack([np.asarray(pairs[index].condition, dtype=np.float32) for index in selected])
    ).to(device)
    predicted_coefficients = composer(embeddings["matched"][selected].to(device))
    language_tokens = compose_tokens(predicted_coefficients, basis, token_shape)
    oracle_tokens = compose_tokens(targets["matched"][selected].to(device), basis, token_shape)
    intercept_tokens = compose_tokens(
        torch.zeros(len(selected), basis.shape[0] - 1, device=device), basis, token_shape
    )
    denominator = F.mse_loss(intercept_tokens, canonical).clamp_min(1e-12)
    language_mse = F.mse_loss(language_tokens, canonical)
    oracle_mse = F.mse_loss(oracle_tokens, canonical)
    return {
        "intercept_mse": float(denominator),
        "language_mse": float(language_mse),
        "oracle_basis_mse": float(oracle_mse),
        "language_mse_ratio_vs_intercept": float(language_mse / denominator),
        "oracle_basis_mse_ratio_vs_intercept": float(oracle_mse / denominator),
    }


@torch.no_grad()
def graph_flow_metrics(
    composer: LanguagePropertyComposer,
    basis: torch.Tensor,
    model: nn.Module,
    representation: nn.Module,
    pairs: Sequence[object],
    embeddings: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    indices: Sequence[int],
    token_shape: tuple[int, int],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> dict[str, float]:
    totals: defaultdict[str, float] = defaultdict(float)
    count = 0
    batch_size = int(preregistration["probe_batch_size"])
    base.seed_everything(int(preregistration["probe_seed"]))
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    for start in range(0, len(indices), batch_size):
        chosen = list(indices[start : start + batch_size])
        collated = base.pair_collate([pairs[index] for index in chosen])
        source = base.move_graph_batch(collated["source"], device)
        target_graph = base.move_graph_batch(collated["target"], device)
        canonical_tokens = collated["condition"].to(device).float()
        matched_coefficients = composer(embeddings["matched"][chosen].to(device))
        reversed_coefficients = composer(embeddings["reversed"][chosen].to(device))
        matched_tokens = compose_tokens(matched_coefficients, basis, token_shape)
        reversed_tokens = compose_tokens(reversed_coefficients, basis, token_shape)
        oracle_tokens = compose_tokens(targets["matched"][chosen].to(device), basis, token_shape)
        intercept_tokens = compose_tokens(
            torch.zeros(len(chosen), basis.shape[0] - 1, device=device), basis, token_shape
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            source_node, source_edge = representation.encode(source)
            target_node, target_edge = representation.encode(target_graph)
            teacher_condition = model.route_condition(canonical_tokens)
            endpoint = model.posterior_endpoint(
                source,
                target_graph,
                source_node,
                source_edge,
                target_node,
                target_edge,
                teacher_condition,
            ).float()
        noise = torch.randn_like(endpoint)
        flow_time = torch.full(
            (len(chosen),), float(preregistration["probe_flow_time"]), device=device
        )
        current = (1.0 - flow_time[:, None]) * noise + flow_time[:, None] * endpoint
        target_velocity = endpoint - noise
        token_sets = {
            "canonical": canonical_tokens,
            "oracle_basis": oracle_tokens,
            "language_basis": matched_tokens,
            "reversed_language": reversed_tokens,
            "intercept": intercept_tokens,
        }
        for name, tokens in token_sets.items():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                velocity = model.transport_velocity(
                    current,
                    flow_time.to(source_node.dtype),
                    source_node,
                    source["node_mask"],
                    tokens,
                ).float()
            totals[f"{name}_flow_mse"] += float(
                F.mse_loss(velocity, target_velocity, reduction="sum")
            )
        count += int(target_velocity.numel())
    metrics = {name: value / max(1, count) for name, value in totals.items()}
    metrics["matched_flow_advantage"] = (
        metrics["reversed_language_flow_mse"] - metrics["language_basis_flow_mse"]
    )
    metrics["language_flow_ratio_vs_intercept"] = (
        metrics["language_basis_flow_mse"] / max(metrics["intercept_flow_mse"], 1e-12)
    )
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preregistration = read_preregistration(args.protocol_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed property-basis probe exists: {summary_path}")
    input_hashes = check_locked_inputs(
        preregistration,
        {
            "prepare_summary_sha256": args.prepare_summary,
            "fit_probe_bundle_sha256": args.fit_probe_bundle,
            "embedding_cache_sha256": args.embedding_cache,
            "representation_checkpoint_sha256": args.representation_checkpoint,
            "representation_summary_sha256": args.representation_summary,
            "canonical_checkpoint_sha256": args.canonical_checkpoint,
        },
    )
    prepare = read_json(args.prepare_summary)
    if int(prepare.get("fit_probe_source_overlap", -1)) != 0:
        raise ValueError("Property-basis probe requires source-disjoint fit/probe")
    bundle = torch.load(args.fit_probe_bundle, map_location="cpu", weights_only=False)
    cache = torch.load(args.embedding_cache, map_location="cpu", weights_only=False)
    if bundle.get("protocol") != PREDECESSOR_PROTOCOL or cache.get("protocol") != PREDECESSOR_PROTOCOL:
        raise ValueError("Property-basis predecessor protocol drift")
    pairs = list(bundle["pairs"])
    train_indices = list(bundle["train_indices"])
    validation_indices = list(bundle["validation_indices"])
    if len(pairs) != int(preregistration["fit_probe_conditions"]):
        raise ValueError("Property-basis pair count drift")
    if len(train_indices) != int(preregistration["fit_conditions"]):
        raise ValueError("Property-basis fit count drift")
    if len(validation_indices) != int(preregistration["probe_conditions"]):
        raise ValueError("Property-basis probe count drift")
    property_columns = [str(name) for name in unified.PROPERTY_COLUMNS]
    if len(property_columns) != int(preregistration["property_count"]):
        raise ValueError("Property vocabulary size drift")
    embeddings = {name: tensor.float() for name, tensor in dict(cache["embeddings"]).items()}
    if set(embeddings) != set(VARIANTS):
        raise ValueError(f"Embedding variant drift: {sorted(embeddings)}")
    targets = coefficient_targets(pairs, property_columns)
    device = base.resolve_device(str(args.device))
    base.seed_everything(int(preregistration["training_seed"]))
    composer = LanguagePropertyComposer(
        int(embeddings["matched"].shape[1]),
        int(preregistration["composer_hidden_dim"]),
        len(property_columns),
    ).to(device)
    history = train_language_composer(
        composer,
        embeddings,
        targets,
        train_indices,
        preregistration,
        device,
    )
    coefficients = coefficient_metrics(
        composer, embeddings, targets, validation_indices, device
    )
    token_shape = tuple(int(value) for value in np.asarray(pairs[0].condition).shape)
    expected_shape = (
        int(preregistration["token_count"]),
        int(preregistration["condition_dim"]),
    )
    if token_shape != expected_shape:
        raise ValueError(f"Property token shape drift: {token_shape}")
    basis = fit_property_token_basis(
        pairs,
        train_indices,
        targets["matched"],
        float(preregistration["basis_ridge"]),
    )
    tokens = token_metrics(
        composer,
        basis,
        pairs,
        embeddings,
        targets,
        validation_indices,
        token_shape,
        device,
    )
    model, representation, _config, _summary = previous.load_graph_stack(
        args, preregistration, bundle, device
    )
    flows = graph_flow_metrics(
        composer,
        basis,
        model,
        representation,
        pairs,
        embeddings,
        targets,
        validation_indices,
        token_shape,
        preregistration,
        device,
    )
    gates = dict(preregistration["representation_gates"])
    checks = {
        "matched_active_sign_accuracy": float(dict(coefficients["matched"])["active_sign_accuracy"])
        >= float(gates["matched_active_sign_accuracy"]),
        "reversed_active_sign_accuracy": float(dict(coefficients["reversed"])["active_sign_accuracy"])
        >= float(gates["reversed_active_sign_accuracy"]),
        "property_swap_active_sign_accuracy": float(dict(coefficients["property_swap"])["active_sign_accuracy"])
        >= float(gates["property_swap_active_sign_accuracy"]),
        "aligned_vs_shuffled_mse_gain": float(dict(coefficients["aligned_vs_shuffled"])["mse_gain"])
        >= float(gates["aligned_vs_shuffled_mse_gain"]),
        "scrambled_mean_abs_coefficient": float(dict(coefficients["scrambled"])["mean_abs_coefficient"])
        <= float(gates["scrambled_mean_abs_coefficient"]),
        "oracle_basis_mse_ratio_vs_intercept": tokens["oracle_basis_mse_ratio_vs_intercept"]
        <= float(gates["oracle_basis_mse_ratio_vs_intercept"]),
        "language_mse_ratio_vs_intercept": tokens["language_mse_ratio_vs_intercept"]
        <= float(gates["language_mse_ratio_vs_intercept"]),
        "matched_flow_advantage": flows["matched_flow_advantage"]
        >= float(gates["matched_flow_advantage"]),
        "language_flow_ratio_vs_intercept": flows["language_flow_ratio_vs_intercept"]
        <= float(gates["language_flow_ratio_vs_intercept"]),
    }
    passed = all(checks.values())
    checkpoint_path = args.output_dir / "property_factorized_language_basis.pt"
    torch.save(
        {
            "protocol": PROTOCOL,
            "composer_state_dict": composer.cpu().state_dict(),
            "basis": basis.cpu(),
            "property_columns": property_columns,
            "embedding_dim": int(embeddings["matched"].shape[1]),
            "hidden_dim": int(preregistration["composer_hidden_dim"]),
            "token_shape": token_shape,
        },
        checkpoint_path,
    )
    summary = {
        "protocol": PROTOCOL,
        "stage": "source_group_heldout_representation_probe",
        "decision": "advance_factorized_basis_to_target_isolated_generation" if passed else "stop_factorized_language_basis",
        "fit_conditions": len(train_indices),
        "probe_conditions": len(validation_indices),
        "property_columns": property_columns,
        "training": history,
        "probe_coefficients": coefficients,
        "probe_tokens": tokens,
        "probe_graph_flow": flows,
        "representation_gate": {"passed": passed, "checks": checks, "thresholds": gates},
        "artifacts": {
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "locked_inputs": input_hashes,
        },
        "contract": {
            "common_llm_prompt_contains_source": False,
            "source_group_overlap": 0,
            "fit_target_access": False,
            "probe_target_access_for_postfit_flow_diagnostic": True,
            "molecule_generation": False,
            "molecular_candidate_ranking": False,
            "oracle_selection": False,
            "official_test_access": False,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
