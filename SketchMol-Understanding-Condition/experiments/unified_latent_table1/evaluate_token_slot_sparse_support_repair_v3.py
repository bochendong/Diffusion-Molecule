#!/usr/bin/env python3
"""Re-evaluate frozen v3 with exact-zero inactive property-slot support.

The v3 language compiler passed every semantic gate, but its graph-flow probe
fed dense floating-point basis reconstructions into a graph model whose active
slot contract is a strict nonzero test.  Ridge leakage therefore activated all
17 slots even when oracle tokens were numerically equal to canonical tokens.

This evaluation-only repair freezes the trained LoRA and slot decoder.  It uses
the preregistered binary direction heads' standard probability threshold 0.5
to determine support, sets inactive coefficients and property slots exactly to
zero, and recomputes the token/flow diagnostics once.  It does not train,
generate, rank, repair, or evaluate molecules.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
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

import property_factorized_language_graph_basis_v1 as v1  # noqa: E402
import token_slot_lora_property_compiler_v3 as v3  # noqa: E402


PROTOCOL = "frozen_token_slot_sparse_support_repair_v3"
base = v3.base
semantic = v3.semantic
unified = v3.unified


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", required=True, type=Path)
    parser.add_argument("--v3-manifest", required=True, type=Path)
    parser.add_argument("--v3-summary", required=True, type=Path)
    parser.add_argument("--decoder-checkpoint", required=True, type=Path)
    parser.add_argument("--lora-adapter-dir", required=True, type=Path)
    parser.add_argument("--prepare-summary", required=True, type=Path)
    parser.add_argument("--fit-probe-bundle", required=True, type=Path)
    parser.add_argument("--representation-checkpoint", required=True, type=Path)
    parser.add_argument("--representation-summary", required=True, type=Path)
    parser.add_argument("--canonical-checkpoint", required=True, type=Path)
    parser.add_argument("--sft-adapter-dir", required=True, type=Path)
    parser.add_argument("--e1-manifest", required=True, type=Path)
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
        "status": "preregistered_before_first_repair_evaluation",
        "single_mechanism_change": "exact_zero_inactive_property_slots",
        "training": False,
        "support_probability_threshold": 0.5,
        "threshold_search": False,
        "molecule_generation": False,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "official_test_access": False,
        "single_seed": True,
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"Sparse-support repair preregistration drift: {drift}")
    actual = file_sha256(Path(__file__).resolve())
    if payload.get("implementation_sha256") != actual:
        raise ValueError(
            "Sparse-support implementation drift: "
            f"expected {payload.get('implementation_sha256')}, found {actual}"
        )
    return payload


def check_locked_inputs(
    preregistration: Mapping[str, object], paths: Mapping[str, Path]
) -> dict[str, str]:
    locks = dict(preregistration["locked_inputs"])
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing sparse-support inputs: {missing}")
    actual = {name: file_sha256(path) for name, path in paths.items()}
    drift = {
        name: {"expected": locks.get(name), "actual": digest}
        for name, digest in actual.items()
        if locks.get(name) != digest
    }
    if drift:
        raise ValueError(f"Sparse-support locked-input drift: {drift}")
    return actual


def load_frozen_compiler(
    args: argparse.Namespace,
    v3_manifest: Mapping[str, object],
    device: torch.device,
) -> tuple[object, object, v3.TokenPropertySlotDecoder]:
    try:
        import peft
    except ImportError as exc:
        raise RuntimeError(f"Missing PEFT for sparse-support evaluation: {exc}") from exc
    llm_args = SimpleNamespace(sft_adapter_dir=args.sft_adapter_dir)
    model, tokenizer = semantic.operator.load_common_llm(
        llm_args, v3_manifest, device, sft=True, latent_lora=False
    )
    model = model.merge_and_unload()
    model = peft.PeftModel.from_pretrained(
        model, args.lora_adapter_dir, is_trainable=False
    ).to(device)
    model.eval().requires_grad_(False)
    checkpoint = torch.load(
        args.decoder_checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint.get("protocol") != v3.PROTOCOL:
        raise ValueError("Frozen token-slot decoder protocol drift")
    decoder = v3.TokenPropertySlotDecoder(
        int(checkpoint["llm_hidden_dim"]),
        int(checkpoint["slot_dim"]),
        len(checkpoint["property_columns"]),
    ).to(device)
    decoder.load_state_dict(dict(checkpoint["state_dict"]), strict=True)
    decoder.eval().requires_grad_(False)
    return model, tokenizer, decoder


@torch.no_grad()
def predict_with_support(
    model: object,
    decoder: v3.TokenPropertySlotDecoder,
    tokenizer: object,
    examples: Sequence[v3.TextExample],
    v3_manifest: Mapping[str, object],
    threshold: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokenized = v3.tokenize_examples(
        tokenizer,
        examples,
        int(v3_manifest["property_count"]),
        int(v3_manifest["llm_max_length"]),
    )
    raw_rows = []
    sparse_rows = []
    support_rows = []
    batch_size = int(v3_manifest["probe_batch_size"])
    for start in range(0, len(examples), batch_size):
        raw, logits, _attention = v3.model_forward(
            model,
            decoder,
            tokenized["input_ids"][start : start + batch_size].to(device),
            tokenized["attention_mask"][start : start + batch_size].to(device),
            device,
        )
        support = torch.sigmoid(logits).amax(dim=-1).ge(float(threshold))
        sparse = torch.where(support, raw, torch.zeros_like(raw))
        raw_rows.append(raw.cpu())
        sparse_rows.append(sparse.cpu())
        support_rows.append(support.cpu())
    return (
        torch.cat(raw_rows, dim=0),
        torch.cat(sparse_rows, dim=0),
        torch.cat(support_rows, dim=0),
    )


def support_metrics(
    support: torch.Tensor, examples: Sequence[v3.TextExample]
) -> dict[str, float]:
    target = torch.stack([example.target for example in examples]).ne(0)
    true_positive = int((support & target).sum())
    false_positive = int((support & ~target).sum())
    false_negative = int((~support & target).sum())
    return {
        "precision": true_positive / max(1, true_positive + false_positive),
        "recall": true_positive / max(1, true_positive + false_negative),
        "exact_support_rate": float(support.eq(target).all(dim=-1).float().mean()),
        "mean_predicted_active_slots": float(support.float().sum(dim=-1).mean()),
        "mean_target_active_slots": float(target.float().sum(dim=-1).mean()),
    }


def compose_sparse_tokens(
    coefficients: torch.Tensor,
    basis: torch.Tensor,
    shape: tuple[int, int],
) -> torch.Tensor:
    tokens = v1.compose_tokens(coefficients, basis, shape)
    if tokens.shape[1] != coefficients.shape[1] + 1:
        raise ValueError("Sparse-support token/property shape mismatch")
    property_support = coefficients.ne(0).unsqueeze(-1)
    tokens = tokens.clone()
    tokens[:, 1:, :] = torch.where(
        property_support, tokens[:, 1:, :], torch.zeros_like(tokens[:, 1:, :])
    )
    return tokens


@torch.no_grad()
def token_metrics(
    basis: torch.Tensor,
    pairs: Sequence[object],
    validation_indices: Sequence[int],
    matched_prediction: torch.Tensor,
    matched_target: torch.Tensor,
    token_shape: tuple[int, int],
    device: torch.device,
) -> dict[str, float]:
    canonical = torch.from_numpy(
        np.stack(
            [np.asarray(pairs[index].condition, dtype=np.float32) for index in validation_indices]
        )
    ).to(device)
    language_tokens = compose_sparse_tokens(
        matched_prediction.to(device), basis, token_shape
    )
    oracle_tokens = compose_sparse_tokens(matched_target.to(device), basis, token_shape)
    intercept_tokens = compose_sparse_tokens(
        torch.zeros(len(validation_indices), basis.shape[0] - 1, device=device),
        basis,
        token_shape,
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
        "oracle_canonical_max_abs": float((oracle_tokens - canonical).abs().max()),
    }


@torch.no_grad()
def graph_flow_metrics(
    graph_model: nn.Module,
    representation: nn.Module,
    basis: torch.Tensor,
    pairs: Sequence[object],
    validation_indices: Sequence[int],
    matched_prediction: torch.Tensor,
    reversed_prediction: torch.Tensor,
    matched_target: torch.Tensor,
    token_shape: tuple[int, int],
    v3_manifest: Mapping[str, object],
    device: torch.device,
) -> dict[str, float]:
    totals: defaultdict[str, float] = defaultdict(float)
    count = 0
    batch_size = int(v3_manifest["graph_probe_batch_size"])
    base.seed_everything(int(v3_manifest["graph_probe_seed"]))
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    max_oracle_token_delta = 0.0
    max_oracle_velocity_delta = 0.0
    for start in range(0, len(validation_indices), batch_size):
        chosen = list(validation_indices[start : start + batch_size])
        local = slice(start, start + len(chosen))
        collated = base.pair_collate([pairs[index] for index in chosen])
        source = base.move_graph_batch(collated["source"], device)
        target_graph = base.move_graph_batch(collated["target"], device)
        canonical_tokens = collated["condition"].to(device).float()
        matched_tokens = compose_sparse_tokens(
            matched_prediction[local].to(device), basis, token_shape
        )
        reversed_tokens = compose_sparse_tokens(
            reversed_prediction[local].to(device), basis, token_shape
        )
        oracle_tokens = compose_sparse_tokens(
            matched_target[local].to(device), basis, token_shape
        )
        intercept_tokens = compose_sparse_tokens(
            torch.zeros(len(chosen), basis.shape[0] - 1, device=device),
            basis,
            token_shape,
        )
        max_oracle_token_delta = max(
            max_oracle_token_delta,
            float((oracle_tokens - canonical_tokens).abs().max()),
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            source_node, source_edge = representation.encode(source)
            target_node, target_edge = representation.encode(target_graph)
            teacher_condition = graph_model.route_condition(canonical_tokens)
            endpoint = graph_model.posterior_endpoint(
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
            (len(chosen),), float(v3_manifest["probe_flow_time"]), device=device
        )
        current = (1.0 - flow_time[:, None]) * noise + flow_time[:, None] * endpoint
        target_velocity = endpoint - noise
        velocities = {}
        for name, tokens in {
            "canonical": canonical_tokens,
            "oracle_basis": oracle_tokens,
            "language_basis": matched_tokens,
            "reversed_language": reversed_tokens,
            "intercept": intercept_tokens,
        }.items():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                velocity = graph_model.transport_velocity(
                    current,
                    flow_time.to(source_node.dtype),
                    source_node,
                    source["node_mask"],
                    tokens,
                ).float()
            velocities[name] = velocity
            totals[f"{name}_flow_mse"] += float(
                F.mse_loss(velocity, target_velocity, reduction="sum")
            )
        max_oracle_velocity_delta = max(
            max_oracle_velocity_delta,
            float((velocities["oracle_basis"] - velocities["canonical"]).abs().max()),
        )
        count += int(target_velocity.numel())
    metrics = {name: value / max(1, count) for name, value in totals.items()}
    metrics["matched_flow_advantage"] = (
        metrics["reversed_language_flow_mse"] - metrics["language_basis_flow_mse"]
    )
    metrics["language_flow_ratio_vs_intercept"] = (
        metrics["language_basis_flow_mse"] / max(metrics["intercept_flow_mse"], 1e-12)
    )
    metrics["oracle_canonical_token_max_abs"] = max_oracle_token_delta
    metrics["oracle_canonical_velocity_max_abs"] = max_oracle_velocity_delta
    metrics["oracle_canonical_flow_mse_abs_delta"] = abs(
        metrics["oracle_basis_flow_mse"] - metrics["canonical_flow_mse"]
    )
    metrics["oracle_canonical_flow_mse_ratio"] = (
        metrics["oracle_basis_flow_mse"]
        / max(metrics["canonical_flow_mse"], 1e-12)
    )
    metrics["oracle_canonical_flow_relative_error"] = (
        metrics["oracle_canonical_flow_mse_abs_delta"]
        / max(metrics["canonical_flow_mse"], 1e-12)
    )
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preregistration = read_preregistration(args.protocol_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed sparse-support evaluation exists: {summary_path}")
    input_hashes = check_locked_inputs(
        preregistration,
        {
            "v3_manifest_sha256": args.v3_manifest,
            "v3_summary_sha256": args.v3_summary,
            "decoder_checkpoint_sha256": args.decoder_checkpoint,
            "lora_adapter_config_sha256": args.lora_adapter_dir / "adapter_config.json",
            "lora_adapter_model_sha256": args.lora_adapter_dir / "adapter_model.safetensors",
            "prepare_summary_sha256": args.prepare_summary,
            "fit_probe_bundle_sha256": args.fit_probe_bundle,
            "representation_checkpoint_sha256": args.representation_checkpoint,
            "representation_summary_sha256": args.representation_summary,
            "canonical_checkpoint_sha256": args.canonical_checkpoint,
            "sft_adapter_config_sha256": args.sft_adapter_dir / "adapter_config.json",
            "sft_adapter_model_sha256": args.sft_adapter_dir / "adapter_model.safetensors",
            "e1_manifest_sha256": args.e1_manifest,
        },
    )
    original = read_json(args.v3_summary)
    if original.get("protocol") != v3.PROTOCOL:
        raise ValueError("Sparse-support repair requires the locked v3 summary")
    original_checks = dict(dict(original["representation_gate"])["checks"])
    nonflow_checks = {
        key: bool(value)
        for key, value in original_checks.items()
        if key != "matched_flow_advantage"
    }
    if not all(nonflow_checks.values()):
        raise ValueError(f"Sparse-support repair cannot bypass failed language gates: {nonflow_checks}")
    v3_manifest = read_json(args.v3_manifest)
    bundle = torch.load(args.fit_probe_bundle, map_location="cpu", weights_only=False)
    pairs = list(bundle["pairs"])
    train_indices = list(bundle["train_indices"])
    validation_indices = list(bundle["validation_indices"])
    property_columns = [str(name) for name in unified.PROPERTY_COLUMNS]
    e1 = read_json(args.e1_manifest)
    graph_examples = v3.graph_probe_examples(
        pairs,
        validation_indices,
        property_columns,
        dict(e1["property_names"]),
        int(v3_manifest["scramble_seed"]),
    )
    device = base.resolve_device(str(args.device))
    model, tokenizer, decoder = load_frozen_compiler(args, v3_manifest, device)
    raw_predictions = {}
    sparse_predictions = {}
    supports = {}
    threshold = float(preregistration["support_probability_threshold"])
    for variant, examples in graph_examples.items():
        raw, sparse, support = predict_with_support(
            model,
            decoder,
            tokenizer,
            examples,
            v3_manifest,
            threshold,
            device,
        )
        raw_predictions[variant] = raw
        sparse_predictions[variant] = sparse
        supports[variant] = support
    support_report = {
        variant: support_metrics(supports[variant], graph_examples[variant])
        for variant in graph_examples
    }
    del model, decoder
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    targets = v1.coefficient_targets(pairs, property_columns)["matched"]
    basis = v1.fit_property_token_basis(
        pairs,
        train_indices,
        targets,
        float(v3_manifest["basis_ridge"]),
    )
    token_shape = tuple(int(value) for value in np.asarray(pairs[0].condition).shape)
    matched_target = torch.stack(
        [example.target for example in graph_examples["matched"]]
    )
    tokens = token_metrics(
        basis,
        pairs,
        validation_indices,
        sparse_predictions["matched"],
        matched_target,
        token_shape,
        device,
    )
    graph_model, representation, _config, _summary = semantic.load_graph_stack(
        args, v3_manifest, bundle, device
    )
    flows = graph_flow_metrics(
        graph_model,
        representation,
        basis,
        pairs,
        validation_indices,
        sparse_predictions["matched"],
        sparse_predictions["reversed"],
        matched_target,
        token_shape,
        v3_manifest,
        device,
    )
    gates = dict(preregistration["repair_gates"])
    checks = {
        "original_nonflow_gates_passed": all(nonflow_checks.values()),
        "matched_support_precision": support_report["matched"]["precision"]
        >= float(gates["matched_support_precision"]),
        "matched_support_recall": support_report["matched"]["recall"]
        >= float(gates["matched_support_recall"]),
        "reversed_support_precision": support_report["reversed"]["precision"]
        >= float(gates["reversed_support_precision"]),
        "reversed_support_recall": support_report["reversed"]["recall"]
        >= float(gates["reversed_support_recall"]),
        "oracle_basis_mse_ratio_vs_intercept": tokens["oracle_basis_mse_ratio_vs_intercept"]
        <= float(gates["oracle_basis_mse_ratio_vs_intercept"]),
        "oracle_canonical_velocity_max_abs": flows["oracle_canonical_velocity_max_abs"]
        <= float(gates["oracle_canonical_velocity_max_abs"]),
        "oracle_canonical_flow_relative_error": flows[
            "oracle_canonical_flow_relative_error"
        ]
        <= float(gates["oracle_canonical_flow_relative_error"]),
        "language_mse_ratio_vs_intercept": tokens["language_mse_ratio_vs_intercept"]
        <= float(gates["language_mse_ratio_vs_intercept"]),
        "matched_flow_advantage": flows["matched_flow_advantage"]
        >= float(gates["matched_flow_advantage"]),
        "language_flow_ratio_vs_intercept": flows["language_flow_ratio_vs_intercept"]
        <= float(gates["language_flow_ratio_vs_intercept"]),
    }
    passed = all(checks.values())
    summary = {
        "protocol": PROTOCOL,
        "stage": "frozen_sparse_support_graph_flow_repair",
        "decision": (
            "advance_frozen_token_slot_compiler_to_target_isolated_generation"
            if passed
            else "stop_after_sparse_support_repair"
        ),
        "original_v3_decision": original["decision"],
        "support_probability_threshold": threshold,
        "support_metrics": support_report,
        "raw_coefficient_metrics": {
            variant: v3.coefficient_summary(
                raw_predictions[variant],
                torch.stack([example.target for example in graph_examples[variant]]),
            )
            for variant in graph_examples
        },
        "sparse_coefficient_metrics": {
            variant: v3.coefficient_summary(
                sparse_predictions[variant],
                torch.stack([example.target for example in graph_examples[variant]]),
            )
            for variant in graph_examples
        },
        "corrected_tokens": tokens,
        "corrected_flow": flows,
        "repair_gate": {"passed": passed, "checks": checks, "thresholds": gates},
        "artifacts": {"locked_inputs": input_hashes},
        "contract": {
            "training": False,
            "frozen_v3_lora": True,
            "frozen_v3_decoder": True,
            "support_probability_threshold": threshold,
            "threshold_search": False,
            "common_llm_prompt_contains_source": False,
            "molecule_generation": False,
            "molecular_candidate_ranking": False,
            "oracle_selection": False,
            "official_test_access": False,
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
