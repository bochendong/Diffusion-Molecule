#!/usr/bin/env python3
"""Test an anchored, contrastive Common-LLM property-coordinate interface.

This is a representation-only follow-up to the v1 factorized basis probe.  A
frozen Common LLM embeds constraint-only text and atomic property-direction
anchor prompts.  A small shared metric head predicts one signed coefficient
per property.  Training explicitly enforces direction labels, paraphrase
agreement, reversal antisymmetry, and separation from property-swapped,
scrambled, and shuffled controls.  The resulting coefficients compose the
same frozen graph-condition basis and are tested on source-group-held-out
conditions.

The experiment cannot generate, rank, repair, or evaluate molecules.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
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


PROTOCOL = "train_only_property_anchor_contrastive_graph_basis_v2"
PREDECESSOR_PROTOCOL = v1.PREDECESSOR_PROTOCOL
DIRECTIONS = (-1, 1)
TRAIN_VARIANTS = ("matched", "paraphrase", "reversed", "property_swap", "scrambled")
base = v1.base
unified = v1.unified
semantic = v1.previous


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", required=True, type=Path)
    parser.add_argument("--prepare-summary", required=True, type=Path)
    parser.add_argument("--fit-probe-bundle", required=True, type=Path)
    parser.add_argument("--embedding-cache", required=True, type=Path)
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
        "status": "preregistered_before_first_run",
        "single_mechanism_change": "anchored_contrastive_property_coefficients",
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
        raise ValueError(f"Anchor-contrastive preregistration drift: {drift}")
    actual = file_sha256(Path(__file__).resolve())
    if payload.get("implementation_sha256") != actual:
        raise ValueError(
            "Anchor-contrastive implementation drift: "
            f"expected {payload.get('implementation_sha256')}, found {actual}"
        )
    return payload


def check_locked_inputs(
    preregistration: Mapping[str, object], paths: Mapping[str, Path]
) -> dict[str, str]:
    locks = dict(preregistration["locked_inputs"])
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing anchor-contrastive inputs: {missing}")
    actual = {name: file_sha256(path) for name, path in paths.items()}
    drift = {
        name: {"expected": locks.get(name), "actual": digest}
        for name, digest in actual.items()
        if locks.get(name) != digest
    }
    if drift:
        raise ValueError(f"Anchor-contrastive locked-input drift: {drift}")
    return actual


def paraphrase_constraint(
    specs: Sequence[tuple[str, int]],
    property_names: Mapping[str, object],
    *,
    heldout: bool,
) -> str:
    """Render a compositional synonym without source or target information."""
    if heldout:
        direction = {1: "push upward", -1: "push downward"}
        prefix = "Adjust the compound so as to"
    else:
        direction = {1: "make higher", -1: "make lower"}
        prefix = "Please"
    parts = [
        f"{direction[int(sign)]} {property_names.get(prop, prop)}"
        for prop, sign in specs
    ]
    if not parts:
        raise ValueError("Paraphrase requires at least one property")
    return f"{prefix} {'; '.join(parts)}."


def atomic_anchor_texts(
    property_columns: Sequence[str], property_names: Mapping[str, object]
) -> list[str]:
    texts = []
    for prop in property_columns:
        readable = str(property_names.get(prop, prop))
        for sign in DIRECTIONS:
            verb = "increase" if sign > 0 else "decrease"
            schema_direction = "positive" if sign > 0 else "negative"
            texts.extend(
                [
                    f"Modify the molecule to {verb} {readable}.",
                    f"Property: {readable}. Requested change: {schema_direction}.",
                ]
            )
    return texts


@torch.no_grad()
def load_supplementary_embeddings(
    pairs: Sequence[object],
    validation_indices: set[int],
    property_columns: Sequence[str],
    property_names: Mapping[str, object],
    preregistration: Mapping[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    paraphrases = []
    for index, pair in enumerate(pairs):
        paraphrases.append(
            paraphrase_constraint(
                semantic.specs_for_row(pair.row),
                property_names,
                heldout=index in validation_indices,
            )
        )
    anchor_text = atomic_anchor_texts(property_columns, property_names)
    llm_args = SimpleNamespace(sft_adapter_dir=args.sft_adapter_dir)
    llm, tokenizer = semantic.operator.load_common_llm(
        llm_args, preregistration, device, sft=True, latent_lora=False
    )
    try:
        paraphrase_embeddings = semantic.embed_texts(
            llm, tokenizer, paraphrases, preregistration, device
        )
        flat_anchor_embeddings = semantic.embed_texts(
            llm, tokenizer, anchor_text, preregistration, device
        )
    finally:
        del llm
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    anchor_embeddings = flat_anchor_embeddings.view(
        len(property_columns), len(DIRECTIONS), 2, -1
    ).mean(dim=2)
    prompt_payload = json.dumps(
        {"paraphrases": paraphrases, "anchors": anchor_text},
        sort_keys=True,
        separators=(",", ":"),
    )
    manifest = {
        "prompt_sha256": hashlib.sha256(prompt_payload.encode("utf-8")).hexdigest(),
        "prompt_contains_source": False,
        "paraphrase_count": len(paraphrases),
        "atomic_anchor_count": len(anchor_text),
        "anchor_templates_per_direction": 2,
        "embedding_dim": int(paraphrase_embeddings.shape[1]),
    }
    return paraphrase_embeddings, anchor_embeddings, manifest


def coefficient_targets(
    pairs: Sequence[object], property_columns: Sequence[str]
) -> dict[str, torch.Tensor]:
    targets = v1.coefficient_targets(pairs, property_columns)
    targets["paraphrase"] = targets["matched"].clone()
    return targets


def direction_labels(coefficients: torch.Tensor) -> torch.Tensor:
    """Return [property, decrease/increase] multi-label targets."""
    return torch.stack(
        [(coefficients < 0).float(), (coefficients > 0).float()], dim=-1
    )


class AnchoredPropertyComposer(nn.Module):
    """Predict signed coordinates by matching text against atomic anchors."""

    def __init__(
        self,
        anchor_embeddings: torch.Tensor,
        hidden_dim: int,
        temperature: float,
        inactive_logit_bias: float,
    ) -> None:
        super().__init__()
        if anchor_embeddings.ndim != 3 or anchor_embeddings.shape[1] != 2:
            raise ValueError(f"Unexpected anchor shape: {tuple(anchor_embeddings.shape)}")
        embedding_dim = int(anchor_embeddings.shape[-1])
        self.register_buffer("anchor_embeddings", anchor_embeddings.float())
        self.query_projector = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, int(hidden_dim), bias=False),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim), bias=False),
        )
        self.anchor_projector = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, int(hidden_dim), bias=False),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim), bias=False),
        )
        self.logit_bias = nn.Parameter(
            torch.full(
                (int(anchor_embeddings.shape[0]), 2),
                float(inactive_logit_bias),
            )
        )
        self.temperature = float(temperature)

    def logits(self, embedding: torch.Tensor) -> torch.Tensor:
        query = F.normalize(self.query_projector(embedding.float()), dim=-1)
        anchors = F.normalize(
            self.anchor_projector(self.anchor_embeddings), dim=-1
        )
        return torch.einsum("bd,psd->bps", query, anchors) / self.temperature + self.logit_bias

    def forward_with_logits(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.logits(embedding)
        probabilities = torch.sigmoid(logits)
        coefficients = probabilities[..., 1] - probabilities[..., 0]
        return coefficients, logits

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.forward_with_logits(embedding)[0]


def per_row_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).square().mean(dim=-1)


def train_composer(
    composer: AnchoredPropertyComposer,
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
    selected = list(train_indices)
    batch_size = int(preregistration["composer_batch_size"])
    history = []
    for epoch in range(1, int(preregistration["composer_epochs"]) + 1):
        order = list(selected)
        random.Random(int(preregistration["training_seed"]) + epoch).shuffle(order)
        totals: defaultdict[str, float] = defaultdict(float)
        batches = 0
        composer.train()
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            predictions: dict[str, torch.Tensor] = {}
            losses: defaultdict[str, torch.Tensor] = defaultdict(
                lambda: torch.zeros((), device=device)
            )
            for variant in TRAIN_VARIANTS:
                target = targets[variant][indices].to(device)
                prediction, logits = composer.forward_with_logits(
                    embeddings[variant][indices].to(device)
                )
                predictions[variant] = prediction
                active = target.abs()
                inactive = 1.0 - active
                losses["coefficient"] = losses["coefficient"] + (
                    ((prediction - target).square() * active).sum()
                    / active.sum().clamp_min(1.0)
                    + float(preregistration["inactive_loss_weight"])
                    * (prediction.square() * inactive).sum()
                    / inactive.sum().clamp_min(1.0)
                )
                labels = direction_labels(target)
                bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
                weight = 1.0 + float(preregistration["positive_direction_weight"]) * labels
                losses["direction_bce"] = losses["direction_bce"] + (bce * weight).mean()
            divisor = float(len(TRAIN_VARIANTS))
            losses["coefficient"] = losses["coefficient"] / divisor
            losses["direction_bce"] = losses["direction_bce"] / divisor
            losses["paraphrase_consistency"] = F.mse_loss(
                predictions["matched"], predictions["paraphrase"]
            )
            losses["reversal_antisymmetry"] = (
                predictions["matched"] + predictions["reversed"]
            ).square().mean()
            matched_target = targets["matched"][indices].to(device)
            correct = per_row_mse(predictions["matched"], matched_target)
            negative_mse = torch.stack(
                [
                    per_row_mse(
                        predictions["matched"], targets["reversed"][indices].to(device)
                    ),
                    per_row_mse(
                        predictions["matched"], targets["property_swap"][indices].to(device)
                    ),
                    per_row_mse(
                        predictions["matched"],
                        matched_target[torch.arange(len(indices) - 1, -1, -1, device=device)],
                    ),
                ],
                dim=-1,
            )
            losses["contrastive_margin"] = F.relu(
                float(preregistration["contrastive_mse_margin"])
                + correct[:, None]
                - negative_mse
            ).mean()
            loss = (
                float(preregistration["coefficient_loss_weight"]) * losses["coefficient"]
                + float(preregistration["direction_bce_weight"]) * losses["direction_bce"]
                + float(preregistration["paraphrase_consistency_weight"])
                * losses["paraphrase_consistency"]
                + float(preregistration["reversal_antisymmetry_weight"])
                * losses["reversal_antisymmetry"]
                + float(preregistration["contrastive_loss_weight"])
                * losses["contrastive_margin"]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                composer.parameters(), float(preregistration["composer_grad_clip"])
            )
            optimizer.step()
            totals["loss"] += float(loss.detach())
            for name, value in losses.items():
                totals[name] += float(value.detach())
            batches += 1
        row = {
            "epoch": epoch,
            **{name: value / max(1, batches) for name, value in totals.items()},
        }
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise FloatingPointError(f"Non-finite anchor-composer metrics: {row}")
        history.append(row)
        print(json.dumps({"stage": "anchor_composer_epoch", **row}, sort_keys=True), flush=True)
    composer.eval()
    return history


@torch.no_grad()
def coefficient_metrics(
    composer: AnchoredPropertyComposer,
    embeddings: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    indices: Sequence[int],
    device: torch.device,
) -> dict[str, object]:
    selected = list(indices)
    output: dict[str, object] = {}
    predictions: dict[str, torch.Tensor] = {}
    for variant in TRAIN_VARIANTS:
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
    matched_target = targets["matched"][selected]
    aligned_mse = float(F.mse_loss(predictions["matched"], matched_target))
    shuffled_mse = float(
        F.mse_loss(predictions["matched"][shuffled_order], matched_target)
    )
    output["aligned_vs_shuffled"] = {
        "aligned_mse": aligned_mse,
        "shuffled_mse": shuffled_mse,
        "mse_gain": shuffled_mse - aligned_mse,
    }
    output["paraphrase_consistency_mse"] = float(
        F.mse_loss(predictions["matched"], predictions["paraphrase"])
    )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preregistration = read_preregistration(args.protocol_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed anchor-contrastive probe exists: {summary_path}")
    input_hashes = check_locked_inputs(
        preregistration,
        {
            "prepare_summary_sha256": args.prepare_summary,
            "fit_probe_bundle_sha256": args.fit_probe_bundle,
            "embedding_cache_sha256": args.embedding_cache,
            "representation_checkpoint_sha256": args.representation_checkpoint,
            "representation_summary_sha256": args.representation_summary,
            "canonical_checkpoint_sha256": args.canonical_checkpoint,
            "sft_adapter_config_sha256": args.sft_adapter_dir / "adapter_config.json",
            "sft_adapter_model_sha256": args.sft_adapter_dir / "adapter_model.safetensors",
            "e1_manifest_sha256": args.e1_manifest,
        },
    )
    prepare = read_json(args.prepare_summary)
    if int(prepare.get("fit_probe_source_overlap", -1)) != 0:
        raise ValueError("Anchor-contrastive probe requires source-disjoint fit/probe")
    bundle = torch.load(args.fit_probe_bundle, map_location="cpu", weights_only=False)
    cache = torch.load(args.embedding_cache, map_location="cpu", weights_only=False)
    if bundle.get("protocol") != PREDECESSOR_PROTOCOL or cache.get("protocol") != PREDECESSOR_PROTOCOL:
        raise ValueError("Anchor-contrastive predecessor protocol drift")
    pairs = list(bundle["pairs"])
    train_indices = list(bundle["train_indices"])
    validation_indices = list(bundle["validation_indices"])
    if len(pairs) != int(preregistration["fit_probe_conditions"]):
        raise ValueError("Anchor-contrastive pair count drift")
    if len(train_indices) != int(preregistration["fit_conditions"]):
        raise ValueError("Anchor-contrastive fit count drift")
    if len(validation_indices) != int(preregistration["probe_conditions"]):
        raise ValueError("Anchor-contrastive probe count drift")
    property_columns = [str(name) for name in unified.PROPERTY_COLUMNS]
    if len(property_columns) != int(preregistration["property_count"]):
        raise ValueError("Anchor-contrastive property vocabulary drift")
    original_embeddings = {
        name: tensor.float() for name, tensor in dict(cache["embeddings"]).items()
    }
    if set(original_embeddings) != set(v1.VARIANTS):
        raise ValueError(f"Embedding variant drift: {sorted(original_embeddings)}")
    device = base.resolve_device(str(args.device))
    e1 = read_json(args.e1_manifest)
    paraphrase_embeddings, anchor_embeddings, prompt_manifest = load_supplementary_embeddings(
        pairs,
        set(validation_indices),
        property_columns,
        dict(e1["property_names"]),
        preregistration,
        args,
        device,
    )
    embeddings = {**original_embeddings, "paraphrase": paraphrase_embeddings.float()}
    if any(int(tensor.shape[0]) != len(pairs) for tensor in embeddings.values()):
        raise ValueError("Anchor-contrastive embedding row drift")
    targets = coefficient_targets(pairs, property_columns)
    base.seed_everything(int(preregistration["training_seed"]))
    composer = AnchoredPropertyComposer(
        anchor_embeddings,
        int(preregistration["composer_hidden_dim"]),
        float(preregistration["anchor_temperature"]),
        float(preregistration["inactive_logit_bias"]),
    ).to(device)
    history = train_composer(
        composer, embeddings, targets, train_indices, preregistration, device
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
        raise ValueError(f"Anchor-contrastive token shape drift: {token_shape}")
    basis = v1.fit_property_token_basis(
        pairs,
        train_indices,
        targets["matched"],
        float(preregistration["basis_ridge"]),
    )
    tokens = v1.token_metrics(
        composer,
        basis,
        pairs,
        embeddings,
        targets,
        validation_indices,
        token_shape,
        device,
    )
    model, representation, _config, _summary = semantic.load_graph_stack(
        args, preregistration, bundle, device
    )
    flows = v1.graph_flow_metrics(
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
        "paraphrase_active_sign_accuracy": float(dict(coefficients["paraphrase"])["active_sign_accuracy"])
        >= float(gates["paraphrase_active_sign_accuracy"]),
        "reversed_active_sign_accuracy": float(dict(coefficients["reversed"])["active_sign_accuracy"])
        >= float(gates["reversed_active_sign_accuracy"]),
        "property_swap_active_sign_accuracy": float(dict(coefficients["property_swap"])["active_sign_accuracy"])
        >= float(gates["property_swap_active_sign_accuracy"]),
        "aligned_vs_shuffled_mse_gain": float(dict(coefficients["aligned_vs_shuffled"])["mse_gain"])
        >= float(gates["aligned_vs_shuffled_mse_gain"]),
        "scrambled_mean_abs_coefficient": float(dict(coefficients["scrambled"])["mean_abs_coefficient"])
        <= float(gates["scrambled_mean_abs_coefficient"]),
        "paraphrase_consistency_mse": float(coefficients["paraphrase_consistency_mse"])
        <= float(gates["paraphrase_consistency_mse"]),
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
    checkpoint_path = args.output_dir / "property_anchor_contrastive_basis.pt"
    torch.save(
        {
            "protocol": PROTOCOL,
            "composer_state_dict": composer.cpu().state_dict(),
            "basis": basis.cpu(),
            "property_columns": property_columns,
            "embedding_dim": int(embeddings["matched"].shape[1]),
            "hidden_dim": int(preregistration["composer_hidden_dim"]),
            "token_shape": token_shape,
            "prompt_manifest": prompt_manifest,
        },
        checkpoint_path,
    )
    summary = {
        "protocol": PROTOCOL,
        "stage": "source_group_heldout_anchor_contrastive_representation_probe",
        "decision": (
            "advance_anchor_contrastive_basis_to_target_isolated_generation"
            if passed
            else "stop_anchor_contrastive_language_basis"
        ),
        "fit_conditions": len(train_indices),
        "probe_conditions": len(validation_indices),
        "property_columns": property_columns,
        "training": history,
        "probe_coefficients": coefficients,
        "probe_tokens": tokens,
        "probe_graph_flow": flows,
        "representation_gate": {
            "passed": passed,
            "checks": checks,
            "thresholds": gates,
        },
        "artifacts": {
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "locked_inputs": input_hashes,
            "prompt_manifest": prompt_manifest,
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
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
