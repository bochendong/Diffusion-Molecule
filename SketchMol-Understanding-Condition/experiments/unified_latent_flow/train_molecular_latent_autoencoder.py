#!/usr/bin/env python3
"""Train and gate the molecular latent representation before fitting any flow.

This is deliberately a representation-only stage. It sees a molecule, encodes
it into a fixed set of continuous latent tokens, and decodes those tokens back
to SMILES. No property condition, candidate library, oracle, selector, or
benchmark target is available to the model. A later conditional flow is only
eligible to warm-start from this checkpoint when every representation gate
passes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from unified_latent_flow import (
    UnifiedMolecularLatentFlow,
    file_sha256,
    pad_sequences,
    resolve_device,
    seed_everything,
    unified,
)


PROTOCOL = "molecular_latent_autoencoder_gate_v2"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=30000)
    parser.add_argument("--validation-limit", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-smiles-length", type=int, default=160)
    parser.add_argument("--latent-tokens", type=int, default=16)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--decoder-corruption", type=float, default=0.40)
    parser.add_argument("--latent-noise", type=float, default=0.03)
    parser.add_argument("--latent-swap-weight", type=float, default=0.25)
    parser.add_argument("--latent-swap-margin", type=float, default=0.20)
    parser.add_argument("--geometry-weight", type=float, default=0.10)
    parser.add_argument("--variance-weight", type=float, default=0.01)
    parser.add_argument("--fingerprint-bits", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--gate-clean-validity", type=float, default=0.95)
    parser.add_argument("--gate-clean-exact", type=float, default=0.25)
    parser.add_argument("--gate-clean-tanimoto", type=float, default=0.80)
    parser.add_argument("--gate-clean-scaffold", type=float, default=0.70)
    parser.add_argument("--gate-noisy-validity", type=float, default=0.90)
    parser.add_argument("--gate-noisy-tanimoto", type=float, default=0.70)
    parser.add_argument("--gate-latent-usage-gap", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1717)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stable_subset(values: Sequence[str], limit: int, seed: int) -> list[str]:
    ordered = sorted(
        values,
        key=lambda value: hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest(),
    )
    if limit <= 0:
        return ordered
    return ordered[: int(limit)]


def canonical_molecules(rows: Sequence[Mapping[str, str]]) -> tuple[set[str], dict[str, int]]:
    molecules: set[str] = set()
    counts = {"raw_values": 0, "invalid": 0}
    for row in rows:
        for column in ("source_smiles", "target_smiles"):
            raw = str(row.get(column, "") or "").strip()
            if not raw:
                continue
            counts["raw_values"] += 1
            canonical = unified.safe_canonical_smiles(raw)
            if canonical:
                molecules.add(canonical)
            else:
                counts["invalid"] += 1
    counts["unique_canonical"] = len(molecules)
    return molecules, counts


def molecule_example(smiles: str, vocab, max_length: int, fingerprint_bits: int) -> dict[str, object] | None:
    tokens = unified.tokenize_smiles(smiles)
    if len(tokens) > int(max_length):
        return None
    if any(token not in vocab.token_to_id for token in tokens):
        return None
    try:
        from sketchmol_understanding_condition.chem import morgan_fingerprint_bits

        fingerprint = morgan_fingerprint_bits(smiles, radius=2, n_bits=int(fingerprint_bits))
    except Exception:
        fingerprint = None
    if fingerprint is None:
        return None
    return {
        "smiles": smiles,
        "encoder_ids": np.asarray(vocab.encode(tokens, add_bos=True, add_eos=True), dtype=np.int64),
        "decoder_input_ids": np.asarray(vocab.encode(tokens, add_bos=True), dtype=np.int64),
        "target_ids": np.asarray(vocab.encode(tokens, add_eos=True), dtype=np.int64),
        "fingerprint": np.asarray(fingerprint, dtype=np.float32),
    }


def build_examples(
    molecules: Sequence[str], vocab, *, max_length: int, fingerprint_bits: int
) -> tuple[list[dict[str, object]], int]:
    examples: list[dict[str, object]] = []
    omitted = 0
    for smiles in molecules:
        example = molecule_example(smiles, vocab, max_length, fingerprint_bits)
        if example is None:
            omitted += 1
        else:
            examples.append(example)
    return examples, omitted


def collate_autoencoder(items: Sequence[dict[str, object]], pad_id: int) -> dict[str, object]:
    encoder_ids, encoder_mask = pad_sequences([item["encoder_ids"] for item in items], pad_id)
    decoder_input_ids, _ = pad_sequences([item["decoder_input_ids"] for item in items], pad_id)
    target_ids, _ = pad_sequences([item["target_ids"] for item in items], pad_id)
    return {
        "encoder_ids": encoder_ids,
        "encoder_mask": encoder_mask,
        "decoder_input_ids": decoder_input_ids,
        "target_ids": target_ids,
        "fingerprint": torch.from_numpy(np.stack([item["fingerprint"] for item in items])),
        "smiles": [str(item["smiles"]) for item in items],
    }


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def per_example_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> torch.Tensor:
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=int(pad_id),
        reduction="none",
    ).reshape_as(targets)
    mask = targets.ne(int(pad_id))
    return (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def corrupt_decoder_inputs(
    values: torch.Tensor, *, probability: float, pad_id: int, unknown_id: int
) -> torch.Tensor:
    probability = max(0.0, min(1.0, float(probability)))
    if probability <= 0:
        return values
    corruptible = values.ne(int(pad_id))
    corruptible[:, 0] = False
    corrupt = torch.rand_like(values, dtype=torch.float32).lt(probability) & corruptible
    return torch.where(corrupt, torch.full_like(values, int(unknown_id)), values)


def fingerprint_geometry_loss(latent: torch.Tensor, fingerprint: torch.Tensor) -> torch.Tensor:
    if latent.shape[0] < 2:
        return latent.sum() * 0.0
    pooled = F.normalize(latent.mean(dim=1), dim=-1)
    predicted = pooled @ pooled.transpose(0, 1)
    intersection = fingerprint @ fingerprint.transpose(0, 1)
    bit_counts = fingerprint.sum(dim=1)
    union = bit_counts[:, None] + bit_counts[None, :] - intersection
    target = intersection / union.clamp_min(1.0)
    off_diagonal = ~torch.eye(latent.shape[0], dtype=torch.bool, device=latent.device)
    return F.mse_loss(predicted.masked_select(off_diagonal), target.masked_select(off_diagonal))


def latent_variance_loss(latent: torch.Tensor) -> torch.Tensor:
    pooled = latent.mean(dim=1)
    if pooled.shape[0] < 2:
        return pooled.sum() * 0.0
    standard_deviation = torch.sqrt(pooled.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(0.5 - standard_deviation).mean()


def enable_autoencoder_parameters(model: UnifiedMolecularLatentFlow) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (
        model.token_embedding,
        model.decoder,
        model.output,
        model.molecule_encoder,
        model.latent_pool,
        model.latent_norm,
    ):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.latent_queries.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def train_autoencoder(
    model: UnifiedMolecularLatentFlow,
    dataset: Sequence[dict[str, object]],
    vocab,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float]]:
    parameters = enable_autoencoder_parameters(model)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    history: list[dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        order = list(range(len(dataset)))
        random.Random(int(args.seed) + epoch).shuffle(order)
        totals: dict[str, float] = defaultdict(float)
        batches = 0
        model.train()
        for start in range(0, len(order), int(args.batch_size)):
            batch = move_batch(
                collate_autoencoder(
                    [dataset[index] for index in order[start : start + int(args.batch_size)]],
                    model.pad_id,
                ),
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                latent = model.encode_molecule(batch["encoder_ids"], batch["encoder_mask"])
                noisy_latent = latent + torch.randn_like(latent) * float(args.latent_noise)
                corrupted = corrupt_decoder_inputs(
                    batch["decoder_input_ids"],
                    probability=float(args.decoder_corruption),
                    pad_id=model.pad_id,
                    unknown_id=int(vocab.token_to_id[unified.UNK]),
                )
                correct_per_example = per_example_cross_entropy(
                    model.decode(noisy_latent, corrupted), batch["target_ids"], model.pad_id
                )
                wrong_per_example = per_example_cross_entropy(
                    model.decode(torch.roll(noisy_latent, shifts=1, dims=0), corrupted),
                    batch["target_ids"],
                    model.pad_id,
                )
                reconstruction = correct_per_example.mean()
                latent_swap = F.relu(
                    float(args.latent_swap_margin) + correct_per_example - wrong_per_example
                ).mean()
                geometry = fingerprint_geometry_loss(latent, batch["fingerprint"])
                variance = latent_variance_loss(latent)
                objective = (
                    reconstruction
                    + float(args.latent_swap_weight) * latent_swap
                    + float(args.geometry_weight) * geometry
                    + float(args.variance_weight) * variance
                )
            objective.backward()
            if float(args.grad_clip) > 0:
                nn.utils.clip_grad_norm_(parameters, float(args.grad_clip))
            optimizer.step()
            with torch.no_grad():
                token_mask = batch["target_ids"].ne(model.pad_id)
                clean_logits = model.decode(latent, batch["decoder_input_ids"])
                token_accuracy = clean_logits.argmax(dim=-1).eq(batch["target_ids"])
                token_accuracy = token_accuracy.masked_select(token_mask).float().mean()
                latent_usage_gap = (wrong_per_example - correct_per_example).mean()
            values = {
                "loss": objective,
                "reconstruction": reconstruction,
                "latent_swap": latent_swap,
                "geometry": geometry,
                "variance": variance,
                "token_accuracy": token_accuracy,
                "latent_usage_gap": latent_usage_gap,
            }
            for key, value in values.items():
                totals[key] += float(value.detach().cpu())
            batches += 1
        record = {
            "epoch": float(epoch),
            **{key: value / max(batches, 1) for key, value in totals.items()},
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    return history


def decode_generated(ids: Sequence[int], vocab) -> tuple[str, str]:
    raw = unified.detokenize_smiles(vocab.decode(ids))
    return raw, unified.safe_canonical_smiles(raw)


def scaffold_match(reference: str, candidate: str) -> bool | None:
    try:
        from sketchmol_understanding_condition.chem import scaffold_smiles

        reference_scaffold = scaffold_smiles(reference)
        candidate_scaffold = scaffold_smiles(candidate)
    except Exception:
        return None
    if reference_scaffold is None:
        return candidate_scaffold is None
    return reference_scaffold == candidate_scaffold


def reconstruction_metrics(rows: Sequence[Mapping[str, object]], prefix: str) -> dict[str, float | int]:
    count = len(rows)
    valid_rows = [row for row in rows if bool(row[f"{prefix}_valid"])]
    similarities = [float(row[f"{prefix}_tanimoto"]) for row in valid_rows]
    scaffold_values = [
        bool(row[f"{prefix}_scaffold_match"])
        for row in valid_rows
        if row[f"{prefix}_scaffold_match"] is not None
    ]
    outputs = [str(row[f"{prefix}_canonical"]) for row in valid_rows]
    return {
        "molecules": count,
        "validity": len(valid_rows) / max(count, 1),
        "exact_reconstruction": sum(bool(row[f"{prefix}_exact"]) for row in rows) / max(count, 1),
        "mean_tanimoto": float(np.mean(similarities)) if similarities else 0.0,
        "median_tanimoto": float(np.median(similarities)) if similarities else 0.0,
        "scaffold_match": sum(scaffold_values) / max(len(scaffold_values), 1),
        "unique_valid": len(set(outputs)),
        "unique_valid_rate": len(set(outputs)) / max(len(outputs), 1),
    }


def rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def pairwise_geometry_correlation(latent: np.ndarray, fingerprint: np.ndarray) -> float:
    if latent.shape[0] < 3:
        return 0.0
    latent_norm = latent / np.maximum(np.linalg.norm(latent, axis=1, keepdims=True), 1e-8)
    latent_similarity = latent_norm @ latent_norm.T
    intersection = fingerprint @ fingerprint.T
    counts = fingerprint.sum(axis=1)
    fingerprint_similarity = intersection / np.maximum(counts[:, None] + counts[None, :] - intersection, 1.0)
    indices = np.triu_indices(latent.shape[0], k=1)
    left = rank_values(latent_similarity[indices])
    right = rank_values(fingerprint_similarity[indices])
    correlation = np.corrcoef(left, right)[0, 1]
    return float(correlation) if math.isfinite(float(correlation)) else 0.0


@torch.no_grad()
def evaluate_autoencoder(
    model: UnifiedMolecularLatentFlow,
    dataset: Sequence[dict[str, object]],
    vocab,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    model.eval()
    rows: list[dict[str, object]] = []
    usage_gaps: list[float] = []
    latent_values: list[np.ndarray] = []
    fingerprint_values: list[np.ndarray] = []
    eval_generator = torch.Generator(device=device)
    eval_generator.manual_seed(int(args.seed) + 1000)
    for start in range(0, len(dataset), int(args.eval_batch_size)):
        items = dataset[start : start + int(args.eval_batch_size)]
        batch = move_batch(collate_autoencoder(items, model.pad_id), device)
        latent = model.encode_molecule(batch["encoder_ids"], batch["encoder_mask"])
        noisy_latent = latent + torch.randn(
            latent.shape,
            generator=eval_generator,
            device=latent.device,
            dtype=latent.dtype,
        ) * float(args.latent_noise)
        clean_ids = model.generate(
            latent,
            vocab=vocab,
            max_new_tokens=int(args.max_new_tokens),
            temperature=0.0,
            top_k=0,
            top_p=1.0,
        )
        noisy_ids = model.generate(
            noisy_latent,
            vocab=vocab,
            max_new_tokens=int(args.max_new_tokens),
            temperature=0.0,
            top_k=0,
            top_p=1.0,
        )
        corrupted = corrupt_decoder_inputs(
            batch["decoder_input_ids"],
            probability=0.50,
            pad_id=model.pad_id,
            unknown_id=int(vocab.token_to_id[unified.UNK]),
        )
        correct_loss = per_example_cross_entropy(
            model.decode(latent, corrupted), batch["target_ids"], model.pad_id
        )
        wrong_loss = per_example_cross_entropy(
            model.decode(torch.roll(latent, shifts=1, dims=0), corrupted),
            batch["target_ids"],
            model.pad_id,
        )
        usage_gaps.extend((wrong_loss - correct_loss).detach().float().cpu().tolist())
        latent_values.append(latent.mean(dim=1).detach().float().cpu().numpy())
        fingerprint_values.append(batch["fingerprint"].detach().float().cpu().numpy())
        for item, clean_sequence, noisy_sequence in zip(
            items,
            clean_ids.detach().cpu().tolist(),
            noisy_ids.detach().cpu().tolist(),
            strict=True,
        ):
            reference = str(item["smiles"])
            row: dict[str, object] = {"reference_smiles": reference}
            for prefix, sequence in (("clean", clean_sequence), ("noisy", noisy_sequence)):
                raw, canonical = decode_generated(sequence, vocab)
                valid = bool(canonical)
                similarity = unified.morgan_tanimoto(reference, canonical) if valid else math.nan
                row.update(
                    {
                        f"{prefix}_raw": raw,
                        f"{prefix}_canonical": canonical,
                        f"{prefix}_valid": valid,
                        f"{prefix}_exact": valid and canonical == reference,
                        f"{prefix}_tanimoto": similarity if math.isfinite(similarity) else None,
                        f"{prefix}_scaffold_match": scaffold_match(reference, canonical) if valid else None,
                    }
                )
            rows.append(row)
        print(
            json.dumps({"evaluated": len(rows), "total": len(dataset)}, sort_keys=True),
            flush=True,
        )

    latent_matrix = np.concatenate(latent_values, axis=0)
    fingerprint_matrix = np.concatenate(fingerprint_values, axis=0)
    geometry_correlation = pairwise_geometry_correlation(latent_matrix, fingerprint_matrix)
    return rows, {
        "clean": reconstruction_metrics(rows, "clean"),
        "noisy": reconstruction_metrics(rows, "noisy"),
        "corrupted_prefix_latent_usage_gap": float(np.mean(usage_gaps)) if usage_gaps else 0.0,
        "pairwise_latent_fingerprint_spearman": geometry_correlation,
    }


def build_gate(metrics: Mapping[str, object], args: argparse.Namespace) -> dict[str, object]:
    clean = metrics["clean"]
    noisy = metrics["noisy"]
    checks = {
        "clean_validity": {"value": clean["validity"], "threshold": float(args.gate_clean_validity)},
        "clean_exact_reconstruction": {
            "value": clean["exact_reconstruction"],
            "threshold": float(args.gate_clean_exact),
        },
        "clean_mean_tanimoto": {
            "value": clean["mean_tanimoto"],
            "threshold": float(args.gate_clean_tanimoto),
        },
        "clean_scaffold_match": {
            "value": clean["scaffold_match"],
            "threshold": float(args.gate_clean_scaffold),
        },
        "noisy_validity": {"value": noisy["validity"], "threshold": float(args.gate_noisy_validity)},
        "noisy_mean_tanimoto": {
            "value": noisy["mean_tanimoto"],
            "threshold": float(args.gate_noisy_tanimoto),
        },
        "latent_usage_gap": {
            "value": metrics["corrupted_prefix_latent_usage_gap"],
            "threshold": float(args.gate_latent_usage_gap),
        },
    }
    failures = [name for name, check in checks.items() if float(check["value"]) < float(check["threshold"])]
    return {"passed": not failures, "checks": checks, "failures": failures}


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["reference_smiles"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    seed_everything(int(args.seed))
    device = resolve_device(str(args.device))
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    vocab = unified.SmilesVocabulary.from_dict(checkpoint["vocab"])
    model_config = dict(checkpoint["model_config"])
    base_model = unified.ConditionedSmilesDecoder(**model_config)
    base_model.load_state_dict(checkpoint["model_state"])
    model = UnifiedMolecularLatentFlow(
        base_model=base_model,
        model_config=model_config,
        latent_tokens=int(args.latent_tokens),
        encoder_layers=int(args.encoder_layers),
        flow_layers=1,
    ).to(device)

    train_rows = read_rows(args.train_csv)
    validation_rows = read_rows(args.validation_csv)
    train_molecules, train_counts = canonical_molecules(train_rows)
    validation_molecules, validation_counts = canonical_molecules(validation_rows)
    overlap = train_molecules & validation_molecules
    train_molecules -= validation_molecules
    selected_train = stable_subset(sorted(train_molecules), int(args.train_limit), int(args.seed))
    selected_validation = stable_subset(
        sorted(validation_molecules), int(args.validation_limit), int(args.seed) + 1
    )
    train_dataset, train_omitted = build_examples(
        selected_train,
        vocab,
        max_length=int(args.max_smiles_length),
        fingerprint_bits=int(args.fingerprint_bits),
    )
    validation_dataset, validation_omitted = build_examples(
        selected_validation,
        vocab,
        max_length=int(args.max_smiles_length),
        fingerprint_bits=int(args.fingerprint_bits),
    )
    if not train_dataset or not validation_dataset:
        raise RuntimeError("Empty molecular autoencoder train or validation dataset.")

    manifest = {
        "protocol": PROTOCOL,
        "seed": int(args.seed),
        "device": str(device),
        "representation_stage_only": True,
        "condition_access": False,
        "property_oracle_access": False,
        "candidate_library": False,
        "selector": False,
        "finalizer": False,
        "benchmark_generation_target_access": False,
        "representation_validation_inputs_include_source_and_target_columns": True,
        "train_csv": str(args.train_csv),
        "train_csv_sha256": file_sha256(args.train_csv),
        "validation_csv": str(args.validation_csv),
        "validation_csv_sha256": file_sha256(args.validation_csv),
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": file_sha256(args.base_checkpoint),
        "raw_train_counts": train_counts,
        "raw_validation_counts": validation_counts,
        "raw_canonical_overlap_removed_from_train": len(overlap),
        "train_validation_canonical_overlap_after_filter": 0,
        "selected_train_molecules": len(train_dataset),
        "selected_validation_molecules": len(validation_dataset),
        "train_omitted_unsupported_or_long": train_omitted,
        "validation_omitted_unsupported_or_long": validation_omitted,
        "latent_tokens": int(args.latent_tokens),
        "encoder_layers": int(args.encoder_layers),
        "latent_noise": float(args.latent_noise),
        "decoder_corruption": float(args.decoder_corruption),
        "fingerprint_bits": int(args.fingerprint_bits),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)

    history = train_autoencoder(model, train_dataset, vocab, args, device)
    checkpoint_path = args.output_dir / "molecular_latent_autoencoder.pt"
    torch.save(
        {
            "stage": "molecular_latent_autoencoder",
            "model_state": model.state_dict(),
            "model_config": model_config,
            "latent_tokens": int(args.latent_tokens),
            "encoder_layers": int(args.encoder_layers),
            "vocab": vocab.to_dict(),
            "history": history,
            "manifest": manifest,
        },
        checkpoint_path,
    )
    reconstruction_rows, metrics = evaluate_autoencoder(model, validation_dataset, vocab, args, device)
    gate = build_gate(metrics, args)
    summary = {
        "protocol": PROTOCOL,
        "checkpoint": str(checkpoint_path),
        "training": history,
        "representation": metrics,
        "gate": gate,
        "next_stage": "conditional_latent_flow" if gate["passed"] else "stop_before_flow",
    }
    write_rows(args.output_dir / "reconstructions.csv", reconstruction_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
