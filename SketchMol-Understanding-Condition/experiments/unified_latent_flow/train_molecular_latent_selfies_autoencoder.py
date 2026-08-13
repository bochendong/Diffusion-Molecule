#!/usr/bin/env python3
"""Replace the fragile SMILES output layer with a latent-conditioned SELFIES decoder.

The molecular encoder is warm-started from the representation-only v2 gate.
The output language is SELFIES, so chemical syntax validity comes from the
representation rather than candidate filtering or a repair finalizer.
"""

from __future__ import annotations

import argparse
import csv
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

import train_molecular_latent_autoencoder as stage_a
from unified_latent_flow import UnifiedMolecularLatentFlow, file_sha256, resolve_device, seed_everything, unified


PROTOCOL = "molecular_latent_selfies_autoencoder_gate_v3"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--latent-checkpoint", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=30000)
    parser.add_argument("--validation-limit", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-smiles-length", type=int, default=160)
    parser.add_argument("--max-selfies-length", type=int, default=192)
    parser.add_argument("--decoder-corruption", type=float, default=0.25)
    parser.add_argument("--latent-noise", type=float, default=0.03)
    parser.add_argument("--latent-swap-weight", type=float, default=0.25)
    parser.add_argument("--latent-swap-margin", type=float, default=0.20)
    parser.add_argument("--geometry-weight", type=float, default=0.05)
    parser.add_argument("--variance-weight", type=float, default=0.01)
    parser.add_argument("--fingerprint-bits", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--gate-clean-validity", type=float, default=0.95)
    parser.add_argument("--gate-clean-exact", type=float, default=0.25)
    parser.add_argument("--gate-clean-tanimoto", type=float, default=0.80)
    parser.add_argument("--gate-clean-scaffold", type=float, default=0.70)
    parser.add_argument("--gate-noisy-validity", type=float, default=0.90)
    parser.add_argument("--gate-noisy-tanimoto", type=float, default=0.70)
    parser.add_argument("--gate-latent-usage-gap", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1718)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def load_selfies_module():
    try:
        import selfies
    except ImportError as exc:
        raise RuntimeError("The SELFIES package is required for the v3 representation gate.") from exc
    return selfies


def build_selfies_vocabulary():
    selfies = load_selfies_module()
    vocab = unified.SmilesVocabulary()
    vocab.update([sorted(selfies.get_semantic_robust_alphabet())])
    return vocab


def selfies_tokens(smiles: str) -> list[str] | None:
    selfies = load_selfies_module()
    try:
        encoded = selfies.encoder(smiles)
        return list(selfies.split_selfies(encoded))
    except Exception:
        return None


def build_selfies_examples(
    molecules: Sequence[str],
    encoder_vocab,
    decoder_vocab,
    *,
    max_smiles_length: int,
    max_selfies_length: int,
    fingerprint_bits: int,
) -> tuple[list[dict[str, object]], int]:
    examples: list[dict[str, object]] = []
    omitted = 0
    for smiles in molecules:
        smiles_tokens = unified.tokenize_smiles(smiles)
        output_tokens = selfies_tokens(smiles)
        if (
            output_tokens is None
            or len(smiles_tokens) > int(max_smiles_length)
            or len(output_tokens) > int(max_selfies_length)
            or any(token not in encoder_vocab.token_to_id for token in smiles_tokens)
            or any(token not in decoder_vocab.token_to_id for token in output_tokens)
        ):
            omitted += 1
            continue
        try:
            from sketchmol_understanding_condition.chem import morgan_fingerprint_bits

            fingerprint = morgan_fingerprint_bits(smiles, radius=2, n_bits=int(fingerprint_bits))
        except Exception:
            fingerprint = None
        if fingerprint is None:
            omitted += 1
            continue
        examples.append(
            {
                "smiles": smiles,
                "encoder_ids": np.asarray(
                    encoder_vocab.encode(smiles_tokens, add_bos=True, add_eos=True), dtype=np.int64
                ),
                "decoder_input_ids": np.asarray(
                    decoder_vocab.encode(output_tokens, add_bos=True), dtype=np.int64
                ),
                "target_ids": np.asarray(
                    decoder_vocab.encode(output_tokens, add_eos=True), dtype=np.int64
                ),
                "fingerprint": np.asarray(fingerprint, dtype=np.float32),
            }
        )
    return examples, omitted


class SelfiesLatentAutoencoder(nn.Module):
    def __init__(self, latent_model: UnifiedMolecularLatentFlow, decoder_vocab_size: int) -> None:
        super().__init__()
        self.latent_model = latent_model
        self.pad_id = 0
        self.token_embedding = nn.Embedding(
            int(decoder_vocab_size), latent_model.d_model, padding_idx=self.pad_id
        )
        # SELFIES sequences are often longer than canonical SMILES.  Keep this
        # decoder's positional capacity independent from the 176-token warm
        # start buffer; the sinusoidal buffer has no trainable state.
        self.position = unified.PositionalEncoding(latent_model.d_model, max_len=512)
        self.decoder = latent_model.decoder
        self.output = nn.Linear(latent_model.d_model, int(decoder_vocab_size))
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.output.bias)

    def encode_molecule(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.latent_model.encode_molecule(ids, mask)

    def decode(self, latent: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        target = self.position(self.token_embedding(decoder_input_ids))
        length = decoder_input_ids.shape[1]
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=decoder_input_ids.device), diagonal=1
        )
        decoded = self.decoder(
            target,
            latent,
            tgt_mask=causal,
            tgt_key_padding_mask=decoder_input_ids.eq(self.pad_id),
        )
        return self.output(decoded)

    @torch.no_grad()
    def generate(self, latent: torch.Tensor, vocab, max_new_tokens: int) -> torch.Tensor:
        generated = torch.full(
            (latent.shape[0], 1), int(vocab.bos_id), dtype=torch.long, device=latent.device
        )
        finished = torch.zeros(latent.shape[0], dtype=torch.bool, device=latent.device)
        blocked_ids = [int(vocab.bos_id), int(vocab.pad_id), int(vocab.token_to_id[unified.UNK])]
        for _ in range(max(1, int(max_new_tokens))):
            logits = self.decode(latent, generated)[:, -1, :]
            logits[:, blocked_ids] = -torch.inf
            next_ids = logits.argmax(dim=-1)
            next_ids = torch.where(finished, torch.full_like(next_ids, int(vocab.eos_id)), next_ids)
            generated = torch.cat([generated, next_ids[:, None]], dim=1)
            finished |= next_ids.eq(int(vocab.eos_id))
            if bool(finished.all()):
                break
        return generated


def enable_parameters(model: SelfiesLatentAutoencoder) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (
        model.latent_model.token_embedding,
        model.latent_model.molecule_encoder,
        model.latent_model.latent_pool,
        model.latent_model.latent_norm,
        model.token_embedding,
        model.decoder,
        model.output,
    ):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.latent_model.latent_queries.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def train_model(
    model: SelfiesLatentAutoencoder,
    dataset: Sequence[dict[str, object]],
    decoder_vocab,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float]]:
    parameters = enable_parameters(model)
    optimizer = torch.optim.AdamW(
        parameters, lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
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
            batch = stage_a.move_batch(
                stage_a.collate_autoencoder(
                    [dataset[index] for index in order[start : start + int(args.batch_size)]],
                    model.pad_id,
                ),
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                latent = model.encode_molecule(batch["encoder_ids"], batch["encoder_mask"])
                noisy_latent = latent + torch.randn_like(latent) * float(args.latent_noise)
                decoder_input = stage_a.corrupt_decoder_inputs(
                    batch["decoder_input_ids"],
                    probability=float(args.decoder_corruption),
                    pad_id=model.pad_id,
                    unknown_id=int(decoder_vocab.token_to_id[unified.UNK]),
                )
                correct = stage_a.per_example_cross_entropy(
                    model.decode(noisy_latent, decoder_input), batch["target_ids"], model.pad_id
                )
                wrong = stage_a.per_example_cross_entropy(
                    model.decode(torch.roll(noisy_latent, shifts=1, dims=0), decoder_input),
                    batch["target_ids"],
                    model.pad_id,
                )
                reconstruction = correct.mean()
                latent_swap = F.relu(float(args.latent_swap_margin) + correct - wrong).mean()
                geometry = stage_a.fingerprint_geometry_loss(latent, batch["fingerprint"])
                variance = stage_a.latent_variance_loss(latent)
                objective = (
                    reconstruction
                    + float(args.latent_swap_weight) * latent_swap
                    + float(args.geometry_weight) * geometry
                    + float(args.variance_weight) * variance
                )
            objective.backward()
            nn.utils.clip_grad_norm_(parameters, float(args.grad_clip))
            optimizer.step()
            with torch.no_grad():
                clean_logits = model.decode(latent, batch["decoder_input_ids"])
                mask = batch["target_ids"].ne(model.pad_id)
                token_accuracy = clean_logits.argmax(dim=-1).eq(batch["target_ids"])
                token_accuracy = token_accuracy.masked_select(mask).float().mean()
                latent_usage_gap = (wrong - correct).mean()
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


def generated_selfies_to_smiles(ids: Sequence[int], vocab) -> tuple[str, str, str]:
    selfies = load_selfies_module()
    tokens = vocab.decode(ids)
    encoded = "".join(token for token in tokens if token not in unified.SPECIAL_TOKENS)
    try:
        raw_smiles = selfies.decoder(encoded)
    except Exception:
        raw_smiles = ""
    return encoded, raw_smiles, unified.safe_canonical_smiles(raw_smiles)


@torch.no_grad()
def evaluate(
    model: SelfiesLatentAutoencoder,
    dataset: Sequence[dict[str, object]],
    decoder_vocab,
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
        batch = stage_a.move_batch(stage_a.collate_autoencoder(items, model.pad_id), device)
        latent = model.encode_molecule(batch["encoder_ids"], batch["encoder_mask"])
        noisy_latent = latent + torch.randn(
            latent.shape,
            generator=eval_generator,
            device=latent.device,
            dtype=latent.dtype,
        ) * float(args.latent_noise)
        clean_ids = model.generate(latent, decoder_vocab, int(args.max_selfies_length))
        noisy_ids = model.generate(noisy_latent, decoder_vocab, int(args.max_selfies_length))
        corrupted = stage_a.corrupt_decoder_inputs(
            batch["decoder_input_ids"],
            probability=0.50,
            pad_id=model.pad_id,
            unknown_id=int(decoder_vocab.token_to_id[unified.UNK]),
        )
        correct = stage_a.per_example_cross_entropy(
            model.decode(latent, corrupted), batch["target_ids"], model.pad_id
        )
        wrong = stage_a.per_example_cross_entropy(
            model.decode(torch.roll(latent, shifts=1, dims=0), corrupted),
            batch["target_ids"],
            model.pad_id,
        )
        usage_gaps.extend((wrong - correct).detach().float().cpu().tolist())
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
                encoded, raw, canonical = generated_selfies_to_smiles(sequence, decoder_vocab)
                valid = bool(canonical)
                similarity = unified.morgan_tanimoto(reference, canonical) if valid else math.nan
                row.update(
                    {
                        f"{prefix}_selfies": encoded,
                        f"{prefix}_raw": raw,
                        f"{prefix}_canonical": canonical,
                        f"{prefix}_valid": valid,
                        f"{prefix}_exact": valid and canonical == reference,
                        f"{prefix}_tanimoto": similarity if math.isfinite(similarity) else None,
                        f"{prefix}_scaffold_match": stage_a.scaffold_match(reference, canonical)
                        if valid
                        else None,
                    }
                )
            rows.append(row)
        print(json.dumps({"evaluated": len(rows), "total": len(dataset)}, sort_keys=True), flush=True)
    latent_matrix = np.concatenate(latent_values, axis=0)
    fingerprint_matrix = np.concatenate(fingerprint_values, axis=0)
    return rows, {
        "clean": stage_a.reconstruction_metrics(rows, "clean"),
        "noisy": stage_a.reconstruction_metrics(rows, "noisy"),
        "corrupted_prefix_latent_usage_gap": float(np.mean(usage_gaps)) if usage_gaps else 0.0,
        "pairwise_latent_fingerprint_spearman": stage_a.pairwise_geometry_correlation(
            latent_matrix, fingerprint_matrix
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    seed_everything(int(args.seed))
    device = resolve_device(str(args.device))
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    latent_checkpoint = torch.load(args.latent_checkpoint, map_location="cpu", weights_only=False)
    encoder_vocab = unified.SmilesVocabulary.from_dict(base_checkpoint["vocab"])
    resume_checkpoint = None
    if args.resume_checkpoint is not None and args.resume_checkpoint.is_file():
        resume_checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
    decoder_vocab = (
        unified.SmilesVocabulary.from_dict(resume_checkpoint["decoder_vocab"])
        if resume_checkpoint is not None
        else build_selfies_vocabulary()
    )
    model_config = dict(base_checkpoint["model_config"])
    base_model = unified.ConditionedSmilesDecoder(**model_config)
    base_model.load_state_dict(base_checkpoint["model_state"])
    latent_model = UnifiedMolecularLatentFlow(
        base_model=base_model,
        model_config=model_config,
        latent_tokens=int(latent_checkpoint["latent_tokens"]),
        encoder_layers=int(latent_checkpoint["encoder_layers"]),
        flow_layers=1,
    )
    latent_model.load_state_dict(latent_checkpoint["model_state"])
    model = SelfiesLatentAutoencoder(latent_model, len(decoder_vocab.token_to_id)).to(device)
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state"])

    train_molecules, train_counts = stage_a.canonical_molecules(stage_a.read_rows(args.train_csv))
    validation_molecules, validation_counts = stage_a.canonical_molecules(
        stage_a.read_rows(args.validation_csv)
    )
    overlap = train_molecules & validation_molecules
    train_molecules -= validation_molecules
    selected_train = stage_a.stable_subset(sorted(train_molecules), int(args.train_limit), int(args.seed))
    selected_validation = stage_a.stable_subset(
        sorted(validation_molecules), int(args.validation_limit), int(args.seed) + 1
    )
    train_dataset, train_omitted = build_selfies_examples(
        selected_train,
        encoder_vocab,
        decoder_vocab,
        max_smiles_length=int(args.max_smiles_length),
        max_selfies_length=int(args.max_selfies_length),
        fingerprint_bits=int(args.fingerprint_bits),
    )
    validation_dataset, validation_omitted = build_selfies_examples(
        selected_validation,
        encoder_vocab,
        decoder_vocab,
        max_smiles_length=int(args.max_smiles_length),
        max_selfies_length=int(args.max_selfies_length),
        fingerprint_bits=int(args.fingerprint_bits),
    )
    if not train_dataset or not validation_dataset:
        raise RuntimeError("Empty SELFIES autoencoder train or validation dataset.")

    manifest = {
        "protocol": PROTOCOL,
        "seed": int(args.seed),
        "device": str(device),
        "representation_stage_only": True,
        "decoder_representation": "SELFIES",
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
        "latent_checkpoint": str(args.latent_checkpoint),
        "latent_checkpoint_sha256": file_sha256(args.latent_checkpoint),
        "raw_train_counts": train_counts,
        "raw_validation_counts": validation_counts,
        "raw_canonical_overlap_removed_from_train": len(overlap),
        "train_validation_canonical_overlap_after_filter": 0,
        "selected_train_molecules": len(train_dataset),
        "selected_validation_molecules": len(validation_dataset),
        "train_omitted_unsupported_or_long": train_omitted,
        "validation_omitted_unsupported_or_long": validation_omitted,
        "decoder_vocabulary_size": len(decoder_vocab.token_to_id),
        "latent_noise": float(args.latent_noise),
        "decoder_corruption": float(args.decoder_corruption),
        "resumed_from_checkpoint": str(args.resume_checkpoint) if resume_checkpoint is not None else None,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    history = (
        list(resume_checkpoint.get("history", []))
        if resume_checkpoint is not None
        else train_model(model, train_dataset, decoder_vocab, args, device)
    )
    checkpoint_path = args.output_dir / "molecular_latent_selfies_autoencoder.pt"
    torch.save(
        {
            "stage": "molecular_latent_selfies_autoencoder",
            "model_state": model.state_dict(),
            "model_config": model_config,
            "encoder_vocab": encoder_vocab.to_dict(),
            "decoder_vocab": decoder_vocab.to_dict(),
            "history": history,
            "manifest": manifest,
        },
        checkpoint_path,
    )
    reconstruction_rows, metrics = evaluate(model, validation_dataset, decoder_vocab, args, device)
    gate = stage_a.build_gate(metrics, args)
    summary = {
        "protocol": PROTOCOL,
        "checkpoint": str(checkpoint_path),
        "training": history,
        "representation": metrics,
        "gate": gate,
        "next_stage": "conditional_latent_flow" if gate["passed"] else "stop_before_flow",
    }
    stage_a.write_rows(args.output_dir / "reconstructions.csv", reconstruction_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
