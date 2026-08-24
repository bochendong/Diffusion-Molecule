#!/usr/bin/env python3
"""Standalone unified SMILES generator experiment line.

This file intentionally copies the direct-SMILES building blocks into the new
experiment folder instead of importing repo-internal training utilities.  The
only hard non-stdlib dependencies are numpy and torch. RDKit and TDC are used
opportunistically for molecular/reward scoring when available.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sketchmol_understanding_condition import direct_condition_tokens as direct_cond  # noqa: E402

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]
DE_NOVO_MODE = "de_novo"
EDIT_MODE = "edit"

PROPERTY_COLUMNS = [
    "MW",
    "LogP",
    "QED",
    "TPSA",
    "HBD",
    "HBA",
    "RB",
    "SA",
    "BBBP",
    "DRD2",
    "GSK3B",
    "JNK3",
    "HIA",
    "mutagenicity",
    "hERG",
    "DILI",
    "PAMPA",
]
PROPERTY_ALIASES = {
    "mw": "MW",
    "molwt": "MW",
    "molecular_weight": "MW",
    "logp": "LogP",
    "plogp": "LogP",
    "qed": "QED",
    "tpsa": "TPSA",
    "hbd": "HBD",
    "hba": "HBA",
    "rb": "RB",
    "rotatable": "RB",
    "rotbonds": "RB",
    "sa": "SA",
    "sas": "SA",
    "bbbp": "BBBP",
    "drd2": "DRD2",
    "gsk3b": "GSK3B",
    "gsk3β": "GSK3B",
    "gsk3": "GSK3B",
    "jnk3": "JNK3",
    "hia": "HIA",
    "mutagenicity": "mutagenicity",
    "ames": "mutagenicity",
    "herg": "hERG",
    "dili": "DILI",
    "pampa": "PAMPA",
}
PROPERTY_NORMALIZERS = {
    "MW": 500.0,
    "LogP": 6.0,
    "QED": 1.0,
    "TPSA": 160.0,
    "HBD": 8.0,
    "HBA": 12.0,
    "RB": 12.0,
    "SA": 8.0,
    "BBBP": 1.0,
    "DRD2": 1.0,
    "GSK3B": 1.0,
    "JNK3": 1.0,
    "HIA": 1.0,
    "mutagenicity": 1.0,
    "hERG": 1.0,
    "DILI": 1.0,
    "PAMPA": 1.0,
}
STRICT_TOLERANCE = {
    "MW": 35.0,
    "LogP": 1.0,
    "QED": 0.10,
    "TPSA": 20.0,
    "HBD": 1.0,
    "HBA": 1.0,
    "RB": 1.0,
    "SA": 1.0,
    "BBBP": 0.5,
    "DRD2": 0.5,
    "GSK3B": 0.5,
    "JNK3": 0.5,
    "HIA": 0.5,
    "mutagenicity": 0.5,
    "hERG": 0.5,
    "DILI": 0.5,
    "PAMPA": 0.5,
}
SMILES_TOKEN_RE = re.compile(
    r"(\[[^\]]+\]|"
    r"Br|Cl|Si|Se|Na|Li|Mg|Ca|Al|Fe|Zn|Cu|Mn|"
    r"@@?|%\d{2}|\d|"
    r"\.|=|#|-|/|\\|\+|:|~|\(|\)|"
    r"[BCNOFPSIHK]|[bcnops]|.)"
)
_TDC_ORACLE_CACHE: dict[str, object | None] = {}
_PROPERTY_VALUE_CACHE: dict[tuple[str, str], float | None] = {}
_FILE_SHA256_CACHE: dict[str, str] = {}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train and optionally sample a unified generator.")
    add_common_data_args(train)
    add_model_args(train)
    add_sampling_args(train)
    train.add_argument("--train-csv", required=True, type=Path)
    train.add_argument("--eval-csv", type=Path, default=None)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--eval-batch-size", type=int, default=128)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument(
        "--trainable-scope",
        choices=("all", "source_only"),
        default="all",
        help="source_only freezes the complete source-free de-novo path and trains only source-conditioned modules.",
    )
    train.add_argument("--limit", type=int, default=0)
    train.add_argument("--eval-limit", type=int, default=0)
    train.add_argument("--resume-checkpoint", type=Path, default=None)
    train.add_argument(
        "--reset-training-state",
        action="store_true",
        help="Load model/vocab weights from --resume-checkpoint but restart epoch, history, and optimizer state.",
    )
    train.add_argument(
        "--sampling-mode",
        choices=("random", "task_balanced"),
        default="random",
        help="random shuffles rows once per epoch; task_balanced balances de_novo/edit modes and their task groups.",
    )
    train.add_argument(
        "--samples-per-epoch",
        type=int,
        default=0,
        help="Number of sampled rows per epoch for task_balanced training. 0 uses the input row count.",
    )
    train.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=None,
        help="Optional frozen teacher checkpoint used for protected de novo logit distillation.",
    )
    train.add_argument("--distill-weight", type=float, default=0.0)
    train.add_argument("--distill-temperature", type=float, default=1.0)
    train.add_argument(
        "--distill-control",
        choices=("fixed", "adaptive"),
        default="fixed",
        help="Keep a fixed teacher-KL weight or update it as a dual variable around --distill-target-kl.",
    )
    train.add_argument("--distill-target-kl", type=float, default=0.02)
    train.add_argument("--distill-dual-lr", type=float, default=0.5)
    train.add_argument("--distill-min-weight", type=float, default=0.0)
    train.add_argument("--distill-max-weight", type=float, default=2.0)
    train.add_argument(
        "--allow-architecture-warmstart",
        action="store_true",
        help="Load compatible legacy decoder weights while initializing newly requested source-aware modules.",
    )
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--device", default="auto")

    group_rl = subparsers.add_parser("group-rl", help="Task-aware group-relative RL for the unified generator.")
    add_common_data_args(group_rl)
    add_sampling_args(group_rl)
    add_group_rl_args(group_rl)
    group_rl.add_argument("--train-csv", required=True, type=Path)
    group_rl.add_argument("--eval-csv", type=Path, default=None)
    group_rl.add_argument("--output-dir", required=True, type=Path)
    group_rl.add_argument("--resume-checkpoint", required=True, type=Path)
    group_rl.add_argument("--epochs", type=int, default=1)
    group_rl.add_argument("--batch-size", type=int, default=8)
    group_rl.add_argument("--eval-batch-size", type=int, default=32)
    group_rl.add_argument("--lr", type=float, default=1e-6)
    group_rl.add_argument("--weight-decay", type=float, default=1e-4)
    group_rl.add_argument("--grad-clip", type=float, default=1.0)
    group_rl.add_argument(
        "--trainable-scope",
        choices=("all", "source_only"),
        default="all",
        help="source_only freezes the complete source-free de-novo path during group-relative RL.",
    )
    group_rl.add_argument("--limit", type=int, default=0)
    group_rl.add_argument("--eval-limit", type=int, default=0)
    group_rl.add_argument("--seed", type=int, default=7)
    group_rl.add_argument("--device", default="auto")

    sample = subparsers.add_parser("sample", help="Sample selected/candidate SMILES from a checkpoint.")
    add_common_data_args(sample)
    add_sampling_args(sample)
    sample.add_argument("--checkpoint", required=True, type=Path)
    sample.add_argument("--eval-csv", required=True, type=Path)
    sample.add_argument("--output-dir", required=True, type=Path)
    sample.add_argument("--eval-limit", type=int, default=0)
    sample.add_argument("--seed", type=int, default=7)
    sample.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--condition-features-dir", type=Path, default=None)
    parser.add_argument("--eval-condition-features-dir", type=Path, default=None)
    parser.add_argument("--condition-feature-array", choices=("query_tokens", "pooled"), default="query_tokens")
    parser.add_argument(
        "--condition-feature-variant",
        default="full",
        help="Feature variant to read from exported condition features, e.g. full or text_only.",
    )
    parser.add_argument(
        "--input-modality",
        default="",
        help="Optional report label such as with_image or no_image. Derived from variant when omitted.",
    )
    parser.add_argument("--condition-dim", type=int, default=256)
    parser.add_argument("--max-smiles-length", type=int, default=160)
    parser.add_argument("--max-source-tokens", type=int, default=96)
    parser.add_argument("--method", default="unified_smiles_generator")
    parser.add_argument(
        "--condition-layout",
        choices=("unified", "transformation", "direct_compat", "direct_edit_compat", "property_program_only"),
        default="unified",
        help=(
            "Condition token layout. `unified` adds explicit mode tokens; "
            "`transformation` uses one goal plus optional-source contract while preserving the legacy de novo goal layout; "
            "`direct_compat` matches the earlier direct-SMILES property-program layout "
            "for direct checkpoint warm-starts (edit rows append source SMILES tokens); "
            "`direct_edit_compat` is an alias of `direct_compat` for source-edit eval; "
            "`property_program_only` drops frozen VLM features and keeps only the property-program tokens."
        ),
    )


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--source-aware",
        action="store_true",
        help="Route source-marked condition tokens through a dedicated molecular memory before shared decoding.",
    )
    parser.add_argument("--source-encoder-layers", type=int, default=2)
    parser.add_argument("--source-residual-scale", type=float, default=1.0)
    parser.add_argument(
        "--source-copy-aware",
        action="store_true",
        help=(
            "Mix the shared vocabulary distribution with a token-level pointer distribution over "
            "the source SMILES. The pointer is source-gated and is exactly inactive for de novo rows."
        ),
    )
    parser.add_argument("--source-adapter-layers", type=int, default=2)
    parser.add_argument("--source-adapter-bottleneck", type=int, default=64)
    parser.add_argument(
        "--source-copy-initial-vocab-bias",
        type=float,
        default=2.0,
        help="Initial logit bias for the shared-vocabulary side of the pointer-generator gate.",
    )


def add_sampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prediction-csv", type=Path, default=None)
    parser.add_argument("--candidate-output-csv", type=Path, default=None)
    parser.add_argument(
        "--decoding-mode",
        choices=("sample", "beam", "sample_beam"),
        default="sample",
        help="Candidate generation mode: stochastic sampling, deterministic beam search, or both.",
    )
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--beam-size", type=int, default=20)
    parser.add_argument("--beam-expand-size", type=int, default=64)
    parser.add_argument("--beam-length-penalty", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--parallel-samples", type=int, default=16)
    parser.add_argument("--max-parallel-sequences", type=int, default=512)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=6)
    parser.add_argument("--min-new-tokens", type=int, default=6)
    parser.add_argument(
        "--smiles-grammar-constraint",
        action="store_true",
        help="Mask tokens that would create unbalanced branches/rings or impossible SMILES token transitions.",
    )
    parser.add_argument("--top-k-candidates", type=int, default=40)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Maximum unique candidates written per row. 0 preserves the legacy --top-k-candidates limit.",
    )
    parser.add_argument("--disable-finalizer", action="store_true")
    parser.add_argument(
        "--include-source-copy-candidate",
        action="store_true",
        help="Append the source molecule as a diagnostic candidate for edit rows. Disabled by default for fair evaluation.",
    )
    parser.add_argument("--source-similarity-threshold", type=float, default=0.4)


def add_group_rl_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rl-objective", choices=("group_pg", "grpo"), default="group_pg")
    parser.add_argument("--grpo-clip-eps", type=float, default=0.2)
    parser.add_argument("--grpo-update-epochs", type=int, default=1)
    parser.add_argument("--rollouts-per-prompt", type=int, default=16)
    parser.add_argument("--sft-weight", type=float, default=0.25)
    parser.add_argument("--advantage-mode", choices=("group_center", "group_zscore"), default="group_zscore")
    parser.add_argument("--advantage-clip", type=float, default=3.0)
    parser.add_argument("--sequence-logprob-reduction", choices=("sum", "mean"), default="mean")
    parser.add_argument("--reference-kl-weight", type=float, default=0.05)
    parser.add_argument(
        "--reward-mode",
        choices=("auto", "property_strict", "table1_edit"),
        default="auto",
        help="auto routes de_novo rows to property_strict and edit rows to table1_edit.",
    )
    parser.add_argument("--reward-valid-weight", type=float, default=0.25)
    parser.add_argument("--reward-strict-weight", type=float, default=2.0)
    parser.add_argument("--reward-distance-weight", type=float, default=0.05)
    parser.add_argument("--reward-distance-clip", type=float, default=10.0)
    parser.add_argument(
        "--reward-aggregation",
        choices=("mean", "joint_bottleneck", "dense_softmin"),
        default="mean",
        help=(
            "mean preserves the legacy average-property reward; joint_bottleneck uses a sparse "
            "all-success bonus; dense_softmin supplies a smooth satisfaction score and soft worst-margin signal."
        ),
    )
    parser.add_argument("--reward-joint-bonus-weight", type=float, default=2.0)
    parser.add_argument("--reward-bottleneck-weight", type=float, default=0.5)
    parser.add_argument("--reward-softmin-weight", type=float, default=1.0)
    parser.add_argument("--reward-softmin-temperature", type=float, default=0.25)
    parser.add_argument("--reward-source-similarity-weight", type=float, default=0.5)
    parser.add_argument("--reward-source-similarity-threshold", type=float, default=None)
    parser.add_argument("--reward-source-copy-penalty", type=float, default=0.5)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    seed_everything(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(args.device))

    if args.command == "train":
        return train_command(args, device)
    if args.command == "group-rl":
        return group_rl_command(args, device)
    if args.command == "sample":
        return sample_command(args, device)
    raise ValueError(f"Unsupported command: {args.command}")


def train_command(args: argparse.Namespace, device: torch.device) -> int:
    train_rows = read_rows(args.train_csv, limit=int(args.limit))
    eval_rows = read_rows(args.eval_csv, limit=int(args.eval_limit)) if args.eval_csv else []
    train_store = FeatureStore(
        args.condition_features_dir,
        array_name=str(args.condition_feature_array),
        variant=str(args.condition_feature_variant),
    )
    eval_store = FeatureStore(
        args.eval_condition_features_dir or args.condition_features_dir,
        array_name=str(args.condition_feature_array),
        variant=str(args.condition_feature_variant),
    )
    checkpoint = load_checkpoint(args.resume_checkpoint)

    if checkpoint:
        vocab = SmilesVocabulary.from_dict(checkpoint["vocab"])
        config = dict(checkpoint["model_config"])
        requested_source_aware = bool(args.source_aware or args.source_copy_aware)
        checkpoint_source_aware = bool(config.get("source_aware", False))
        requested_source_copy = bool(args.source_copy_aware)
        checkpoint_source_copy = bool(config.get("source_copy_aware", False))
        architecture_upgrade = (requested_source_aware and not checkpoint_source_aware) or (
            requested_source_copy and not checkpoint_source_copy
        )
        if architecture_upgrade:
            if not bool(args.allow_architecture_warmstart):
                raise ValueError(
                    "Adding source-aware or source-copy modules to a checkpoint requires "
                    "--allow-architecture-warmstart"
                )
            config.update(
                {
                    "source_aware": True,
                    "source_encoder_layers": int(args.source_encoder_layers),
                    "source_residual_scale": float(args.source_residual_scale),
                    "source_copy_aware": requested_source_copy,
                    "source_adapter_layers": int(args.source_adapter_layers),
                    "source_adapter_bottleneck": int(args.source_adapter_bottleneck),
                    "source_copy_initial_vocab_bias": float(args.source_copy_initial_vocab_bias),
                }
            )
    else:
        vocab = build_vocabulary(
            [row.get("target_smiles", "") for row in train_rows + eval_rows]
            + [row.get("source_smiles", "") for row in train_rows + eval_rows]
        )
        condition_dim = infer_condition_dim(train_store, eval_store, default=int(args.condition_dim))
        config = {
            "vocab_size": len(vocab.token_to_id),
            "condition_dim": condition_dim,
            "d_model": int(args.d_model),
            "num_layers": int(args.num_layers),
            "num_heads": int(args.num_heads),
            "dim_feedforward": int(args.dim_feedforward),
            "dropout": float(args.dropout),
            "pad_id": vocab.pad_id,
            "max_length": int(args.max_smiles_length) + 8,
            "source_aware": bool(args.source_aware or args.source_copy_aware),
            "source_encoder_layers": int(args.source_encoder_layers),
            "source_residual_scale": float(args.source_residual_scale),
            "source_copy_aware": bool(args.source_copy_aware),
            "source_adapter_layers": int(args.source_adapter_layers),
            "source_adapter_bottleneck": int(args.source_adapter_bottleneck),
            "source_copy_initial_vocab_bias": float(args.source_copy_initial_vocab_bias),
        }

    model = ConditionedSmilesDecoder(**config).to(device)
    if checkpoint:
        incompatible = model.load_state_dict(
            checkpoint["model_state"],
            strict=not bool(args.allow_architecture_warmstart),
        )
        if bool(args.allow_architecture_warmstart) and incompatible.unexpected_keys:
            raise ValueError(f"Unexpected warm-start parameters: {incompatible.unexpected_keys}")
    trainable_scope = configure_trainable_scope(model, str(args.trainable_scope))

    teacher_model = build_distillation_teacher(
        args.teacher_checkpoint,
        expected_vocab=vocab,
        expected_config=config,
        device=device,
        allow_architecture_mismatch=bool(args.allow_architecture_warmstart),
    )
    if float(args.distill_weight) > 0 and teacher_model is None:
        raise ValueError("--distill-weight requires --teacher-checkpoint")

    train_dataset = build_dataset(
        train_rows,
        vocab,
        train_store,
        int(config["condition_dim"]),
        max_smiles_length=int(args.max_smiles_length),
        max_source_tokens=int(args.max_source_tokens),
        condition_layout=str(args.condition_layout),
    )
    eval_dataset = build_dataset(
        eval_rows,
        vocab,
        eval_store,
        int(config["condition_dim"]),
        max_smiles_length=int(args.max_smiles_length),
        max_source_tokens=int(args.max_source_tokens),
        condition_layout=str(args.condition_layout),
    )
    if not train_dataset:
        raise ValueError("No trainable rows found. Rows need target_smiles.")

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    warm_start = (
        bool(getattr(args, "reset_training_state", False))
        or bool(getattr(args, "allow_architecture_warmstart", False))
    ) and checkpoint is not None
    if checkpoint and checkpoint.get("optimizer_state") and not warm_start:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    history = [] if warm_start else (list(checkpoint.get("history", [])) if checkpoint else [])
    start_epoch = 1 if warm_start else (int(checkpoint.get("epoch", 0)) + 1 if checkpoint else 1)

    current_distill_weight = float(args.distill_weight)
    for epoch in range(start_epoch, int(args.epochs) + 1):
        record = train_epoch(
            model,
            teacher_model,
            train_dataset,
            optimizer,
            batch_size=int(args.batch_size),
            grad_clip=float(args.grad_clip),
            device=device,
            seed=int(args.seed) + epoch,
            sampling_mode=str(args.sampling_mode),
            samples_per_epoch=int(args.samples_per_epoch),
            distill_weight=current_distill_weight,
            distill_temperature=float(args.distill_temperature),
        )
        record["epoch"] = epoch
        record["distill_weight"] = current_distill_weight
        if str(args.distill_control) == "adaptive" and teacher_model is not None:
            current_distill_weight = update_adaptive_distill_weight(
                current_distill_weight,
                observed_kl=float(record["distill_loss"]),
                target_kl=float(args.distill_target_kl),
                dual_lr=float(args.distill_dual_lr),
                min_weight=float(args.distill_min_weight),
                max_weight=float(args.distill_max_weight),
            )
        record["next_distill_weight"] = current_distill_weight
        if eval_dataset:
            eval_record = evaluate_loss(model, eval_dataset, batch_size=int(args.eval_batch_size), device=device)
            record.update({f"eval_{key}": value for key, value in eval_record.items()})
        history.append(record)
        save_checkpoint(args.output_dir / "latest_checkpoint.pt", model, optimizer, vocab, config, epoch, history, args)
        save_checkpoint(
            args.output_dir / f"checkpoint_epoch_{epoch:03d}.pt",
            model,
            optimizer,
            vocab,
            config,
            epoch,
            history,
            args,
        )

    checkpoint_path = args.output_dir / "unified_smiles_generator.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        vocab,
        config,
        int(history[-1]["epoch"]) if history else 0,
        history,
        args,
    )

    prediction_summary = None
    if eval_rows:
        prediction_summary = write_predictions(
            model,
            eval_rows,
            eval_store,
            vocab,
            int(config["condition_dim"]),
            args=args,
            device=device,
        )

    summary = {
        "checkpoint": str(checkpoint_path),
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv) if args.eval_csv else None,
        "train_rows": len(train_dataset),
        "eval_rows": len(eval_rows),
        "condition_dim": int(config["condition_dim"]),
        "condition_feature_variant": str(args.condition_feature_variant),
        "condition_layout": str(args.condition_layout),
        "input_modality": input_modality_for_args(args),
        "sampling_mode": str(args.sampling_mode),
        "samples_per_epoch": int(args.samples_per_epoch),
        "teacher_checkpoint": str(args.teacher_checkpoint) if args.teacher_checkpoint else None,
        "distill_weight": float(args.distill_weight),
        "distill_temperature": float(args.distill_temperature),
        "distill_control": str(args.distill_control),
        "distill_target_kl": float(args.distill_target_kl),
        "distill_dual_lr": float(args.distill_dual_lr),
        "final_distill_weight": current_distill_weight,
        "source_aware": bool(config.get("source_aware", False)),
        "source_encoder_layers": int(config.get("source_encoder_layers", 0)),
        "source_residual_scale": float(config.get("source_residual_scale", 0.0)),
        "source_copy_aware": bool(config.get("source_copy_aware", False)),
        "source_adapter_layers": int(config.get("source_adapter_layers", 0)),
        "source_adapter_bottleneck": int(config.get("source_adapter_bottleneck", 0)),
        "trainable_scope": trainable_scope,
        "task_mode_counts": task_mode_counts(train_dataset),
        "training_group_counts": training_group_counts(train_dataset),
        "vocab_size": len(vocab.token_to_id),
        "history": history,
        "prediction_summary": prediction_summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def group_rl_command(args: argparse.Namespace, device: torch.device) -> int:
    train_rows = read_rows(args.train_csv, limit=int(args.limit))
    eval_rows = read_rows(args.eval_csv, limit=int(args.eval_limit)) if args.eval_csv else []
    train_store = FeatureStore(
        args.condition_features_dir,
        array_name=str(args.condition_feature_array),
        variant=str(args.condition_feature_variant),
    )
    eval_store = FeatureStore(
        args.eval_condition_features_dir or args.condition_features_dir,
        array_name=str(args.condition_feature_array),
        variant=str(args.condition_feature_variant),
    )
    checkpoint = load_checkpoint(args.resume_checkpoint)
    if not checkpoint:
        raise ValueError(f"Missing warm-start checkpoint: {args.resume_checkpoint}")
    vocab = SmilesVocabulary.from_dict(checkpoint["vocab"])
    config = dict(checkpoint["model_config"])

    model = ConditionedSmilesDecoder(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    trainable_scope = configure_trainable_scope(model, str(args.trainable_scope))
    reference_model = None
    if float(args.reference_kl_weight) > 0:
        reference_model = ConditionedSmilesDecoder(**config).to(device)
        reference_model.load_state_dict(checkpoint["model_state"])
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad_(False)

    train_dataset = build_dataset(
        train_rows,
        vocab,
        train_store,
        int(config["condition_dim"]),
        max_smiles_length=int(args.max_smiles_length),
        max_source_tokens=int(args.max_source_tokens),
        condition_layout=str(args.condition_layout),
    )
    eval_dataset = build_dataset(
        eval_rows,
        vocab,
        eval_store,
        int(config["condition_dim"]),
        max_smiles_length=int(args.max_smiles_length),
        max_source_tokens=int(args.max_source_tokens),
        condition_layout=str(args.condition_layout),
    )
    if not train_dataset:
        raise ValueError("No trainable rows found. Rows need target_smiles.")

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    history: list[dict[str, object]] = []
    for epoch in range(1, int(args.epochs) + 1):
        record = train_epoch_group_rl(
            model,
            reference_model,
            train_dataset,
            optimizer,
            vocab,
            batch_size=int(args.batch_size),
            device=device,
            rollouts_per_prompt=int(args.rollouts_per_prompt),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            parallel_samples=int(args.parallel_samples),
            max_parallel_sequences=int(args.max_parallel_sequences),
            repetition_penalty=float(args.repetition_penalty),
            no_repeat_ngram_size=int(args.no_repeat_ngram_size),
            min_new_tokens=int(args.min_new_tokens),
            smiles_grammar_constraint=bool(args.smiles_grammar_constraint),
            sft_weight=float(args.sft_weight),
            rl_objective=str(args.rl_objective),
            grpo_clip_eps=float(args.grpo_clip_eps),
            grpo_update_epochs=int(args.grpo_update_epochs),
            advantage_mode=str(args.advantage_mode),
            advantage_clip=float(args.advantage_clip),
            sequence_logprob_reduction=str(args.sequence_logprob_reduction),
            reference_kl_weight=float(args.reference_kl_weight),
            reward_mode=str(args.reward_mode),
            reward_valid_weight=float(args.reward_valid_weight),
            reward_strict_weight=float(args.reward_strict_weight),
            reward_distance_weight=float(args.reward_distance_weight),
            reward_distance_clip=float(args.reward_distance_clip),
            reward_aggregation=str(args.reward_aggregation),
            reward_joint_bonus_weight=float(args.reward_joint_bonus_weight),
            reward_bottleneck_weight=float(args.reward_bottleneck_weight),
            reward_softmin_weight=float(args.reward_softmin_weight),
            reward_softmin_temperature=float(args.reward_softmin_temperature),
            reward_source_similarity_weight=float(args.reward_source_similarity_weight),
            reward_source_similarity_threshold=effective_reward_source_similarity_threshold(args),
            reward_source_copy_penalty=float(args.reward_source_copy_penalty),
            grad_clip=float(args.grad_clip),
            seed=int(args.seed) + epoch,
        )
        record["epoch"] = epoch
        if eval_dataset:
            eval_record = evaluate_group_rl(
                model,
                eval_dataset,
                vocab,
                batch_size=int(args.eval_batch_size),
                device=device,
                rollouts_per_prompt=int(args.rollouts_per_prompt),
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_k=int(args.top_k),
                top_p=float(args.top_p),
                parallel_samples=int(args.parallel_samples),
                max_parallel_sequences=int(args.max_parallel_sequences),
                repetition_penalty=float(args.repetition_penalty),
                no_repeat_ngram_size=int(args.no_repeat_ngram_size),
                min_new_tokens=int(args.min_new_tokens),
                smiles_grammar_constraint=bool(args.smiles_grammar_constraint),
                reward_mode=str(args.reward_mode),
                reward_valid_weight=float(args.reward_valid_weight),
                reward_strict_weight=float(args.reward_strict_weight),
                reward_distance_weight=float(args.reward_distance_weight),
                reward_distance_clip=float(args.reward_distance_clip),
                reward_aggregation=str(args.reward_aggregation),
                reward_joint_bonus_weight=float(args.reward_joint_bonus_weight),
                reward_bottleneck_weight=float(args.reward_bottleneck_weight),
                reward_softmin_weight=float(args.reward_softmin_weight),
                reward_softmin_temperature=float(args.reward_softmin_temperature),
                reward_source_similarity_weight=float(args.reward_source_similarity_weight),
                reward_source_similarity_threshold=effective_reward_source_similarity_threshold(args),
                reward_source_copy_penalty=float(args.reward_source_copy_penalty),
            )
            record.update({f"eval_{key}": value for key, value in eval_record.items()})
        history.append(record)
        save_checkpoint(args.output_dir / "latest_group_rl_checkpoint.pt", model, optimizer, vocab, config, epoch, history, args)

    checkpoint_path = args.output_dir / "unified_smiles_generator_group_rl.pt"
    save_checkpoint(checkpoint_path, model, optimizer, vocab, config, len(history), history, args)

    prediction_summary = None
    if eval_rows:
        prediction_summary = write_predictions(
            model,
            eval_rows,
            eval_store,
            vocab,
            int(config["condition_dim"]),
            args=args,
            device=device,
        )

    summary = {
        "checkpoint": str(checkpoint_path),
        "warm_start_checkpoint": str(args.resume_checkpoint),
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv) if args.eval_csv else None,
        "train_rows": len(train_dataset),
        "eval_rows": len(eval_dataset),
        "task_mode_counts": task_mode_counts(train_dataset),
        "eval_task_mode_counts": task_mode_counts(eval_dataset),
        "condition_dim": int(config["condition_dim"]),
        "condition_feature_variant": str(args.condition_feature_variant),
        "condition_layout": str(args.condition_layout),
        "input_modality": input_modality_for_args(args),
        "vocab_size": len(vocab.token_to_id),
        "rl_objective": str(args.rl_objective),
        "grpo_clip_eps": float(args.grpo_clip_eps),
        "grpo_update_epochs": int(args.grpo_update_epochs),
        "reward_mode": str(args.reward_mode),
        "reward_aggregation": str(args.reward_aggregation),
        "reward_joint_bonus_weight": float(args.reward_joint_bonus_weight),
        "reward_bottleneck_weight": float(args.reward_bottleneck_weight),
        "reward_softmin_weight": float(args.reward_softmin_weight),
        "reward_softmin_temperature": float(args.reward_softmin_temperature),
        "smiles_grammar_constraint": bool(args.smiles_grammar_constraint),
        "reward_source_similarity_threshold": effective_reward_source_similarity_threshold(args),
        "trainable_scope": trainable_scope,
        "history": history,
        "prediction_summary": prediction_summary,
    }
    (args.output_dir / "group_rl_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def sample_command(args: argparse.Namespace, device: torch.device) -> int:
    checkpoint = load_checkpoint(args.checkpoint)
    if not checkpoint:
        raise ValueError(f"Missing checkpoint: {args.checkpoint}")
    vocab = SmilesVocabulary.from_dict(checkpoint["vocab"])
    config = dict(checkpoint["model_config"])
    model = ConditionedSmilesDecoder(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    rows = read_rows(args.eval_csv, limit=int(args.eval_limit))
    store = FeatureStore(
        args.eval_condition_features_dir or args.condition_features_dir,
        array_name=str(args.condition_feature_array),
        variant=str(args.condition_feature_variant),
    )
    summary = write_predictions(
        model,
        rows,
        store,
        vocab,
        int(config["condition_dim"]),
        args=args,
        device=device,
    )
    (args.output_dir / "sample_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def read_rows(path: Path | None, *, limit: int = 0) -> list[dict[str, str]]:
    if path is None:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:limit] if limit and limit > 0 else rows


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_checkpoint(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def build_distillation_teacher(
    checkpoint_path: Path | None,
    *,
    expected_vocab: SmilesVocabulary,
    expected_config: Mapping[str, object],
    device: torch.device,
    allow_architecture_mismatch: bool = False,
) -> ConditionedSmilesDecoder | None:
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is None:
        return None
    teacher_vocab = SmilesVocabulary.from_dict(checkpoint["vocab"])
    if teacher_vocab.to_dict() != expected_vocab.to_dict():
        raise ValueError("Teacher and student vocabularies differ; protected distillation requires identical token ids.")
    teacher_config = dict(checkpoint["model_config"])
    if teacher_config != dict(expected_config) and not allow_architecture_mismatch:
        raise ValueError("Teacher and student model configs differ; protected distillation requires matching architectures.")
    for key in ("vocab_size", "condition_dim", "d_model", "num_layers", "num_heads", "dim_feedforward"):
        if teacher_config.get(key) != dict(expected_config).get(key):
            raise ValueError(f"Teacher/student core config mismatch for {key}")
    teacher = ConditionedSmilesDecoder(**teacher_config).to(device)
    teacher.load_state_dict(checkpoint["model_state"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


SOURCE_ONLY_PREFIXES = (
    "source_condition_proj.",
    "source_encoder.",
    "source_type",
    "null_source",
    "source_gate.",
    "source_output.",
    "source_adapters.",
    "source_copy_query.",
    "source_copy_key.",
    "source_copy_gate.",
)


def configure_trainable_scope(model: ConditionedSmilesDecoder, scope: str) -> dict[str, object]:
    normalized = str(scope or "all")
    if normalized == "all":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    elif normalized == "source_only":
        if not bool(model.source_aware):
            raise ValueError("source_only training requires a source-aware checkpoint")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(SOURCE_ONLY_PREFIXES))
    else:
        raise ValueError(f"Unsupported trainable scope: {scope}")
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable <= 0:
        raise ValueError(f"Trainable scope {normalized} selected no parameters")
    return {
        "scope": normalized,
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(total - trainable),
        "trainable_prefixes": list(SOURCE_ONLY_PREFIXES) if normalized == "source_only" else ["*"],
        "trainable_parameter_names": trainable_names,
    }


class FeatureStore:
    def __init__(self, feature_dir: Path | None, *, array_name: str = "query_tokens", variant: str = "full") -> None:
        self.feature_dir = Path(feature_dir) if feature_dir else None
        self.array_name = array_name
        self.variant = str(variant or "").strip()
        self.features: np.ndarray | None = None
        self.index: dict[str, int] = {}
        self.input_hidden_dim: int | None = None
        if self.feature_dir is not None:
            self._load()

    def _load(self) -> None:
        assert self.feature_dir is not None
        index_path = self.feature_dir / "index.csv"
        array_path = self.feature_dir / ("pooled.npy" if self.array_name == "pooled" else "query_tokens.npy")
        if not index_path.exists():
            raise FileNotFoundError(index_path)
        if not array_path.exists():
            raise FileNotFoundError(array_path)
        with index_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        features = np.load(array_path).astype(np.float32)
        if len(rows) != int(features.shape[0]):
            raise ValueError(f"Feature row mismatch: {index_path} vs {array_path}")
        self.features = features
        self.input_hidden_dim = int(features.shape[-1])
        for idx, row in enumerate(rows):
            row_variant = str(row.get("variant", "") or "").strip()
            if self.variant and row_variant and row_variant != self.variant:
                continue
            for key in ("variant_id", "condition_id", "sample_id", "pair_id"):
                value = str(row.get(key, "") or "").strip()
                if value and value not in self.index:
                    self.index[value] = idx

    def get(self, row: Mapping[str, str]) -> np.ndarray | None:
        if self.features is None:
            return None
        explicit_variant_id = str(row.get("variant_id", "") or "").strip()
        if explicit_variant_id:
            explicit_parts = [explicit_variant_id]
        else:
            condition_id = str(row.get("condition_id", "") or row.get("sample_id", "") or row.get("pair_id", "") or "").strip()
            explicit_parts = [f"{condition_id}:{self.variant}"] if condition_id and self.variant else []
        for value in explicit_parts:
            if value in self.index:
                arr = np.asarray(self.features[self.index[value]], dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[None, :]
                return arr
        for key in ("condition_id", "sample_id", "pair_id"):
            value = str(row.get(key, "") or "").strip()
            if value and value in self.index:
                arr = np.asarray(self.features[self.index[value]], dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[None, :]
                return arr
        return None


def infer_condition_dim(*stores: FeatureStore, default: int) -> int:
    for store in stores:
        if store.input_hidden_dim is not None:
            return int(store.input_hidden_dim)
    return int(default)


def tokenize_smiles(smiles: str) -> list[str]:
    text = str(smiles or "").strip()
    if not text:
        return []
    return [token for token in SMILES_TOKEN_RE.findall(text) if token]


def detokenize_smiles(tokens: Iterable[str]) -> str:
    skip = {PAD, BOS, EOS}
    return "".join(token for token in tokens if token not in skip)


@dataclass
class SmilesVocabulary:
    token_to_id: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for token in SPECIAL_TOKENS:
            self.add(token)

    @property
    def id_to_token(self) -> list[str]:
        return [token for token, _ in sorted(self.token_to_id.items(), key=lambda item: item[1])]

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS]

    def add(self, token: str) -> int:
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.token_to_id)
        return self.token_to_id[token]

    def update(self, token_sequences: Iterable[Iterable[str]]) -> None:
        for tokens in token_sequences:
            for token in tokens:
                self.add(token)

    def encode(self, tokens: Iterable[str], *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.token_to_id.get(token, self.token_to_id[UNK]) for token in tokens)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int]) -> list[str]:
        tokens = self.id_to_token
        out = []
        for value in ids:
            idx = int(value)
            if idx == self.eos_id:
                break
            out.append(tokens[idx] if 0 <= idx < len(tokens) else UNK)
        return out

    def to_dict(self) -> dict[str, int]:
        return dict(self.token_to_id)

    @classmethod
    def from_dict(cls, payload: dict[str, int]) -> "SmilesVocabulary":
        vocab = cls()
        vocab.token_to_id = dict(payload)
        return vocab


def build_vocabulary(smiles_values: Sequence[str]) -> SmilesVocabulary:
    vocab = SmilesVocabulary()
    vocab.update(tokenize_smiles(value) for value in smiles_values)
    return vocab


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512) -> None:
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / max(dim, 1)))
        pe = torch.zeros(max_len, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(positions * div_term)
        if dim > 1:
            pe[:, 1::2] = torch.cos(positions * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.pe[:, : values.shape[1]].to(dtype=values.dtype, device=values.device)


class SourceGatedAdapter(nn.Module):
    """Small edit-only residual block; the null-source path is an exact identity."""

    def __init__(self, d_model: int, bottleneck: int) -> None:
        super().__init__()
        hidden = max(1, int(bottleneck))
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model * 2, hidden)
        self.up = nn.Linear(hidden, d_model)
        self.gate = nn.Linear(d_model * 2, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(
        self,
        decoded: torch.Tensor,
        source_summary: torch.Tensor,
        source_present: torch.Tensor,
    ) -> torch.Tensor:
        expanded_source = source_summary[:, None, :].expand(-1, decoded.shape[1], -1)
        joined = torch.cat([self.norm(decoded), expanded_source], dim=-1)
        delta = self.up(F.gelu(self.down(joined)))
        gate = torch.sigmoid(self.gate(joined))
        return decoded + source_present[:, None, None].to(decoded.dtype) * gate * delta


class ConditionedSmilesDecoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        condition_dim: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pad_id: int = 0,
        max_length: int = 192,
        source_aware: bool = False,
        source_encoder_layers: int = 2,
        source_residual_scale: float = 1.0,
        source_copy_aware: bool = False,
        source_adapter_layers: int = 0,
        source_adapter_bottleneck: int = 64,
        source_copy_initial_vocab_bias: float = 2.0,
    ) -> None:
        super().__init__()
        self.pad_id = int(pad_id)
        self.source_aware = bool(source_aware)
        self.source_copy_aware = bool(source_copy_aware)
        if self.source_copy_aware and not self.source_aware:
            raise ValueError("source_copy_aware requires source_aware")
        self.source_residual_scale = float(source_residual_scale)
        self.condition_proj = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        if self.source_aware:
            self.source_condition_proj = nn.Sequential(
                nn.LayerNorm(condition_dim),
                nn.Linear(condition_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
            source_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.source_encoder = nn.TransformerEncoder(
                source_layer,
                num_layers=max(1, int(source_encoder_layers)),
            )
            self.source_type = nn.Parameter(torch.zeros(1, 1, d_model))
            self.null_source = nn.Parameter(torch.zeros(1, 1, d_model))
            self.source_gate = nn.Linear(d_model * 2, d_model)
            self.source_output = nn.Linear(d_model, vocab_size, bias=False)
            self.source_adapters = nn.ModuleList(
                SourceGatedAdapter(d_model, source_adapter_bottleneck)
                for _ in range(max(0, int(source_adapter_layers)))
            )
            if self.source_copy_aware:
                self.source_copy_query = nn.Linear(d_model, d_model, bias=False)
                self.source_copy_key = nn.Linear(d_model, d_model, bias=False)
                self.source_copy_gate = nn.Linear(d_model * 2, 1)
                nn.init.constant_(self.source_copy_gate.bias, float(source_copy_initial_vocab_bias))
            nn.init.normal_(self.source_type, std=0.02)
            nn.init.normal_(self.null_source, std=0.02)
            nn.init.zeros_(self.source_output.weight)
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position = PositionalEncoding(d_model, max_len=max_length + 8)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        condition_tokens: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        *,
        condition_mask: torch.Tensor | None = None,
        source_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory, memory_padding, source_summary, source_present, source_memory, source_valid = self.encode_condition_memory(
            condition_tokens,
            condition_mask=condition_mask,
        )
        target = self.position(self.token_embedding(decoder_input_ids))
        seq_len = decoder_input_ids.shape[1]
        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=decoder_input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        target_padding = decoder_input_ids.eq(self.pad_id)
        decoded = self.decoder(
            target,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=target_padding,
            memory_key_padding_mask=memory_padding,
        )
        if self.source_aware and source_summary is not None and source_present is not None:
            for adapter in self.source_adapters:
                decoded = adapter(decoded, source_summary, source_present)
        logits = self.output(decoded)
        if self.source_aware and source_summary is not None and source_present is not None:
            expanded_source = source_summary[:, None, :].expand(-1, decoded.shape[1], -1)
            gate = torch.sigmoid(self.source_gate(torch.cat([decoded, expanded_source], dim=-1)))
            source_delta = self.source_output(torch.tanh(decoded + gate * expanded_source))
            logits = logits + (
                float(self.source_residual_scale)
                * source_present[:, None, None].to(dtype=logits.dtype)
                * source_delta
            )
        if self.source_copy_aware and source_present is not None and bool(source_present.any()):
            if source_token_ids is None:
                raise ValueError("source_copy_aware edit rows require source_token_ids")
            if source_memory is None or source_valid is None:
                raise RuntimeError("source-copy memory was not constructed")
            if source_token_ids.shape != source_valid.shape:
                raise ValueError(
                    f"source_token_ids shape {tuple(source_token_ids.shape)} does not match "
                    f"source memory {tuple(source_valid.shape)}"
                )
            query = self.source_copy_query(decoded)
            key = self.source_copy_key(source_memory)
            attention_logits = torch.einsum("btd,bsd->bts", query, key) / math.sqrt(max(query.shape[-1], 1))
            attention_logits = attention_logits.masked_fill(~source_valid[:, None, :], -torch.inf)
            copy_attention = torch.softmax(attention_logits, dim=-1)
            copy_attention = torch.nan_to_num(copy_attention, nan=0.0, posinf=0.0, neginf=0.0)
            copy_probs = logits.new_zeros(logits.shape)
            copy_index = source_token_ids[:, None, :].expand(-1, decoded.shape[1], -1)
            copy_probs.scatter_add_(dim=-1, index=copy_index, src=copy_attention)
            expanded_source = source_summary[:, None, :].expand(-1, decoded.shape[1], -1)
            vocab_gate = torch.sigmoid(self.source_copy_gate(torch.cat([decoded, expanded_source], dim=-1)))
            base_probs = torch.softmax(logits, dim=-1)
            mixed_probs = vocab_gate * base_probs + (1.0 - vocab_gate) * copy_probs
            mixed_logits = mixed_probs.clamp_min(torch.finfo(mixed_probs.dtype).tiny).log()
            logits = torch.where(source_present[:, None, None], mixed_logits, logits)
        return logits

    def encode_condition_memory(
        self,
        condition_tokens: torch.Tensor,
        *,
        condition_mask: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        valid = (
            torch.ones(condition_tokens.shape[:2], dtype=torch.bool, device=condition_tokens.device)
            if condition_mask is None
            else condition_mask.bool()
        )
        base_memory = self.condition_proj(condition_tokens)
        if not self.source_aware:
            return base_memory, ~valid, None, None, None, None

        # Source token features are emitted by source_token_feature with an exact
        # leading marker of 2.0. Goal/VLM/property tokens stay in the other stream.
        source_valid = valid & condition_tokens[..., 0].sub(2.0).abs().lt(1e-6)
        goal_valid = valid & ~source_valid
        source_present = source_valid.any(dim=1)

        safe_source_valid = source_valid.clone()
        missing = ~source_present
        if bool(missing.any()):
            safe_source_valid[missing, 0] = True
        source_memory = self.source_condition_proj(condition_tokens) + self.source_type
        if bool(missing.any()):
            source_memory = source_memory.clone()
            source_memory[missing, 0, :] = self.null_source[0, 0, :]
        source_memory = self.source_encoder(
            source_memory,
            src_key_padding_mask=~safe_source_valid,
        )

        source_weights = source_valid.to(dtype=source_memory.dtype)
        source_summary = (source_memory * source_weights[..., None]).sum(dim=1)
        source_summary = source_summary / source_weights.sum(dim=1, keepdim=True).clamp_min(1.0)

        # Keep the null source out of decoder attention so de novo behavior is
        # exactly the legacy goal-conditioned path after a compatible warm-start.
        memory = torch.cat([base_memory, source_memory], dim=1)
        memory_valid = torch.cat([goal_valid, source_valid], dim=1)
        return memory, ~memory_valid, source_summary, source_present, source_memory, source_valid

    @torch.no_grad()
    def generate(
        self,
        condition_tokens: torch.Tensor,
        *,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int,
        condition_mask: torch.Tensor | None = None,
        source_token_ids: torch.Tensor | None = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        min_new_tokens: int = 0,
        blocked_token_ids: Sequence[int] | None = None,
        smiles_token_text: Sequence[str] | None = None,
    ) -> torch.Tensor:
        batch = condition_tokens.shape[0]
        device = condition_tokens.device
        generated = torch.full((batch, 1), int(bos_id), dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        blocked_ids = {int(bos_id), self.pad_id}
        blocked_ids.update(int(value) for value in (blocked_token_ids or ()))
        for step in range(max(1, int(max_new_tokens))):
            logits = self(
                condition_tokens,
                generated,
                condition_mask=condition_mask,
                source_token_ids=source_token_ids,
            )[:, -1, :]
            logits[:, list(blocked_ids)] = -torch.inf
            if smiles_token_text is not None:
                apply_smiles_grammar_mask_(
                    logits,
                    generated,
                    token_text=smiles_token_text,
                    eos_id=int(eos_id),
                )
            if step < max(0, int(min_new_tokens)):
                logits[:, int(eos_id)] = -torch.inf
            if repetition_penalty and repetition_penalty > 1.0:
                apply_repetition_penalty_(logits, generated, float(repetition_penalty))
            if no_repeat_ngram_size and no_repeat_ngram_size > 0:
                mask_repeated_ngrams_(logits, generated, int(no_repeat_ngram_size))
            if temperature and temperature > 0:
                logits = logits / float(temperature)
                if top_k > 0 and top_k < logits.shape[-1]:
                    threshold = torch.topk(logits, int(top_k), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, -torch.inf)
                logits = top_p_filter(logits, top_p=float(top_p))
                probs = torch.softmax(logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                zero_rows = probs.sum(dim=-1).le(0)
                if bool(zero_rows.any()):
                    fallback = torch.zeros_like(probs)
                    fallback[:, int(eos_id)] = 1.0
                    probs = torch.where(zero_rows[:, None], fallback, probs)
                next_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_ids = logits.argmax(dim=-1)
            next_ids = torch.where(finished, torch.full_like(next_ids, int(eos_id)), next_ids)
            generated = torch.cat([generated, next_ids[:, None]], dim=1)
            finished |= next_ids.eq(int(eos_id))
            if bool(finished.all()):
                break
        return generated

    @torch.no_grad()
    def beam_search(
        self,
        condition_tokens: torch.Tensor,
        *,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int,
        condition_mask: torch.Tensor | None = None,
        source_token_ids: torch.Tensor | None = None,
        beam_size: int = 20,
        expand_size: int = 64,
        length_penalty: float = 0.8,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        min_new_tokens: int = 0,
        blocked_token_ids: Sequence[int] | None = None,
        smiles_token_text: Sequence[str] | None = None,
    ) -> torch.Tensor:
        if condition_tokens.shape[0] != 1:
            raise ValueError("beam_search expects one condition row at a time")
        device = condition_tokens.device
        beam_size = max(1, int(beam_size))
        expand_size = max(1, int(expand_size))
        beams: list[tuple[list[int], float, bool]] = [([int(bos_id)], 0.0, False)]
        blocked_ids = {int(bos_id), self.pad_id}
        blocked_ids.update(int(value) for value in (blocked_token_ids or ()))
        for step in range(max(1, int(max_new_tokens))):
            active: list[tuple[list[int], float, bool]] = []
            finished: list[tuple[list[int], float, bool]] = []
            for sequence, score, done in beams:
                if done:
                    finished.append((sequence, score, True))
                    continue
                seq_tensor = torch.tensor([sequence], dtype=torch.long, device=device)
                logits = self(
                    condition_tokens,
                    seq_tensor,
                    condition_mask=condition_mask,
                    source_token_ids=source_token_ids,
                )[:, -1, :]
                logits[:, list(blocked_ids)] = -torch.inf
                if smiles_token_text is not None:
                    apply_smiles_grammar_mask_(
                        logits,
                        seq_tensor,
                        token_text=smiles_token_text,
                        eos_id=int(eos_id),
                    )
                if step < max(0, int(min_new_tokens)):
                    logits[:, int(eos_id)] = -torch.inf
                if repetition_penalty and repetition_penalty > 1.0:
                    apply_repetition_penalty_(logits, seq_tensor, float(repetition_penalty))
                if no_repeat_ngram_size and no_repeat_ngram_size > 0:
                    mask_repeated_ngrams_(logits, seq_tensor, int(no_repeat_ngram_size))
                log_probs = torch.log_softmax(logits, dim=-1)
                top_count = min(max(expand_size, beam_size), log_probs.shape[-1])
                values, indices = torch.topk(log_probs, top_count, dim=-1)
                for value, token_id in zip(values[0].tolist(), indices[0].tolist()):
                    if not math.isfinite(float(value)):
                        continue
                    next_sequence = sequence + [int(token_id)]
                    next_done = int(token_id) == int(eos_id)
                    active.append((next_sequence, score + float(value), next_done))
            pool = finished + active
            if not pool:
                break
            beams = sorted(
                pool,
                key=lambda item: normalized_beam_score(item[1], len(item[0]), length_penalty),
                reverse=True,
            )[:beam_size]
            if all(done for _, _, done in beams):
                break
        if not beams:
            beams = [([int(bos_id), int(eos_id)], 0.0, True)]
        max_len = max(len(sequence) for sequence, _, _ in beams)
        output = torch.full((len(beams), max_len), int(eos_id), dtype=torch.long, device=device)
        for idx, (sequence, _, _) in enumerate(beams):
            output[idx, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        return output


SMILES_BOND_TOKENS = frozenset({"-", "=", "#", "/", "\\", ":", "~"})


def smiles_token_kind(token: str) -> str:
    if token.startswith("[") and token.endswith("]"):
        return "atom"
    if re.fullmatch(r"(?:[A-Z][a-z]?|[bcnops]|\*)", token):
        return "atom"
    if token in SMILES_BOND_TOKENS:
        return "bond"
    if token == "(":
        return "branch_open"
    if token == ")":
        return "branch_close"
    if re.fullmatch(r"(?:\d|%\d{2})", token):
        return "ring"
    if token == ".":
        return "dot"
    return "other"


def smiles_grammar_allowed_ids(
    sequence: Sequence[int],
    *,
    token_text: Sequence[str],
    eos_id: int,
) -> set[int]:
    depth = 0
    open_rings: set[str] = set()
    previous = "start"
    has_atom = False
    for token_id in sequence:
        index = int(token_id)
        if not 0 <= index < len(token_text):
            continue
        token = token_text[index]
        if token in SPECIAL_TOKENS:
            continue
        kind = smiles_token_kind(token)
        if kind == "atom":
            has_atom = True
        elif kind == "branch_open":
            depth += 1
        elif kind == "branch_close":
            depth = max(0, depth - 1)
        elif kind == "ring":
            if token in open_rings:
                open_rings.remove(token)
            else:
                open_rings.add(token)
        previous = kind

    expecting_atom = previous in {"start", "bond", "branch_open", "dot", "other"}
    allowed: set[int] = set()
    for token_id, token in enumerate(token_text):
        kind = smiles_token_kind(token)
        if kind == "atom":
            allowed.add(token_id)
        elif not expecting_atom and kind in {"bond", "branch_open", "ring", "dot"}:
            allowed.add(token_id)
        elif not expecting_atom and kind == "branch_close" and depth > 0:
            allowed.add(token_id)
    if has_atom and not expecting_atom and depth == 0 and not open_rings:
        allowed.add(int(eos_id))
    return allowed


def apply_smiles_grammar_mask_(
    logits: torch.Tensor,
    generated: torch.Tensor,
    *,
    token_text: Sequence[str],
    eos_id: int,
) -> None:
    for row_index in range(generated.shape[0]):
        allowed = smiles_grammar_allowed_ids(
            generated[row_index].tolist(),
            token_text=token_text,
            eos_id=eos_id,
        )
        if not allowed:
            continue
        blocked = [token_id for token_id in range(logits.shape[-1]) if token_id not in allowed]
        logits[row_index, blocked] = -torch.inf


def apply_repetition_penalty_(logits: torch.Tensor, generated: torch.Tensor, penalty: float) -> None:
    for row_idx in range(generated.shape[0]):
        for token_id in set(int(value) for value in generated[row_idx].tolist()):
            value = logits[row_idx, token_id]
            logits[row_idx, token_id] = value / penalty if value > 0 else value * penalty


def normalized_beam_score(score: float, length: int, length_penalty: float) -> float:
    length_value = max(int(length) - 1, 1)
    if length_penalty <= 0:
        return float(score)
    return float(score) / (float(length_value) ** float(length_penalty))


def mask_repeated_ngrams_(logits: torch.Tensor, generated: torch.Tensor, ngram_size: int) -> None:
    if ngram_size <= 0:
        return
    for row_idx in range(generated.shape[0]):
        banned = banned_ngram_tokens(generated[row_idx].tolist(), ngram_size)
        if banned:
            logits[row_idx, sorted(banned)] = -torch.inf


def banned_ngram_tokens(sequence: list[int], ngram_size: int) -> set[int]:
    if len(sequence) + 1 < ngram_size:
        return set()
    prefix_len = ngram_size - 1
    prefix = tuple(sequence[-prefix_len:]) if prefix_len > 0 else tuple()
    banned: set[int] = set()
    for idx in range(0, len(sequence) - ngram_size + 1):
        ngram = tuple(sequence[idx : idx + ngram_size])
        if prefix_len == 0 or ngram[:-1] == prefix:
            banned.add(int(ngram[-1]))
    return banned


def top_p_filter(logits: torch.Tensor, *, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative > float(top_p)
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    filtered = logits.clone()
    filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits.masked_fill(remove, -torch.inf))
    return filtered


def build_dataset(
    rows: list[dict[str, str]],
    vocab: SmilesVocabulary,
    store: FeatureStore,
    condition_dim: int,
    *,
    max_smiles_length: int,
    max_source_tokens: int,
    condition_layout: str = "unified",
) -> list[dict[str, object]]:
    dataset = []
    for row in rows:
        target = str(row.get("target_smiles", "") or "").strip()
        if not target:
            continue
        tokens = tokenize_smiles(target)[: max(1, int(max_smiles_length))]
        decoder_input = vocab.encode(tokens, add_bos=True, add_eos=False)
        target_ids = vocab.encode(tokens, add_bos=False, add_eos=True)
        condition = condition_array_for_row(
            row,
            store,
            condition_dim,
            max_source_tokens=max_source_tokens,
            condition_layout=condition_layout,
        )
        source_token_ids = source_token_ids_for_condition(
            row,
            vocab,
            condition,
            max_source_tokens=max_source_tokens,
        )
        dataset.append(
            {
                "row": dict(row),
                "condition": condition.astype(np.float32),
                "source_token_ids": source_token_ids,
                "decoder_input_ids": np.asarray(decoder_input, dtype=np.int64),
                "target_ids": np.asarray(target_ids, dtype=np.int64),
                "task_mode": task_mode_for_row(row),
            }
        )
    return dataset


def condition_array_for_row(
    row: Mapping[str, str],
    store: FeatureStore,
    condition_dim: int,
    *,
    max_source_tokens: int,
    condition_layout: str = "unified",
) -> np.ndarray:
    layout = str(condition_layout or "unified")
    if layout == "p6_transition":
        base = store.get(row)
        if base is None:
            base = direct_cond.fallback_condition_features(row, condition_dim)
        if int(base.shape[-1]) != int(condition_dim):
            raise ValueError(f"Condition feature dim mismatch: {base.shape[-1]} != {condition_dim}")
        source_text = str(row.get("source_smiles", "") or row.get("molecule_smiles", "") or "").strip()
        # This is an initial-graph state observation, not a task router: it is
        # derived entirely from whether the supplied graph is empty and feeds
        # the same decoder/head/interpreter in both cases.
        initial_state = expand_condition_token(
            [4.0, float(not source_text), float(bool(source_text)), 0.0],
            condition_dim,
        )[None, :].astype(np.float32)
        program = direct_cond.property_program_tokens(row, condition_dim)
        if source_text:
            source = source_smiles_condition_tokens(row, condition_dim, max_source_tokens=max_source_tokens)
            return np.concatenate([base, initial_state, source, program], axis=0)
        return np.concatenate([base, initial_state, program], axis=0)
    if layout in {"transformation", "direct_compat", "direct_edit_compat", "property_program_only"}:
        if layout == "property_program_only":
            return direct_cond.property_program_tokens(row, condition_dim)
        base = store.get(row)
        if base is None:
            base = direct_cond.fallback_condition_features(row, condition_dim)
        if int(base.shape[-1]) != int(condition_dim):
            raise ValueError(f"Condition feature dim mismatch: {base.shape[-1]} != {condition_dim}")
        program = direct_cond.property_program_tokens(row, condition_dim)
        if task_mode_for_row(row) == EDIT_MODE:
            source = source_smiles_condition_tokens(row, condition_dim, max_source_tokens=max_source_tokens)
            return np.concatenate([base, source, program], axis=0)
        return np.concatenate([base, program], axis=0)

    mode = task_mode_for_row(row)
    mode_token = mode_condition_token(mode, condition_dim)
    program = property_program_tokens(row, condition_dim)
    base = store.get(row)
    if base is None:
        base = fallback_condition_features(row, condition_dim)
    if int(base.shape[-1]) != int(condition_dim):
        raise ValueError(f"Condition feature dim mismatch: {base.shape[-1]} != {condition_dim}")
    if mode == EDIT_MODE:
        source = source_smiles_condition_tokens(row, condition_dim, max_source_tokens=max_source_tokens)
        return np.concatenate([base, mode_token, source, program], axis=0)
    return np.concatenate([base, mode_token, program], axis=0)


def task_mode_for_row(row: Mapping[str, str]) -> str:
    raw = str(row.get("task_mode", "") or row.get("unified_task_mode", "") or "").strip().lower()
    normalized = raw.replace("-", "_").replace(" ", "_")
    if normalized in {"de_novo", "denovo", "generate", "generation"}:
        return DE_NOVO_MODE
    if normalized in {"edit", "conditional_edit", "source_edit", "edit_generation"}:
        return EDIT_MODE
    source = str(row.get("source_smiles", "") or row.get("molecule_smiles", "") or "").strip()
    return EDIT_MODE if source else DE_NOVO_MODE


def mode_condition_token(mode: str, condition_dim: int) -> np.ndarray:
    is_edit = 1.0 if mode == EDIT_MODE else 0.0
    is_denovo = 1.0 - is_edit
    return expand_condition_token([3.0, is_denovo, is_edit, 0.0], condition_dim)[None, :].astype(np.float32)


def fallback_condition_features(row: Mapping[str, str], condition_dim: int) -> np.ndarray:
    values = []
    active_props = set(selected_properties(row))
    for prop in PROPERTY_COLUMNS:
        value = parse_float(first_present(row, [f"target_{prop}", f"target_{prop.lower()}"]))
        normalizer = PROPERTY_NORMALIZERS.get(prop, 1.0)
        values.append(0.0 if math.isnan(value) else float(value) / normalizer)
    for prop in PROPERTY_COLUMNS:
        active = truthy(first_present(row, [f"{prop}_active", f"{prop.lower()}_active"]))
        values.append(1.0 if (active if active is not None else prop in active_props) else 0.0)
    for prop in PROPERTY_COLUMNS:
        direction = property_direction(row, prop)
        values.append(float(direction))
    values.append(float(len(active_props)) / max(len(PROPERTY_COLUMNS), 1))
    values.extend(hash_text_features(str(row.get("instruction", "") or row.get("prompt", "")), 16))
    return expand_condition_token(values, condition_dim)[None, :].astype(np.float32)


def property_program_tokens(row: Mapping[str, str], condition_dim: int) -> np.ndarray:
    selected = selected_properties(row)
    selected_set = set(selected)
    count_norm = float(len(selected)) / max(len(PROPERTY_COLUMNS), 1)
    direction_values = [property_direction(row, prop) for prop in PROPERTY_COLUMNS]
    positive_fraction = sum(1 for value in direction_values if value > 0) / max(len(direction_values), 1)
    negative_fraction = sum(1 for value in direction_values if value < 0) / max(len(direction_values), 1)
    raw_task_text = ",".join(
        str(row.get(key, "") or "")
        for key in ("condition_properties", "external_task_properties", "external_property_directions_json")
    )
    tokens = [
        expand_condition_token(
            [
                0.25,
                count_norm,
                positive_fraction,
                negative_fraction,
                float(task_mode_for_row(row) == EDIT_MODE),
            ]
            + hash_text_features(raw_task_text, 11),
            condition_dim,
        )
    ]
    for idx, prop in enumerate(PROPERTY_COLUMNS):
        target = parse_float(first_present(row, [f"target_{prop}", f"target_{prop.lower()}"]))
        normalizer = PROPERTY_NORMALIZERS.get(prop, 1.0)
        direction = property_direction(row, prop)
        tokens.append(
            expand_condition_token(
                [
                    1.0,
                    float(idx + 1) / max(len(PROPERTY_COLUMNS), 1),
                    1.0 if prop in selected_set else 0.0,
                    0.0 if math.isnan(target) else float(target) / max(normalizer, 1e-8),
                    float(direction),
                    STRICT_TOLERANCE.get(prop, normalizer) / max(normalizer, 1e-8),
                    count_norm,
                    0.0 if math.isnan(target) else 1.0,
                ],
                condition_dim,
            )
        )
    return np.stack(tokens, axis=0).astype(np.float32)


def source_smiles_condition_tokens(
    row: Mapping[str, str],
    condition_dim: int,
    *,
    max_source_tokens: int,
) -> np.ndarray:
    source_smiles = str(row.get("source_smiles", "") or row.get("molecule_smiles", "") or "").strip()
    if not source_smiles:
        return np.zeros((1, max(1, int(condition_dim))), dtype=np.float32)
    tokens = tokenize_smiles(source_smiles)[: max(1, int(max_source_tokens))]
    if not tokens:
        return np.zeros((1, max(1, int(condition_dim))), dtype=np.float32)
    source_length = max(len(tokens), 1)
    rows = [source_token_feature(token, idx, source_length, condition_dim) for idx, token in enumerate(tokens)]
    return np.stack(rows, axis=0).astype(np.float32)


def source_token_ids_for_condition(
    row: Mapping[str, str],
    vocab: SmilesVocabulary,
    condition: np.ndarray,
    *,
    max_source_tokens: int,
) -> np.ndarray:
    """Align source SMILES vocabulary ids with the marked source-memory slots."""
    source_smiles = str(row.get("source_smiles", "") or row.get("molecule_smiles", "") or "").strip()
    tokens = tokenize_smiles(source_smiles)[: max(1, int(max_source_tokens))] if source_smiles else []
    aligned = np.full(int(condition.shape[0]), int(vocab.pad_id), dtype=np.int64)
    if not tokens:
        return aligned
    source_positions = np.flatnonzero(np.isclose(condition[:, 0], 2.0, atol=1e-6))
    token_ids = vocab.encode(tokens)
    count = min(len(source_positions), len(token_ids))
    if count:
        aligned[source_positions[:count]] = np.asarray(token_ids[:count], dtype=np.int64)
    return aligned


def source_token_feature(token: str, index: int, source_length: int, condition_dim: int) -> np.ndarray:
    dim = max(1, int(condition_dim))
    vec = np.zeros(dim, dtype=np.float32)
    safe_set(vec, 0, 2.0)
    safe_set(vec, 1, float(index + 1) / max(float(source_length), 1.0))
    safe_set(vec, 2, float(source_length) / 160.0)
    safe_set(vec, 3, float(len(str(token))) / 16.0)
    safe_set(vec, 4, 1.0 if is_atom_token(token) else 0.0)
    safe_set(vec, 5, 1.0 if str(token).islower() else 0.0)
    safe_set(vec, 6, 1.0 if is_bond_token(token) else 0.0)
    safe_set(vec, 7, 1.0 if str(token).isdigit() or str(token).startswith("%") else 0.0)
    safe_set(vec, 8, 1.0 if str(token) in {"(", ")"} else 0.0)
    safe_set(vec, 9, 1.0 if str(token).startswith("[") and str(token).endswith("]") else 0.0)
    bucket_space = max(dim - 16, 1)
    token_hash = stable_hash_int(str(token))
    bucket = 16 + (token_hash % bucket_space)
    if bucket < dim:
        vec[bucket] = 1.0
    second_bucket = 16 + ((token_hash // max(bucket_space, 1)) % bucket_space)
    if second_bucket < dim:
        vec[second_bucket] = max(vec[second_bucket], 0.5)
    return vec


def selected_properties(row: Mapping[str, str]) -> list[str]:
    raw_values = [
        row.get("condition_properties", ""),
        row.get("external_task_properties", ""),
        row.get("property_name", ""),
        row.get("objective", ""),
    ]
    out = []
    for raw in raw_values:
        text = str(raw or "").replace(";", ",").replace("|", ",")
        for part in text.split(","):
            prop = canonical_prop(part)
            if prop and prop in PROPERTY_COLUMNS and prop not in out:
                out.append(prop)
    for prop, _direction in instruction_task_specs(row):
        if prop and prop in PROPERTY_COLUMNS and prop not in out:
            out.append(prop)
    for prop, _direction in external_direction_specs(row):
        if prop and prop in PROPERTY_COLUMNS and prop not in out:
            out.append(prop)
    for prop in PROPERTY_COLUMNS:
        active = truthy(first_present(row, [f"{prop}_active", f"{prop.lower()}_active"]))
        if active and prop not in out:
            out.append(prop)
    return out


def canonical_prop(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return PROPERTY_ALIASES.get(text.lower(), text)


def parse_direction_value(value: object) -> int:
    text = str(value or "").strip().lower()
    if text in {"increase", "up", "+", "higher", "improve", "maximize", "max", "positive"}:
        return 1
    if text in {"decrease", "down", "-", "lower", "minimize", "min", "negative", "reduce"}:
        return -1
    return 0


def property_direction(row: Mapping[str, str], prop: str) -> int:
    canonical = canonical_prop(prop)
    direct = parse_direction_value(first_present(row, [f"{canonical}_direction", f"{canonical.lower()}_direction"]))
    if direct:
        return direct
    for item_prop, direction in instruction_task_specs(row):
        if item_prop == canonical and direction:
            return direction
    for item_prop, direction in external_direction_specs(row):
        if item_prop == canonical and direction:
            return direction
    return 0


def instruction_task_specs(row: Mapping[str, str]) -> list[tuple[str, int]]:
    raw = str(row.get("instruction_tasks", "") or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    out: list[tuple[str, int]] = []
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, MappingABC):
                continue
            prop = canonical_prop(item.get("property", "") or item.get("name", "") or item.get("prop", ""))
            direction = parse_direction_value(item.get("direction", "") or item.get("operation", "") or item.get("trend", ""))
            if prop:
                out.append((prop, direction))
    elif isinstance(parsed, MappingABC):
        for key, value in parsed.items():
            prop = canonical_prop(key)
            direction = parse_direction_value(value)
            if prop:
                out.append((prop, direction))
    return out


def external_direction_specs(row: Mapping[str, str]) -> list[tuple[str, int]]:
    raw = str(row.get("external_property_directions_json", "") or row.get("property_directions_json", "") or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    out: list[tuple[str, int]] = []
    if isinstance(parsed, MappingABC):
        for key, value in parsed.items():
            prop = canonical_prop(key)
            direction = parse_direction_value(value)
            if prop:
                out.append((prop, direction))
    elif isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, MappingABC):
                prop = canonical_prop(item.get("property", "") or item.get("name", "") or item.get("prop", ""))
                direction = parse_direction_value(item.get("direction", "") or item.get("operation", "") or item.get("trend", ""))
                if prop:
                    out.append((prop, direction))
    return out


def expand_condition_token(values: Sequence[float], condition_dim: int) -> np.ndarray:
    source = np.asarray(list(values), dtype=np.float32)
    if source.size == 0:
        return np.zeros(max(1, int(condition_dim)), dtype=np.float32)
    repeats = int(math.ceil(max(1, int(condition_dim)) / max(source.size, 1)))
    return np.tile(source, repeats)[: max(1, int(condition_dim))].astype(np.float32)


def hash_text_features(text: str, dim: int) -> list[float]:
    if dim <= 0:
        return []
    digest = hashlib.sha256(str(text or "").encode("utf-8")).digest()
    values = []
    for idx in range(dim):
        byte = digest[idx % len(digest)]
        values.append((float(byte) / 127.5) - 1.0)
    return values


def stable_hash_int(text: str) -> int:
    digest = hashlib.sha1(str(text).encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def safe_set(vec: np.ndarray, index: int, value: float) -> None:
    if 0 <= int(index) < vec.shape[0]:
        vec[int(index)] = float(value)


def is_atom_token(token: str) -> bool:
    text = str(token)
    if not text:
        return False
    if text.startswith("[") and text.endswith("]"):
        return True
    return any(ch.isalpha() for ch in text)


def is_bond_token(token: str) -> bool:
    return str(token) in {"=", "#", "-", "/", "\\", ":", "~", "."}


def first_present(row: Mapping[str, str], keys: Sequence[str]) -> object:
    for key in keys:
        value = row.get(key, "")
        if value not in ("", None):
            return value
    return ""


def parse_float(value: object) -> float:
    try:
        text = str(value).strip()
        if not text:
            return math.nan
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def truthy(value: object) -> bool | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "y", "active"}:
        return True
    if text in {"0", "false", "no", "n", "inactive"}:
        return False
    return None


def collate_batch(rows: list[dict[str, object]], pad_id: int) -> dict[str, torch.Tensor]:
    max_condition = max(np.asarray(row["condition"]).shape[0] for row in rows)
    condition_dim = np.asarray(rows[0]["condition"]).shape[-1]
    max_len = max(len(row["decoder_input_ids"]) for row in rows)
    condition = np.zeros((len(rows), max_condition, condition_dim), dtype=np.float32)
    condition_mask = np.zeros((len(rows), max_condition), dtype=bool)
    source_token_ids = np.full((len(rows), max_condition), int(pad_id), dtype=np.int64)
    decoder_input = np.full((len(rows), max_len), int(pad_id), dtype=np.int64)
    target = np.full((len(rows), max_len), int(pad_id), dtype=np.int64)
    task_is_denovo = np.zeros(len(rows), dtype=bool)
    for idx, row in enumerate(rows):
        cond = np.asarray(row["condition"], dtype=np.float32)
        seq_in = np.asarray(row["decoder_input_ids"], dtype=np.int64)
        seq_out = np.asarray(row["target_ids"], dtype=np.int64)
        condition[idx, : cond.shape[0], :] = cond
        condition_mask[idx, : cond.shape[0]] = True
        source_ids = np.asarray(row.get("source_token_ids", np.full(cond.shape[0], pad_id)), dtype=np.int64)
        source_token_ids[idx, : source_ids.shape[0]] = source_ids
        decoder_input[idx, : seq_in.shape[0]] = seq_in
        target[idx, : seq_out.shape[0]] = seq_out
        task_is_denovo[idx] = str(row.get("task_mode", "")) == DE_NOVO_MODE
    return {
        "condition": torch.from_numpy(condition),
        "condition_mask": torch.from_numpy(condition_mask),
        "source_token_ids": torch.from_numpy(source_token_ids),
        "decoder_input_ids": torch.from_numpy(decoder_input),
        "target_ids": torch.from_numpy(target),
        "task_is_denovo": torch.from_numpy(task_is_denovo),
    }


def training_group_key(item: Mapping[str, object]) -> str:
    mode = str(item.get("task_mode", "") or "unknown")
    raw_row = item.get("row", {})
    row = raw_row if isinstance(raw_row, MappingABC) else {}
    if mode == DE_NOVO_MODE:
        count = parse_float(row.get("property_count", ""))
        if math.isnan(count):
            count = float(len(selected_properties(row)))
        return f"{DE_NOVO_MODE}:{int(count) if math.isfinite(count) else 0}p"
    specs = sorted((prop, direction) for prop, direction in instruction_task_specs(row) if prop)
    if not specs:
        specs = sorted((prop, property_direction(row, prop)) for prop in selected_properties(row))
    task = "+".join(f"{prop}:{direction:+d}" for prop, direction in specs) or str(
        row.get("benchmark_task", "edit") or "edit"
    )
    return f"{EDIT_MODE}:{task}"


def training_group_counts(dataset: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in dataset:
        key = training_group_key(item)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def build_epoch_order(
    dataset: Sequence[Mapping[str, object]],
    *,
    sampling_mode: str,
    samples_per_epoch: int,
    seed: int,
) -> list[int]:
    rng = random.Random(seed)
    if sampling_mode == "random":
        order = list(range(len(dataset)))
        rng.shuffle(order)
        if samples_per_epoch > 0:
            if not order:
                return []
            return [order[idx % len(order)] for idx in range(int(samples_per_epoch))]
        return order
    if sampling_mode != "task_balanced":
        raise ValueError(f"Unsupported sampling_mode={sampling_mode!r}")

    modes: dict[str, dict[str, list[int]]] = {}
    for index, item in enumerate(dataset):
        mode = str(item.get("task_mode", "") or "unknown")
        modes.setdefault(mode, {}).setdefault(training_group_key(item), []).append(index)
    if not modes:
        return []

    mode_names = sorted(modes)
    target_size = int(samples_per_epoch) if samples_per_epoch > 0 else len(dataset)
    pools: dict[tuple[str, str], list[int]] = {}
    cursors: dict[tuple[str, str], int] = {}
    group_names: dict[str, list[str]] = {}
    for mode, groups in modes.items():
        group_names[mode] = sorted(groups)
        for group, indices in groups.items():
            pool = list(indices)
            rng.shuffle(pool)
            pools[(mode, group)] = pool
            cursors[(mode, group)] = 0

    order: list[int] = []
    mode_steps = {mode: 0 for mode in mode_names}
    for step in range(target_size):
        mode = mode_names[step % len(mode_names)]
        groups = group_names[mode]
        group = groups[mode_steps[mode] % len(groups)]
        mode_steps[mode] += 1
        key = (mode, group)
        cursor = cursors[key]
        pool = pools[key]
        if cursor >= len(pool):
            rng.shuffle(pool)
            cursor = 0
        order.append(pool[cursor])
        cursors[key] = cursor + 1
    return order


def de_novo_distillation_loss(
    student_logits: torch.Tensor,
    teacher_model: ConditionedSmilesDecoder,
    batch: Mapping[str, torch.Tensor],
    *,
    pad_id: int,
    temperature: float,
) -> tuple[torch.Tensor, int]:
    temp = max(float(temperature), 1e-6)
    denovo_rows = batch["task_is_denovo"].bool()
    if not bool(denovo_rows.any()):
        return student_logits.new_zeros(()), 0
    with torch.no_grad():
        teacher_logits = teacher_model(
            batch["condition"][denovo_rows],
            batch["decoder_input_ids"][denovo_rows],
            condition_mask=batch["condition_mask"][denovo_rows],
        )
    student_logits = student_logits[denovo_rows]
    target_ids = batch["target_ids"][denovo_rows]
    token_mask = target_ids.ne(int(pad_id))
    token_count = int(token_mask.sum().item())
    if token_count == 0:
        return student_logits.new_zeros(()), 0
    # A structured-action policy may append edit-only tokens to the student's
    # vocabulary while retaining the legacy de-novo teacher.  Existing token
    # ids stay prefix-compatible, so compare the teacher against that prefix.
    # The log-softmax is intentionally computed over the *full* student vocab:
    # probability leaked into edit-only tokens is therefore still penalized.
    teacher_vocab_size = int(teacher_logits.shape[-1])
    if int(student_logits.shape[-1]) < teacher_vocab_size:
        raise ValueError(
            "Student vocabulary is smaller than the protected de-novo teacher vocabulary: "
            f"{student_logits.shape[-1]} < {teacher_vocab_size}"
        )
    student_log_probs = F.log_softmax(student_logits / temp, dim=-1)[..., :teacher_vocab_size]
    teacher_probs = F.softmax(teacher_logits / temp, dim=-1)
    token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1) * (temp * temp)
    return token_kl.masked_select(token_mask).mean(), token_count


def update_adaptive_distill_weight(
    current_weight: float,
    *,
    observed_kl: float,
    target_kl: float,
    dual_lr: float,
    min_weight: float,
    max_weight: float,
) -> float:
    """Dual ascent for the de novo retention constraint E[KL] <= target."""
    lower = float(min(min_weight, max_weight))
    upper = float(max(min_weight, max_weight))
    updated = float(current_weight) + float(dual_lr) * (float(observed_kl) - float(target_kl))
    return max(lower, min(upper, updated))


def train_epoch(
    model: ConditionedSmilesDecoder,
    teacher_model: ConditionedSmilesDecoder | None,
    dataset: list[dict[str, object]],
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int,
    grad_clip: float,
    device: torch.device,
    seed: int,
    sampling_mode: str = "random",
    samples_per_epoch: int = 0,
    distill_weight: float = 0.0,
    distill_temperature: float = 1.0,
) -> dict[str, float]:
    model.train()
    order = build_epoch_order(
        dataset,
        sampling_mode=sampling_mode,
        samples_per_epoch=samples_per_epoch,
        seed=seed,
    )
    total_loss = 0.0
    total_distill_loss = 0.0
    total_distill_tokens = 0
    total_tokens = 0
    for start in range(0, len(order), batch_size):
        batch_rows = [dataset[idx] for idx in order[start : start + batch_size]]
        batch = {key: value.to(device) for key, value in collate_batch(batch_rows, model.pad_id).items()}
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            batch["condition"],
            batch["decoder_input_ids"],
            condition_mask=batch["condition_mask"],
            source_token_ids=batch["source_token_ids"],
        )
        sft_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            batch["target_ids"].reshape(-1),
            ignore_index=model.pad_id,
            reduction="sum",
        )
        token_count = int(batch["target_ids"].ne(model.pad_id).sum().item())
        objective = sft_loss / max(token_count, 1)
        distill_loss = logits.new_zeros(())
        distill_tokens = 0
        if teacher_model is not None and float(distill_weight) > 0:
            distill_loss, distill_tokens = de_novo_distillation_loss(
                logits,
                teacher_model,
                batch,
                pad_id=model.pad_id,
                temperature=float(distill_temperature),
            )
            objective = objective + float(distill_weight) * distill_loss
        objective.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()
        total_loss += float(sft_loss.item())
        total_distill_loss += float(distill_loss.item()) * max(distill_tokens, 1)
        total_distill_tokens += distill_tokens
        total_tokens += token_count
    return {
        "loss": total_loss / max(total_tokens, 1),
        "distill_loss": total_distill_loss / max(total_distill_tokens, 1),
        "distill_tokens": float(total_distill_tokens),
        "tokens": float(total_tokens),
        "sampled_rows": float(len(order)),
    }


@torch.no_grad()
def evaluate_loss(
    model: ConditionedSmilesDecoder,
    dataset: list[dict[str, object]],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, len(dataset), batch_size):
        batch_rows = dataset[start : start + batch_size]
        batch = {key: value.to(device) for key, value in collate_batch(batch_rows, model.pad_id).items()}
        logits = model(
            batch["condition"],
            batch["decoder_input_ids"],
            condition_mask=batch["condition_mask"],
            source_token_ids=batch["source_token_ids"],
        )
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            batch["target_ids"].reshape(-1),
            ignore_index=model.pad_id,
            reduction="sum",
        )
        total_loss += float(loss.item())
        total_tokens += int(batch["target_ids"].ne(model.pad_id).sum().item())
    return {"loss": total_loss / max(total_tokens, 1), "tokens": float(total_tokens)}


def train_epoch_group_rl(
    model: ConditionedSmilesDecoder,
    reference_model: ConditionedSmilesDecoder | None,
    dataset: list[dict[str, object]],
    optimizer: torch.optim.Optimizer,
    vocab: SmilesVocabulary,
    *,
    batch_size: int,
    device: torch.device,
    rollouts_per_prompt: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    parallel_samples: int,
    max_parallel_sequences: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    min_new_tokens: int,
    smiles_grammar_constraint: bool,
    sft_weight: float,
    rl_objective: str,
    grpo_clip_eps: float,
    grpo_update_epochs: int,
    advantage_mode: str,
    advantage_clip: float,
    sequence_logprob_reduction: str,
    reference_kl_weight: float,
    reward_mode: str,
    reward_valid_weight: float,
    reward_strict_weight: float,
    reward_distance_weight: float,
    reward_distance_clip: float,
    reward_aggregation: str,
    reward_joint_bonus_weight: float,
    reward_bottleneck_weight: float,
    reward_softmin_weight: float,
    reward_softmin_temperature: float,
    reward_source_similarity_weight: float,
    reward_source_similarity_threshold: float,
    reward_source_copy_penalty: float,
    grad_clip: float,
    seed: int,
) -> dict[str, object]:
    model.train()
    order = list(range(len(dataset)))
    random.Random(seed).shuffle(order)
    totals = {
        "loss": 0.0,
        "pg_loss": 0.0,
        "sft_loss": 0.0,
        "reference_loss": 0.0,
        "mean_reward": 0.0,
        "mean_policy_ratio": 0.0,
        "clip_fraction": 0.0,
        "batches": 0.0,
        "updates": 0.0,
    }
    mode_rewards: dict[str, list[float]] = {DE_NOVO_MODE: [], EDIT_MODE: []}
    objective = str(rl_objective)
    update_epochs = max(1, int(grpo_update_epochs)) if objective == "grpo" else 1
    for start in range(0, len(order), batch_size):
        batch_rows = [dataset[idx] for idx in order[start : start + batch_size]]
        batch = {key: value.to(device) for key, value in collate_batch(batch_rows, model.pad_id).items()}
        metadata_rows = [dict(row.get("row", {})) for row in batch_rows]
        generated = sample_group_rollouts(
            model,
            batch,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            rollouts_per_prompt=rollouts_per_prompt,
            parallel_samples=parallel_samples,
            max_parallel_sequences=max_parallel_sequences,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            min_new_tokens=min_new_tokens,
            smiles_token_text=vocab.id_to_token if smiles_grammar_constraint else None,
        )
        expanded = repeat_generation_batch(batch, repeats=rollouts_per_prompt)
        old_seq_logprob = None
        if objective == "grpo":
            with torch.no_grad():
                old_seq_logprob = sequence_logprobs(
                    model,
                    expanded["condition"],
                    expanded["condition_mask"],
                    expanded["source_token_ids"],
                    generated.to(device),
                    eos_id=vocab.eos_id,
                    reduction=sequence_logprob_reduction,
                ).detach()
        rewards = compute_group_rl_rewards(
            metadata_rows,
            generated,
            vocab,
            reward_mode=reward_mode,
            reward_valid_weight=reward_valid_weight,
            reward_strict_weight=reward_strict_weight,
            reward_distance_weight=reward_distance_weight,
            reward_distance_clip=reward_distance_clip,
            reward_aggregation=reward_aggregation,
            reward_joint_bonus_weight=reward_joint_bonus_weight,
            reward_bottleneck_weight=reward_bottleneck_weight,
            reward_softmin_weight=reward_softmin_weight,
            reward_softmin_temperature=reward_softmin_temperature,
            reward_source_similarity_weight=reward_source_similarity_weight,
            reward_source_similarity_threshold=reward_source_similarity_threshold,
            reward_source_copy_penalty=reward_source_copy_penalty,
        ).to(device)
        reward_groups = rewards.view(len(batch_rows), rollouts_per_prompt)
        advantages = group_relative_advantages(
            reward_groups,
            mode=advantage_mode,
            clip=advantage_clip,
        ).reshape(-1)
        for _update_idx in range(update_epochs):
            seq_logprob = sequence_logprobs(
                model,
                expanded["condition"],
                expanded["condition_mask"],
                expanded["source_token_ids"],
                generated.to(device),
                eos_id=vocab.eos_id,
                reduction=sequence_logprob_reduction,
            )
            pg_loss, pg_stats = policy_gradient_loss(
                seq_logprob,
                advantages,
                objective=objective,
                old_seq_logprob=old_seq_logprob,
                grpo_clip_eps=grpo_clip_eps,
            )
            reference_loss = torch.zeros((), dtype=pg_loss.dtype, device=device)
            if reference_model is not None and float(reference_kl_weight) > 0:
                with torch.no_grad():
                    ref_logprob = sequence_logprobs(
                        reference_model,
                        expanded["condition"],
                        expanded["condition_mask"],
                        expanded["source_token_ids"],
                        generated.to(device),
                        eos_id=vocab.eos_id,
                        reduction=sequence_logprob_reduction,
                    )
                reference_loss = float(reference_kl_weight) * (seq_logprob - ref_logprob).pow(2).mean()

            logits = model(
                batch["condition"],
                batch["decoder_input_ids"],
                condition_mask=batch["condition_mask"],
                source_token_ids=batch["source_token_ids"],
            )
            sft_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                batch["target_ids"].reshape(-1),
                ignore_index=model.pad_id,
            )
            loss = pg_loss + float(sft_weight) * sft_loss + reference_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

            totals["loss"] += float(loss.detach().cpu())
            totals["pg_loss"] += float(pg_loss.detach().cpu())
            totals["sft_loss"] += float(sft_loss.detach().cpu())
            totals["reference_loss"] += float(reference_loss.detach().cpu())
            totals["mean_policy_ratio"] += float(pg_stats["mean_policy_ratio"])
            totals["clip_fraction"] += float(pg_stats["clip_fraction"])
            totals["updates"] += 1.0
        totals["mean_reward"] += float(rewards.mean().detach().cpu())
        totals["batches"] += 1.0
        for row_idx, row in enumerate(metadata_rows):
            mode = task_mode_for_row(row)
            mode_rewards.setdefault(mode, []).append(float(reward_groups[row_idx].mean().detach().cpu()))
    batch_denom = max(totals["batches"], 1.0)
    update_denom = max(totals["updates"], 1.0)
    update_scaled = {"loss", "pg_loss", "sft_loss", "reference_loss", "mean_policy_ratio", "clip_fraction"}
    out: dict[str, object] = {}
    for key, value in totals.items():
        if key in update_scaled:
            out[key] = value / update_denom
        elif key == "batches" or key == "updates":
            out[key] = int(value)
        else:
            out[key] = value / batch_denom
    out["rl_objective"] = objective
    out["grpo_clip_eps"] = float(grpo_clip_eps)
    out["grpo_update_epochs"] = int(update_epochs)
    for mode, values in mode_rewards.items():
        if values:
            out[f"mean_reward_{mode}"] = sum(values) / len(values)
    return out


@torch.no_grad()
def evaluate_group_rl(
    model: ConditionedSmilesDecoder,
    dataset: list[dict[str, object]],
    vocab: SmilesVocabulary,
    *,
    batch_size: int,
    device: torch.device,
    rollouts_per_prompt: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    parallel_samples: int,
    max_parallel_sequences: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    min_new_tokens: int,
    smiles_grammar_constraint: bool,
    reward_mode: str,
    reward_valid_weight: float,
    reward_strict_weight: float,
    reward_distance_weight: float,
    reward_distance_clip: float,
    reward_aggregation: str,
    reward_joint_bonus_weight: float,
    reward_bottleneck_weight: float,
    reward_softmin_weight: float,
    reward_softmin_temperature: float,
    reward_source_similarity_weight: float,
    reward_source_similarity_threshold: float,
    reward_source_copy_penalty: float,
) -> dict[str, object]:
    model.eval()
    rewards_out: list[float] = []
    mode_rewards: dict[str, list[float]] = {DE_NOVO_MODE: [], EDIT_MODE: []}
    for start in range(0, len(dataset), batch_size):
        batch_rows = dataset[start : start + batch_size]
        batch = {key: value.to(device) for key, value in collate_batch(batch_rows, model.pad_id).items()}
        metadata_rows = [dict(row.get("row", {})) for row in batch_rows]
        generated = sample_group_rollouts(
            model,
            batch,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            rollouts_per_prompt=rollouts_per_prompt,
            parallel_samples=parallel_samples,
            max_parallel_sequences=max_parallel_sequences,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            min_new_tokens=min_new_tokens,
            smiles_token_text=vocab.id_to_token if smiles_grammar_constraint else None,
        )
        rewards = compute_group_rl_rewards(
            metadata_rows,
            generated,
            vocab,
            reward_mode=reward_mode,
            reward_valid_weight=reward_valid_weight,
            reward_strict_weight=reward_strict_weight,
            reward_distance_weight=reward_distance_weight,
            reward_distance_clip=reward_distance_clip,
            reward_aggregation=reward_aggregation,
            reward_joint_bonus_weight=reward_joint_bonus_weight,
            reward_bottleneck_weight=reward_bottleneck_weight,
            reward_softmin_weight=reward_softmin_weight,
            reward_softmin_temperature=reward_softmin_temperature,
            reward_source_similarity_weight=reward_source_similarity_weight,
            reward_source_similarity_threshold=reward_source_similarity_threshold,
            reward_source_copy_penalty=reward_source_copy_penalty,
        )
        grouped = rewards.view(len(batch_rows), rollouts_per_prompt)
        rewards_out.extend(float(value) for value in rewards.tolist())
        for row_idx, row in enumerate(metadata_rows):
            mode_rewards.setdefault(task_mode_for_row(row), []).append(float(grouped[row_idx].mean()))
    out: dict[str, object] = {"mean_reward": sum(rewards_out) / max(len(rewards_out), 1)}
    for mode, values in mode_rewards.items():
        if values:
            out[f"mean_reward_{mode}"] = sum(values) / len(values)
    return out


@torch.no_grad()
def sample_group_rollouts(
    model: ConditionedSmilesDecoder,
    batch: Mapping[str, torch.Tensor],
    *,
    bos_id: int,
    eos_id: int,
    rollouts_per_prompt: int,
    parallel_samples: int,
    max_parallel_sequences: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    min_new_tokens: int,
    smiles_token_text: Sequence[str] | None = None,
) -> torch.Tensor:
    prompt_count = int(batch["condition"].shape[0])
    remaining = max(1, int(rollouts_per_prompt))
    chunks: list[torch.Tensor] = []
    parallel_samples = max(1, int(parallel_samples))
    max_parallel_sequences = max(1, int(max_parallel_sequences))
    while remaining > 0:
        chunk_limit = max(1, max_parallel_sequences // max(prompt_count, 1))
        chunk = min(remaining, parallel_samples, chunk_limit)
        expanded = repeat_generation_batch(batch, repeats=chunk)
        generated = model.generate(
            expanded["condition"],
            bos_id=bos_id,
            eos_id=eos_id,
            max_new_tokens=max_new_tokens,
            condition_mask=expanded["condition_mask"],
            source_token_ids=expanded["source_token_ids"],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            min_new_tokens=min_new_tokens,
            smiles_token_text=smiles_token_text,
        ).detach().cpu()
        chunks.append(generated.view(prompt_count, chunk, generated.shape[1]))
        remaining -= chunk
    max_len = max(tensor.shape[-1] for tensor in chunks)
    padded = []
    for tensor in chunks:
        if tensor.shape[-1] < max_len:
            pad = torch.full(
                (tensor.shape[0], tensor.shape[1], max_len - tensor.shape[-1]),
                int(eos_id),
                dtype=tensor.dtype,
            )
            tensor = torch.cat([tensor, pad], dim=-1)
        padded.append(tensor)
    merged = torch.cat(padded, dim=1)
    return merged.reshape(prompt_count * merged.shape[1], max_len)


def repeat_generation_batch(batch: Mapping[str, torch.Tensor], *, repeats: int) -> dict[str, torch.Tensor]:
    out = {}
    for key in ("condition", "condition_mask", "source_token_ids"):
        out[key] = batch[key].repeat_interleave(max(1, int(repeats)), dim=0)
    return out


def sequence_logprobs(
    model: ConditionedSmilesDecoder,
    condition: torch.Tensor,
    condition_mask: torch.Tensor,
    source_token_ids: torch.Tensor,
    generated: torch.Tensor,
    *,
    eos_id: int,
    reduction: str,
) -> torch.Tensor:
    generated = generated.to(condition.device)
    decoder_input = generated[:, :-1]
    target_ids = generated[:, 1:]
    logits = model(
        condition,
        decoder_input,
        condition_mask=condition_mask,
        source_token_ids=source_token_ids,
    )
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    eos_hits = target_ids.eq(int(eos_id)).cumsum(dim=1)
    token_mask = eos_hits.le(1).to(token_log_probs.dtype)
    seq_logprob = (token_log_probs * token_mask).sum(dim=1)
    if reduction == "mean":
        seq_logprob = seq_logprob / token_mask.sum(dim=1).clamp_min(1.0)
    return seq_logprob


def policy_gradient_loss(
    seq_logprob: torch.Tensor,
    advantages: torch.Tensor,
    *,
    objective: str,
    old_seq_logprob: torch.Tensor | None,
    grpo_clip_eps: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    detached_advantages = advantages.detach()
    if objective == "grpo":
        if old_seq_logprob is None:
            raise ValueError("GRPO objective requires rollout-time old sequence log-probabilities.")
        log_ratio = (seq_logprob - old_seq_logprob.detach()).clamp(min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)
        eps = max(0.0, float(grpo_clip_eps))
        if eps > 0:
            clipped_ratio = ratio.clamp(min=1.0 - eps, max=1.0 + eps)
            surrogate = torch.minimum(ratio * detached_advantages, clipped_ratio * detached_advantages)
            clip_fraction = ratio.sub(1.0).abs().gt(eps).to(seq_logprob.dtype).mean()
        else:
            surrogate = ratio * detached_advantages
            clip_fraction = torch.zeros((), dtype=seq_logprob.dtype, device=seq_logprob.device)
        loss = -surrogate.mean()
        return loss, {
            "mean_policy_ratio": float(ratio.detach().mean().cpu()),
            "clip_fraction": float(clip_fraction.detach().cpu()),
        }
    loss = -(detached_advantages * seq_logprob).mean()
    return loss, {"mean_policy_ratio": 1.0, "clip_fraction": 0.0}


def group_relative_advantages(rewards: torch.Tensor, *, mode: str, clip: float) -> torch.Tensor:
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    if mode == "group_zscore":
        centered = centered / rewards.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    if clip and clip > 0:
        centered = torch.clamp(centered, min=-float(clip), max=float(clip))
    return centered


def compute_group_rl_rewards(
    rows: Sequence[Mapping[str, str]],
    generated: torch.Tensor,
    vocab: SmilesVocabulary,
    *,
    reward_mode: str,
    reward_valid_weight: float,
    reward_strict_weight: float,
    reward_distance_weight: float,
    reward_distance_clip: float,
    reward_aggregation: str,
    reward_joint_bonus_weight: float,
    reward_bottleneck_weight: float,
    reward_softmin_weight: float,
    reward_softmin_temperature: float,
    reward_source_similarity_weight: float,
    reward_source_similarity_threshold: float,
    reward_source_copy_penalty: float,
) -> torch.Tensor:
    rollout_count = max(1, int(generated.shape[0] // max(len(rows), 1)))
    rewards: list[float] = []
    for row_idx, row in enumerate(rows):
        start = row_idx * rollout_count
        end = start + rollout_count
        for ids in generated[start:end]:
            smiles = detokenize_smiles(vocab.decode(ids.tolist()))
            rewards.append(
                reward_for_smiles(
                    row,
                    smiles,
                    reward_mode=reward_mode,
                    reward_valid_weight=reward_valid_weight,
                    reward_strict_weight=reward_strict_weight,
                    reward_distance_weight=reward_distance_weight,
                    reward_distance_clip=reward_distance_clip,
                    reward_aggregation=reward_aggregation,
                    reward_joint_bonus_weight=reward_joint_bonus_weight,
                    reward_bottleneck_weight=reward_bottleneck_weight,
                    reward_softmin_weight=reward_softmin_weight,
                    reward_softmin_temperature=reward_softmin_temperature,
                    reward_source_similarity_weight=reward_source_similarity_weight,
                    reward_source_similarity_threshold=reward_source_similarity_threshold,
                    reward_source_copy_penalty=reward_source_copy_penalty,
                )
            )
    return torch.as_tensor(rewards, dtype=torch.float32)


def reward_for_smiles(
    row: Mapping[str, str],
    smiles: str,
    *,
    reward_mode: str,
    reward_valid_weight: float,
    reward_strict_weight: float,
    reward_distance_weight: float,
    reward_distance_clip: float,
    reward_source_similarity_weight: float,
    reward_source_similarity_threshold: float,
    reward_source_copy_penalty: float,
    reward_aggregation: str = "mean",
    reward_joint_bonus_weight: float = 2.0,
    reward_bottleneck_weight: float = 0.5,
    reward_softmin_weight: float = 1.0,
    reward_softmin_temperature: float = 0.25,
) -> float:
    canonical = safe_canonical_smiles(smiles)
    if not canonical:
        return -1.0
    mode = task_mode_for_row(row)
    routed_reward = reward_mode_for_row(row, reward_mode)
    scoring_mode = EDIT_MODE if routed_reward == "table1_edit" else DE_NOVO_MODE
    components = property_reward_components(row, canonical, mode=scoring_mode)
    if reward_aggregation == "joint_bottleneck":
        strict_fraction = components.success_fraction
        distance = components.mean_distance
    elif reward_aggregation == "dense_softmin":
        strict_fraction = components.mean_satisfaction(float(reward_softmin_temperature))
        distance = components.mean_violation
    else:
        strict_fraction = components.legacy_success_fraction
        distance = components.mean_distance
    distance = min(float(distance), float(reward_distance_clip))
    reward = float(reward_valid_weight)
    reward += float(reward_strict_weight) * float(strict_fraction)
    reward -= float(reward_distance_weight) * float(distance)
    if reward_aggregation == "joint_bottleneck":
        reward += float(reward_joint_bonus_weight) * float(components.all_success)
        reward -= float(reward_bottleneck_weight) * min(
            float(components.worst_violation),
            float(reward_distance_clip),
        )
    elif reward_aggregation == "dense_softmin":
        reward += float(reward_softmin_weight) * max(
            -float(reward_distance_clip),
            min(1.0, components.softmin_margin(float(reward_softmin_temperature))),
        )
        reward += float(reward_joint_bonus_weight) * float(components.all_success)
    if mode == EDIT_MODE:
        reward += float(reward_source_similarity_weight) * source_similarity_component(
            row,
            canonical,
            threshold=float(reward_source_similarity_threshold),
        )
        reward -= float(reward_source_copy_penalty) * source_copy_component(row, canonical)
    return float(reward)


def reward_mode_for_row(row: Mapping[str, str], reward_mode: str) -> str:
    if reward_mode != "auto":
        return str(reward_mode)
    return "table1_edit" if task_mode_for_row(row) == EDIT_MODE else "property_strict"


def source_similarity_component(row: Mapping[str, str], smiles: str, *, threshold: float) -> float:
    source = str(row.get("source_smiles", "") or row.get("molecule_smiles", "") or "").strip()
    if not source:
        return 0.0
    similarity = morgan_tanimoto(source, smiles)
    if not math.isfinite(similarity):
        return 0.0
    threshold = max(0.0, min(0.999, float(threshold)))
    if threshold <= 0:
        return max(0.0, min(1.0, float(similarity)))
    return (max(0.0, min(1.0, float(similarity))) - threshold) / max(1.0 - threshold, 1e-6)


def source_copy_component(row: Mapping[str, str], smiles: str) -> float:
    source = str(row.get("source_smiles", "") or row.get("molecule_smiles", "") or "").strip()
    if not source:
        return 0.0
    source_canonical = safe_canonical_smiles(source)
    smiles_canonical = safe_canonical_smiles(smiles)
    return 1.0 if source_canonical and source_canonical == smiles_canonical else 0.0


def effective_reward_source_similarity_threshold(args: argparse.Namespace) -> float:
    value = getattr(args, "reward_source_similarity_threshold", None)
    if value is None:
        value = getattr(args, "source_similarity_threshold", 0.4)
    return float(value)


def task_mode_counts(dataset: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in dataset:
        mode = str(row.get("task_mode", ""))
        counts[mode] = counts.get(mode, 0) + 1
    return counts


def input_modality_for_args(args: argparse.Namespace) -> str:
    explicit = str(getattr(args, "input_modality", "") or "").strip()
    if explicit:
        return explicit
    variant = str(getattr(args, "condition_feature_variant", "") or "").strip()
    if variant in {"full", "image_only"}:
        return "with_image"
    if variant in {"text_only", "caption_bottleneck"}:
        return "no_image"
    if variant == "random_query":
        return "random_query"
    return variant or "unknown"


def method_for_args(args: argparse.Namespace) -> str:
    method = str(getattr(args, "method", "") or "unified_smiles_generator")
    modality = input_modality_for_args(args)
    if method == "unified_smiles_generator" and modality not in {"", "unknown"}:
        return f"{method}_{modality}"
    return method


def file_sha256(path: Path) -> str:
    resolved = str(path.resolve())
    if resolved in _FILE_SHA256_CACHE:
        return _FILE_SHA256_CACHE[resolved]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _FILE_SHA256_CACHE[resolved] = value
    return value


def checkpoint_fingerprint_for_args(args: argparse.Namespace) -> str:
    for name in ("checkpoint", "resume_checkpoint"):
        value = getattr(args, name, None)
        if value:
            path = Path(value)
            if path.is_file():
                return file_sha256(path)
            return hashlib.sha256(str(path).encode("utf-8")).hexdigest()
    return hashlib.sha256(str(getattr(args, "output_dir", "in_memory")).encode("utf-8")).hexdigest()


def candidate_pool_id_for_row(row: Mapping[str, str], args: argparse.Namespace) -> str:
    row_id = first_present(row, ["condition_id", "sample_id", "example_id", "pair_id"])
    row_sha256 = hashlib.sha256(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {
        "checkpoint_sha256": checkpoint_fingerprint_for_args(args),
        "row_id": str(row_id or ""),
        "row_sha256": row_sha256,
        "seed": int(args.seed),
        "decoding_mode": str(args.decoding_mode),
        "condition_layout": str(args.condition_layout),
        "condition_feature_variant": str(args.condition_feature_variant),
        "condition_feature_array": str(args.condition_feature_array),
        "num_samples": int(args.num_samples),
        "beam_size": int(args.beam_size),
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "source_similarity_threshold": float(args.source_similarity_threshold),
        "include_source_copy_candidate": bool(args.include_source_copy_candidate),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def sampling_seed_for_row(row: Mapping[str, str], base_seed: int) -> int:
    row_id = first_present(row, ["condition_id", "sample_id", "example_id", "pair_id"])
    digest = hashlib.sha256(f"{base_seed}|{row_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def candidate_pool_hash(candidates: Sequence["Candidate"]) -> str:
    ordered = sorted(candidates, key=lambda candidate: candidate.generation_rank)
    payload = [
        {"generation_rank": candidate.generation_rank, "smiles": candidate.smiles}
        for candidate in ordered
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_predictions(
    model: ConditionedSmilesDecoder,
    rows: list[dict[str, str]],
    store: FeatureStore,
    vocab: SmilesVocabulary,
    condition_dim: int,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    prediction_csv = args.prediction_csv or args.output_dir / "unified_smiles_predictions.csv"
    candidate_csv = args.candidate_output_csv or args.output_dir / "unified_smiles_candidate_predictions.csv"
    model.eval()
    selected_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        seed_everything(sampling_seed_for_row(row, int(args.seed)))
        candidates = sample_candidates_for_row(
            model,
            row,
            store,
            vocab,
            condition_dim,
            args=args,
            device=device,
        )
        ranked = rank_candidates(row, candidates, source_similarity_threshold=float(args.source_similarity_threshold))
        if bool(args.disable_finalizer):
            ranked = candidates
        candidate_limit = int(args.max_candidates) if int(args.max_candidates) > 0 else int(args.top_k_candidates)
        candidate_limit = max(1, candidate_limit)
        pool_id = candidate_pool_id_for_row(row, args)
        pool_hash = candidate_pool_hash(candidates)
        selected = ranked[0] if ranked else Candidate("", 0.0, {}, 0)
        out = dict(row)
        out["task_mode"] = task_mode_for_row(row)
        out["condition_feature_variant"] = str(args.condition_feature_variant)
        out["condition_layout"] = str(args.condition_layout)
        out["input_modality"] = input_modality_for_args(args)
        out["method"] = method_for_args(args)
        out["generated_smiles"] = selected.smiles
        out["candidate_rank"] = 1
        out["generation_rank"] = selected.generation_rank
        out["candidate_pool_id"] = pool_id
        out["candidate_pool_hash"] = pool_hash
        out["candidate_selected"] = "True"
        out.update(selected.metrics)
        selected_rows.append(out)
        for rank, candidate in enumerate(ranked[:candidate_limit], start=1):
            c_row = dict(row)
            c_row["task_mode"] = task_mode_for_row(row)
            c_row["condition_feature_variant"] = str(args.condition_feature_variant)
            c_row["condition_layout"] = str(args.condition_layout)
            c_row["input_modality"] = input_modality_for_args(args)
            c_row["method"] = method_for_args(args)
            c_row["generated_smiles"] = candidate.smiles
            c_row["candidate_rank"] = rank
            c_row["generation_rank"] = candidate.generation_rank
            c_row["candidate_pool_id"] = pool_id
            c_row["candidate_pool_hash"] = pool_hash
            c_row["candidate_selected"] = "True" if rank == 1 else "False"
            c_row.update(candidate.metrics)
            candidate_rows.append(c_row)
        if row_index % 100 == 0 and row_index:
            print(json.dumps({"sampled_rows": row_index, "selected": len(selected_rows), "candidates": len(candidate_rows)}))
    write_csv(prediction_csv, selected_rows)
    write_csv(candidate_csv, candidate_rows)
    return {
        "rows": len(rows),
        "selected_rows": len(selected_rows),
        "candidate_rows": len(candidate_rows),
        "prediction_csv": str(prediction_csv),
        "candidate_output_csv": str(candidate_csv),
        "condition_layout": str(args.condition_layout),
        "max_candidates": int(args.max_candidates) if int(args.max_candidates) > 0 else int(args.top_k_candidates),
        "include_source_copy_candidate": bool(args.include_source_copy_candidate),
    }


@dataclass
class Candidate:
    smiles: str
    score: float
    metrics: dict[str, object]
    generation_rank: int


@torch.no_grad()
def sample_candidates_for_row(
    model: ConditionedSmilesDecoder,
    row: Mapping[str, str],
    store: FeatureStore,
    vocab: SmilesVocabulary,
    condition_dim: int,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> list[Candidate]:
    condition = condition_array_for_row(
        row,
        store,
        condition_dim,
        max_source_tokens=int(args.max_source_tokens),
        condition_layout=str(args.condition_layout),
    )
    source_token_ids = source_token_ids_for_condition(
        row,
        vocab,
        condition,
        max_source_tokens=int(args.max_source_tokens),
    )
    total = max(1, int(args.num_samples))
    batch_size = max(1, min(int(args.parallel_samples), int(args.max_parallel_sequences), total))
    seen: dict[str, Candidate] = {}
    decoding_mode = str(args.decoding_mode)
    # Normal sampling represents a molecule as SMILES. Structured edit
    # programs use their grammar-constrained ranking path, so appended action
    # tokens must never leak into the de-novo SMILES contract.
    action_token_ids = [
        int(token_id)
        for token, token_id in vocab.token_to_id.items()
        if token.startswith("<") and token not in SPECIAL_TOKENS
    ]
    if decoding_mode in {"sample", "sample_beam"}:
        for start in range(0, total, batch_size):
            current = min(batch_size, total - start)
            condition_batch = np.repeat(condition[None, :, :], current, axis=0)
            condition_mask = np.ones(condition_batch.shape[:2], dtype=bool)
            source_ids_batch = np.repeat(source_token_ids[None, :], current, axis=0)
            generated = model.generate(
                torch.from_numpy(condition_batch).to(device),
                bos_id=vocab.bos_id,
                eos_id=vocab.eos_id,
                max_new_tokens=int(args.max_new_tokens),
                condition_mask=torch.from_numpy(condition_mask).to(device),
                source_token_ids=torch.from_numpy(source_ids_batch).to(device),
                temperature=float(args.temperature),
                top_k=int(args.top_k),
                top_p=float(args.top_p),
                repetition_penalty=float(args.repetition_penalty),
                no_repeat_ngram_size=int(args.no_repeat_ngram_size),
                min_new_tokens=int(args.min_new_tokens),
                blocked_token_ids=action_token_ids,
                smiles_token_text=vocab.id_to_token if bool(args.smiles_grammar_constraint) else None,
            )
            add_generated_sequences(
                seen,
                generated.detach().cpu().tolist(),
                row,
                vocab,
                source_similarity_threshold=float(args.source_similarity_threshold),
                candidate_source="sample",
            )
    if decoding_mode in {"beam", "sample_beam"}:
        condition_batch = condition[None, :, :]
        condition_mask = np.ones(condition_batch.shape[:2], dtype=bool)
        source_ids_batch = source_token_ids[None, :]
        generated = model.beam_search(
            torch.from_numpy(condition_batch).to(device),
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            max_new_tokens=int(args.max_new_tokens),
            condition_mask=torch.from_numpy(condition_mask).to(device),
            source_token_ids=torch.from_numpy(source_ids_batch).to(device),
            beam_size=int(args.beam_size),
            expand_size=int(args.beam_expand_size),
            length_penalty=float(args.beam_length_penalty),
            repetition_penalty=float(args.repetition_penalty),
            no_repeat_ngram_size=int(args.no_repeat_ngram_size),
            min_new_tokens=int(args.min_new_tokens),
            blocked_token_ids=action_token_ids,
            smiles_token_text=vocab.id_to_token if bool(args.smiles_grammar_constraint) else None,
        )
        add_generated_sequences(
            seen,
            generated.detach().cpu().tolist(),
            row,
            vocab,
            source_similarity_threshold=float(args.source_similarity_threshold),
            candidate_source="beam",
        )
    source = str(row.get("source_smiles", "") or "").strip()
    if bool(args.include_source_copy_candidate) and task_mode_for_row(row) == EDIT_MODE and source:
        canonical_source = safe_canonical_smiles(source) or source
        if canonical_source and canonical_source not in seen:
            metrics = candidate_metrics(row, canonical_source, source_similarity_threshold=float(args.source_similarity_threshold))
            metrics["candidate_source_copy"] = "True"
            metrics["candidate_generation_mode"] = "source_copy"
            seen[canonical_source] = Candidate(
                canonical_source,
                float(metrics.get("unified_finalizer_score", 0.0)),
                metrics,
                len(seen) + 1,
            )
    return list(seen.values())


def add_generated_sequences(
    seen: dict[str, Candidate],
    sequences: Sequence[Sequence[int]],
    row: Mapping[str, str],
    vocab: SmilesVocabulary,
    *,
    source_similarity_threshold: float,
    candidate_source: str,
) -> None:
    for ids in sequences:
        smiles = detokenize_smiles(vocab.decode(ids))
        canonical = safe_canonical_smiles(smiles) or smiles
        if not canonical or canonical in seen:
            continue
        metrics = candidate_metrics(row, canonical, source_similarity_threshold=source_similarity_threshold)
        metrics["candidate_generation_mode"] = candidate_source
        seen[canonical] = Candidate(
            canonical,
            float(metrics.get("unified_finalizer_score", 0.0)),
            metrics,
            len(seen) + 1,
        )


def rank_candidates(row: Mapping[str, str], candidates: list[Candidate], *, source_similarity_threshold: float) -> list[Candidate]:
    rescored = []
    for candidate in candidates:
        metrics = dict(candidate.metrics)
        metrics.update(candidate_metrics(row, candidate.smiles, source_similarity_threshold=source_similarity_threshold))
        rescored.append(
            Candidate(
                candidate.smiles,
                float(metrics.get("unified_finalizer_score", 0.0)),
                metrics,
                candidate.generation_rank,
            )
        )
    return sorted(
        rescored,
        key=lambda item: (
            item.metrics.get("valid_smiles") == "True",
            item.metrics.get("table1_strict_success") == "True",
            item.metrics.get("table1_instruction_success") == "True",
            float(item.metrics.get("unified_property_success_fraction", 0.0)),
            item.metrics.get("source_similarity_success", "") == "True",
            float(item.metrics.get("source_tanimoto", -1.0) or -1.0),
            item.score,
        ),
        reverse=True,
    )


def candidate_metrics(
    row: Mapping[str, str],
    smiles: str,
    *,
    source_similarity_threshold: float,
) -> dict[str, object]:
    mode = task_mode_for_row(row)
    valid = bool(safe_canonical_smiles(smiles))
    source = str(row.get("source_smiles", "") or "").strip()
    similarity = morgan_tanimoto(source, smiles) if source and mode == EDIT_MODE else math.nan
    source_success = bool(math.isfinite(similarity) and similarity >= source_similarity_threshold)
    task_specs = instruction_task_specs(row) if mode == EDIT_MODE else []
    instruction_metrics: dict[str, object] = {}
    if task_specs:
        property_fraction, property_distance, evaluated, all_success = instruction_success_and_distance(
            row,
            smiles,
            task_specs=task_specs,
        )
        strict_success = bool(valid and all_success and source_success)
        instruction_metrics = {
            "table1_instruction_property_count": len(task_specs),
            "table1_instruction_evaluated_count": evaluated,
            "table1_instruction_success": "True" if all_success else "False",
            "table1_strict_success": "True" if strict_success else "False",
        }
    else:
        property_fraction, property_distance = property_success_and_distance(row, smiles, mode=mode)
        all_success = False
        strict_success = False
    score = 0.0
    score += 100.0 if valid else -100.0
    score += 50.0 * property_fraction
    score -= 5.0 * property_distance
    if mode == EDIT_MODE:
        if task_specs:
            # Match the official MolEdit Table1 predicate: every requested
            # property must improve strictly relative to the source, and the
            # source-similarity gate must pass. Large lexicographic bonuses
            # keep offline selection faithful when it sorts by this scalar.
            score += 400.0 if strict_success else 0.0
            score += 200.0 if all_success else 0.0
            score += 100.0 if source_success else 0.0
        else:
            score += 25.0 if source_success else 0.0
        score += 10.0 * (similarity if math.isfinite(similarity) else 0.0)
    metrics = {
        "valid_smiles": "True" if valid else "False",
        "unified_finalizer_score": format_float(score),
        "unified_property_success_fraction": format_float(property_fraction),
        "unified_property_distance": format_float(property_distance),
        "source_tanimoto": "" if not math.isfinite(similarity) else format_float(similarity),
        "source_similarity_success": "True" if source_success else "False",
    }
    metrics.update(instruction_metrics)
    return metrics


def instruction_success_and_distance(
    row: Mapping[str, str],
    smiles: str,
    *,
    task_specs: Sequence[tuple[str, int]] | None = None,
) -> tuple[float, float, int, bool]:
    """Score an edit with the exact source-relative MolEdit Table1 predicate."""
    specs = list(task_specs if task_specs is not None else instruction_task_specs(row))
    if not specs:
        return 0.0, 0.0, 0, False
    if not safe_canonical_smiles(smiles):
        return 0.0, 1e6, 0, False
    source = str(row.get("source_smiles", "") or "").strip()
    if not safe_canonical_smiles(source):
        return 0.0, 1e6, 0, False
    successes = 0
    evaluated = 0
    distances: list[float] = []
    for prop, direction in specs:
        if not direction:
            distances.append(1.0)
            continue
        source_value = score_property(source, prop)
        predicted_value = score_property(smiles, prop)
        if source_value is None or predicted_value is None:
            distances.append(1.0)
            continue
        if not math.isfinite(float(source_value)) or not math.isfinite(float(predicted_value)):
            distances.append(1.0)
            continue
        evaluated += 1
        signed_delta = float(direction) * (float(predicted_value) - float(source_value))
        normalizer = max(float(PROPERTY_NORMALIZERS.get(prop, 1.0)), 1e-8)
        distances.append(max(0.0, -signed_delta) / normalizer)
        if signed_delta > 0.0:
            successes += 1
    property_count = len(specs)
    fraction = successes / max(property_count, 1)
    all_success = evaluated == property_count and successes == property_count
    return fraction, sum(distances) / max(property_count, 1), evaluated, all_success


@dataclass(frozen=True)
class PropertyRewardComponents:
    legacy_success_fraction: float
    success_fraction: float
    mean_distance: float
    worst_violation: float
    mean_violation: float
    margins: tuple[float, ...]
    evaluated_count: int
    property_count: int
    all_success: bool

    def mean_satisfaction(self, temperature: float) -> float:
        if not self.margins:
            return 0.0
        temp = max(float(temperature), 1e-6)
        return sum(0.5 * (math.tanh(float(margin) / temp) + 1.0) for margin in self.margins) / len(self.margins)

    def softmin_margin(self, temperature: float) -> float:
        if not self.margins:
            return 0.0
        temp = max(float(temperature), 1e-6)
        minimum = min(self.margins)
        exp_sum = sum(math.exp(-(float(value) - minimum) / temp) for value in self.margins)
        return float(minimum) - temp * math.log(max(exp_sum, 1e-12))


def property_success_and_distance(row: Mapping[str, str], smiles: str, *, mode: str) -> tuple[float, float]:
    components = property_reward_components(row, smiles, mode=mode)
    return components.legacy_success_fraction, components.mean_distance


def property_reward_components(
    row: Mapping[str, str],
    smiles: str,
    *,
    mode: str,
) -> PropertyRewardComponents:
    if not safe_canonical_smiles(smiles):
        return PropertyRewardComponents(0.0, 0.0, 1e6, 1e6, 1e6, (-1e6,), 0, 0, False)
    selected = selected_properties(row)
    if not selected:
        return PropertyRewardComponents(0.0, 0.0, 0.0, 0.0, 0.0, (), 0, 0, False)
    source = str(row.get("source_smiles", "") or row.get("molecule_smiles", "") or "").strip()
    successes = 0
    distances: list[float] = []
    violations: list[float] = []
    margins: list[float] = []
    evaluated = 0
    for prop in selected:
        value = score_property(smiles, prop)
        if value is None or math.isnan(float(value)):
            violations.append(1.0)
            margins.append(-1.0)
            continue
        target = parse_float(first_present(row, [f"target_{prop}", f"target_{prop.lower()}"]))
        direction = property_direction(row, prop)
        if not math.isnan(target):
            tolerance = STRICT_TOLERANCE.get(prop, PROPERTY_NORMALIZERS.get(prop, 1.0))
            distance = abs(float(value) - float(target)) / max(tolerance, 1e-8)
            success = distance <= 1.0
            margin = 1.0 - float(distance)
            violation = max(0.0, -margin)
        elif mode == EDIT_MODE and direction and source:
            source_value = score_property(source, prop)
            if source_value is None or math.isnan(float(source_value)):
                violations.append(1.0)
                margins.append(-1.0)
                continue
            delta = float(value) - float(source_value)
            distance = max(0.0, -float(direction) * delta)
            success = (delta * float(direction)) > 0.0
            normalizer = max(float(PROPERTY_NORMALIZERS.get(prop, 1.0)), 1e-8)
            margin = float(direction) * delta / normalizer
            violation = max(0.0, -margin)
        else:
            violations.append(1.0)
            margins.append(-1.0)
            continue
        evaluated += 1
        distances.append(float(distance))
        violations.append(float(violation))
        margins.append(float(margin))
        successes += 1 if success else 0
    property_count = len(selected)
    legacy_fraction = successes / evaluated if evaluated else 0.0
    success_fraction = successes / max(property_count, 1)
    mean_distance = sum(distances) / max(len(distances), 1) if evaluated else 0.0
    all_success = evaluated == property_count and successes == property_count
    return PropertyRewardComponents(
        legacy_success_fraction=legacy_fraction,
        success_fraction=success_fraction,
        mean_distance=mean_distance,
        worst_violation=max(violations, default=0.0),
        mean_violation=sum(violations) / max(len(violations), 1),
        margins=tuple(margins),
        evaluated_count=evaluated,
        property_count=property_count,
        all_success=all_success,
    )


def score_property(smiles: str, prop: str) -> float | None:
    canonical_prop_name = canonical_prop(prop)
    canonical_smiles = safe_canonical_smiles(smiles)
    if not canonical_smiles or not canonical_prop_name:
        return None
    key = (canonical_smiles, canonical_prop_name)
    if key in _PROPERTY_VALUE_CACHE:
        return _PROPERTY_VALUE_CACHE[key]
    props = molecular_properties(canonical_smiles)
    if canonical_prop_name in props:
        value = float(props[canonical_prop_name])
        _PROPERTY_VALUE_CACHE[key] = value if math.isfinite(value) else None
        return _PROPERTY_VALUE_CACHE[key]
    oracle = tdc_oracle(canonical_prop_name)
    if oracle is None:
        _PROPERTY_VALUE_CACHE[key] = None
        return None
    try:
        value = float(oracle(canonical_smiles))  # type: ignore[operator]
    except Exception:
        _PROPERTY_VALUE_CACHE[key] = None
        return None
    _PROPERTY_VALUE_CACHE[key] = value if math.isfinite(value) else None
    return _PROPERTY_VALUE_CACHE[key]


def tdc_oracle(prop: str):
    canonical_prop_name = canonical_prop(prop)
    if canonical_prop_name in _TDC_ORACLE_CACHE:
        return _TDC_ORACLE_CACHE[canonical_prop_name]
    if canonical_prop_name == "SA":
        try:
            import importlib.util

            from rdkit import Chem
            from rdkit.Chem import RDConfig

            scorer_path = Path(RDConfig.RDContribDir) / "SA_Score" / "sascorer.py"
            spec = importlib.util.spec_from_file_location("unified_sascorer", scorer_path)
            if not scorer_path.is_file() or spec is None or spec.loader is None:
                raise FileNotFoundError(scorer_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            def score_sa(smiles: str) -> float:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    raise ValueError("invalid SMILES")
                return float(module.calculateScore(mol))

            _TDC_ORACLE_CACHE[canonical_prop_name] = score_sa
        except Exception:
            _TDC_ORACLE_CACHE[canonical_prop_name] = None
        return _TDC_ORACLE_CACHE[canonical_prop_name]
    pinned_env = {
        "GSK3B": "SUCC_GSK3B_ORACLE_PATH",
        "DRD2": "SUCC_DRD2_ORACLE_PATH",
    }.get(canonical_prop_name)
    if pinned_env and str(os.environ.get(pinned_env, "")).strip():
        try:
            import legacy_gsk3b_oracle

            if canonical_prop_name == "GSK3B":
                _TDC_ORACLE_CACHE[canonical_prop_name] = legacy_gsk3b_oracle.configured_oracle()
            else:
                _TDC_ORACLE_CACHE[canonical_prop_name] = legacy_gsk3b_oracle.configured_oracle_for(
                    canonical_prop_name
                )
        except Exception:
            _TDC_ORACLE_CACHE[canonical_prop_name] = None
        return _TDC_ORACLE_CACHE[canonical_prop_name]
    try:
        ensure_rdkit_six_compat()
        from tdc import Oracle

        _TDC_ORACLE_CACHE[canonical_prop_name] = Oracle(name=canonical_prop_name)
    except Exception:
        _TDC_ORACLE_CACHE[canonical_prop_name] = None
    return _TDC_ORACLE_CACHE[canonical_prop_name]


def configured_oracle_provenance() -> dict[str, object]:
    """Return explicit benchmark-oracle provenance for experiment manifests."""
    import legacy_gsk3b_oracle

    provenance = {}
    for prop, env_name in legacy_gsk3b_oracle.PINNED_ORACLE_ENVS.items():
        if str(os.environ.get(env_name, "")).strip():
            provenance[prop] = legacy_gsk3b_oracle.configured_provenance_for(prop)
    return provenance


def ensure_rdkit_six_compat() -> None:
    if "rdkit.six" in sys.modules:
        return
    try:
        from rdkit.six import iteritems  # noqa: F401
    except ModuleNotFoundError:
        import types

        six_mod = types.ModuleType("rdkit.six")
        six_mod.iteritems = dict.items
        sys.modules["rdkit.six"] = six_mod


def molecular_properties(smiles: str) -> dict[str, float]:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, QED, rdMolDescriptors
    except Exception:
        return {}
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return {}
    return {
        "MW": float(Descriptors.MolWt(mol)),
        "LogP": float(Descriptors.MolLogP(mol)),
        "QED": float(QED.qed(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "HBD": float(rdMolDescriptors.CalcNumHBD(mol)),
        "HBA": float(rdMolDescriptors.CalcNumHBA(mol)),
        "RB": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
    }


def safe_canonical_smiles(smiles: str) -> str:
    try:
        from rdkit import Chem
    except Exception:
        return str(smiles or "").strip()
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True)


def morgan_tanimoto(left: str, right: str) -> float:
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except Exception:
        return math.nan
    left_mol = Chem.MolFromSmiles(str(left or ""))
    right_mol = Chem.MolFromSmiles(str(right or ""))
    if left_mol is None or right_mol is None:
        return math.nan
    left_fp = AllChem.GetMorganFingerprintAsBitVect(left_mol, 2, nBits=2048)
    right_fp = AllChem.GetMorganFingerprintAsBitVect(right_mol, 2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(left_fp, right_fp))


def format_float(value: float, digits: int = 6) -> str:
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):.{digits}g}"


def write_csv(path: Path, rows: list[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def save_checkpoint(
    path: Path,
    model: ConditionedSmilesDecoder,
    optimizer: torch.optim.Optimizer | None,
    vocab: SmilesVocabulary,
    config: Mapping[str, object],
    epoch: int,
    history: list[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "vocab": vocab.to_dict(),
        "model_config": dict(config),
        "epoch": int(epoch),
        "history": history,
        "args": vars(args),
    }
    torch.save(payload, path)


if __name__ == "__main__":
    raise SystemExit(main())
