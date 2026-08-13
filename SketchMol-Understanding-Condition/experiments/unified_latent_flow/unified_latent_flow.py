#!/usr/bin/env python3
"""Minimal language-conditioned molecular latent-flow kill test.

The experiment deliberately has no retrieval materializer and no finalizer. A
frozen Common-LLM feature sequence conditions a continuous source-to-target
latent flow; the resulting latent tokens are decoded directly to SMILES.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
UNIFIED_GENERATOR_PATH = PROJECT_DIR / "experiments" / "unified_smiles_generator" / "unified_smiles_generator.py"


def load_unified_module():
    spec = importlib.util.spec_from_file_location("unified_latent_flow_base", UNIFIED_GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load unified generator module: {UNIFIED_GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


unified = load_unified_module()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--train-features-dir", type=Path, required=True)
    parser.add_argument("--validation-features-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-variant", default="full")
    parser.add_argument("--train-limit", type=int, default=12000)
    parser.add_argument("--validation-per-mode", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-smiles-length", type=int, default=160)
    parser.add_argument("--latent-tokens", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--flow-layers", type=int, default=3)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--edit-noise", type=float, default=0.08)
    parser.add_argument("--denovo-noise", type=float, default=0.60)
    parser.add_argument("--flow-loss-weight", type=float, default=1.0)
    parser.add_argument("--endpoint-loss-weight", type=float, default=0.25)
    parser.add_argument("--latent-reg-weight", type=float, default=1e-4)
    parser.add_argument("--decoder-corruption", type=float, default=0.30)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--sample-batch-size", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.90)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1715)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def constraint_array_for_row(
    row: Mapping[str, str],
    store,
    condition_dim: int,
) -> np.ndarray:
    """Build condition memory without leaking target SMILES or hiding source in it.

    The Common-LLM query sequence and explicit property tokens are retained.
    Molecular source structure is handled exclusively by the molecular encoder.
    """

    base = store.get(row)
    if base is None:
        base = unified.fallback_condition_features(row, condition_dim)
    if int(base.shape[-1]) != int(condition_dim):
        raise ValueError(f"Condition feature dim mismatch: {base.shape[-1]} != {condition_dim}")
    mode = unified.mode_condition_token(unified.task_mode_for_row(row), condition_dim)
    program = unified.property_program_tokens(row, condition_dim)
    return np.concatenate([base, mode, program], axis=0).astype(np.float32)


def molecule_token_ids(smiles: str, vocab, max_length: int) -> list[int]:
    tokens = unified.tokenize_smiles(str(smiles or ""))[: max(1, int(max_length))]
    return vocab.encode(tokens, add_bos=True, add_eos=True)


def build_dataset(
    rows: Sequence[dict[str, str]],
    store,
    vocab,
    condition_dim: int,
    *,
    max_smiles_length: int,
) -> list[dict[str, object]]:
    dataset: list[dict[str, object]] = []
    for row in rows:
        target = str(row.get("target_smiles", "") or "").strip()
        if not target:
            continue
        mode = unified.task_mode_for_row(row)
        source = str(row.get("source_smiles", "") or "").strip()
        target_tokens = unified.tokenize_smiles(target)[: max(1, int(max_smiles_length))]
        dataset.append(
            {
                "row": dict(row),
                "mode": mode,
                "source_present": mode == unified.EDIT_MODE and bool(source),
                "source_ids": np.asarray(
                    molecule_token_ids(source if source else target, vocab, max_smiles_length), dtype=np.int64
                ),
                "target_encoder_ids": np.asarray(
                    molecule_token_ids(target, vocab, max_smiles_length), dtype=np.int64
                ),
                "decoder_input_ids": np.asarray(vocab.encode(target_tokens, add_bos=True), dtype=np.int64),
                "target_ids": np.asarray(vocab.encode(target_tokens, add_eos=True), dtype=np.int64),
                "condition": constraint_array_for_row(row, store, condition_dim),
            }
        )
    return dataset


def balanced_subset(dataset: Sequence[dict[str, object]], limit: int, seed: int) -> list[dict[str, object]]:
    if limit <= 0 or limit >= len(dataset):
        return list(dataset)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(dataset):
        groups[unified.training_group_key({"task_mode": item["mode"], "row": item["row"]})].append(index)
    rng = random.Random(seed)
    for indices in groups.values():
        rng.shuffle(indices)
    names = sorted(groups)
    cursors = {name: 0 for name in names}
    selected: list[int] = []
    while len(selected) < limit:
        progressed = False
        for name in names:
            cursor = cursors[name]
            if cursor >= len(groups[name]):
                continue
            selected.append(groups[name][cursor])
            cursors[name] = cursor + 1
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    rng.shuffle(selected)
    return [dataset[index] for index in selected]


def validation_subset(
    dataset: Sequence[dict[str, object]], per_mode: int, seed: int
) -> list[dict[str, object]]:
    by_mode: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in dataset:
        by_mode[str(item["mode"])].append(item)
    rng = random.Random(seed)
    selected: list[dict[str, object]] = []
    for mode in (unified.DE_NOVO_MODE, unified.EDIT_MODE):
        values = list(by_mode.get(mode, []))
        rng.shuffle(values)
        selected.extend(values[: max(0, int(per_mode))])
    return selected


def pad_sequences(values: Sequence[np.ndarray], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(len(value) for value in values)
    output = np.full((len(values), max_length), int(pad_id), dtype=np.int64)
    mask = np.zeros((len(values), max_length), dtype=bool)
    for index, value in enumerate(values):
        output[index, : len(value)] = value
        mask[index, : len(value)] = True
    return torch.from_numpy(output), torch.from_numpy(mask)


def collate(items: Sequence[dict[str, object]], pad_id: int) -> dict[str, object]:
    source_ids, source_mask = pad_sequences([item["source_ids"] for item in items], pad_id)
    encoder_ids, encoder_mask = pad_sequences([item["target_encoder_ids"] for item in items], pad_id)
    decoder_ids, _ = pad_sequences([item["decoder_input_ids"] for item in items], pad_id)
    target_ids, _ = pad_sequences([item["target_ids"] for item in items], pad_id)
    condition_length = max(np.asarray(item["condition"]).shape[0] for item in items)
    condition_dim = np.asarray(items[0]["condition"]).shape[-1]
    condition = np.zeros((len(items), condition_length, condition_dim), dtype=np.float32)
    condition_mask = np.zeros((len(items), condition_length), dtype=bool)
    for index, item in enumerate(items):
        value = np.asarray(item["condition"], dtype=np.float32)
        condition[index, : value.shape[0]] = value
        condition_mask[index, : value.shape[0]] = True
    return {
        "source_ids": source_ids,
        "source_mask": source_mask,
        "target_encoder_ids": encoder_ids,
        "target_encoder_mask": encoder_mask,
        "decoder_input_ids": decoder_ids,
        "target_ids": target_ids,
        "condition": torch.from_numpy(condition),
        "condition_mask": torch.from_numpy(condition_mask),
        "source_present": torch.tensor([bool(item["source_present"]) for item in items], dtype=torch.bool),
        "rows": [item["row"] for item in items],
        "modes": [item["mode"] for item in items],
    }


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = max(1, self.dim // 2)
        frequencies = torch.exp(
            torch.arange(half, device=time.device, dtype=time.dtype) * (-math.log(10000.0) / max(half - 1, 1))
        )
        angles = time[:, None] * frequencies[None, :]
        output = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if output.shape[-1] < self.dim:
            output = F.pad(output, (0, self.dim - output.shape[-1]))
        return output[:, : self.dim]


class UnifiedMolecularLatentFlow(nn.Module):
    def __init__(
        self,
        *,
        base_model,
        model_config: Mapping[str, object],
        latent_tokens: int,
        encoder_layers: int,
        flow_layers: int,
    ) -> None:
        super().__init__()
        self.pad_id = int(model_config["pad_id"])
        self.d_model = int(model_config["d_model"])
        self.token_embedding = base_model.token_embedding
        self.position = base_model.position
        self.decoder = base_model.decoder
        self.output = base_model.output
        self.condition_proj = base_model.condition_proj

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(model_config["num_heads"]),
            dim_feedforward=int(model_config["dim_feedforward"]),
            dropout=float(model_config["dropout"]),
            batch_first=True,
            norm_first=True,
        )
        self.molecule_encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(encoder_layers)))
        self.latent_queries = nn.Parameter(torch.randn(1, int(latent_tokens), self.d_model) * 0.02)
        self.latent_pool = nn.MultiheadAttention(
            self.d_model, int(model_config["num_heads"]), dropout=float(model_config["dropout"]), batch_first=True
        )
        self.latent_norm = nn.LayerNorm(self.d_model)
        self.de_novo_prior = nn.Parameter(torch.randn(1, int(latent_tokens), self.d_model) * 0.02)

        condition_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(model_config["num_heads"]),
            dim_feedforward=int(model_config["dim_feedforward"]),
            dropout=float(model_config["dropout"]),
            batch_first=True,
            norm_first=True,
        )
        self.constraint_encoder = nn.TransformerEncoder(condition_layer, num_layers=1)
        self.condition_type = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.source_type = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.mode_embedding = nn.Embedding(2, self.d_model)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        flow_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=int(model_config["num_heads"]),
            dim_feedforward=int(model_config["dim_feedforward"]),
            dropout=float(model_config["dropout"]),
            batch_first=True,
            norm_first=True,
        )
        self.flow = nn.TransformerDecoder(flow_layer, num_layers=max(1, int(flow_layers)))
        self.flow_output = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.d_model))
        nn.init.normal_(self.flow_output[-1].weight, std=0.01)
        nn.init.zeros_(self.flow_output[-1].bias)

    def encode_molecule(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.position(self.token_embedding(ids))
        hidden = self.molecule_encoder(hidden, src_key_padding_mask=~mask.bool())
        queries = self.latent_queries.expand(ids.shape[0], -1, -1)
        pooled, _ = self.latent_pool(queries, hidden, hidden, key_padding_mask=~mask.bool(), need_weights=False)
        return self.latent_norm(pooled)

    def encode_constraints(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        memory = self.condition_proj(values) + self.condition_type
        return self.constraint_encoder(memory, src_key_padding_mask=~mask.bool())

    def vector_field(
        self,
        latent: torch.Tensor,
        time: torch.Tensor,
        source_anchor: torch.Tensor,
        source_present: torch.Tensor,
        condition_memory: torch.Tensor,
        condition_mask: torch.Tensor,
    ) -> torch.Tensor:
        mode_ids = source_present.long()
        time_context = self.time_embedding(time).unsqueeze(1)
        flow_input = latent + time_context + self.mode_embedding(mode_ids).unsqueeze(1)
        source_memory = source_anchor + self.source_type
        memory = torch.cat([condition_memory, source_memory], dim=1)
        source_mask = torch.ones(source_anchor.shape[:2], dtype=torch.bool, device=source_anchor.device)
        memory_padding = ~torch.cat([condition_mask.bool(), source_mask], dim=1)
        return self.flow_output(self.flow(flow_input, memory, memory_key_padding_mask=memory_padding))

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

    def training_losses(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        edit_noise: float,
        denovo_noise: float,
        decoder_corruption: float,
        unknown_id: int,
    ) -> dict[str, torch.Tensor]:
        source_latent = self.encode_molecule(batch["source_ids"], batch["source_mask"])
        target_latent = self.encode_molecule(batch["target_encoder_ids"], batch["target_encoder_mask"])
        source_present = batch["source_present"].bool()
        prior = self.de_novo_prior.expand(source_latent.shape[0], -1, -1)
        base = torch.where(source_present[:, None, None], source_latent, prior)
        noise_scale = torch.where(
            source_present,
            torch.full_like(source_present, float(edit_noise), dtype=target_latent.dtype),
            torch.full_like(source_present, float(denovo_noise), dtype=target_latent.dtype),
        )
        start = base + torch.randn_like(base) * noise_scale[:, None, None]
        time = torch.rand(target_latent.shape[0], device=target_latent.device, dtype=target_latent.dtype)
        target_velocity = target_latent.detach() - start.detach()
        interpolated = start.detach() + time[:, None, None] * target_velocity
        condition_memory = self.encode_constraints(batch["condition"], batch["condition_mask"])
        predicted_velocity = self.vector_field(
            interpolated,
            time,
            base.detach(),
            source_present,
            condition_memory,
            batch["condition_mask"],
        )
        predicted_endpoint = interpolated + (1.0 - time[:, None, None]) * predicted_velocity
        decoder_input_ids = batch["decoder_input_ids"]
        corruption_probability = max(0.0, min(1.0, float(decoder_corruption)))
        if corruption_probability > 0:
            corruptible = decoder_input_ids.ne(self.pad_id)
            corruptible[:, 0] = False
            corrupt = torch.rand_like(decoder_input_ids, dtype=torch.float32).lt(corruption_probability)
            decoder_input_ids = torch.where(
                corrupt & corruptible,
                torch.full_like(decoder_input_ids, int(unknown_id)),
                decoder_input_ids,
            )
        logits = self.decode(target_latent, decoder_input_ids)
        reconstruction = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            batch["target_ids"].reshape(-1),
            ignore_index=self.pad_id,
        )
        flow_loss = F.mse_loss(predicted_velocity, target_velocity)
        endpoint_loss = F.mse_loss(predicted_endpoint, target_latent.detach())
        latent_reg = target_latent.square().mean() + source_latent.square().mean()
        with torch.no_grad():
            token_mask = batch["target_ids"].ne(self.pad_id)
            token_accuracy = logits.argmax(dim=-1).eq(batch["target_ids"]).masked_select(token_mask).float().mean()
        return {
            "reconstruction": reconstruction,
            "flow": flow_loss,
            "endpoint": endpoint_loss,
            "latent_reg": latent_reg,
            "token_accuracy": token_accuracy,
        }

    @torch.no_grad()
    def sample_latent(
        self,
        *,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        source_present: torch.Tensor,
        condition: torch.Tensor,
        condition_mask: torch.Tensor,
        steps: int,
        edit_noise: float,
        denovo_noise: float,
    ) -> torch.Tensor:
        source_latent = self.encode_molecule(source_ids, source_mask)
        prior = self.de_novo_prior.expand(source_latent.shape[0], -1, -1)
        base = torch.where(source_present[:, None, None], source_latent, prior)
        scale = torch.where(
            source_present,
            torch.full_like(source_present, float(edit_noise), dtype=base.dtype),
            torch.full_like(source_present, float(denovo_noise), dtype=base.dtype),
        )
        latent = base + torch.randn_like(base) * scale[:, None, None]
        condition_memory = self.encode_constraints(condition, condition_mask)
        step_count = max(1, int(steps))
        for index in range(step_count):
            time = torch.full(
                (latent.shape[0],), (index + 0.5) / step_count, device=latent.device, dtype=latent.dtype
            )
            latent = latent + self.vector_field(
                latent, time, base, source_present, condition_memory, condition_mask
            ) / step_count
        return latent

    @torch.no_grad()
    def generate(
        self,
        latent: torch.Tensor,
        *,
        vocab,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        generated = torch.full(
            (latent.shape[0], 1), int(vocab.bos_id), dtype=torch.long, device=latent.device
        )
        finished = torch.zeros(latent.shape[0], dtype=torch.bool, device=latent.device)
        blocked = [int(vocab.bos_id), int(vocab.pad_id)]
        for _ in range(max(1, int(max_new_tokens))):
            logits = self.decode(latent, generated)[:, -1, :]
            logits[:, blocked] = -torch.inf
            unified.apply_smiles_grammar_mask_(
                logits,
                generated,
                token_text=vocab.id_to_token,
                eos_id=int(vocab.eos_id),
            )
            if temperature > 0:
                logits = logits / float(temperature)
                if top_k > 0 and top_k < logits.shape[-1]:
                    threshold = torch.topk(logits, int(top_k), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, -torch.inf)
                logits = unified.top_p_filter(logits, top_p=float(top_p))
                probabilities = torch.softmax(logits, dim=-1)
                probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
                empty = probabilities.sum(dim=-1).le(0)
                if bool(empty.any()):
                    fallback = torch.zeros_like(probabilities)
                    fallback[:, int(vocab.eos_id)] = 1.0
                    probabilities = torch.where(empty[:, None], fallback, probabilities)
                next_ids = torch.multinomial(probabilities, 1).squeeze(1)
            else:
                next_ids = logits.argmax(dim=-1)
            next_ids = torch.where(finished, torch.full_like(next_ids, int(vocab.eos_id)), next_ids)
            generated = torch.cat([generated, next_ids[:, None]], dim=1)
            finished |= next_ids.eq(int(vocab.eos_id))
            if bool(finished.all()):
                break
        return generated


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def train_model(
    model: UnifiedMolecularLatentFlow,
    dataset: Sequence[dict[str, object]],
    vocab,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    history: list[dict[str, float]] = []
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    for epoch in range(1, int(args.epochs) + 1):
        order = list(range(len(dataset)))
        random.Random(int(args.seed) + epoch).shuffle(order)
        totals: dict[str, float] = defaultdict(float)
        batches = 0
        model.train()
        for start in range(0, len(order), int(args.batch_size)):
            items = [dataset[index] for index in order[start : start + int(args.batch_size)]]
            batch = move_batch(collate(items, model.pad_id), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                losses = model.training_losses(
                    batch,
                    edit_noise=float(args.edit_noise),
                    denovo_noise=float(args.denovo_noise),
                    decoder_corruption=float(args.decoder_corruption),
                    unknown_id=int(vocab.token_to_id[unified.UNK]),
                )
                objective = (
                    losses["reconstruction"]
                    + float(args.flow_loss_weight) * losses["flow"]
                    + float(args.endpoint_loss_weight) * losses["endpoint"]
                    + float(args.latent_reg_weight) * losses["latent_reg"]
                )
            objective.backward()
            if float(args.grad_clip) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            totals["loss"] += float(objective.detach().cpu())
            for key, value in losses.items():
                totals[key] += float(value.detach().cpu())
            batches += 1
        record = {"epoch": float(epoch), **{key: value / max(batches, 1) for key, value in totals.items()}}
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    return history


def repeated_batch(item: dict[str, object], repeats: int, pad_id: int, device: torch.device) -> dict[str, object]:
    return move_batch(collate([item] * int(repeats), pad_id), device)


@torch.no_grad()
def evaluate_direct_n20(
    model: UnifiedMolecularLatentFlow,
    dataset: Sequence[dict[str, object]],
    vocab,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    model.eval()
    predictions: list[dict[str, object]] = []
    condition_results: list[dict[str, object]] = []
    for condition_index, item in enumerate(dataset):
        row = dict(item["row"])
        # The target molecule is intentionally unavailable to generation.
        generation_item = dict(item)
        generation_item["row"] = {key: value for key, value in row.items() if key != "target_smiles"}
        attempts: list[dict[str, object]] = []
        for start in range(0, int(args.num_samples), int(args.sample_batch_size)):
            count = min(int(args.sample_batch_size), int(args.num_samples) - start)
            batch = repeated_batch(generation_item, count, model.pad_id, device)
            latent = model.sample_latent(
                source_ids=batch["source_ids"],
                source_mask=batch["source_mask"],
                source_present=batch["source_present"],
                condition=batch["condition"],
                condition_mask=batch["condition_mask"],
                steps=int(args.flow_steps),
                edit_noise=float(args.edit_noise),
                denovo_noise=float(args.denovo_noise),
            )
            generated = model.generate(
                latent,
                vocab=vocab,
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_k=int(args.top_k),
                top_p=float(args.top_p),
            )
            for local_index, ids in enumerate(generated.detach().cpu().tolist()):
                raw_smiles = unified.detokenize_smiles(vocab.decode(ids))
                canonical = unified.safe_canonical_smiles(raw_smiles)
                valid = bool(canonical)
                generated_smiles = canonical or raw_smiles
                mode = str(item["mode"])
                property_success, property_distance = (
                    unified.property_success_and_distance(row, generated_smiles, mode=mode)
                    if valid
                    else (0.0, math.inf)
                )
                source_similarity = (
                    unified.morgan_tanimoto(str(row.get("source_smiles", "") or ""), generated_smiles)
                    if valid and mode == unified.EDIT_MODE
                    else math.nan
                )
                strict_success = bool(property_success >= 1.0) and (
                    mode != unified.EDIT_MODE
                    or (math.isfinite(source_similarity) and float(source_similarity) >= 0.65)
                )
                attempt = {
                    "condition_index": condition_index,
                    "condition_id": row.get("condition_id", row.get("sample_id", "")),
                    "mode": mode,
                    "attempt_index": start + local_index,
                    "generated_smiles": generated_smiles,
                    "valid": valid,
                    "property_success": bool(property_success >= 1.0),
                    "strict_success": strict_success,
                    "property_distance": property_distance if math.isfinite(property_distance) else None,
                    "source_similarity": source_similarity if math.isfinite(source_similarity) else None,
                }
                attempts.append(attempt)
                predictions.append(attempt)
        valid_smiles = [str(value["generated_smiles"]) for value in attempts if value["valid"]]
        condition_results.append(
            {
                "mode": str(item["mode"]),
                "attempted": len(attempts),
                "valid": sum(bool(value["valid"]) for value in attempts),
                "unique_valid": len(set(valid_smiles)),
                "property_any20": any(bool(value["property_success"]) for value in attempts),
                "strict_any20": any(bool(value["strict_success"]) for value in attempts),
            }
        )
        print(
            json.dumps(
                {
                    "eval_condition": condition_index + 1,
                    "total_conditions": len(dataset),
                    **condition_results[-1],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def aggregate(values: Sequence[dict[str, object]]) -> dict[str, float | int]:
        attempts = sum(int(value["attempted"]) for value in values)
        return {
            "conditions": len(values),
            "attempted": attempts,
            "validity": sum(int(value["valid"]) for value in values) / max(attempts, 1),
            "mean_unique_valid": sum(int(value["unique_valid"]) for value in values) / max(len(values), 1),
            "property_any20": sum(bool(value["property_any20"]) for value in values) / max(len(values), 1),
            "strict_any20": sum(bool(value["strict_any20"]) for value in values) / max(len(values), 1),
        }

    by_mode = {
        mode: aggregate([value for value in condition_results if value["mode"] == mode])
        for mode in (unified.DE_NOVO_MODE, unified.EDIT_MODE)
    }
    summary = {
        "protocol": "unified_language_conditioned_molecular_latent_flow_kill_test_v1",
        "candidate_contract": {
            "exact_attempts_per_condition": int(args.num_samples),
            "candidate_library": False,
            "retrieval_materializer": False,
            "finalizer": False,
            "oracle_reranking": False,
            "evaluation_target_access": False,
        },
        "overall": aggregate(condition_results),
        "by_mode": by_mode,
    }
    return predictions, summary


def write_predictions(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["condition_id", "generated_smiles"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    seed_everything(int(args.seed))
    device = resolve_device(str(args.device))
    if device.type != "cuda":
        print("WARNING: CUDA is unavailable; this pilot is intended for a GPU node.", file=sys.stderr)
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
        flow_layers=int(args.flow_layers),
    ).to(device)

    train_store = unified.FeatureStore(
        args.train_features_dir, array_name="query_tokens", variant=str(args.feature_variant)
    )
    validation_store = unified.FeatureStore(
        args.validation_features_dir, array_name="query_tokens", variant=str(args.feature_variant)
    )
    condition_dim = int(model_config["condition_dim"])
    train_rows = read_rows(args.train_csv)
    validation_rows = read_rows(args.validation_csv)
    train_dataset = build_dataset(
        train_rows,
        train_store,
        vocab,
        condition_dim,
        max_smiles_length=int(args.max_smiles_length),
    )
    validation_dataset = build_dataset(
        validation_rows,
        validation_store,
        vocab,
        condition_dim,
        max_smiles_length=int(args.max_smiles_length),
    )
    train_dataset = balanced_subset(train_dataset, int(args.train_limit), int(args.seed))
    validation_dataset = validation_subset(
        validation_dataset, int(args.validation_per_mode), int(args.seed) + 1
    )
    if not train_dataset or not validation_dataset:
        raise RuntimeError("Empty train or validation dataset after filtering.")

    manifest = {
        "protocol": "unified_language_conditioned_molecular_latent_flow_kill_test_v1",
        "seed": int(args.seed),
        "device": str(device),
        "train_rows": len(train_dataset),
        "validation_rows": len(validation_dataset),
        "train_mode_counts": dict(
            sorted(
                (mode, sum(str(item["mode"]) == mode for item in train_dataset))
                for mode in (unified.DE_NOVO_MODE, unified.EDIT_MODE)
            )
        ),
        "validation_mode_counts": dict(
            sorted(
                (mode, sum(str(item["mode"]) == mode for item in validation_dataset))
                for mode in (unified.DE_NOVO_MODE, unified.EDIT_MODE)
            )
        ),
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": file_sha256(args.base_checkpoint),
        "train_csv": str(args.train_csv),
        "train_csv_sha256": file_sha256(args.train_csv),
        "validation_csv": str(args.validation_csv),
        "validation_csv_sha256": file_sha256(args.validation_csv),
        "common_llm_feature_variant": str(args.feature_variant),
        "latent_tokens": int(args.latent_tokens),
        "flow_steps": int(args.flow_steps),
        "decoder_corruption": float(args.decoder_corruption),
        "exact_n": int(args.num_samples),
        "evaluation_target_access": False,
        "candidate_library": False,
        "finalizer": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)

    history = train_model(model, train_dataset, vocab, args, device)
    checkpoint_path = args.output_dir / "unified_latent_flow.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model_config,
            "latent_tokens": int(args.latent_tokens),
            "encoder_layers": int(args.encoder_layers),
            "flow_layers": int(args.flow_layers),
            "vocab": vocab.to_dict(),
            "history": history,
            "manifest": manifest,
        },
        checkpoint_path,
    )
    predictions, evaluation = evaluate_direct_n20(model, validation_dataset, vocab, args, device)
    evaluation["training"] = history
    evaluation["checkpoint"] = str(checkpoint_path)
    write_predictions(args.output_dir / "predictions.csv", predictions)
    (args.output_dir / "summary.json").write_text(
        json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(evaluation, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
