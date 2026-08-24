#!/usr/bin/env python3
"""Train, sample and audit one empty/source masked SELFIES denoiser."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
P6_DIR = PROJECT_DIR / "experiments" / "p6_unified_molecular_transition_policy"
UNIFIED_DIR = PROJECT_DIR / "experiments" / "unified_smiles_generator"
for path in (P6_DIR, UNIFIED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import p6_transition_program as p6  # noqa: E402
import unified_smiles_generator as unified  # noqa: E402

import selfies  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")


PAD = "<PAD>"
MASK = "<MASK>"
EOS = "<EOS>"
UNK = "<UNK>"
SPECIAL = (PAD, MASK, EOS, UNK)


def canonical(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(str(smiles or ""))
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True) if molecule is not None else ""


def sf_tokens(smiles: str) -> list[str]:
    encoded = selfies.encoder(str(smiles or ""))
    return list(selfies.split_selfies(encoded))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{str(key): str(value or "") for key, value in row.items()} for row in csv.DictReader(handle)]


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def source_for_row(row: Mapping[str, str]) -> str:
    return str(row.get("source_smiles", "") or row.get("molecule_smiles", "")).strip()


def target_for_row(row: Mapping[str, str]) -> str:
    return str(row.get("target_smiles", "") or row.get("policy_target_smiles", "")).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Vocabulary:
    token_to_id: dict[str, int]

    @classmethod
    def build(cls, rows: Sequence[Mapping[str, str]]) -> "Vocabulary":
        tokens = set(selfies.get_semantic_robust_alphabet())
        for row in rows:
            for smiles in (source_for_row(row), target_for_row(row)):
                if not smiles:
                    continue
                try:
                    tokens.update(sf_tokens(canonical(smiles)))
                except Exception:
                    continue
        ordered = [*SPECIAL, *sorted(tokens)]
        return cls({token: index for index, token in enumerate(ordered)})

    @property
    def id_to_token(self) -> list[str]:
        return [token for token, _index in sorted(self.token_to_id.items(), key=lambda item: item[1])]

    def encode_tokens(self, tokens: Sequence[str], max_tokens: int) -> np.ndarray:
        values = [self.token_to_id.get(token, self.token_to_id[UNK]) for token in tokens[: max_tokens - 1]]
        values.append(self.token_to_id[EOS])
        values.extend([self.token_to_id[PAD]] * (max_tokens - len(values)))
        return np.asarray(values[:max_tokens], dtype=np.int64)

    def decode_ids(self, values: Sequence[int]) -> str:
        tokens = []
        id_to_token = self.id_to_token
        for value in values:
            token = id_to_token[int(value)]
            if token == EOS:
                break
            if token not in SPECIAL:
                tokens.append(token)
        if not tokens:
            return ""
        try:
            return canonical(selfies.decoder("".join(tokens)))
        except Exception:
            return ""


class MaskedMoleculeDenoiser(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        condition_dim: int,
        max_tokens: int,
        d_model: int = 192,
        layers: int = 4,
        heads: int = 6,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_tokens = int(max_tokens)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.anchor_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_tokens, d_model)
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(condition_dim), nn.Linear(condition_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.time_projection = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        canvas: torch.Tensor,
        anchor: torch.Tensor,
        condition: torch.Tensor,
        noise_level: torch.Tensor,
    ) -> torch.Tensor:
        positions = torch.arange(canvas.shape[1], device=canvas.device)[None, :]
        context = self.condition_projection(condition) + self.time_projection(noise_level[:, None])
        hidden = (
            self.token_embedding(canvas)
            + self.anchor_embedding(anchor)
            + self.position_embedding(positions)
            + context[:, None, :]
        )
        return self.output(self.encoder(hidden))


def mode_agnostic_condition(
    row: Mapping[str, str], store: unified.FeatureStore, condition_dim: int
) -> np.ndarray:
    base = store.get(row)
    if base is None:
        base = unified.fallback_condition_features(row, condition_dim)
    program = unified.property_program_tokens(row, condition_dim).copy()
    # Remove the explicit edit/de-novo bit. The initial canvas is the only task-state signal.
    if program.shape[0] and program.shape[1] > 4:
        program[0, 4] = 0.0
    return np.concatenate([base, program], axis=0).mean(axis=0).astype(np.float32)


def make_dataset(
    rows: Sequence[Mapping[str, str]],
    store: unified.FeatureStore,
    vocab: Vocabulary,
    *,
    condition_dim: int,
    max_tokens: int,
) -> list[dict[str, object]]:
    output = []
    mask_id = vocab.token_to_id[MASK]
    for row in rows:
        target = canonical(target_for_row(row))
        if not target:
            continue
        try:
            target_ids = vocab.encode_tokens(sf_tokens(target), max_tokens)
            source = canonical(source_for_row(row)) if source_for_row(row) else ""
            anchor_ids = (
                vocab.encode_tokens(sf_tokens(source), max_tokens)
                if source
                else np.full(max_tokens, mask_id, dtype=np.int64)
            )
        except Exception:
            continue
        output.append({
            "row": dict(row),
            "target": target_ids,
            "anchor": anchor_ids,
            "condition": mode_agnostic_condition(row, store, condition_dim),
            "is_edit": bool(source),
        })
    return output


def corrupt_canvas(
    item: Mapping[str, object], vocab: Vocabulary, rng: random.Random,
    *, edit_mask_fraction: float,
) -> tuple[np.ndarray, float]:
    target = np.asarray(item["target"], dtype=np.int64)
    anchor = np.asarray(item["anchor"], dtype=np.int64)
    mask_id, pad_id = vocab.token_to_id[MASK], vocab.token_to_id[PAD]
    if not bool(item["is_edit"]):
        probability = rng.uniform(0.35, 1.0)
        if rng.random() < 0.35:
            probability = 1.0
        canvas = target.copy()
        selected = np.asarray([rng.random() < probability for _ in canvas], dtype=bool)
        canvas[selected] = mask_id
        return canvas, probability
    canvas = anchor.copy()
    active = int(np.argmax(anchor == pad_id)) if bool(np.any(anchor == pad_id)) else len(anchor)
    active = max(active, 1)
    fraction = min(max(float(edit_mask_fraction) * rng.uniform(0.7, 1.3), 0.1), 0.9)
    span = max(1, int(round(active * fraction)))
    start = rng.randrange(max(1, active - span + 1))
    canvas[start : start + span] = mask_id
    # A few masked tail slots let the same fixed canvas learn insertions/length changes.
    tail_end = min(len(canvas), active + 5)
    canvas[active:tail_end] = mask_id
    return canvas, fraction


def train_command(args: argparse.Namespace) -> int:
    unified.seed_everything(int(args.seed))
    rng = random.Random(int(args.seed))
    device = unified.resolve_device(str(args.device))
    rows = read_rows(args.train_csv)
    vocab = Vocabulary.build(rows)
    store = unified.FeatureStore(args.train_features_dir, array_name="query_tokens", variant="full")
    condition_dim = int(store.input_hidden_dim or 768)
    dataset = make_dataset(rows, store, vocab, condition_dim=condition_dim, max_tokens=int(args.max_tokens))
    if not dataset:
        raise ValueError("No masked-molecule training rows")
    config = {
        "vocab_size": len(vocab.token_to_id), "condition_dim": condition_dim,
        "max_tokens": int(args.max_tokens), "d_model": int(args.d_model),
        "layers": int(args.layers), "heads": int(args.heads), "ff_dim": int(args.ff_dim),
        "dropout": float(args.dropout),
    }
    model = MaskedMoleculeDenoiser(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        order = list(range(len(dataset)))
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), int(args.batch_size)):
            items = [dataset[index] for index in order[start : start + int(args.batch_size)]]
            corrupted = [
                corrupt_canvas(item, vocab, rng, edit_mask_fraction=float(args.edit_mask_fraction))
                for item in items
            ]
            canvas = torch.from_numpy(np.stack([value[0] for value in corrupted])).to(device)
            anchor = torch.from_numpy(np.stack([item["anchor"] for item in items])).to(device)
            target = torch.from_numpy(np.stack([item["target"] for item in items])).to(device)
            condition = torch.from_numpy(np.stack([item["condition"] for item in items])).to(device)
            noise = torch.tensor([value[1] for value in corrupted], dtype=torch.float32, device=device)
            logits = model(canvas, anchor, condition, noise)
            token_loss = F.cross_entropy(logits.transpose(1, 2), target, reduction="none")
            weights = torch.where(target.eq(vocab.token_to_id[PAD]), 0.15, 1.0)
            loss = (token_loss * weights).sum() / weights.sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        record = {"epoch": epoch, "train_loss": sum(losses) / max(len(losses), 1)}
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "masked_molecule_policy.pt"
    torch.save({
        "model_state": model.state_dict(), "model_config": config,
        "vocab": vocab.token_to_id, "history": history,
        "protocol": "p8_1_8_single_masked_selfies_denoiser",
        "output_language": "SELFIES", "seed": int(args.seed),
        "explicit_task_router": False, "property_reranking": False,
    }, checkpoint_path)
    summary = {
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
        "train_rows": len(dataset), "edit_rows": sum(bool(item["is_edit"]) for item in dataset),
        "denovo_rows": sum(not bool(item["is_edit"]) for item in dataset),
        "vocab_size": len(vocab.token_to_id), "history": history,
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def allowed_generation_ids(vocab: Vocabulary) -> list[int]:
    blocked = {PAD, MASK, UNK}
    return [index for token, index in vocab.token_to_id.items() if token not in blocked]


@torch.no_grad()
def refine(
    model: MaskedMoleculeDenoiser,
    canvas: torch.Tensor,
    anchor: torch.Tensor,
    condition: torch.Tensor,
    *,
    vocab: Vocabulary,
    steps: int,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    allowed = allowed_generation_ids(vocab)
    mask_id = vocab.token_to_id[MASK]
    current = canvas.clone()
    for step in range(max(1, int(steps))):
        noise = torch.full(
            (current.shape[0],), 1.0 - step / max(int(steps), 1),
            dtype=torch.float32, device=current.device,
        )
        logits = model(current, anchor, condition, noise) / max(float(temperature), 1e-6)
        blocked = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
        blocked[allowed] = False
        logits[..., blocked] = -torch.inf
        if 0 < int(top_k) < logits.shape[-1]:
            threshold = torch.topk(logits, int(top_k), dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < threshold, -torch.inf)
        probabilities = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(probabilities.reshape(-1, probabilities.shape[-1]), 1).reshape(current.shape)
        if step + 1 == int(steps):
            current = sampled
            break
        confidence = probabilities.gather(-1, sampled[..., None]).squeeze(-1)
        remask_fraction = max(0.0, 1.0 - (step + 1) / max(int(steps), 1))
        count = max(1, int(round(current.shape[1] * remask_fraction)))
        lowest = confidence.topk(count, dim=1, largest=False).indices
        current = sampled
        current.scatter_(1, lowest, mask_id)
    return current


def initial_edit_canvas(
    anchor: np.ndarray, vocab: Vocabulary, rng: random.Random, fraction: float
) -> np.ndarray:
    canvas = anchor.copy()
    pad_id, mask_id = vocab.token_to_id[PAD], vocab.token_to_id[MASK]
    active = int(np.argmax(anchor == pad_id)) if bool(np.any(anchor == pad_id)) else len(anchor)
    active = max(active, 1)
    span = max(1, min(active, int(round(active * float(fraction)))))
    start = rng.randrange(max(1, active - span + 1))
    canvas[start : start + span] = mask_id
    canvas[active : min(len(canvas), active + 5)] = mask_id
    return canvas


def sample_command(args: argparse.Namespace) -> int:
    unified.seed_everything(int(args.seed))
    rng = random.Random(int(args.seed))
    device = unified.resolve_device(str(args.device))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vocab = Vocabulary(dict(checkpoint["vocab"]))
    config = dict(checkpoint["model_config"])
    model = MaskedMoleculeDenoiser(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    store = unified.FeatureStore(args.eval_features_dir, array_name="query_tokens", variant="full")
    rows = read_rows(args.eval_csv)
    output: list[dict[str, object]] = []
    valid = identity = strict = 0
    for row_index, row in enumerate(rows):
        source = canonical(source_for_row(row)) if source_for_row(row) else ""
        anchor_np = (
            vocab.encode_tokens(sf_tokens(source), int(config["max_tokens"]))
            if source
            else np.full(int(config["max_tokens"]), vocab.token_to_id[MASK], dtype=np.int64)
        )
        anchors = np.repeat(anchor_np[None, :], int(args.num_samples), axis=0)
        if source:
            canvases = np.stack([
                initial_edit_canvas(anchor_np, vocab, rng, float(args.edit_mask_fraction))
                for _ in range(int(args.num_samples))
            ])
        else:
            canvases = anchors.copy()
        condition_np = mode_agnostic_condition(row, store, int(config["condition_dim"]))
        conditions = np.repeat(condition_np[None, :], int(args.num_samples), axis=0)
        generated = refine(
            model,
            torch.from_numpy(canvases).to(device),
            torch.from_numpy(anchors).to(device),
            torch.from_numpy(conditions).to(device),
            vocab=vocab, steps=int(args.steps), temperature=float(args.temperature), top_k=int(args.top_k),
        ).cpu().numpy()
        for candidate_index, values in enumerate(generated):
            smiles = vocab.decode_ids(values.tolist())
            metrics = unified.candidate_metrics(row, smiles, source_similarity_threshold=0.65)
            is_valid = bool(smiles)
            is_identity = bool(source and smiles == source)
            is_strict = metrics.get("table1_strict_success") == "True"
            valid += int(is_valid)
            identity += int(is_identity)
            strict += int(is_strict)
            item = dict(row)
            item.update({
                "generated_smiles": smiles, "candidate_smiles": smiles,
                "direct_candidate_index": candidate_index,
                "direct_candidate_raw_smiles": smiles,
                "direct_candidate_canonical_smiles": smiles,
                "generation_rank": candidate_index + 1, "candidate_rank": candidate_index + 1,
                "method": "p8_1_8_masked_selfies_policy",
                "p818_identity": str(is_identity),
                "p818_edit_mask_fraction": float(args.edit_mask_fraction),
                "p818_denoising_steps": int(args.steps),
            })
            item.update(metrics)
            item["direct_candidate_strict_fraction"] = item["unified_property_success_fraction"]
            output.append(item)
        if (row_index + 1) % 20 == 0 or row_index + 1 == len(rows):
            print(f"[p8.1.8-sample] {row_index + 1}/{len(rows)}", flush=True)
    write_rows(args.candidate_output_csv, output)
    total = len(rows) * int(args.num_samples)
    summary = {
        "protocol": "p8_1_8_raw_masked_selfies_denoising",
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
        "eval_rows": len(rows), "num_samples": int(args.num_samples),
        "candidate_validity": valid / max(total, 1),
        "candidate_identity_fraction": identity / max(total, 1),
        "candidate_strict_fraction": strict / max(total, 1),
        "edit_mask_fraction": float(args.edit_mask_fraction),
        "denoising_steps": int(args.steps), "temperature": float(args.temperature),
        "output_language": "SELFIES", "single_checkpoint": True,
        "property_reranking": False, "target_molecule_used_at_inference": False,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def percentile(values: Sequence[int], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, max(0, int(math.ceil(q * len(ordered))) - 1))])


def preflight_command(args: argparse.Namespace) -> int:
    train_rows = read_rows(args.train_csv)
    vocab = Vocabulary.build(train_rows)
    train_tokens = set(vocab.token_to_id)
    payload: dict[str, object] = {
        "protocol": "p8_1_8_representation_support_preflight",
        "output_language": "SELFIES", "same_canvas_language": True,
        "denovo_initial_state": "all_mask", "edit_initial_state": "source_span_corruption",
        "explicit_task_router": False, "property_reranking": False,
    }
    for label, path in (("train", args.train_csv), ("denovo", args.denovo_eval_csv), ("edit", args.edit_eval_csv)):
        lengths = []
        valid = supported = 0
        oov = set()
        rows = read_rows(path)
        for row in rows:
            target = canonical(target_for_row(row))
            if not target:
                continue
            valid += 1
            tokens = sf_tokens(target)
            lengths.append(len(tokens) + 1)
            oov.update(token for token in tokens if token not in train_tokens)
            supported += int(len(tokens) + 1 <= int(args.max_tokens) and not any(token not in train_tokens for token in tokens))
        payload[label] = {
            "rows": len(rows), "valid_targets": valid,
            "max_tokens": max(lengths, default=0), "p95_tokens": percentile(lengths, 0.95),
            "fixed_canvas_support": supported / max(valid, 1), "train_vocab_oov_tokens": sorted(oov),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def audit_command(args: argparse.Namespace) -> int:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_hash = sha256(args.checkpoint)
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.summaries]
    checks = {
        "one_checkpoint_hash": all(item["checkpoint_sha256"] == checkpoint_hash for item in summaries),
        "one_output_language": checkpoint.get("output_language") == "SELFIES" and all(item["output_language"] == "SELFIES" for item in summaries),
        "one_decoder_checkpoint": checkpoint.get("protocol") == "p8_1_8_single_masked_selfies_denoiser",
        "no_explicit_router": checkpoint.get("explicit_task_router") is False,
        "no_property_reranking": all(item.get("property_reranking") is False for item in summaries),
        "no_target_at_inference": all(item.get("target_molecule_used_at_inference") is False for item in summaries),
    }
    payload = {"status": "pass" if all(checks.values()) else "fail", "checks": checks, "checkpoint_sha256": checkpoint_hash}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--train-csv", required=True, type=Path)
    pre.add_argument("--denovo-eval-csv", required=True, type=Path)
    pre.add_argument("--edit-eval-csv", required=True, type=Path)
    pre.add_argument("--max-tokens", type=int, default=80)
    pre.add_argument("--output", required=True, type=Path)
    train = sub.add_parser("train")
    train.add_argument("--train-csv", required=True, type=Path)
    train.add_argument("--train-features-dir", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--max-tokens", type=int, default=80)
    train.add_argument("--d-model", type=int, default=192)
    train.add_argument("--layers", type=int, default=4)
    train.add_argument("--heads", type=int, default=6)
    train.add_argument("--ff-dim", type=int, default=512)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--edit-mask-fraction", type=float, default=0.35)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--device", default="auto")
    sample = sub.add_parser("sample")
    sample.add_argument("--checkpoint", required=True, type=Path)
    sample.add_argument("--eval-csv", required=True, type=Path)
    sample.add_argument("--eval-features-dir", required=True, type=Path)
    sample.add_argument("--candidate-output-csv", required=True, type=Path)
    sample.add_argument("--summary-json", required=True, type=Path)
    sample.add_argument("--num-samples", type=int, default=20)
    sample.add_argument("--edit-mask-fraction", type=float, default=0.35)
    sample.add_argument("--steps", type=int, default=4)
    sample.add_argument("--temperature", type=float, default=0.8)
    sample.add_argument("--top-k", type=int, default=16)
    sample.add_argument("--seed", type=int, default=7)
    sample.add_argument("--device", default="auto")
    audit = sub.add_parser("audit")
    audit.add_argument("--checkpoint", required=True, type=Path)
    audit.add_argument("--summaries", required=True, nargs="+", type=Path)
    audit.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "preflight":
        return preflight_command(args)
    if args.command == "train":
        return train_command(args)
    if args.command == "sample":
        return sample_command(args)
    if args.command == "audit":
        return audit_command(args)
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
