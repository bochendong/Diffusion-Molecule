"""Optional PyTorch training loop for JSONL dual-stream examples."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

from .config import get_section, load_config, merged_config
from .data import read_jsonl, write_summary
from .model import SmilesDualStreamModel, TORCH_AVAILABLE
from .tokenization import BOS, EOS, PAD, SmilesVocabulary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--train-jsonl", type=Path, default=None)
    parser.add_argument("--eval-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--alignment-loss-weight", type=float, default=None)
    parser.add_argument("--reconstruction-loss-weight", type=float, default=None)
    parser.add_argument("--molecule-alignment-weight", type=float, default=None)
    parser.add_argument("--token-alignment-weight", type=float, default=None)
    parser.add_argument("--fragment-alignment-weight", type=float, default=None)
    parser.add_argument("--fragment-chunk-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--eval-fraction", type=float, default=None)
    parser.add_argument("--max-sequence-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args(argv)

    config = _resolve_train_config(args)
    train_jsonl = _path_required(config, "train_jsonl")
    output_dir = _path_required(config, "output_dir")
    eval_jsonl = _path_or_none(config.get("eval_jsonl"))
    epochs = int(config.get("epochs", 3))
    batch_size = int(config.get("batch_size", 32))
    eval_batch_size = int(config.get("eval_batch_size") or batch_size)
    seed = int(config.get("seed", 7))
    max_sequence_length = _optional_int(config.get("max_sequence_length"))
    eval_fraction = float(config.get("eval_fraction", 0.0) or 0.0)
    save_every = max(1, int(config.get("save_every", 1)))
    eval_every = max(1, int(config.get("eval_every", 1)))
    resume = bool(config.get("resume", False))

    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for training. Install torch or run prepare_manifest/run_smoke only.")

    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    rows = _filter_rows(read_jsonl(train_jsonl), max_sequence_length=max_sequence_length)
    if not rows:
        raise ValueError(f"No training rows found in {train_jsonl}")
    train_rows, eval_rows = _split_rows(rows, eval_fraction=eval_fraction, seed=seed)
    if eval_jsonl is not None:
        eval_rows = _filter_rows(read_jsonl(eval_jsonl), max_sequence_length=max_sequence_length)

    output_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint = output_dir / "latest_checkpoint.pt"
    checkpoint_payload = None
    if resume and latest_checkpoint.exists():
        checkpoint_payload = torch.load(latest_checkpoint, map_location="cpu")

    if checkpoint_payload is not None:
        vocab = SmilesVocabulary.from_dict(checkpoint_payload["vocab"])
    else:
        vocab = build_vocab(train_rows + eval_rows)
    pad_id = vocab.token_to_id[PAD]
    bos_id = vocab.token_to_id[BOS]
    eos_id = vocab.token_to_id[EOS]
    train_dataset = [_encode_row(row, vocab, bos_id=bos_id, eos_id=eos_id) for row in train_rows]
    eval_dataset = [_encode_row(row, vocab, bos_id=bos_id, eos_id=eos_id) for row in eval_rows]

    model = SmilesDualStreamModel(
        len(vocab.token_to_id),
        embed_dim=int(config.get("embed_dim", 128)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        pad_id=pad_id,
        fragment_chunk_size=int(config.get("fragment_chunk_size", 8)),
    )
    device = _resolve_device(str(config.get("device", "auto")))
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )

    start_epoch = 1
    global_step = 0
    history: list[dict[str, float]] = []
    if checkpoint_payload is not None:
        model.load_state_dict(checkpoint_payload["model_state"])
        optimizer.load_state_dict(checkpoint_payload["optimizer_state"])
        start_epoch = int(checkpoint_payload.get("epoch", 0)) + 1
        global_step = int(checkpoint_payload.get("global_step", 0))
        history = list(checkpoint_payload.get("history", []))

    run_config = {
        **config,
        "train_jsonl": str(train_jsonl),
        "eval_jsonl": str(eval_jsonl) if eval_jsonl else None,
        "output_dir": str(output_dir),
        "train_rows": len(train_dataset),
        "eval_rows": len(eval_dataset),
        "vocab_size": len(vocab.token_to_id),
        "device": str(device),
        "resume_from": str(latest_checkpoint) if checkpoint_payload is not None else None,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")

    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.get("fp16", False)) and device.type == "cuda")
    grad_accum = max(1, int(config.get("gradient_accumulation_steps", 1)))
    grad_clip = float(config.get("grad_clip", 0.0) or 0.0)
    reconstruction_weight = float(config.get("reconstruction_loss_weight", 1.0))
    alignment_weight = float(config.get("alignment_loss_weight", 1.0))
    molecule_alignment_weight = float(config.get("molecule_alignment_weight", 1.0))
    token_alignment_weight = float(config.get("token_alignment_weight", 0.1))
    fragment_alignment_weight = float(config.get("fragment_alignment_weight", 0.2))
    log_path = output_dir / "train_log.jsonl"
    if checkpoint_payload is None:
        log_path.write_text("", encoding="utf-8")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        shuffled = list(train_dataset)
        random.Random(seed + epoch).shuffle(shuffled)
        totals = {"loss": 0.0, "reconstruction_loss": 0.0, "alignment_loss": 0.0}
        total_batches = 0
        optimizer_steps = 0
        skipped_optimizer_steps = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch_rows in enumerate(_batches(shuffled, batch_size), start=1):
            batch = _collate(batch_rows, pad_id=pad_id, device=device)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                output = model(
                    batch["input_ids"],
                    batch["decoder_input_ids"],
                    batch["target_ids"],
                    reconstruction_loss_weight=reconstruction_weight,
                    alignment_loss_weight=alignment_weight,
                    molecule_alignment_weight=molecule_alignment_weight,
                    token_alignment_weight=token_alignment_weight,
                    fragment_alignment_weight=fragment_alignment_weight,
                )
                loss = output["loss"] / grad_accum
            scaler.scale(loss).backward()
            if batch_index % grad_accum == 0:
                skipped_optimizer_steps += _optimizer_step(
                    model,
                    optimizer,
                    scaler,
                    grad_clip=grad_clip,
                )
                optimizer_steps += 1
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            totals["loss"] += float(output["loss"].detach().cpu())
            totals["reconstruction_loss"] += float(output["reconstruction_loss"].detach().cpu())
            totals["alignment_loss"] += float(output["alignment_loss"].detach().cpu())
            for optional_key in ("molecule_alignment_loss", "token_alignment_loss", "fragment_alignment_loss"):
                totals[optional_key] = totals.get(optional_key, 0.0) + float(output[optional_key].detach().cpu())
            total_batches += 1

        if total_batches and total_batches % grad_accum != 0:
            skipped_optimizer_steps += _optimizer_step(
                model,
                optimizer,
                scaler,
                grad_clip=grad_clip,
            )
            optimizer_steps += 1
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        record = {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_steps": optimizer_steps,
            "skipped_optimizer_steps": skipped_optimizer_steps,
            "loss": totals["loss"] / max(total_batches, 1),
            "reconstruction_loss": totals["reconstruction_loss"] / max(total_batches, 1),
            "alignment_loss": totals["alignment_loss"] / max(total_batches, 1),
            "molecule_alignment_loss": totals.get("molecule_alignment_loss", 0.0) / max(total_batches, 1),
            "token_alignment_loss": totals.get("token_alignment_loss", 0.0) / max(total_batches, 1),
            "fragment_alignment_loss": totals.get("fragment_alignment_loss", 0.0) / max(total_batches, 1),
        }
        if eval_dataset and epoch % eval_every == 0:
            record.update(
                {
                    f"eval_{key}": value
                    for key, value in _evaluate(
                        model,
                        eval_dataset,
                        eval_batch_size,
                        pad_id,
                        device,
                        seed=seed + epoch,
                        use_autocast=scaler.is_enabled(),
                        reconstruction_loss_weight=reconstruction_weight,
                        alignment_loss_weight=alignment_weight,
                        molecule_alignment_weight=molecule_alignment_weight,
                        token_alignment_weight=token_alignment_weight,
                        fragment_alignment_weight=fragment_alignment_weight,
                    ).items()
                }
            )
        history.append(record)
        _append_jsonl(log_path, record)

        payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "vocab": vocab.to_dict(),
            "config": {
                "embed_dim": int(config.get("embed_dim", 128)),
                "hidden_dim": int(config.get("hidden_dim", 256)),
                "pad_id": pad_id,
                "fragment_chunk_size": int(config.get("fragment_chunk_size", 8)),
            },
            "epoch": epoch,
            "global_step": global_step,
            "history": history,
        }
        torch.save(payload, latest_checkpoint)
        if epoch % save_every == 0 or epoch == epochs:
            torch.save(payload, output_dir / f"checkpoint_epoch_{epoch:04d}.pt")

    final_checkpoint = output_dir / "smiles_dual_stream.pt"
    if latest_checkpoint.exists():
        torch.save(torch.load(latest_checkpoint, map_location="cpu"), final_checkpoint)
    write_summary(
        {
            "train_rows": len(train_dataset),
            "eval_rows": len(eval_dataset),
            "history": history,
            "vocab_size": len(vocab.token_to_id),
            "latest_checkpoint": str(latest_checkpoint),
            "final_checkpoint": str(final_checkpoint),
        },
        output_dir / "summary.json",
    )
    return 0


def _resolve_train_config(args: argparse.Namespace) -> dict[str, object]:
    config = get_section(load_config(args.config), "train")
    paths = get_section(load_config(args.config), "paths") if args.config else {}
    defaults = {
        "train_jsonl": paths.get("manifest_jsonl"),
        "eval_jsonl": paths.get("eval_jsonl"),
        "output_dir": paths.get("train_output_dir"),
        "epochs": 3,
        "batch_size": 32,
        "eval_batch_size": None,
        "embed_dim": 128,
        "hidden_dim": 256,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "alignment_loss_weight": 1.0,
        "reconstruction_loss_weight": 1.0,
        "molecule_alignment_weight": 1.0,
        "token_alignment_weight": 0.1,
        "fragment_alignment_weight": 0.2,
        "fragment_chunk_size": 8,
        "gradient_accumulation_steps": 1,
        "grad_clip": 1.0,
        "eval_fraction": 0.0,
        "max_sequence_length": None,
        "seed": 7,
        "device": "auto",
        "save_every": 1,
        "eval_every": 1,
        "resume": False,
        "fp16": False,
    }
    cli = {
        "train_jsonl": args.train_jsonl,
        "eval_jsonl": args.eval_jsonl,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "alignment_loss_weight": args.alignment_loss_weight,
        "reconstruction_loss_weight": args.reconstruction_loss_weight,
        "molecule_alignment_weight": args.molecule_alignment_weight,
        "token_alignment_weight": args.token_alignment_weight,
        "fragment_alignment_weight": args.fragment_alignment_weight,
        "fragment_chunk_size": args.fragment_chunk_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "grad_clip": args.grad_clip,
        "eval_fraction": args.eval_fraction,
        "max_sequence_length": args.max_sequence_length,
        "seed": args.seed,
        "device": args.device,
        "save_every": args.save_every,
        "eval_every": args.eval_every,
        "resume": True if args.resume else None,
        "fp16": True if args.fp16 else None,
    }
    if args.no_resume:
        cli["resume"] = False
    return merged_config(merged_config(defaults, config), cli)


def build_vocab(rows: Iterable[dict[str, object]]) -> SmilesVocabulary:
    vocab = SmilesVocabulary()
    for row in rows:
        vocab.update(
            [
                _tokens(row, "corrupted_tokens"),
                _tokens(row, "target_tokens"),
                _tokens(row, "source_tokens"),
            ]
        )
    return vocab


def _encode_row(row: dict[str, object], vocab: SmilesVocabulary, *, bos_id: int, eos_id: int) -> dict[str, list[int]]:
    target = vocab.encode(_tokens(row, "target_tokens"), add_eos=True)
    return {
        "input_ids": vocab.encode(_tokens(row, "corrupted_tokens"), add_eos=True),
        "decoder_input_ids": [bos_id] + target[:-1],
        "target_ids": target,
    }


def _tokens(row: dict[str, object], key: str) -> list[str]:
    value = row.get(key)
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _batches(rows: list[dict[str, list[int]]], batch_size: int) -> Iterable[list[dict[str, list[int]]]]:
    for start in range(0, len(rows), max(1, batch_size)):
        yield rows[start : start + max(1, batch_size)]


def _collate(rows: list[dict[str, list[int]]], *, pad_id: int, device=None):
    import torch

    keys = ["input_ids", "decoder_input_ids", "target_ids"]
    output = {}
    for key in keys:
        max_len = max(len(row[key]) for row in rows)
        output[key] = torch.tensor(
            [row[key] + [pad_id] * (max_len - len(row[key])) for row in rows],
            dtype=torch.long,
            device=device,
        )
    return output


def _evaluate(
    model,
    dataset: list[dict[str, list[int]]],
    batch_size: int,
    pad_id: int,
    device,
    *,
    seed: int,
    use_autocast: bool,
    reconstruction_loss_weight: float,
    alignment_loss_weight: float,
    molecule_alignment_weight: float,
    token_alignment_weight: float,
    fragment_alignment_weight: float,
) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    model.eval()
    shuffled = list(dataset)
    random.Random(seed).shuffle(shuffled)
    totals = {
        "loss": 0.0,
        "reconstruction_loss": 0.0,
        "alignment_loss": 0.0,
        "molecule_alignment_loss": 0.0,
        "token_alignment_loss": 0.0,
        "fragment_alignment_loss": 0.0,
    }
    token_correct = 0
    token_total = 0
    molecule_cosine_sum = 0.0
    molecule_cosine_count = 0
    total_batches = 0
    with torch.no_grad():
        for batch_rows in _batches(shuffled, batch_size):
            batch = _collate(batch_rows, pad_id=pad_id, device=device)
            with torch.cuda.amp.autocast(enabled=use_autocast and device.type == "cuda"):
                output = model(
                    batch["input_ids"],
                    batch["decoder_input_ids"],
                    batch["target_ids"],
                    reconstruction_loss_weight=reconstruction_loss_weight,
                    alignment_loss_weight=alignment_loss_weight,
                    molecule_alignment_weight=molecule_alignment_weight,
                    token_alignment_weight=token_alignment_weight,
                    fragment_alignment_weight=fragment_alignment_weight,
                )
            for key in totals:
                totals[key] += float(output[key].detach().cpu())
            logits = output["logits"]
            preds = logits.argmax(dim=-1)
            mask = batch["target_ids"].ne(pad_id)
            token_correct += int(((preds == batch["target_ids"]) & mask).sum().item())
            token_total += int(mask.sum().item())
            _, input_pooled = model.encode(batch["input_ids"], model.encoder)
            _, target_pooled = model.encode(batch["target_ids"], model.target_encoder)
            input_proj = F.normalize(model.projection(input_pooled.float()), dim=-1)
            target_proj = F.normalize(model.projection(target_pooled.float()), dim=-1)
            molecule_cosine_sum += float(F.cosine_similarity(input_proj, target_proj, dim=-1).sum().item())
            molecule_cosine_count += int(input_proj.shape[0])
            total_batches += 1
    metrics = {key: value / max(total_batches, 1) for key, value in totals.items()}
    metrics["token_accuracy"] = token_correct / max(token_total, 1)
    metrics["molecule_cosine"] = molecule_cosine_sum / max(molecule_cosine_count, 1)
    metrics["rows"] = float(len(dataset))
    return metrics


def _optimizer_step(model, optimizer, scaler, *, grad_clip: float) -> int:
    """Apply one optimizer step. Returns 1 when GradScaler skips the update."""

    import torch

    reference = next(model.parameters()).detach().clone()
    if grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    updated = next(model.parameters()).detach()
    if reference.device != updated.device:
        updated = updated.to(reference.device)
    if reference.dtype != updated.dtype:
        updated = updated.to(reference.dtype)
    return int(torch.equal(reference, updated))


def _split_rows(rows: list[dict[str, object]], *, eval_fraction: float, seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    explicit_eval = [row for row in rows if str(row.get("split", "")).lower() in {"eval", "valid", "validation", "test"}]
    explicit_train = [row for row in rows if str(row.get("split", "")).lower() not in {"eval", "valid", "validation", "test"}]
    if explicit_eval:
        return explicit_train, explicit_eval
    if eval_fraction <= 0:
        return rows, []
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    eval_count = max(1, int(len(shuffled) * eval_fraction))
    return shuffled[eval_count:], shuffled[:eval_count]


def _filter_rows(rows: list[dict[str, object]], *, max_sequence_length: int | None) -> list[dict[str, object]]:
    if max_sequence_length is None or max_sequence_length <= 0:
        return rows
    return [row for row in rows if _row_length(row) <= max_sequence_length]


def _row_length(row: dict[str, object]) -> int:
    return max(len(_tokens(row, "corrupted_tokens")), len(_tokens(row, "target_tokens")), len(_tokens(row, "source_tokens")))


def _resolve_device(value: str):
    import torch

    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _path_required(config: dict[str, object], key: str) -> Path:
    path = _path_or_none(config.get(key))
    if path is None:
        raise ValueError(f"Missing required training path: {key}")
    return path


def _path_or_none(value: object) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value))


def _optional_int(value: object) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def _append_jsonl(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True))
        handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
