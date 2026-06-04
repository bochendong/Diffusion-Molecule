"""Image-molecule contrastive reranker for saved SketchMolEnd2End candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

from sketch_smiles.audit_pairs import _load_pillow, _load_rdkit, _resolve_image_path
from sketch_smiles.phase5a0_oracle_baseline import _fraction, _write_rows
from sketch_smiles.phase5a1_learned_smiles_decoder import _load_numpy, _load_torch, _make_fingerprint_fn, _scaffold_match, _set_rdkit_error_logging, _set_seeds, _tanimoto
from sketch_smiles.phase5b_joint_decoder import _load_ink_tensor


def run_contrastive_reranker(
    train_pairs_csv: str | Path,
    pair_dir: str | Path,
    predictions_csv: str | Path,
    output_dir: str | Path,
    fingerprint_bits: int = 2048,
    image_size: int = 128,
    embedding_dim: int = 256,
    encoder_channels: int = 64,
    hidden_dim: int = 512,
    epochs: int = 5,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    train_limit: int | None = None,
    eval_limit: int | None = None,
    seed: int = 7,
    device: str = "auto",
) -> dict[str, Any]:
    """Train a lightweight CLIP-style reranker and apply it to saved candidates."""

    train_pairs_path = Path(train_pairs_csv)
    pair_path = Path(pair_dir)
    predictions_path = Path(predictions_csv)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rdkit = _load_rdkit()
    pillow = _load_pillow()
    if not rdkit:
        raise RuntimeError("RDKit is required for contrastive reranking.")
    if not pillow:
        raise RuntimeError("Pillow is required for contrastive reranking.")
    _set_rdkit_error_logging(enabled=False)

    torch = _load_torch()
    np = _load_numpy()
    _set_seeds(seed, torch=torch, np=np)
    resolved_device = _resolve_device(device, torch)
    fingerprint_fn = _make_fingerprint_fn(rdkit, np=np, fingerprint_bits=fingerprint_bits)

    train_rows = _read_rows(train_pairs_path)
    if train_limit is not None:
        train_rows = train_rows[: int(train_limit)]
    train_examples = _prepare_train_examples(
        rows=train_rows,
        pair_dir=pair_path,
        fingerprint_fn=fingerprint_fn,
        pillow=pillow,
        np=np,
    )
    if not train_examples:
        raise RuntimeError("No train examples available for contrastive reranking.")

    model = ImageMoleculeContrastiveReranker(
        fingerprint_bits=fingerprint_bits,
        embedding_dim=embedding_dim,
        encoder_channels=encoder_channels,
        hidden_dim=hidden_dim,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = _train_model(
        model=model,
        examples=train_examples,
        optimizer=optimizer,
        torch=torch,
        np=np,
        pillow=pillow,
        device=resolved_device,
        batch_size=batch_size,
        epochs=epochs,
        image_size=image_size,
        seed=seed,
    )

    model_path = output_path / "contrastive_reranker.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "fingerprint_bits": int(fingerprint_bits),
                "embedding_dim": int(embedding_dim),
                "encoder_channels": int(encoder_channels),
                "hidden_dim": int(hidden_dim),
                "image_size": int(image_size),
            },
        },
        model_path,
    )
    (output_path / "train_history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    prediction_rows = _read_rows(predictions_path)
    if eval_limit is not None:
        prediction_rows = prediction_rows[: int(eval_limit)]
    reranked_rows = _evaluate_reranker(
        model=model,
        prediction_rows=prediction_rows,
        fingerprint_fn=fingerprint_fn,
        rdkit=rdkit,
        pillow=pillow,
        torch=torch,
        np=np,
        device=resolved_device,
        image_size=image_size,
    )
    reranked_path = output_path / "contrastive_reranked_predictions.csv"
    _write_rows(reranked_path, reranked_rows)
    summary = _summarize_rows(
        rows=reranked_rows,
        train_examples=len(train_examples),
        train_pairs_csv=train_pairs_path,
        pair_dir=pair_path,
        predictions_csv=predictions_path,
        output_dir=output_path,
        reranked_predictions=reranked_path,
        model_path=model_path,
        image_size=image_size,
        fingerprint_bits=fingerprint_bits,
        embedding_dim=embedding_dim,
        encoder_channels=encoder_channels,
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        device=str(resolved_device),
    )
    (output_path / "contrastive_rerank_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_rows(output_path / "contrastive_rerank_summary.csv", [summary])
    return summary


class ImageMoleculeContrastiveReranker:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        torch = _load_torch()
        nn = torch.nn
        functional = torch.nn.functional

        class _Model(nn.Module):
            def __init__(self, fingerprint_bits: int, embedding_dim: int, encoder_channels: int, hidden_dim: int) -> None:
                super().__init__()
                c1 = int(encoder_channels)
                c2 = max(c1 * 2, int(hidden_dim) // 2)
                self.image_encoder = nn.Sequential(
                    nn.Conv2d(1, c1, kernel_size=5, stride=2, padding=2),
                    nn.GELU(),
                    nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
                    nn.GELU(),
                    nn.Conv2d(c2, int(hidden_dim), kernel_size=3, stride=2, padding=1),
                    nn.GELU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(int(hidden_dim), int(embedding_dim)),
                )
                self.molecule_encoder = nn.Sequential(
                    nn.Linear(int(fingerprint_bits), int(hidden_dim)),
                    nn.GELU(),
                    nn.LayerNorm(int(hidden_dim)),
                    nn.Linear(int(hidden_dim), int(embedding_dim)),
                )
                self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07), dtype=torch.float32))

            def encode_image(self, images: Any) -> Any:
                return functional.normalize(self.image_encoder(images), dim=-1)

            def encode_molecule(self, fingerprints: Any) -> Any:
                return functional.normalize(self.molecule_encoder(fingerprints), dim=-1)

            def forward(self, images: Any, fingerprints: Any) -> Any:
                image_embeddings = self.encode_image(images)
                molecule_embeddings = self.encode_molecule(fingerprints)
                scale = self.logit_scale.exp().clamp(max=100.0)
                return scale * image_embeddings @ molecule_embeddings.t()

        return _Model(*args, **kwargs)


def _prepare_train_examples(rows: list[dict[str, str]], pair_dir: Path, fingerprint_fn: Any, pillow: dict[str, Any], np: Any) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        smiles = row.get("canonical_smiles") or row.get("input_smiles", "")
        if not smiles:
            continue
        image_path = _resolve_image_path(row.get("image_path", ""), pair_dir)
        if image_path is None or not image_path.exists():
            continue
        fingerprint = fingerprint_fn(smiles)
        if fingerprint is None:
            continue
        examples.append(
            {
                "pair_id": row.get("pair_id", ""),
                "smiles": smiles,
                "image_path": str(image_path),
                "fingerprint": np.asarray(fingerprint, dtype=np.float32),
            }
        )
    return examples


def _train_model(
    model: Any,
    examples: list[dict[str, Any]],
    optimizer: Any,
    torch: Any,
    np: Any,
    pillow: dict[str, Any],
    device: Any,
    batch_size: int,
    epochs: int,
    image_size: int,
    seed: int,
) -> list[dict[str, float]]:
    loss_fn = torch.nn.CrossEntropyLoss()
    rng = random.Random(seed)
    history: list[dict[str, float]] = []
    for epoch in range(1, int(epochs) + 1):
        order = list(range(len(examples)))
        rng.shuffle(order)
        total_loss = 0.0
        total_correct_i2m = 0
        total_correct_m2i = 0
        total_seen = 0
        model.train()
        for start in range(0, len(order), int(batch_size)):
            batch = _load_batch(
                examples=[examples[idx] for idx in order[start : start + int(batch_size)]],
                pillow=pillow,
                np=np,
                image_size=image_size,
            )
            if not batch:
                continue
            images = torch.as_tensor(np.stack([row["image"] for row in batch]), dtype=torch.float32, device=device)
            fingerprints = torch.as_tensor(np.stack([row["fingerprint"] for row in batch]), dtype=torch.float32, device=device)
            labels = torch.arange(images.shape[0], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images, fingerprints)
            loss_i2m = loss_fn(logits, labels)
            loss_m2i = loss_fn(logits.t(), labels)
            loss = 0.5 * (loss_i2m + loss_m2i)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_size_actual = int(images.shape[0])
            total_loss += float(loss.item()) * batch_size_actual
            total_correct_i2m += int((logits.argmax(dim=1) == labels).sum().item())
            total_correct_m2i += int((logits.argmax(dim=0) == labels).sum().item())
            total_seen += batch_size_actual
        row = {
            "epoch": float(epoch),
            "train_loss": float(total_loss / max(1, total_seen)),
            "train_i2m_top1": float(total_correct_i2m / max(1, total_seen)),
            "train_m2i_top1": float(total_correct_m2i / max(1, total_seen)),
            "train_examples": float(total_seen),
        }
        history.append(row)
        print(
            "  epoch={epoch} train_loss={loss:.4f} i2m_top1={i2m:.3f} m2i_top1={m2i:.3f}".format(
                epoch=epoch,
                loss=row["train_loss"],
                i2m=row["train_i2m_top1"],
                m2i=row["train_m2i_top1"],
            ),
            flush=True,
        )
    return history


def _load_batch(examples: list[dict[str, Any]], pillow: dict[str, Any], np: Any, image_size: int) -> list[dict[str, Any]]:
    batch: list[dict[str, Any]] = []
    for example in examples:
        image = _load_ink_tensor(Path(example["image_path"]), pillow=pillow, image_size=image_size, np=np)
        if image is None:
            continue
        batch.append({**example, "image": image})
    return batch


def _evaluate_reranker(
    model: Any,
    prediction_rows: list[dict[str, str]],
    fingerprint_fn: Any,
    rdkit: dict[str, Any],
    pillow: dict[str, Any],
    torch: Any,
    np: Any,
    device: Any,
    image_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for row in prediction_rows:
            candidates = _candidate_list(row)
            target_smiles = row.get("target_smiles", "")
            image_path = _best_target_image_path(row)
            image = _load_ink_tensor(image_path, pillow=pillow, image_size=image_size, np=np)
            scores = _score_candidates(
                model=model,
                image=image,
                candidates=candidates,
                fingerprint_fn=fingerprint_fn,
                torch=torch,
                np=np,
                device=device,
            )
            ranked = _rank_descending(candidates, scores)
            top1 = ranked[0] if ranked else ""
            target_mol = rdkit["Chem"].MolFromSmiles(target_smiles) if target_smiles else None
            top1_mol = rdkit["Chem"].MolFromSmiles(top1) if top1 else None
            top1_tanimoto = _tanimoto(target_mol, top1_mol, rdkit)
            best_tanimoto = max((_tanimoto(target_mol, rdkit["Chem"].MolFromSmiles(smiles), rdkit) for smiles in candidates), default=0.0)
            rows.append(
                {
                    "pair_id": row.get("pair_id", ""),
                    "rerank_mode": "contrastive_image_molecule",
                    "target_smiles": target_smiles,
                    "generated_smiles": top1,
                    "candidate_count": float(len(candidates)),
                    "canonical_candidates": "|".join(ranked),
                    "candidate_scores": "|".join(f"{scores.get(smiles, 0.0):.6f}" for smiles in ranked),
                    "top1_valid": bool(top1),
                    "top1_exact_match": bool(top1 and top1 == target_smiles),
                    "topk_exact_match": bool(target_smiles in candidates),
                    "top1_scaffold_match": bool(_scaffold_match(target_smiles, top1, rdkit)),
                    "top1_target_tanimoto": float(top1_tanimoto),
                    "mean_best_tanimoto": float(best_tanimoto),
                    "oracle_gap_tanimoto": float(best_tanimoto - top1_tanimoto),
                    "source_image_path": row.get("source_image_path", ""),
                    "target_image_path": row.get("target_image_path", ""),
                }
            )
    return rows


def _score_candidates(
    model: Any,
    image: Any | None,
    candidates: list[str],
    fingerprint_fn: Any,
    torch: Any,
    np: Any,
    device: Any,
) -> dict[str, float]:
    if image is None or not candidates:
        return {smiles: 0.0 for smiles in candidates}
    valid_candidates: list[str] = []
    fingerprints: list[Any] = []
    for smiles in candidates:
        fingerprint = fingerprint_fn(smiles)
        if fingerprint is not None:
            valid_candidates.append(smiles)
            fingerprints.append(np.asarray(fingerprint, dtype=np.float32))
    if not valid_candidates:
        return {smiles: 0.0 for smiles in candidates}
    image_tensor = torch.as_tensor(np.asarray(image), dtype=torch.float32, device=device).unsqueeze(0)
    fingerprint_tensor = torch.as_tensor(np.stack(fingerprints), dtype=torch.float32, device=device)
    image_embedding = model.encode_image(image_tensor)
    molecule_embeddings = model.encode_molecule(fingerprint_tensor)
    scores_tensor = (image_embedding @ molecule_embeddings.t())[0].detach().cpu().numpy()
    scores = {smiles: float(score) for smiles, score in zip(valid_candidates, scores_tensor)}
    return {smiles: scores.get(smiles, 0.0) for smiles in candidates}


def _summarize_rows(
    rows: list[dict[str, Any]],
    train_examples: int,
    train_pairs_csv: Path,
    pair_dir: Path,
    predictions_csv: Path,
    output_dir: Path,
    reranked_predictions: Path,
    model_path: Path,
    image_size: int,
    fingerprint_bits: int,
    embedding_dim: int,
    encoder_channels: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    total = len(rows)
    tanimoto_values = [float(row["top1_target_tanimoto"]) for row in rows if row.get("top1_target_tanimoto") != ""]
    best_values = [float(row["mean_best_tanimoto"]) for row in rows if row.get("mean_best_tanimoto") != ""]
    gap_values = [float(row["oracle_gap_tanimoto"]) for row in rows if row.get("oracle_gap_tanimoto") != ""]
    return {
        "phase": "sketchmol_end2end_contrastive_reranker",
        "rerank_mode": "contrastive_image_molecule",
        "train_pairs_csv": str(train_pairs_csv),
        "pair_dir": str(pair_dir),
        "predictions_csv": str(predictions_csv),
        "output_dir": str(output_dir),
        "model": str(model_path),
        "reranked_predictions": str(reranked_predictions),
        "fingerprint_bits": float(fingerprint_bits),
        "image_size": float(image_size),
        "embedding_dim": float(embedding_dim),
        "encoder_channels": float(encoder_channels),
        "hidden_dim": float(hidden_dim),
        "epochs": float(epochs),
        "batch_size": float(batch_size),
        "learning_rate": float(learning_rate),
        "seed": float(seed),
        "device": device,
        "train_examples": float(train_examples),
        "eval_examples": float(total),
        "top1_valid": float(_count(rows, "top1_valid")),
        "top1_valid_fraction": _fraction(_count(rows, "top1_valid"), total),
        "top1_exact_matches": float(_count(rows, "top1_exact_match")),
        "top1_exact_match_fraction": _fraction(_count(rows, "top1_exact_match"), total),
        "topk_exact_matches": float(_count(rows, "topk_exact_match")),
        "topk_exact_match_fraction": _fraction(_count(rows, "topk_exact_match"), total),
        "top1_scaffold_matches": float(_count(rows, "top1_scaffold_match")),
        "top1_scaffold_match_fraction": _fraction(_count(rows, "top1_scaffold_match"), total),
        "top1_target_tanimoto": _mean(tanimoto_values),
        "mean_best_tanimoto": _mean(best_values),
        "mean_oracle_gap_tanimoto": _mean(gap_values),
        "mean_candidate_count": _mean([float(row["candidate_count"]) for row in rows if row.get("candidate_count") != ""]),
    }


def _candidate_list(row: dict[str, str]) -> list[str]:
    raw = row.get("beam_canonical_candidates") or row.get("canonical_candidates") or row.get("raw_samples") or ""
    candidates: list[str] = []
    seen: set[str] = set()
    for value in raw.split("|"):
        smiles = value.strip()
        if smiles and smiles not in seen:
            seen.add(smiles)
            candidates.append(smiles)
    return candidates


def _best_target_image_path(row: dict[str, str]) -> Path | None:
    for key in ("target_image_path", "source_image_path"):
        value = row.get(key, "")
        if value:
            path = Path(value)
            if path.exists():
                return path
    return None


def _rank_descending(candidates: list[str], scores: dict[str, float]) -> list[str]:
    return [smiles for _idx, smiles in sorted(enumerate(candidates), key=lambda item: (scores.get(item[1], 0.0), -item[0]), reverse=True)]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if bool(row.get(key)))


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _resolve_device(device: str, torch: Any) -> Any:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an image-molecule contrastive reranker for SketchMolEnd2End.")
    parser.add_argument("--train-pairs-csv", required=True)
    parser.add_argument("--pair-dir", required=True)
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--encoder-channels", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run_contrastive_reranker(
        train_pairs_csv=args.train_pairs_csv,
        pair_dir=args.pair_dir,
        predictions_csv=args.predictions_csv,
        output_dir=args.output_dir,
        fingerprint_bits=args.fingerprint_bits,
        image_size=args.image_size,
        embedding_dim=args.embedding_dim,
        encoder_channels=args.encoder_channels,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
