"""Phase 1 oracle latent-conditioned SMILES denoising decoder.

This module intentionally removes the JEPA planner from the loop. It asks a
cleaner Phase 1 question: if the decoder receives the target molecule latent
directly, can a small diffusion-style denoising model generate valid and
similar SMILES?
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from .chem import canonicalize_smiles
from .features import MOLECULE_LATENT_VERSION, molecule_latent
from .report import summarize_predictions_csv
from .schema import BenchmarkExample, Candidate, TaskType
from .task_builder import load_molecule_rows
from .verifier import CandidateScore, score_candidates, summarize_scores


@dataclass
class OracleLatentDiffusionConfig:
    condition_dim: int = 256
    hidden_dim: int = 256
    transformer_layers: int = 3
    attention_heads: int = 4
    objective: str = "autoregressive"
    dropout: float = 0.10
    max_length: int = 128
    epochs: int = 20
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    noise_min: float = 0.15
    noise_max: float = 0.85
    sample_steps: int = 16
    samples_per_condition: int = 8
    sample_multiplier: int = 1
    sample_batch_size: int = 256
    temperature: float = 0.9
    top_p: float = 0.95
    validity_bonus: float = 1.0
    train_fraction: float = 0.8
    device: str = "auto"
    seed: int = 7


class SmilesVocabulary:
    pad = "<pad>"
    bos = "<bos>"
    eos = "<eos>"
    mask = "<mask>"
    unk = "<unk>"

    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.index = {token: idx for idx, token in enumerate(tokens)}
        self.pad_id = self.index[self.pad]
        self.bos_id = self.index[self.bos]
        self.eos_id = self.index[self.eos]
        self.mask_id = self.index[self.mask]
        self.unk_id = self.index[self.unk]
        self.first_char_id = len([self.pad, self.bos, self.eos, self.mask, self.unk])

    @classmethod
    def build(cls, smiles: list[str]) -> "SmilesVocabulary":
        alphabet = sorted({token for value in smiles for token in _tokenize_smiles(value)})
        return cls([cls.pad, cls.bos, cls.eos, cls.mask, cls.unk, *alphabet])

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, smiles: str, max_length: int) -> list[int]:
        ids = [self.bos_id]
        ids.extend(self.index.get(token, self.unk_id) for token in _tokenize_smiles(smiles)[: max(0, max_length - 2)])
        ids.append(self.eos_id)
        if len(ids) < max_length:
            ids.extend([self.pad_id] * (max_length - len(ids)))
        return ids[:max_length]

    def decode(self, ids: list[int]) -> str:
        chars: list[str] = []
        for token_id in ids:
            token = self.tokens[int(token_id)] if 0 <= int(token_id) < len(self.tokens) else self.unk
            if token == self.eos:
                break
            if token in {self.pad, self.bos, self.mask, self.unk}:
                continue
            chars.append(token)
        return "".join(chars)

    def to_dict(self) -> dict[str, object]:
        return {"tokens": self.tokens}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SmilesVocabulary":
        return cls([str(token) for token in payload["tokens"]])


class OracleLatentSmilesDiffusion:
    backend_name = "oracle_latent_smiles_diffusion"

    def __init__(self, config: OracleLatentDiffusionConfig | None = None):
        self.config = config or OracleLatentDiffusionConfig()
        self.vocab: SmilesVocabulary | None = None
        self.model: Any | None = None
        self.device_name = "cpu"
        self.history: list[dict[str, float | str]] = []

    def fit(self, smiles: list[str]) -> "OracleLatentSmilesDiffusion":
        torch, nn, DataLoader, TensorDataset = _torch_deps()
        _set_torch_seed(torch, self.config.seed)
        smiles = _canonical_smiles_list(smiles, self.config.max_length)
        if not smiles:
            raise ValueError("No valid training SMILES remain after canonicalization and max_length filtering.")
        self.vocab = SmilesVocabulary.build(smiles)
        tokens = np.asarray([self.vocab.encode(value, self.config.max_length) for value in smiles], dtype=np.int64)
        conditions = np.stack([molecule_latent(value, self.config.condition_dim) for value in smiles]).astype(np.float32)
        device = _resolve_device(torch, self.config.device)
        self.device_name = str(device)

        model = _TokenDenoisingTransformer(
            vocab_size=len(self.vocab),
            condition_dim=self.config.condition_dim,
            hidden_dim=self.config.hidden_dim,
            max_length=self.config.max_length,
            layers=self.config.transformer_layers,
            heads=self.config.attention_heads,
            dropout=self.config.dropout,
            nn=nn,
        ).to(device)
        dataset = TensorDataset(torch.from_numpy(tokens), torch.from_numpy(conditions))
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        model.train()
        history: list[dict[str, float | str]] = []
        for epoch in range(1, self.config.epochs + 1):
            total_loss = 0.0
            total_tokens = 0
            for batch_tokens, batch_conditions in loader:
                batch_tokens = batch_tokens.to(device)
                batch_conditions = batch_conditions.to(device)
                if self.config.objective == "denoising":
                    noise_level = torch.empty((batch_tokens.shape[0], 1), device=device).uniform_(self.config.noise_min, self.config.noise_max)
                    corrupted = _corrupt_tokens(torch, batch_tokens, noise_level, self.vocab)
                    logits = model(corrupted, batch_conditions, noise_level)
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, len(self.vocab)),
                        batch_tokens.reshape(-1),
                        ignore_index=self.vocab.pad_id,
                    )
                elif self.config.objective == "autoregressive":
                    noise_level = torch.zeros((batch_tokens.shape[0], 1), device=device)
                    logits = model(batch_tokens, batch_conditions, noise_level, causal=True)
                    loss = torch.nn.functional.cross_entropy(
                        logits[:, :-1, :].reshape(-1, len(self.vocab)),
                        batch_tokens[:, 1:].reshape(-1),
                        ignore_index=self.vocab.pad_id,
                    )
                else:
                    raise ValueError(f"Unknown oracle latent decoder objective: {self.config.objective}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                valid_tokens = int((batch_tokens != self.vocab.pad_id).sum().detach().cpu())
                total_loss += float(loss.detach().cpu()) * valid_tokens
                total_tokens += valid_tokens
            history.append(
                {
                    "epoch": float(epoch),
                    "token_loss": total_loss / max(1, total_tokens),
                    "backend": self.backend_name,
                    "device": self.device_name,
                }
            )

        self.model = model
        self.history = history
        return self

    def decode(self, condition_latents: np.ndarray, top_k: int | None = None) -> list[list[Candidate]]:
        if self.model is None or self.vocab is None:
            raise RuntimeError("OracleLatentSmilesDiffusion must be fit before decode().")
        top_k = int(top_k or self.config.samples_per_condition)
        sample_count = max(top_k, int(self.config.samples_per_condition) * max(1, int(self.config.sample_multiplier)))
        condition_latents = np.asarray(condition_latents, dtype=np.float32)
        repeated = np.repeat(condition_latents, sample_count, axis=0)
        generated: list[tuple[str, float]] = []
        torch, _, _, _ = _torch_deps()
        _set_torch_seed(torch, self.config.seed)
        for start in range(0, len(repeated), max(1, int(self.config.sample_batch_size))):
            batch = repeated[start : start + int(self.config.sample_batch_size)]
            generated.extend(self._sample_batch(batch))

        out: list[list[Candidate]] = []
        cursor = 0
        for _ in range(len(condition_latents)):
            raw_group = generated[cursor : cursor + sample_count]
            cursor += sample_count
            out.append(_rank_generated_group(raw_group, top_k=top_k, validity_bonus=self.config.validity_bonus))
        return out

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(asdict(self.config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (out_dir / "metadata.json").write_text(
            json.dumps({"model_type": self.backend_name, "device": self.device_name, "history": self.history}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if self.vocab is not None:
            (out_dir / "vocab.json").write_text(json.dumps(self.vocab.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if self.model is not None:
            torch, _, _, _ = _torch_deps()
            torch.save(self.model.state_dict(), out_dir / "model.pt")

    @classmethod
    def load(cls, out_dir: str | Path, device: str = "auto") -> "OracleLatentSmilesDiffusion":
        out_dir = Path(out_dir)
        config_payload = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
        valid_fields = {field.name for field in fields(OracleLatentDiffusionConfig)}
        config = OracleLatentDiffusionConfig(**{key: value for key, value in config_payload.items() if key in valid_fields})
        config.device = device
        vocab = SmilesVocabulary.from_dict(json.loads((out_dir / "vocab.json").read_text(encoding="utf-8")))
        torch, nn, _, _ = _torch_deps()
        target_device = _resolve_device(torch, device)
        model = _TokenDenoisingTransformer(
            vocab_size=len(vocab),
            condition_dim=config.condition_dim,
            hidden_dim=config.hidden_dim,
            max_length=config.max_length,
            layers=config.transformer_layers,
            heads=config.attention_heads,
            dropout=config.dropout,
            nn=nn,
        ).to(target_device)
        model.load_state_dict(torch.load(out_dir / "model.pt", map_location=target_device))
        model.eval()
        obj = cls(config)
        obj.vocab = vocab
        obj.model = model
        obj.device_name = str(target_device)
        metadata_path = out_dir / "metadata.json"
        if metadata_path.exists():
            obj.history = list(json.loads(metadata_path.read_text(encoding="utf-8")).get("history", []))
        return obj

    def _sample_batch(self, condition_latents: np.ndarray) -> list[tuple[str, float]]:
        if self.model is None or self.vocab is None:
            raise RuntimeError("OracleLatentSmilesDiffusion must be fit before sampling.")
        torch, _, _, _ = _torch_deps()
        device = torch.device(self.device_name)
        model = self.model.to(device)
        model.eval()
        conditions = torch.from_numpy(np.asarray(condition_latents, dtype=np.float32)).to(device)
        if self.config.objective == "autoregressive":
            return self._sample_batch_autoregressive(torch, model, conditions)
        return self._sample_batch_denoising(torch, model, conditions)

    def _sample_batch_autoregressive(self, torch: Any, model: Any, conditions: Any) -> list[tuple[str, float]]:
        if self.vocab is None:
            raise RuntimeError("OracleLatentSmilesDiffusion must be fit before sampling.")
        tokens = torch.full((conditions.shape[0], self.config.max_length), self.vocab.pad_id, dtype=torch.long, device=conditions.device)
        tokens[:, 0] = self.vocab.bos_id
        final_log_probs = torch.zeros((conditions.shape[0], self.config.max_length), dtype=torch.float32, device=conditions.device)
        done = torch.zeros((conditions.shape[0],), dtype=torch.bool, device=conditions.device)
        noise_level = torch.zeros((conditions.shape[0], 1), device=conditions.device)

        with torch.no_grad():
            for pos in range(1, self.config.max_length):
                logits = model(tokens, conditions, noise_level, causal=True)[:, pos - 1, :] / max(float(self.config.temperature), 1e-6)
                logits = _mask_special_logits(logits, self.vocab)
                if pos == 1:
                    logits[:, self.vocab.eos_id] = -1e9
                logits = _top_p_filter(torch, logits, self.config.top_p)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1).squeeze(1)
                selected = torch.gather(torch.log(torch.clamp(probs, min=1e-12)), 1, sampled.unsqueeze(1)).squeeze(1)
                sampled = torch.where(done, torch.full_like(sampled, self.vocab.pad_id), sampled)
                selected = torch.where(done, torch.zeros_like(selected), selected)
                tokens[:, pos] = sampled
                final_log_probs[:, pos] = selected
                done = done | (sampled == self.vocab.eos_id)
                if bool(torch.all(done).detach().cpu()):
                    break
        return _decode_sampled_rows(tokens, final_log_probs, self.vocab)

    def _sample_batch_denoising(self, torch: Any, model: Any, conditions: Any) -> list[tuple[str, float]]:
        if self.vocab is None:
            raise RuntimeError("OracleLatentSmilesDiffusion must be fit before sampling.")
        tokens = torch.full((conditions.shape[0], self.config.max_length), self.vocab.mask_id, dtype=torch.long, device=conditions.device)
        tokens[:, 0] = self.vocab.bos_id
        final_log_probs = torch.zeros((conditions.shape[0], self.config.max_length), dtype=torch.float32, device=conditions.device)

        with torch.no_grad():
            for step in range(max(1, int(self.config.sample_steps)), 0, -1):
                noise_level = torch.full((conditions.shape[0], 1), step / max(1, int(self.config.sample_steps)), device=conditions.device)
                logits = model(tokens, conditions, noise_level) / max(float(self.config.temperature), 1e-6)
                logits[:, :, self.vocab.pad_id] = -1e9
                logits[:, :, self.vocab.bos_id] = -1e9
                logits[:, :, self.vocab.mask_id] = -1e9
                logits[:, :, self.vocab.unk_id] = -1e9
                probs = torch.nn.functional.softmax(logits, dim=-1)
                sampled = torch.multinomial(probs.reshape(-1, probs.shape[-1]), num_samples=1).reshape(tokens.shape)
                selected = torch.gather(torch.log(torch.clamp(probs, min=1e-12)), 2, sampled.unsqueeze(-1)).squeeze(-1)
                tokens[:, 1:] = sampled[:, 1:]
                final_log_probs[:, 1:] = selected[:, 1:]
            tokens[:, 0] = self.vocab.bos_id

        return _decode_sampled_rows(tokens, final_log_probs, self.vocab)


def run_oracle_latent_diffusion(
    molecule_csv: str | Path = "data/example_molecules.csv",
    output_dir: str | Path = "outputs/runs/phase1_oracle_latent_diffusion",
    smiles_column: str | None = None,
    limit: int | None = None,
    config: OracleLatentDiffusionConfig | None = None,
) -> dict[str, float]:
    config = config or OracleLatentDiffusionConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    molecules = load_molecule_rows(molecule_csv, smiles_column=smiles_column, limit=limit)
    smiles = _canonical_smiles_list([row.smiles for row in molecules], config.max_length)
    train_smiles, eval_smiles = _split_smiles(smiles, train_fraction=config.train_fraction, seed=config.seed)
    if not train_smiles or not eval_smiles:
        raise ValueError("Need at least one train and one eval molecule after filtering.")

    model = OracleLatentSmilesDiffusion(config).fit(train_smiles)
    eval_latents = np.stack([molecule_latent(value, config.condition_dim) for value in eval_smiles]).astype(np.float32)
    decoded = model.decode(eval_latents, top_k=config.samples_per_condition)
    examples = [
        BenchmarkExample(
            task_id=f"oracle_latent_{idx}",
            task_type=TaskType.DE_NOVO,
            target_smiles=smiles_value,
            instruction="Decode the oracle target molecular latent into a valid molecule.",
        )
        for idx, smiles_value in enumerate(eval_smiles)
    ]
    scores_by_task = [score_candidates(example, candidates) for example, candidates in zip(examples, decoded)]
    metrics = summarize_scores(scores_by_task)
    metrics.update(_oracle_surface_metrics(scores_by_task, train_pool=set(train_smiles), targets=eval_smiles))
    metrics.update(
        {
            "train_molecules": float(len(train_smiles)),
            "eval_molecules": float(len(eval_smiles)),
            "molecules_loaded": float(len(molecules)),
            "molecules_after_length_filter": float(len(smiles)),
            "vocab_size": float(len(model.vocab or [])),
            "max_length": float(config.max_length),
        }
    )

    run_config = {
        "phase": "phase1_oracle_latent_smiles_diffusion",
        "research_question": "Can a decoder generate molecules from an oracle molecular latent before adding a JEPA planner?",
        "molecule_csv": str(molecule_csv),
        "smiles_column": smiles_column,
        "limit": limit,
        "molecule_latent_version": MOLECULE_LATENT_VERSION,
        "decoder": "latent_conditioned_autoregressive_token_decoder" if config.objective == "autoregressive" else "latent_conditioned_token_denoising_diffusion",
        "condition_source": "oracle_target_molecule_latent",
        "ranking": "model_confidence_plus_rdkit_validity_no_target_oracle",
        "config": asdict(config),
        "history": model.history,
    }
    model.save(output_dir / "model")
    _write_smiles_csv(output_dir / "train_smiles.csv", train_smiles)
    _write_smiles_csv(output_dir / "eval_smiles.csv", eval_smiles)
    _write_predictions(output_dir / "predictions.csv", examples, scores_by_task, train_pool=set(train_smiles), targets=eval_smiles)
    summarize_predictions_csv(output_dir / "predictions.csv", out_dir=output_dir)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 oracle latent-conditioned SMILES diffusion.")
    parser.add_argument("--molecule-csv", default="data/example_molecules.csv")
    parser.add_argument("--output-dir", default="outputs/runs/phase1_oracle_latent_diffusion")
    parser.add_argument("--smiles-column", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--condition-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--objective", choices=["autoregressive", "denoising"], default="autoregressive")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--noise-min", type=float, default=0.15)
    parser.add_argument("--noise-max", type=float, default=0.85)
    parser.add_argument("--sample-steps", type=int, default=16)
    parser.add_argument("--samples-per-condition", type=int, default=8)
    parser.add_argument("--sample-multiplier", type=int, default=4)
    parser.add_argument("--sample-batch-size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--validity-bonus", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    config = OracleLatentDiffusionConfig(
        condition_dim=args.condition_dim,
        hidden_dim=args.hidden_dim,
        transformer_layers=args.transformer_layers,
        attention_heads=args.attention_heads,
        objective=args.objective,
        dropout=args.dropout,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        noise_min=args.noise_min,
        noise_max=args.noise_max,
        sample_steps=args.sample_steps,
        samples_per_condition=args.samples_per_condition,
        sample_multiplier=args.sample_multiplier,
        sample_batch_size=args.sample_batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        validity_bonus=args.validity_bonus,
        train_fraction=args.train_fraction,
        device=args.device,
        seed=args.seed,
    )
    metrics = run_oracle_latent_diffusion(
        molecule_csv=args.molecule_csv,
        output_dir=args.output_dir,
        smiles_column=args.smiles_column,
        limit=args.limit,
        config=config,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _TokenDenoisingTransformer(
    vocab_size: int,
    condition_dim: int,
    hidden_dim: int,
    max_length: int,
    layers: int,
    heads: int,
    dropout: float,
    nn: Any,
):
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
            self.position_embedding = nn.Embedding(max_length, hidden_dim)
            self.condition_projection = nn.Sequential(nn.Linear(condition_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
            self.noise_projection = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.output = nn.Linear(hidden_dim, vocab_size)

        def forward(self, input_ids, conditions, noise_level, causal: bool = False):
            import torch

            positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
            x = self.token_embedding(input_ids) + self.position_embedding(positions)
            x = x + self.condition_projection(conditions).unsqueeze(1) + self.noise_projection(noise_level).unsqueeze(1)
            attn_mask = None
            if causal:
                length = int(input_ids.shape[1])
                attn_mask = torch.full((length, length), float("-inf"), device=input_ids.device)
                attn_mask = torch.triu(attn_mask, diagonal=1)
            return self.output(self.encoder(x, mask=attn_mask))

    return Model()


def _corrupt_tokens(torch: Any, tokens: Any, noise_level: Any, vocab: SmilesVocabulary):
    corrupted = tokens.clone()
    valid = (tokens != vocab.pad_id) & (tokens != vocab.bos_id)
    noise = torch.rand(tokens.shape, device=tokens.device)
    corrupt_mask = valid & (noise < noise_level)
    corrupted[corrupt_mask] = vocab.mask_id
    if len(vocab) > vocab.first_char_id:
        random_mask = corrupt_mask & (torch.rand(tokens.shape, device=tokens.device) < 0.15)
        random_tokens = torch.randint(vocab.first_char_id, len(vocab), tokens.shape, device=tokens.device)
        corrupted[random_mask] = random_tokens[random_mask]
    return corrupted


def _rank_generated_group(raw_group: list[tuple[str, float]], top_k: int, validity_bonus: float) -> list[Candidate]:
    best_by_key: dict[str, Candidate] = {}
    for raw_smiles, log_score in raw_group:
        clean = raw_smiles.strip()
        canonical = canonicalize_smiles(clean)
        valid = canonical is not None
        smiles = canonical if canonical is not None else clean
        if not smiles:
            smiles = clean or "<empty>"
        key = canonical or f"invalid:{smiles}"
        score = float(log_score) + (float(validity_bonus) if valid else 0.0)
        candidate = Candidate(smiles=smiles, origin=OracleLatentSmilesDiffusion.backend_name, score=score, rank=0)
        if key not in best_by_key or candidate.score > best_by_key[key].score:
            best_by_key[key] = candidate
    ranked = sorted(best_by_key.values(), key=lambda item: item.score, reverse=True)[:top_k]
    return [Candidate(smiles=item.smiles, origin=item.origin, score=item.score, rank=idx) for idx, item in enumerate(ranked, start=1)]


def _decode_sampled_rows(tokens: Any, log_probs: Any, vocab: SmilesVocabulary) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    token_rows = tokens.detach().cpu().numpy()
    score_rows = log_probs.detach().cpu().numpy()
    for token_row, score_row in zip(token_rows, score_rows):
        token_ids = [int(item) for item in token_row]
        smiles = vocab.decode(token_ids)
        score = _sequence_score(token_ids, [float(item) for item in score_row], vocab.eos_id)
        rows.append((smiles, score))
    return rows


def _mask_special_logits(logits: Any, vocab: SmilesVocabulary):
    logits = logits.clone()
    for token_id in (vocab.pad_id, vocab.bos_id, vocab.mask_id, vocab.unk_id):
        logits[:, token_id] = -1e9
    return logits


def _top_p_filter(torch: Any, logits: Any, top_p: float):
    top_p = float(top_p)
    if top_p <= 0.0 or top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    remove_sorted = cumulative_probs > top_p
    remove_sorted[:, 1:] = remove_sorted[:, :-1].clone()
    remove_sorted[:, 0] = False
    remove = torch.zeros_like(remove_sorted).scatter(1, sorted_indices, remove_sorted)
    return logits.masked_fill(remove, -1e9)


def _sequence_score(token_ids: list[int], log_probs: list[float], eos_id: int) -> float:
    values: list[float] = []
    for token_id, score in zip(token_ids[1:], log_probs[1:]):
        values.append(float(score))
        if int(token_id) == eos_id:
            break
    return float(sum(values) / max(1, len(values)))


def _tokenize_smiles(smiles: str) -> list[str]:
    tokens: list[str] = []
    idx = 0
    while idx < len(smiles):
        if smiles[idx] == "[":
            end = smiles.find("]", idx + 1)
            if end != -1:
                tokens.append(smiles[idx : end + 1])
                idx = end + 1
                continue
        two = smiles[idx : idx + 2]
        if two in {"Cl", "Br"}:
            tokens.append(two)
            idx += 2
            continue
        tokens.append(smiles[idx])
        idx += 1
    return tokens


def _canonical_smiles_list(smiles: list[str], max_length: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in smiles:
        can = canonicalize_smiles(value)
        if not can or can in seen:
            continue
        if len(_tokenize_smiles(can)) + 2 > max_length:
            continue
        seen.add(can)
        out.append(can)
    return out


def _split_smiles(smiles: list[str], train_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(smiles)).tolist()
    train_fraction = min(0.95, max(0.05, float(train_fraction)))
    train_size = int(round(len(indices) * train_fraction))
    train_size = min(max(1, train_size), len(indices) - 1)
    train_idx = set(indices[:train_size])
    train = [value for idx, value in enumerate(smiles) if idx in train_idx]
    eval_smiles = [value for idx, value in enumerate(smiles) if idx not in train_idx]
    return train, eval_smiles


def _oracle_surface_metrics(scores_by_task: list[list[CandidateScore]], train_pool: set[str], targets: list[str]) -> dict[str, float]:
    n = max(1, len(scores_by_task))
    target_canonical = [canonicalize_smiles(value) or value for value in targets]
    top1_exact = 0.0
    topk_exact = 0.0
    top1_train_member = 0.0
    candidate_train_member = 0.0
    candidate_count = 0.0
    for target, scores in zip(target_canonical, scores_by_task):
        if not scores:
            continue
        top_smiles = canonicalize_smiles(scores[0].smiles) or scores[0].smiles
        top1_exact += 1.0 if top_smiles == target else 0.0
        topk_exact += 1.0 if any((canonicalize_smiles(score.smiles) or score.smiles) == target for score in scores) else 0.0
        top1_train_member += 1.0 if top_smiles in train_pool else 0.0
        for score in scores:
            smiles = canonicalize_smiles(score.smiles) or score.smiles
            candidate_train_member += 1.0 if smiles in train_pool else 0.0
            candidate_count += 1.0
    return {
        "top1_exact_match": top1_exact / n,
        "topk_exact_match": topk_exact / n,
        "top1_train_pool_member": top1_train_member / n,
        "candidate_train_pool_member_fraction": candidate_train_member / max(1.0, candidate_count),
        "mean_candidate_count": candidate_count / n,
    }


def _write_smiles_csv(path: Path, smiles: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["smiles"])
        writer.writeheader()
        for value in smiles:
            writer.writerow({"smiles": value})


def _write_predictions(
    path: Path,
    examples: list[BenchmarkExample],
    scores_by_task: list[list[CandidateScore]],
    train_pool: set[str],
    targets: list[str],
) -> None:
    fieldnames = [
        "task_id",
        "task_type",
        "instruction",
        "source_smiles",
        "target_smiles",
        "rank",
        "candidate_smiles",
        "origin",
        "valid",
        "target_tanimoto",
        "scaffold_match",
        "score",
        "property_mae",
        "property_success",
        "exact_match",
        "train_pool_member",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for example, target, scores in zip(examples, targets, scores_by_task):
            target_can = canonicalize_smiles(target) or target
            for score in scores:
                smiles = canonicalize_smiles(score.smiles) or score.smiles
                writer.writerow(
                    {
                        "task_id": example.task_id,
                        "task_type": "oracle_latent_decode",
                        "instruction": example.instruction,
                        "source_smiles": "",
                        "target_smiles": example.target_smiles,
                        "rank": score.rank,
                        "candidate_smiles": score.smiles,
                        "origin": score.origin,
                        "valid": score.valid,
                        "target_tanimoto": f"{score.target_tanimoto:.6f}",
                        "scaffold_match": score.scaffold_match,
                        "score": f"{score.score:.6f}",
                        "property_mae": f"{score.property_mae:.6f}",
                        "property_success": score.property_success,
                        "exact_match": smiles == target_can,
                        "train_pool_member": smiles in train_pool,
                    }
                )


def _resolve_device(torch: Any, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def _set_torch_seed(torch: Any, seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _torch_deps():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:  # pragma: no cover - optional install.
        raise RuntimeError(
            "Phase 1 oracle latent diffusion requires PyTorch. Use SKETCHIMAGE_PYTHON_BIN "
            "with torch installed, for example /scratch/bdong/venvs/phystabmol/bin/python."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


if __name__ == "__main__":
    main()
