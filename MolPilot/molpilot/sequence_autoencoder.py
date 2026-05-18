"""SELFIES/SMILES sequence molecular autoencoder.

This is the first real molecular decoder for MolPilot. It is still intentionally
compact, but unlike the nearest-latent baseline it learns to decode a latent
vector back into a molecular sequence. If the optional `selfies` package is
available, the sequence representation is SELFIES; otherwise it falls back to a
SMILES tokenizer so the project remains runnable in light environments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np

from .artifacts import ensure_dir, load_json, load_lines, save_json, save_lines
from .chem import canonicalize_smiles
from .features import molecule_feature_vector

try:  # pragma: no cover - optional dependency.
    import selfies as sf

    SELFIES_AVAILABLE = True
except Exception:  # pragma: no cover
    sf = None
    SELFIES_AVAILABLE = False

try:  # pragma: no cover - server path.
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False


PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIAL_TOKENS = (PAD, BOS, EOS, UNK)


@dataclass
class SequenceAutoencoderConfig:
    representation: str = "auto"  # auto, selfies, smiles
    feature_dim: int = 256
    latent_dim: int = 64
    embedding_dim: int = 128
    hidden_dim: int = 512
    layers: int = 1
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    max_length: int = 96
    seed: int = 7


if TORCH_AVAILABLE:  # pragma: no cover - exercised on GPU server.

    class _SeqAutoencoder(nn.Module):
        def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, latent_dim: int, layers: int, pad_idx: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
            self.encoder = nn.GRU(
                embedding_dim,
                hidden_dim,
                num_layers=layers,
                batch_first=True,
                dropout=0.0 if layers == 1 else 0.1,
            )
            self.to_latent = nn.Linear(hidden_dim, latent_dim)
            self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * layers)
            self.decoder = nn.GRU(
                embedding_dim,
                hidden_dim,
                num_layers=layers,
                batch_first=True,
                dropout=0.0 if layers == 1 else 0.1,
            )
            self.output = nn.Linear(hidden_dim, vocab_size)
            self.layers = layers
            self.hidden_dim = hidden_dim

        def encode_ids(self, ids):
            emb = self.embedding(ids)
            _, hidden = self.encoder(emb)
            return self.to_latent(hidden[-1])

        def decode_teacher_forced(self, decoder_input_ids, latent):
            emb = self.embedding(decoder_input_ids)
            hidden = self.latent_to_hidden(latent).view(self.layers, latent.shape[0], self.hidden_dim).contiguous()
            out, _ = self.decoder(emb, hidden)
            return self.output(out)

        def forward(self, encoder_ids, decoder_input_ids):
            latent = self.encode_ids(encoder_ids)
            logits = self.decode_teacher_forced(decoder_input_ids, latent)
            return logits, latent


class MolecularSequenceAutoencoder:
    def __init__(self, config: SequenceAutoencoderConfig | None = None):
        self.config = config or SequenceAutoencoderConfig()
        self.representation_: str = _resolve_representation(self.config.representation)
        self.token_to_id: dict[str, int] = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
        self.id_to_token: list[str] = list(SPECIAL_TOKENS)
        self.train_smiles: list[str] = []
        self.train_latents: np.ndarray | None = None
        self.history: list[dict[str, float | str]] = []
        self.model = None

    def fit(self, smiles: list[str]) -> "MolecularSequenceAutoencoder":
        self.train_smiles = [canonicalize_smiles(smi) or str(smi) for smi in smiles]
        tokenized = [self._tokens(smi) for smi in self.train_smiles]
        self._fit_vocab(tokenized)
        if TORCH_AVAILABLE:
            self._fit_torch(tokenized)
        else:
            self.history = [{"epoch": 0.0, "loss": 0.0, "backend": "sequence_nearest_no_torch"}]
        self.train_latents = self.encode_many(self.train_smiles)
        return self

    def encode(self, smiles: str | None) -> np.ndarray:
        return self.encode_many([smiles or ""])[0]

    def encode_many(self, smiles: list[str]) -> np.ndarray:
        if TORCH_AVAILABLE and self.model is not None:
            encoded = [self._encode_ids(self._tokens(smi), include_bos_eos=True) for smi in smiles]
            ids = _pad_sequences(encoded, self.pad_id, self.config.max_length + 2)
            self.model.eval()
            chunks = []
            with torch.no_grad():
                device = next(self.model.parameters()).device
                for start in range(0, len(ids), max(1, self.config.batch_size)):
                    batch = torch.tensor(ids[start : start + self.config.batch_size], dtype=torch.long, device=device)
                    chunks.append(self.model.encode_ids(batch).cpu().numpy())
            return np.concatenate(chunks, axis=0).astype(np.float32)
        # Fallback keeps the staged pipeline runnable without torch. It is not
        # the paper decoder.
        return np.asarray([molecule_feature_vector(smi, self.config.latent_dim) for smi in smiles], dtype=np.float32)

    def decode(self, latent: np.ndarray, top_k: int = 8) -> list[str]:
        out: list[str] = []
        latent = np.asarray(latent, dtype=np.float32)
        if TORCH_AVAILABLE and self.model is not None:
            decoded = self._decode_greedy(latent)
            if decoded:
                out.append(decoded)
        out.extend(self._nearest_smiles(latent, top_k=max(1, top_k - len(out))))
        return _dedupe(out)[:top_k]

    def save(self, out_dir: str | Path) -> None:
        out_dir = ensure_dir(out_dir)
        save_json(asdict(self.config), out_dir / "config.json")
        save_json(
            {
                "codec_type": "sequence",
                "history": self.history,
                "backend": self.backend,
                "representation": self.representation_,
                "token_to_id": self.token_to_id,
            },
            out_dir / "metadata.json",
        )
        save_lines(self.train_smiles, out_dir / "train_smiles.txt")
        if self.train_latents is not None:
            np.save(out_dir / "train_latents.npy", self.train_latents)
        if TORCH_AVAILABLE and self.model is not None:
            torch.save(self.model.state_dict(), out_dir / "sequence_autoencoder.pt")

    @classmethod
    def load(cls, out_dir: str | Path) -> "MolecularSequenceAutoencoder":
        out_dir = Path(out_dir)
        cfg = SequenceAutoencoderConfig(**load_json(out_dir / "config.json"))
        obj = cls(cfg)
        meta = load_json(out_dir / "metadata.json") if (out_dir / "metadata.json").exists() else {}
        obj.representation_ = str(meta.get("representation", _resolve_representation(cfg.representation)))
        obj.token_to_id = {str(k): int(v) for k, v in dict(meta.get("token_to_id", obj.token_to_id)).items()}
        obj.id_to_token = [PAD] * len(obj.token_to_id)
        for token, idx in obj.token_to_id.items():
            if idx >= len(obj.id_to_token):
                obj.id_to_token.extend([PAD] * (idx - len(obj.id_to_token) + 1))
            obj.id_to_token[idx] = token
        obj.train_smiles = load_lines(out_dir / "train_smiles.txt") if (out_dir / "train_smiles.txt").exists() else []
        obj.train_latents = np.load(out_dir / "train_latents.npy") if (out_dir / "train_latents.npy").exists() else None
        if TORCH_AVAILABLE and (out_dir / "sequence_autoencoder.pt").exists():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            obj.model = _SeqAutoencoder(
                vocab_size=len(obj.id_to_token),
                embedding_dim=cfg.embedding_dim,
                hidden_dim=cfg.hidden_dim,
                latent_dim=cfg.latent_dim,
                layers=cfg.layers,
                pad_idx=obj.pad_id,
            )
            obj.model.load_state_dict(torch.load(out_dir / "sequence_autoencoder.pt", map_location="cpu"))
            obj.model.to(device)
            obj.model.eval()
        obj.history = list(meta.get("history", []))
        return obj

    @property
    def backend(self) -> str:
        if TORCH_AVAILABLE and self.model is not None:
            return f"torch_{self.representation_}_sequence_autoencoder"
        return f"{self.representation_}_tokenizer_nearest_no_torch"

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK]

    def _fit_vocab(self, tokenized: list[list[str]]) -> None:
        for tokens in tokenized:
            for token in tokens:
                if token not in self.token_to_id:
                    self.token_to_id[token] = len(self.id_to_token)
                    self.id_to_token.append(token)

    def _fit_torch(self, tokenized: list[list[str]]) -> None:
        cfg = self.config
        torch.manual_seed(cfg.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sequences = [self._encode_ids(tokens, include_bos_eos=True) for tokens in tokenized]
        full = _pad_sequences(sequences, self.pad_id, cfg.max_length + 2)
        encoder_ids = torch.tensor(full, dtype=torch.long)
        decoder_in = torch.tensor(full[:, :-1], dtype=torch.long)
        decoder_target = torch.tensor(full[:, 1:], dtype=torch.long)
        self.model = _SeqAutoencoder(
            vocab_size=len(self.id_to_token),
            embedding_dim=cfg.embedding_dim,
            hidden_dim=cfg.hidden_dim,
            latent_dim=cfg.latent_dim,
            layers=cfg.layers,
            pad_idx=self.pad_id,
        ).to(device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_id)
        loader = DataLoader(TensorDataset(encoder_ids, decoder_in, decoder_target), batch_size=cfg.batch_size, shuffle=True)
        self.history = []
        for epoch in range(cfg.epochs):
            losses = []
            accs = []
            for enc, dec_in, target in loader:
                enc = enc.to(device)
                dec_in = dec_in.to(device)
                target = target.to(device)
                logits, latent = self.model(enc, dec_in)
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
                loss = loss + 1e-4 * torch.mean(latent.pow(2))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach()))
                with torch.no_grad():
                    mask = target != self.pad_id
                    if mask.any():
                        pred = logits.argmax(dim=-1)
                        accs.append(float((pred[mask] == target[mask]).float().mean()))
            self.history.append(
                {
                    "epoch": float(epoch + 1),
                    "loss": float(np.mean(losses)),
                    "token_accuracy": float(np.mean(accs)) if accs else 0.0,
                    "backend": "torch",
                }
            )

    def _tokens(self, smiles: str | None) -> list[str]:
        smiles = canonicalize_smiles(smiles) or str(smiles or "")
        if self.representation_ == "selfies" and SELFIES_AVAILABLE:
            try:
                encoded = sf.encoder(smiles)
                return list(sf.split_selfies(encoded))
            except Exception:
                return _smiles_tokens(smiles)
        return _smiles_tokens(smiles)

    def _encode_ids(self, tokens: list[str], include_bos_eos: bool) -> list[int]:
        if include_bos_eos:
            tokens = [BOS] + tokens[: self.config.max_length] + [EOS]
        else:
            tokens = tokens[: self.config.max_length]
        return [self.token_to_id.get(tok, self.unk_id) for tok in tokens]

    def _decode_greedy(self, latent: np.ndarray) -> str | None:
        if self.model is None:
            return None
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            z = torch.tensor(latent[None, :], dtype=torch.float32, device=device)
            hidden = self.model.latent_to_hidden(z).view(self.config.layers, 1, self.config.hidden_dim).contiguous()
            current = torch.tensor([[self.bos_id]], dtype=torch.long, device=device)
            tokens: list[str] = []
            for _ in range(self.config.max_length):
                emb = self.model.embedding(current)
                out, hidden = self.model.decoder(emb, hidden)
                logits = self.model.output(out[:, -1])
                next_id = int(logits.argmax(dim=-1).item())
                if next_id == self.eos_id:
                    break
                if next_id not in {self.pad_id, self.bos_id, self.unk_id}:
                    tokens.append(self.id_to_token[next_id])
                current = torch.tensor([[next_id]], dtype=torch.long, device=device)
        return self._tokens_to_smiles(tokens)

    def _tokens_to_smiles(self, tokens: list[str]) -> str | None:
        if not tokens:
            return None
        if self.representation_ == "selfies" and SELFIES_AVAILABLE and all(tok.startswith("[") for tok in tokens):
            try:
                decoded = sf.decoder("".join(tokens))
                return canonicalize_smiles(decoded) or decoded
            except Exception:
                return None
        decoded = "".join(tokens)
        return canonicalize_smiles(decoded) or decoded

    def _nearest_smiles(self, latent: np.ndarray, top_k: int) -> list[str]:
        if self.train_latents is None or not self.train_smiles or top_k <= 0:
            return []
        distances = np.mean(np.square(self.train_latents - latent[None, :]), axis=1)
        k = min(top_k, len(self.train_smiles))
        chosen = np.argpartition(distances, k - 1)[:k]
        chosen = chosen[np.argsort(distances[chosen])]
        return [self.train_smiles[int(idx)] for idx in chosen]


def _resolve_representation(representation: str) -> str:
    representation = str(representation or "auto").lower()
    if representation == "auto":
        return "selfies" if SELFIES_AVAILABLE else "smiles"
    if representation == "selfies" and not SELFIES_AVAILABLE:
        return "smiles"
    if representation not in {"selfies", "smiles"}:
        raise ValueError(f"Unsupported sequence representation: {representation}")
    return representation


def _smiles_tokens(smiles: str) -> list[str]:
    pattern = r"(\[[^\]]+\]|Br|Cl|Si|Se|Na|Li|Mg|Ca|Al|@@?|%\d{2}|.)"
    return [tok for tok in re.findall(pattern, str(smiles)) if tok]


def _pad_sequences(sequences: list[list[int]], pad_id: int, max_len: int) -> np.ndarray:
    out = np.full((len(sequences), max_len), pad_id, dtype=np.int64)
    for idx, seq in enumerate(sequences):
        clipped = seq[:max_len]
        out[idx, : len(clipped)] = clipped
    return out


def _dedupe(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out
