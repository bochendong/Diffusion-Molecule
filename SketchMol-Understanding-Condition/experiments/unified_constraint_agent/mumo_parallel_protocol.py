#!/usr/bin/env python3
"""Shared contracts for the parallel MuMO train-only evidence pipeline.

The module deliberately contains no evaluation-oracle client.  It is used by
CPU jobs that prepare train shards, extract matched-pair deltas, and train
property predictors from labels already stored in MuMO's training split.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator, Mapping, Sequence


PROPERTIES = ("bbbp", "drd2", "hia", "mutagenicity", "plogp")
TASK_IDS = ("BDP", "BDQ", "BPQ", "DPQ", "BDPQ", "MPQ", "BDMQ", "BHMQ", "BMPQ", "HMPQ")
IND_TASK_IDS = frozenset(("BDP", "BDQ", "BPQ", "DPQ", "BDPQ"))
OOD_TASK_IDS = frozenset(("MPQ", "BDMQ", "BHMQ", "BMPQ", "HMPQ"))
PROTOCOL_VERSION = "mumo_parallel_train_evidence_v8"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_fraction(value: str, *, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def stable_shard(value: str, *, seed: int, shard_count: int) -> int:
    if int(shard_count) <= 0:
        raise ValueError("shard_count must be positive")
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % int(shard_count)


def iter_json_array(path: Path, *, chunk_size: int = 1024 * 1024) -> Iterator[object]:
    """Stream one top-level JSON array without holding the file in memory."""

    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer = ""
        position = 0
        eof = False

        def refill() -> None:
            nonlocal buffer, position, eof
            if position:
                buffer = buffer[position:]
                position = 0
            chunk = handle.read(int(chunk_size))
            if chunk:
                buffer += chunk
            else:
                eof = True

        refill()
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer):
                break
            if eof:
                raise ValueError(f"Empty JSON file: {path}")
            refill()
        if buffer[position] != "[":
            raise ValueError(f"Expected a top-level JSON array in {path}")
        position += 1

        while True:
            while True:
                while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
                    position += 1
                if position < len(buffer) or eof:
                    break
                refill()
            if position < len(buffer) and buffer[position] == "]":
                position += 1
                break
            if eof and position >= len(buffer):
                raise ValueError(f"Unterminated JSON array in {path}")
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    refill()
                    continue
                position = end
                yield value
                break

        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer):
                raise ValueError(f"Trailing content after JSON array in {path}")
            if eof:
                break
            refill()


def first_value(row: Mapping[str, object], keys: Sequence[str]) -> object | None:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def raw_source_smiles(row: Mapping[str, object]) -> str:
    return str(first_value(row, ("source_smiles", "input_smiles", "input", "source", "src_smiles")) or "").strip()


def raw_target_smiles(row: Mapping[str, object]) -> str:
    return str(first_value(row, ("target_smiles", "output_smiles", "output", "target", "tgt_smiles")) or "").strip()


def normalize_task_token(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "").replace("_", "+").replace("-", "+")


def task_aliases(task_id: str, task_key: str) -> set[str]:
    return {
        normalize_task_token(task_id),
        normalize_task_token(task_key),
        normalize_task_token(f"mumo:{task_id}"),
        normalize_task_token(f"mumo:{task_key}"),
    }


def canonical_pair_key(task_id: str, source_smiles: str, target_smiles: str) -> str:
    return json.dumps(
        [str(task_id), str(source_smiles), str(target_smiles)],
        separators=(",", ":"),
        ensure_ascii=True,
    )


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                value = json.loads(text)
                if not isinstance(value, Mapping):
                    raise ValueError(f"Expected object rows in {path}")
                rows.append(dict(value))
    return rows
