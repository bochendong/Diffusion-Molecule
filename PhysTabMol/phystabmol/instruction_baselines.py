"""Baseline candidate generators for verified molecular editing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .chem import canonicalize_smiles
from .instruction_verifier import score_verification, verify_instruction


def main() -> None:
    args = parse_args()
    dataset = pd.read_csv(args.dataset)
    if args.split != "all" and "split" in dataset.columns:
        dataset = dataset[dataset["split"] == args.split].reset_index(drop=True)
    if args.limit:
        dataset = dataset.head(args.limit).reset_index(drop=True)
    candidates = generate_baseline_candidates(
        dataset,
        baseline=args.baseline,
        samples_per_instruction=args.samples_per_instruction,
        retrieval_pool_size=args.retrieval_pool_size,
        seed=args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out, index=False)
    print(f"Wrote {len(candidates)} {args.baseline} candidates -> {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate instruction-editing baseline candidates.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="outputs/instruction_baseline_candidates.csv")
    parser.add_argument("--baseline", choices=["no_edit", "oracle_target", "random_target", "rule_retrieval"], default="no_edit")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test", "all"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples-per-instruction", type=int, default=1)
    parser.add_argument("--retrieval-pool-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def generate_baseline_candidates(
    dataset: pd.DataFrame,
    baseline: str,
    samples_per_instruction: int = 1,
    retrieval_pool_size: int = 512,
    seed: int = 7,
) -> pd.DataFrame:
    required = {"source_smiles", "target_smiles", "instruction_spec_json"}
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    rng = np.random.default_rng(seed)
    molecule_pool = _molecule_pool(dataset)
    rows = []
    for instruction_idx, row in dataset.iterrows():
        for sample_idx in range(samples_per_instruction):
            if baseline == "no_edit":
                candidate = row["source_smiles"]
            elif baseline == "oracle_target":
                candidate = row["target_smiles"]
            elif baseline == "random_target":
                candidate = _random_candidate(row["source_smiles"], molecule_pool, rng)
            elif baseline == "rule_retrieval":
                candidate = _rule_retrieval_candidate(row, molecule_pool, rng, retrieval_pool_size)
            else:
                raise ValueError(f"Unsupported baseline: {baseline}")
            out = {
                "instruction_id": int(instruction_idx),
                "sample_idx": int(sample_idx),
                "baseline": baseline,
                "candidate_smiles": candidate,
            }
            for col in [
                "pair_id",
                "source_smiles",
                "target_smiles",
                "reference_smiles",
                "reference_role",
                "instruction_text",
                "instruction_spec_json",
                "edit_tags",
                "split",
                "template_id",
            ]:
                if col in row:
                    out[col] = row[col]
            rows.append(out)
    return pd.DataFrame(rows)


def _molecule_pool(dataset: pd.DataFrame) -> list[str]:
    pool = []
    for col in ("target_smiles", "source_smiles"):
        pool.extend(dataset[col].dropna().astype(str).tolist())
    unique = []
    seen = set()
    for smiles in pool:
        can = canonicalize_smiles(smiles)
        if can and can not in seen:
            unique.append(can)
            seen.add(can)
    if not unique:
        raise ValueError("No valid molecules available for baseline pool.")
    return unique


def _random_candidate(source_smiles: str, pool: list[str], rng: np.random.Generator) -> str:
    if len(pool) == 1:
        return pool[0]
    for _ in range(20):
        candidate = pool[int(rng.integers(0, len(pool)))]
        if candidate != source_smiles:
            return candidate
    return pool[0]


def _rule_retrieval_candidate(row, pool: list[str], rng: np.random.Generator, retrieval_pool_size: int) -> str:
    if len(pool) <= retrieval_pool_size:
        candidates = pool
    else:
        candidates = [pool[idx] for idx in rng.choice(len(pool), size=retrieval_pool_size, replace=False)]
    best_smiles = row["source_smiles"]
    best_score = -1.0
    for smiles in candidates:
        if smiles == row["source_smiles"]:
            continue
        result = verify_instruction(row["source_smiles"], smiles, row["instruction_spec_json"])
        score = score_verification(result)
        if result.get("overall_success"):
            score += 100.0
        if score > best_score:
            best_score = score
            best_smiles = smiles
    return best_smiles


if __name__ == "__main__":
    main()
