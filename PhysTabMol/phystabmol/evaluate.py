"""Evaluation helpers for quick PhysTabMol runs."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .chem import molecular_descriptors, passes_druglike_filters, tanimoto
from .schema import TARGET_COLUMNS


def evaluate_smiles(smiles: list[str], train_smiles: list[str] | None = None, target: dict[str, float] | None = None) -> dict[str, float]:
    records = [molecular_descriptors(s) for s in smiles]
    valid_records = [r for r in records if r.valid]
    unique = sorted({r.smiles for r in valid_records})
    train = set(train_smiles or [])
    metrics = {
        "n": float(len(smiles)),
        "validity": len(valid_records) / max(1, len(smiles)),
        "uniqueness": len(unique) / max(1, len(valid_records)),
        "novelty": len([s for s in unique if s not in train]) / max(1, len(unique)),
        "druglike_rate": len([r for r in valid_records if passes_druglike_filters(r.descriptors)]) / max(1, len(valid_records)),
        "mean_pairwise_tanimoto": _mean_pairwise_tanimoto(unique),
    }
    if target:
        for col in TARGET_COLUMNS:
            if col in target:
                metrics[f"{col}_mae"] = float(np.mean([abs(r.descriptors[col] - target[col]) for r in valid_records])) if valid_records else float("nan")
    return metrics


def evaluate_decoded_table(decoded: pd.DataFrame, train_smiles: list[str] | None = None) -> dict[str, float]:
    if decoded.empty:
        return {"n": 0.0, "validity": 0.0}

    smiles = decoded["smiles"].dropna().astype(str).tolist()
    metrics = evaluate_smiles(smiles, train_smiles=train_smiles)
    for col in TARGET_COLUMNS:
        target_col = f"target_{col}"
        actual_col = f"actual_{col}"
        if target_col in decoded.columns and actual_col in decoded.columns:
            metrics[f"{col}_mae"] = float((decoded[actual_col] - decoded[target_col]).abs().mean())
    if "condition_idx" in decoded.columns:
        metrics["conditions"] = float(decoded["condition_idx"].nunique())
        metrics["mean_candidates_per_condition"] = float(decoded.groupby("condition_idx").size().mean())
    return metrics


def _mean_pairwise_tanimoto(smiles: list[str], max_pairs: int = 20000) -> float:
    n = len(smiles)
    if n < 2:
        return 0.0
    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        pairs = list(itertools.combinations(smiles, 2))
    else:
        rng = np.random.default_rng(17)
        pairs = []
        seen = set()
        while len(pairs) < max_pairs:
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n - 1))
            if j >= i:
                j += 1
            a, b = sorted((i, j))
            if (a, b) in seen:
                continue
            seen.add((a, b))
            pairs.append((smiles[a], smiles[b]))
    return float(np.mean([tanimoto(a, b) for a, b in pairs]))
