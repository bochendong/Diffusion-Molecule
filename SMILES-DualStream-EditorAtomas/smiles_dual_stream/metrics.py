"""Pure-SMILES evaluation helpers with optional RDKit support."""

from __future__ import annotations

from difflib import SequenceMatcher
import math
from typing import Iterable, Mapping

from .hierarchy import hierarchy_alignment
from .tokenization import tokenize_smiles


def safe_canonical_smiles(smiles: str) -> str | None:
    """Return RDKit canonical SMILES when RDKit is installed."""

    try:
        from rdkit import Chem
        from rdkit import RDLogger
    except ImportError:
        return str(smiles or "").strip() or None
    RDLogger.DisableLog("rdApp.warning")
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return None
    return str(Chem.MolToSmiles(mol, canonical=True))


def is_valid_smiles(smiles: str) -> bool | None:
    """Return RDKit validity, or None when RDKit is unavailable."""

    try:
        from rdkit import Chem
        from rdkit import RDLogger
    except ImportError:
        return None
    RDLogger.DisableLog("rdApp.warning")
    return Chem.MolFromSmiles(str(smiles or "")) is not None


def token_similarity(left: str, right: str) -> float:
    return float(SequenceMatcher(a=tokenize_smiles(left), b=tokenize_smiles(right), autojunk=False).ratio())


def exact_match(left: str, right: str) -> bool:
    left_canonical = safe_canonical_smiles(left)
    right_canonical = safe_canonical_smiles(right)
    if left_canonical is not None and right_canonical is not None:
        return left_canonical == right_canonical
    return str(left or "").strip() == str(right or "").strip()


def evaluate_prediction_row(row: Mapping[str, object]) -> dict[str, object]:
    generated = str(row.get("generated_smiles") or row.get("prediction") or "")
    target = str(row.get("target_smiles") or "")
    source = str(row.get("source_smiles") or row.get("corrupted_smiles") or "")
    alignment = hierarchy_alignment(generated, target)
    valid = is_valid_smiles(generated)
    return {
        "sample_id": str(row.get("sample_id") or ""),
        "generated_smiles": generated,
        "target_smiles": target,
        "source_smiles": source,
        "valid_smiles": valid,
        "exact_match": exact_match(generated, target),
        "token_similarity": token_similarity(generated, target),
        "source_token_similarity": token_similarity(generated, source) if source else math.nan,
        **alignment,
    }


def summarize_prediction_rows(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    decoded = [evaluate_prediction_row(row) for row in rows]
    if not decoded:
        return {"rows": 0}
    numeric_keys = ["token_similarity", "source_token_similarity", "token_jaccard", "fragment_jaccard", "token_lcs_ratio"]
    summary: dict[str, object] = {
        "rows": len(decoded),
        "exact_match_rate": sum(1 for row in decoded if row["exact_match"]) / len(decoded),
    }
    valid_values = [row["valid_smiles"] for row in decoded if row["valid_smiles"] is not None]
    if valid_values:
        summary["validity"] = sum(1 for value in valid_values if value) / len(valid_values)
    else:
        summary["validity"] = None
    for key in numeric_keys:
        values = [float(row[key]) for row in decoded if not _is_nan(row[key])]
        summary[f"mean_{key}"] = sum(values) / len(values) if values else math.nan
    return summary


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)

