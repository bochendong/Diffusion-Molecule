#!/usr/bin/env python3
"""Build source-preserving agentic revise predictions for external benchmarks.

The direct-SMILES checkpoint is used as a proposal model, then a small local
edit loop searches source-neighbor molecules that keep Tanimoto similarity high
while improving the locally computable task properties.  This is intentionally
reported as an agentic/assisted line, separate from one-shot direct generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sketchmol_understanding_condition.chem import canonical_smiles, molecular_properties, morgan_tanimoto  # noqa: E402


DEFAULT_DIRECTION = {
    "ampa": "increase",
    "bbbp": "increase",
    "drd2": "increase",
    "hia": "increase",
    "plogp": "increase",
    "qed": "increase",
    "carc": "decrease",
    "erg": "decrease",
    "liver": "decrease",
    "mutagenicity": "decrease",
}
DEFAULT_THRESHOLDS = {
    "ampa": 0.1,
    "bbbp": 0.1,
    "carc": 0.1,
    "drd2": 0.2,
    "erg": 0.2,
    "hia": 0.1,
    "liver": 0.1,
    "mutagenicity": 0.1,
    "plogp": 1.0,
    "qed": 0.1,
}
PROPERTY_ALIASES = {
    "ames": "mutagenicity",
    "mutagen": "mutagenicity",
    "mutagenicity": "mutagenicity",
    "bbbp": "bbbp",
    "bbb_martins": "bbbp",
    "hia": "hia",
    "hia_hou": "hia",
    "qed": "qed",
    "plogp": "plogp",
    "penalized_logp": "plogp",
    "logp": "logp",
    "drd2": "drd2",
    "carc": "carc",
    "carcinogenicity": "carc",
    "erg": "erg",
    "herg": "erg",
    "liver": "liver",
    "dili": "liver",
    "ampa": "ampa",
    "pampa": "ampa",
    "sas": "sas",
    "sa": "sas",
}


@dataclass(frozen=True)
class Candidate:
    smiles: str
    action_trace: str
    source: str


@dataclass(frozen=True)
class CandidateScore:
    candidate: Candidate
    canonical_smiles: str
    score: float
    source_tanimoto: float
    source_similarity_success: bool
    local_success_fraction: float
    all_evaluated_local_success: bool
    evaluated_local_property_count: int
    local_property_distance: float
    missing_local_properties: tuple[str, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-csv", required=True, type=Path)
    parser.add_argument("--prediction-csv", required=True, type=Path)
    parser.add_argument("--direct-prediction-csv", type=Path, default=None)
    parser.add_argument("--direct-smiles-column", default="generated_smiles")
    parser.add_argument("--min-source-tanimoto", type=float, default=0.4)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--beam-size", type=int, default=48)
    parser.add_argument("--max-candidates-per-parent", type=int, default=64)
    parser.add_argument("--max-candidates-per-row", type=int, default=256)
    parser.add_argument("--property-weight", type=float, default=100.0)
    parser.add_argument("--distance-weight", type=float, default=10.0)
    parser.add_argument("--similarity-weight", type=float, default=12.0)
    parser.add_argument("--similarity-bonus", type=float, default=25.0)
    parser.add_argument("--copy-penalty", type=float, default=2.0)
    parser.add_argument("--method", default="external_agentic_revise")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rng = random.Random(int(args.seed))
    rows = read_rows(args.rows_csv)
    direct_rows = read_direct_predictions(args.direct_prediction_csv) if args.direct_prediction_csv else {}
    output_rows = []
    for index, row in enumerate(rows):
        direct_row = direct_rows.get(row_key(row), {})
        result = revise_row(row, direct_row=direct_row, args=args, rng=rng)
        output_rows.append(result)
        if (index + 1) % 100 == 0 or index + 1 == len(rows):
            print(f"[agentic-revise] wrote {index + 1}/{len(rows)} rows", flush=True)
    write_rows(args.prediction_csv, output_rows)
    summary = summarize_agentic_rows(output_rows)
    summary_path = args.prediction_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def revise_row(
    row: Mapping[str, str],
    *,
    direct_row: Mapping[str, str],
    args: argparse.Namespace,
    rng: random.Random,
) -> dict[str, object]:
    source_smiles = str(row.get("source_smiles", "") or "").strip()
    candidates = initial_candidates(row, direct_row=direct_row, direct_smiles_column=str(args.direct_smiles_column))
    seen = {safe_canonical(candidate.smiles) for candidate in candidates if safe_canonical(candidate.smiles)}
    beam = rank_candidates(row, candidates, args=args)[: max(1, int(args.beam_size))]
    for step in range(1, max(0, int(args.max_steps)) + 1):
        proposed: list[Candidate] = []
        parents = [item.canonical_smiles for item in beam] or [source_smiles]
        rng.shuffle(parents)
        for parent in parents[: max(1, int(args.beam_size))]:
            for smiles, action in local_edit_candidates(parent, max_candidates=int(args.max_candidates_per_parent)):
                canonical = safe_canonical(smiles)
                if not canonical or canonical in seen:
                    continue
                seen.add(canonical)
                proposed.append(Candidate(canonical, f"step{step}:{action}", "local_edit"))
                if len(seen) >= int(args.max_candidates_per_row):
                    break
            if len(seen) >= int(args.max_candidates_per_row):
                break
        candidates.extend(proposed)
        beam = rank_candidates(row, candidates, args=args)[: max(1, int(args.beam_size))]
        if len(seen) >= int(args.max_candidates_per_row):
            break
    ranked = rank_candidates(row, candidates, args=args)
    best = ranked[0] if ranked else score_candidate(row, Candidate(source_smiles, "source_copy", "source_copy"), args=args)
    direct_smiles = str(direct_row.get(str(args.direct_smiles_column), "") or "").strip()
    out = dict(row)
    out["generated_smiles"] = best.canonical_smiles
    out["method"] = str(args.method)
    out["direct_generated_smiles"] = direct_smiles
    out["agentic_action_trace"] = best.candidate.action_trace
    out["agentic_candidate_source"] = best.candidate.source
    out["agentic_candidate_count"] = len(candidates)
    out["agentic_valid_candidate_count"] = len([item for item in ranked if item.canonical_smiles])
    out["agentic_best_score"] = format_float(best.score, digits=6)
    out["agentic_source_tanimoto"] = "" if math.isnan(best.source_tanimoto) else format_float(best.source_tanimoto, digits=6)
    out["agentic_source_similarity_success"] = "True" if best.source_similarity_success else "False"
    out["agentic_local_success_fraction"] = format_float(best.local_success_fraction, digits=6)
    out["agentic_all_evaluated_local_success"] = "True" if best.all_evaluated_local_success else "False"
    out["agentic_evaluated_local_property_count"] = best.evaluated_local_property_count
    out["agentic_local_property_distance"] = format_float(best.local_property_distance, digits=6)
    out["agentic_missing_local_properties"] = ",".join(best.missing_local_properties)
    return out


def initial_candidates(
    row: Mapping[str, str],
    *,
    direct_row: Mapping[str, str],
    direct_smiles_column: str,
) -> list[Candidate]:
    out = []
    source = str(row.get("source_smiles", "") or "").strip()
    if source:
        out.append(Candidate(source, "source_copy", "source_copy"))
    direct = str(direct_row.get(direct_smiles_column, "") or "").strip()
    if direct:
        out.append(Candidate(direct, "direct_model_proposal", "direct_model"))
    return out


def rank_candidates(row: Mapping[str, str], candidates: Iterable[Candidate], *, args: argparse.Namespace) -> list[CandidateScore]:
    scored = [score_candidate(row, candidate, args=args) for candidate in candidates]
    scored = [item for item in scored if item.canonical_smiles]
    return sorted(
        scored,
        key=lambda item: (
            item.score,
            item.all_evaluated_local_success,
            item.local_success_fraction,
            item.source_tanimoto if math.isfinite(item.source_tanimoto) else -1.0,
        ),
        reverse=True,
    )


def score_candidate(row: Mapping[str, str], candidate: Candidate, *, args: argparse.Namespace) -> CandidateScore:
    canonical = safe_canonical(candidate.smiles)
    if not canonical:
        return CandidateScore(candidate, "", -1e9, math.nan, False, 0.0, False, 0, 1e6, tuple())
    source = str(row.get("source_smiles", "") or "").strip()
    source_canonical = safe_canonical(source)
    tanimoto = safe_tanimoto(source_canonical or source, canonical) if source else math.nan
    sim_success = bool(math.isnan(tanimoto) or tanimoto >= float(args.min_source_tanimoto))
    components = local_property_components(row, canonical)
    score = (
        float(args.property_weight) * components["success_fraction"]
        - float(args.distance_weight) * components["mean_distance"]
        + float(args.similarity_weight) * (0.0 if math.isnan(tanimoto) else tanimoto)
        + (float(args.similarity_bonus) if sim_success else 0.0)
    )
    if source_canonical and canonical == source_canonical:
        score -= float(args.copy_penalty)
    return CandidateScore(
        candidate=candidate,
        canonical_smiles=canonical,
        score=float(score),
        source_tanimoto=float(tanimoto),
        source_similarity_success=sim_success,
        local_success_fraction=float(components["success_fraction"]),
        all_evaluated_local_success=bool(components["all_success"]),
        evaluated_local_property_count=int(components["evaluated_count"]),
        local_property_distance=float(components["mean_distance"]),
        missing_local_properties=tuple(components["missing"]),
    )


def local_property_components(row: Mapping[str, str], smiles: str) -> dict[str, object]:
    task_props = parse_list(row.get("external_task_properties") or row.get("condition_properties"))
    directions = parse_json_dict(row.get("external_property_directions_json"), DEFAULT_DIRECTION)
    thresholds = parse_json_dict(row.get("external_property_thresholds_json"), DEFAULT_THRESHOLDS)
    source_props = source_properties(row)
    generated_props = local_properties(smiles)
    successes = []
    distances = []
    missing = []
    for prop in task_props:
        prop = canonical_prop(prop)
        generated = generated_props.get(prop)
        source = source_props.get(prop)
        target = parse_float(row.get(f"external_target_{prop}"))
        if generated is None or (source is None and target is None):
            missing.append(prop)
            continue
        direction = str(directions.get(prop, DEFAULT_DIRECTION.get(prop, "increase"))).lower()
        threshold = max(float(thresholds.get(prop, DEFAULT_THRESHOLDS.get(prop, 0.0))), 1e-8)
        if target is not None:
            margin = generated - target if direction == "increase" else target - generated
        else:
            delta = generated - float(source)
            margin = delta - threshold if direction == "increase" else -delta - threshold
        success = margin >= 0.0
        successes.append(success)
        distances.append(max(0.0, -margin / threshold))
    evaluated = len(successes)
    success_fraction = sum(1 for item in successes if item) / max(evaluated, 1)
    return {
        "evaluated_count": evaluated,
        "success_fraction": success_fraction,
        "all_success": bool(successes) and all(successes),
        "mean_distance": sum(distances) / max(len(distances), 1),
        "missing": missing,
    }


def source_properties(row: Mapping[str, str]) -> dict[str, float]:
    props = {}
    source = str(row.get("source_smiles", "") or "").strip()
    if source:
        props.update(local_properties(source))
    for key, value in row.items():
        if key.startswith("external_source_"):
            parsed = parse_float(value)
            if parsed is not None:
                props[canonical_prop(key.removeprefix("external_source_"))] = parsed
    return props


def local_properties(smiles: str) -> dict[str, float]:
    try:
        props = molecular_properties(smiles) or {}
    except RuntimeError:
        return {}
    out = {}
    if "QED" in props:
        out["qed"] = float(props["QED"])
    if "LogP" in props:
        out["logp"] = float(props["LogP"])
    if "LogP" in props:
        sa = float(props.get("SA", 0.0))
        out["plogp"] = float(props["LogP"]) - sa
    if "SA" in props:
        out["sas"] = float(props["SA"])
    return out


def local_edit_candidates(smiles: str, *, max_candidates: int) -> list[tuple[str, str]]:
    try:
        from rdkit import Chem
    except ImportError:
        return []
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return []
    out: list[tuple[str, str]] = []
    seen = set()

    def add_candidate(candidate_mol, action: str) -> None:
        if len(out) >= max(1, int(max_candidates)):
            return
        try:
            Chem.SanitizeMol(candidate_mol)
            candidate = Chem.MolToSmiles(candidate_mol, canonical=True)
        except Exception:
            return
        if candidate and candidate not in seen:
            seen.add(candidate)
            out.append((candidate, action))

    atom_symbols = ("C", "N", "O", "F", "Cl")
    for atom in mol.GetAtoms():
        atom_index = atom.GetIdx()
        if len(out) >= max_candidates:
            break
        for symbol in atom_symbols:
            rw = Chem.RWMol(mol)
            new_index = rw.AddAtom(Chem.Atom(symbol))
            rw.AddBond(atom_index, new_index, Chem.BondType.SINGLE)
            add_candidate(rw.GetMol(), f"add_{symbol}@{atom_index}")
        if not atom.GetIsAromatic() and atom.GetAtomicNum() not in {1}:
            for symbol in atom_symbols:
                if atom.GetSymbol() == symbol:
                    continue
                rw = Chem.RWMol(mol)
                rw.GetAtomWithIdx(atom_index).SetAtomicNum(Chem.Atom(symbol).GetAtomicNum())
                add_candidate(rw.GetMol(), f"replace_{atom.GetSymbol()}_to_{symbol}@{atom_index}")
    if mol.GetNumAtoms() > 1:
        for atom in mol.GetAtoms():
            if len(out) >= max_candidates:
                break
            if atom.GetDegree() == 1:
                rw = Chem.RWMol(mol)
                rw.RemoveAtom(atom.GetIdx())
                add_candidate(rw.GetMol(), f"remove_terminal@{atom.GetIdx()}")
    return out[: max(1, int(max_candidates))]


def read_direct_predictions(path: Path) -> dict[str, dict[str, str]]:
    return {row_key(row): row for row in read_rows(path) if row_key(row)}


def row_key(row: Mapping[str, str]) -> str:
    return str(row.get("condition_id") or row.get("sample_id") or row.get("molecule_id") or "").strip()


def summarize_agentic_rows(rows: list[Mapping[str, object]]) -> dict[str, object]:
    n = len(rows)
    return {
        "rows": n,
        "valid_generated": sum(1 for row in rows if str(row.get("generated_smiles") or "").strip()),
        "source_similarity_success_rate": mean_bool(rows, "agentic_source_similarity_success"),
        "all_evaluated_local_success_rate": mean_bool(rows, "agentic_all_evaluated_local_success"),
        "mean_local_success_fraction": mean_float(rows, "agentic_local_success_fraction"),
        "mean_source_tanimoto": mean_float(rows, "agentic_source_tanimoto"),
        "mean_candidate_count": mean_float(rows, "agentic_candidate_count"),
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_canonical(smiles: str) -> str:
    try:
        return canonical_smiles(smiles) or ""
    except RuntimeError:
        return str(smiles or "").strip()


def safe_tanimoto(left: str, right: str) -> float:
    try:
        value = morgan_tanimoto(left, right)
    except RuntimeError:
        return math.nan
    return math.nan if value is None else float(value)


def parse_json_dict(value: object, fallback: Mapping[str, object]) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {canonical_prop(key): val for key, val in value.items()}
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return dict(fallback)
        if isinstance(parsed, Mapping):
            return {canonical_prop(key): val for key, val in parsed.items()}
    return dict(fallback)


def parse_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [canonical_prop(item) for item in value if str(item).strip()]
    return [canonical_prop(item) for item in str(value or "").replace("|", ",").split(",") if item.strip()]


def canonical_prop(value: object) -> str:
    key = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    return PROPERTY_ALIASES.get(key, key)


def parse_float(value: object) -> float | None:
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def mean_bool(rows: Sequence[Mapping[str, object]], key: str) -> float:
    return sum(1 for row in rows if truthy(row.get(key))) / max(len(rows), 1)


def mean_float(rows: Sequence[Mapping[str, object]], key: str) -> float:
    values = [parse_float(row.get(key)) for row in rows]
    finite = [float(value) for value in values if value is not None]
    return sum(finite) / max(len(finite), 1)


def format_float(value: float, *, digits: int = 4) -> str:
    if not math.isfinite(float(value)):
        return ""
    text = f"{float(value):.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


if __name__ == "__main__":
    raise SystemExit(main())
