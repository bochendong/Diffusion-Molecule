"""Structure-prompt benchmark aligned with SketchMol partial-image tasks.

SketchMol supports partial molecular images as prompts for inpainting,
fragment growing, and property optimization. This module builds a deterministic
analogue in molecule space:

    prompt structure + property target -> generated molecule

The chemistry judge is RDKit/rule based. A candidate succeeds only when it
keeps the prompt structure and satisfies the requested property constraints.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import chem as chem_mod
from .chem import canonicalize_smiles, molecular_descriptors, passes_druglike_filters, tanimoto
from .dataset import arrays_from_dataframe
from .decoder import DRUGLIKE_SOFT_PENALTY, DecodedCandidate
from .evaluate import evaluate_smiles
from .features import IMAGE_FEATURE_COLUMNS, descriptor_image, image_array_features
from .mmp_transform_decoder import MMPTransformConfig, MMPTransformIndex
from .retrieval_decoder import RetrievalCandidateIndex, RetrievalDecoderConfig, decode_retrieval_table_row
from .schema import TARGET_COLUMNS, TABLE_COLUMNS
from .sketchmol_benchmark import SKETCHMOL_SUCCESS_TOLERANCE, STRICT_SUCCESS_TOLERANCE


LOCAL_OPTIMIZATION_TASKS = {
    "LogP": 2.5,
    "QED": 0.3,
    "TPSA": -45.0,
}
DIRECT_TRAIN_DECODER_SOURCES = {
    "retrieval_train_pool_decoder",
    "mmp_transform_target_decoder",
}
SOURCE_AWARE_EDIT_DECODER_SOURCES = {
    "mmp_learned_fragment_grow_decoder",
    "mmp_two_step_fragment_grow_decoder",
    "retrieval_mmp_edit_decoder",
    "structure_prompt_assembly_decoder",
    "physics_aware_dynamic_decoder",
}


@dataclass
class StructurePromptBenchmarkConfig:
    conditions_per_task: int = 200
    samples_per_prompt: int = 8
    decode_top_k: int = 2
    seed: int = 7


def run_structure_prompt_benchmark(
    diffusion,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    aligner,
    args,
    compose_conditions_fn,
    output_dir: str | Path,
    config: StructurePromptBenchmarkConfig,
    retrieval_index: RetrievalCandidateIndex | None = None,
    retrieval_config: RetrievalDecoderConfig | None = None,
    mmp_index: MMPTransformIndex | None = None,
    mmp_config: MMPTransformConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    retrieval_config = _light_retrieval_config(retrieval_config)
    mmp_config = _light_mmp_config(mmp_config)
    decoded_parts = []
    summary_parts = []

    for task_name, builder in [
        ("scaffold_prompt", _build_scaffold_conditions),
        ("fragment_prompt", _build_fragment_conditions),
        ("local_optimization_prompt", _build_local_optimization_conditions),
    ]:
        cond_df = builder(train_df, eval_df, config)
        if cond_df.empty:
            continue
        decoded = _generate_decode(
            diffusion=diffusion,
            cond_df=cond_df,
            aligner=aligner,
            args=args,
            compose_conditions_fn=compose_conditions_fn,
            samples_per_prompt=config.samples_per_prompt,
            top_k=config.decode_top_k,
            retrieval_index=retrieval_index,
            retrieval_config=retrieval_config,
            mmp_index=mmp_index,
            mmp_config=mmp_config,
        )
        decoded = _annotate_train_overlap(decoded, train_df)
        decoded_parts.append(decoded)
        summary_parts.append(_summarize(decoded, train_df, task_name))

    all_decoded = pd.concat(decoded_parts, ignore_index=True) if decoded_parts else pd.DataFrame()
    all_summary = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    all_decoded.to_csv(output_dir / "structure_prompt_decoded.csv", index=False)
    all_summary.to_csv(output_dir / "structure_prompt_summary.csv", index=False)
    return all_decoded, all_summary


def _build_scaffold_conditions(train_df: pd.DataFrame, eval_df: pd.DataFrame, config: StructurePromptBenchmarkConfig) -> pd.DataFrame:
    rows = []
    source_df = _condition_frame(eval_df, config.conditions_per_task, config.seed + 11)
    train_medians = {col: float(train_df[col].median()) for col in TARGET_COLUMNS}
    for condition_idx, row in source_df.iterrows():
        source = str(row["smiles"])
        prompt = scaffold_prompt(source)
        if not prompt:
            continue
        out = _condition_row_from_prompt(
            row=row,
            condition_idx=condition_idx,
            benchmark_task="scaffold_prompt",
            prompt_smiles=prompt,
            source_smiles=source,
            target_props={col: float(row[col]) for col in ["MW", "LogP", "QED", "TPSA"]},
            constrained_props=["MW", "LogP", "QED", "TPSA"],
            train_medians=train_medians,
        )
        rows.append(out)
    return pd.DataFrame(rows)


def _build_fragment_conditions(train_df: pd.DataFrame, eval_df: pd.DataFrame, config: StructurePromptBenchmarkConfig) -> pd.DataFrame:
    rows = []
    source_df = _condition_frame(eval_df, config.conditions_per_task, config.seed + 23)
    train_medians = {col: float(train_df[col].median()) for col in TARGET_COLUMNS}
    for condition_idx, row in source_df.iterrows():
        source = str(row["smiles"])
        prompt = fragment_prompt(source)
        if not prompt:
            continue
        prompt_desc = molecular_descriptors(prompt)
        if not prompt_desc.valid:
            continue
        source_mw = float(row["MW"])
        prompt_mw = float(prompt_desc.descriptors.get("MW", 0.0))
        out = _condition_row_from_prompt(
            row=row,
            condition_idx=condition_idx,
            benchmark_task="fragment_prompt",
            prompt_smiles=prompt,
            source_smiles=source,
            target_props={
                "MW": max(source_mw, prompt_mw + 50.0),
                "LogP": float(row["LogP"]),
                "QED": float(row["QED"]),
            },
            constrained_props=["MW", "LogP", "QED"],
            train_medians=train_medians,
        )
        out["fragment_growth_mw_min"] = prompt_mw + 25.0
        rows.append(out)
    return pd.DataFrame(rows)


def _build_local_optimization_conditions(train_df: pd.DataFrame, eval_df: pd.DataFrame, config: StructurePromptBenchmarkConfig) -> pd.DataFrame:
    rows = []
    source_df = _condition_frame(eval_df, config.conditions_per_task, config.seed + 37)
    train_medians = {col: float(train_df[col].median()) for col in TARGET_COLUMNS}
    row_id = 0
    for _, row in source_df.iterrows():
        source = str(row["smiles"])
        prompt = scaffold_prompt(source) or fragment_prompt(source)
        if not prompt:
            continue
        for prop, delta in LOCAL_OPTIMIZATION_TASKS.items():
            target = _clipped_property_target(prop, float(row[prop]) + delta)
            out = _condition_row_from_prompt(
                row=row,
                condition_idx=row_id,
                benchmark_task="local_optimization_prompt",
                prompt_smiles=prompt,
                source_smiles=source,
                target_props={prop: target},
                constrained_props=[prop],
                train_medians=train_medians,
            )
            out["optimization_property"] = prop
            out["requested_delta"] = float(delta)
            out["before_property"] = float(row[prop])
            rows.append(out)
            row_id += 1
    return pd.DataFrame(rows)


def _condition_row_from_prompt(
    row,
    condition_idx: int,
    benchmark_task: str,
    prompt_smiles: str,
    source_smiles: str,
    target_props: dict[str, float],
    constrained_props: list[str],
    train_medians: dict[str, float],
) -> dict[str, Any]:
    prompt_features = _prompt_image_features(prompt_smiles)
    out = {
        "condition_idx": int(condition_idx),
        "benchmark_task": benchmark_task,
        "source_smiles": source_smiles,
        "prompt_smiles": prompt_smiles,
        "target_json": json.dumps(target_props, sort_keys=True),
        "constraint_properties": ",".join(constrained_props),
    }
    for col in TARGET_COLUMNS:
        out[col] = float(target_props.get(col, train_medians[col]))
        out[f"condition_mask_{col}"] = 1.0 if col in target_props else 0.0
    for col in TABLE_COLUMNS:
        if col not in out:
            out[col] = float(row[col]) if col in row else 0.0
    out.update(prompt_features)
    return out


def _generate_decode(
    diffusion,
    cond_df: pd.DataFrame,
    aligner,
    args,
    compose_conditions_fn,
    samples_per_prompt: int,
    top_k: int,
    retrieval_index: RetrievalCandidateIndex | None = None,
    retrieval_config: RetrievalDecoderConfig | None = None,
    mmp_index: MMPTransformIndex | None = None,
    mmp_config: MMPTransformConfig | None = None,
) -> pd.DataFrame:
    image_x, _, base_condition_x, _ = arrays_from_dataframe(cond_df)
    conditions, _ = compose_conditions_fn(
        cond_df,
        base_condition_x,
        image_x,
        aligner,
        args,
        intent=getattr(args, "intent", "default"),
        reference_smiles=getattr(args, "reference_smiles", None),
        use_in_context=bool(getattr(args, "reference_smiles", None) or getattr(args, "reference_image", None) or getattr(args, "intent", "default") != "default"),
        property_mask_mode="benchmark",
    )
    rows = []
    if hasattr(diffusion, "sample_batch"):
        sampled_rows = diffusion.sample_batch(conditions, samples_per_condition=samples_per_prompt)
    else:
        sampled_rows = []
        for condition_idx, condition in enumerate(conditions):
            for sample_idx, table_row in enumerate(diffusion.sample(condition[None, :], n=samples_per_prompt)):
                sampled_rows.append((condition_idx, sample_idx, table_row))
    for condition_idx, sample_idx, table_row in sampled_rows:
        condition_row = cond_df.iloc[int(condition_idx)]
        decode_row = _condition_guided_table_row(table_row, condition_row)
        candidates = _decode_prompt_candidates(
            table_row=decode_row,
            prompt_smiles=str(condition_row["prompt_smiles"]),
            top_k=top_k,
            seed=int(condition_idx) * 1009 + int(sample_idx),
            decoder_mode=getattr(args, "decoder_mode", "physics"),
            retrieval_index=retrieval_index,
            retrieval_config=retrieval_config,
            mmp_index=mmp_index,
            mmp_config=mmp_config,
        )
        for rank, candidate in enumerate(candidates, start=1):
            out = {
                "condition_idx": int(condition_idx),
                "sample_idx": int(sample_idx),
                "rank": int(rank),
                "benchmark_task": condition_row["benchmark_task"],
                "source_smiles": condition_row["source_smiles"],
                "prompt_smiles": condition_row["prompt_smiles"],
                "constraint_properties": condition_row["constraint_properties"],
                "target_json": condition_row["target_json"],
                "smiles": candidate.smiles,
                "valid": candidate.valid,
                "decoder_score": candidate.score,
                "candidate_source": candidate.source,
            }
            for optional_col in ("optimization_property", "requested_delta", "before_property", "fragment_growth_mw_min"):
                if optional_col in condition_row:
                    out[optional_col] = condition_row[optional_col]
            out.update({f"target_{key}": value for key, value in decode_row.items()})
            out.update({f"sampled_{key}": value for key, value in table_row.items() if isinstance(value, (int, float))})
            out.update({f"actual_{key}": value for key, value in candidate.descriptors.items() if isinstance(value, (int, float))})
            out.update(_verify_structure_prompt(condition_row, candidate.smiles))
            rows.append(out)
    return pd.DataFrame(rows)


def _condition_guided_table_row(table_row: dict[str, float], condition_row) -> dict[str, float]:
    """Use explicit benchmark targets for decoder ranking.

    The diffusion sample is still saved for analysis, but structure-prompt
    evaluation judges the explicit prompt constraints in ``target_json``. If we
    rank candidates only against the sampled row, a good source-aware fragment
    can be rejected despite satisfying the actual benchmark instruction.
    """

    out = dict(table_row)
    try:
        target_props = json.loads(str(condition_row["target_json"]))
    except Exception:
        target_props = {}
    for prop, value in target_props.items():
        if prop in TARGET_COLUMNS:
            out[prop] = float(value)
    if "fragment_growth_mw_min" in condition_row and not pd.isna(condition_row["fragment_growth_mw_min"]):
        out["MW"] = max(float(out.get("MW", 0.0)), float(condition_row["fragment_growth_mw_min"]))
    return out


def _decode_prompt_candidates(
    table_row: dict[str, float],
    prompt_smiles: str,
    top_k: int,
    seed: int,
    decoder_mode: str = "physics",
    retrieval_index: RetrievalCandidateIndex | None = None,
    retrieval_config: RetrievalDecoderConfig | None = None,
    mmp_index: MMPTransformIndex | None = None,
    mmp_config: MMPTransformConfig | None = None,
) -> list[DecodedCandidate]:
    candidates = []
    decoder_pool_k = max(top_k, min(4, top_k * 2))
    candidates.extend(
        decode_retrieval_table_row(
            table_row,
            top_k=decoder_pool_k,
            seed=seed,
            mode=decoder_mode,
            index=retrieval_index,
            config=retrieval_config,
            include_dynamic=True,
            prompt_smiles=prompt_smiles,
            mmp_index=mmp_index,
            mmp_config=mmp_config,
        )
    )
    candidates.extend(_prompt_assembly_candidates(prompt_smiles, table_row, seed=seed))
    by_smiles = {}
    for candidate in candidates:
        existing = by_smiles.get(candidate.smiles)
        if existing is None or candidate.score < existing.score:
            by_smiles[candidate.smiles] = candidate
    ranked = sorted(by_smiles.values(), key=lambda item: item.score)
    return ranked[:top_k]


def _prompt_assembly_candidates(prompt_smiles: str, table_row: dict[str, float], seed: int) -> list[DecodedCandidate]:
    variants = {prompt_smiles}
    additions = _additions_for_target(table_row)
    for add in additions:
        variants.add(prompt_smiles + add)
        variants.add(add + prompt_smiles)
    out = []
    for smi in variants:
        rec = molecular_descriptors(smi)
        if not rec.valid:
            continue
        score = _candidate_score(table_row, prompt_smiles, rec.smiles, rec.descriptors, seed=seed)
        out.append(
            DecodedCandidate(
                smiles=rec.smiles,
                score=score,
                valid=True,
                descriptors=rec.descriptors,
                source="structure_prompt_assembly_decoder",
            )
        )
    return out


def _additions_for_target(table_row: dict[str, float]) -> list[str]:
    additions = ["C", "CC", "CCC"]
    if float(table_row.get("O", 0.0)) > 0 or float(table_row.get("HBA", 0.0)) > 2:
        additions.extend(["O", "CO", "OC", "C(=O)O", "C(=O)N"])
    if float(table_row.get("N", 0.0)) > 0 or float(table_row.get("HBD", 0.0)) > 0:
        additions.extend(["N", "CN", "CCN", "NC(=O)"])
    if float(table_row.get("fg_halogen", 0.0)) > 0 or float(table_row.get("LogP", 0.0)) > 3.0:
        additions.extend(["F", "Cl", "Br"])
    return list(dict.fromkeys(additions))


def _candidate_score(table_row: dict[str, float], prompt_smiles: str, candidate_smiles: str, descriptors: dict[str, float], seed: int) -> float:
    scales = {"MW": 120.0, "LogP": 3.0, "QED": 0.35, "TPSA": 60.0, "HBD": 3.0, "HBA": 5.0, "RB": 5.0, "SA": 2.5}
    score = float(np.mean([abs(float(descriptors.get(col, 0.0)) - float(table_row.get(col, 0.0))) / scales[col] for col in TARGET_COLUMNS]))
    if contains_prompt(prompt_smiles, candidate_smiles):
        score -= 3.0
    else:
        score += 6.0
    if not passes_druglike_filters(descriptors):
        score += DRUGLIKE_SOFT_PENALTY
    score += 0.01 * _stable_noise(candidate_smiles, seed)
    return score


def _verify_structure_prompt(condition_row, candidate_smiles: str) -> dict[str, Any]:
    prompt_smiles = str(condition_row["prompt_smiles"])
    target_json = str(condition_row["target_json"])
    target_props = json.loads(target_json)
    desc = molecular_descriptors(candidate_smiles)
    structure_success = bool(desc.valid and contains_prompt(prompt_smiles, candidate_smiles))
    strict_success = _property_success(desc.descriptors if desc.valid else {}, target_props, STRICT_SUCCESS_TOLERANCE)
    sketchmol_success = _property_success(desc.descriptors if desc.valid else {}, target_props, SKETCHMOL_SUCCESS_TOLERANCE)
    growth_success = True
    if "fragment_growth_mw_min" in condition_row and desc.valid:
        growth_success = float(desc.descriptors.get("MW", 0.0)) >= float(condition_row["fragment_growth_mw_min"])
    delta_success = True
    if "optimization_property" in condition_row and desc.valid:
        prop = str(condition_row["optimization_property"])
        requested = float(condition_row["requested_delta"])
        before = float(condition_row["before_property"])
        achieved = float(desc.descriptors.get(prop, 0.0)) - before
        delta_success = (achieved >= requested - SKETCHMOL_SUCCESS_TOLERANCE.get(prop, 1.0)) if requested >= 0 else (
            achieved <= requested + SKETCHMOL_SUCCESS_TOLERANCE.get(prop, 1.0)
        )
    return {
        "structure_prompt_success": structure_success,
        "property_success_strict": bool(strict_success),
        "property_success_sketchmol_tolerance": bool(sketchmol_success),
        "fragment_growth_success": bool(growth_success),
        "optimization_delta_success": bool(delta_success),
        "joint_success_strict": bool(structure_success and strict_success and growth_success and delta_success),
        "joint_success_sketchmol_tolerance": bool(structure_success and sketchmol_success and growth_success and delta_success),
        "prompt_candidate_similarity": float(tanimoto(prompt_smiles, candidate_smiles)) if desc.valid else 0.0,
    }


def _property_success(descriptors: dict[str, float], targets: dict[str, float], tolerance: dict[str, float]) -> bool:
    for prop, target in targets.items():
        if prop not in tolerance:
            continue
        if abs(float(descriptors.get(prop, 0.0)) - float(target)) > tolerance[prop]:
            return False
    return True


def _summarize(decoded: pd.DataFrame, train_df: pd.DataFrame, task_name: str) -> pd.DataFrame:
    if decoded.empty:
        return pd.DataFrame()
    rows = []
    train_smiles = _canonical_smiles_set(train_df["smiles"].dropna().astype(str).tolist())
    for label, frame in decoded.groupby("benchmark_task", dropna=False):
        metrics = evaluate_smiles(frame["smiles"].dropna().astype(str).tolist(), train_smiles=list(train_smiles))
        source_series = frame["candidate_source"].astype(str) if "candidate_source" in frame else pd.Series([], dtype=str)
        sampled_train_overlap = _sampled_train_overlap_metrics(frame["smiles"].dropna().astype(str).tolist(), train_smiles)
        row = {
            "benchmark_task": task_name,
            "benchmark_label": str(label),
            "structure_prompt_success_rate": _mean_bool(frame["structure_prompt_success"]),
            "property_success_strict_in_valid_mols": _mean_bool(frame["property_success_strict"]),
            "property_success_sketchmol_tolerance_in_valid_mols": _mean_bool(frame["property_success_sketchmol_tolerance"]),
            "joint_success_strict": _mean_bool(frame["joint_success_strict"]),
            "joint_success_sketchmol_tolerance": _mean_bool(frame["joint_success_sketchmol_tolerance"]),
            "fragment_growth_success_rate": _mean_bool(frame["fragment_growth_success"]),
            "optimization_delta_success_rate": _mean_bool(frame["optimization_delta_success"]),
            "prompt_candidate_similarity": float(frame["prompt_candidate_similarity"].mean()) if "prompt_candidate_similarity" in frame else 0.0,
            "exact_train_hit_rate": _mean_bool(frame["exact_train_hit"]) if "exact_train_hit" in frame else 0.0,
            "direct_train_decoder_fraction": float(source_series.isin(DIRECT_TRAIN_DECODER_SOURCES).mean()) if len(source_series) else 0.0,
            "source_aware_edit_decoder_fraction": float(source_series.isin(SOURCE_AWARE_EDIT_DECODER_SOURCES).mean()) if len(source_series) else 0.0,
            "mmp_target_decoder_fraction": float((source_series == "mmp_transform_target_decoder").mean()) if len(source_series) else 0.0,
            "mmp_fragment_decoder_fraction": float(
                source_series.isin({"mmp_learned_fragment_grow_decoder", "mmp_two_step_fragment_grow_decoder"}).mean()
            )
            if len(source_series)
            else 0.0,
            "mmp_two_step_fragment_decoder_fraction": float((source_series == "mmp_two_step_fragment_grow_decoder").mean()) if len(source_series) else 0.0,
        }
        row.update(metrics)
        row.update(sampled_train_overlap)
        rows.append(row)
    return pd.DataFrame(rows)


def _annotate_train_overlap(decoded: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    if decoded.empty or "smiles" not in decoded:
        return decoded
    train_smiles = _canonical_smiles_set(train_df["smiles"].dropna().astype(str).tolist())
    out = decoded.copy()
    canonical = []
    exact = []
    for smiles in out["smiles"].fillna("").astype(str):
        can = canonicalize_smiles(smiles)
        canonical.append(can or "")
        exact.append(bool(can and can in train_smiles))
    out["canonical_smiles"] = canonical
    out["exact_train_hit"] = exact
    out["direct_train_decoder"] = out["candidate_source"].astype(str).isin(DIRECT_TRAIN_DECODER_SOURCES) if "candidate_source" in out else False
    out["source_aware_edit_decoder"] = out["candidate_source"].astype(str).isin(SOURCE_AWARE_EDIT_DECODER_SOURCES) if "candidate_source" in out else False
    return out


def _canonical_smiles_set(smiles: list[str]) -> set[str]:
    out = set()
    for smi in smiles:
        can = canonicalize_smiles(str(smi))
        if can:
            out.add(can)
    return out


def _sampled_train_overlap_metrics(candidate_smiles: list[str], train_smiles: set[str], max_candidates: int = 256, max_train: int = 1024) -> dict[str, float]:
    candidate_set = set()
    for smi in candidate_smiles:
        can = canonicalize_smiles(smi)
        if can:
            candidate_set.add(can)
    candidates = sorted(candidate_set)
    train = sorted(train_smiles)
    if not candidates or not train:
        return {
            "sampled_nearest_train_tanimoto": 0.0,
            "sampled_nearest_train_tanimoto_ge_0_90": 0.0,
            "sampled_novelty_at_tanimoto_0_90": 0.0,
        }
    rng = np.random.default_rng(23)
    if len(candidates) > max_candidates:
        candidates = [candidates[int(i)] for i in rng.choice(len(candidates), size=max_candidates, replace=False)]
    if len(train) > max_train:
        train = [train[int(i)] for i in rng.choice(len(train), size=max_train, replace=False)]
    nearest = []
    for candidate in candidates:
        if candidate in train_smiles:
            nearest.append(1.0)
            continue
        sims = [tanimoto(candidate, train_smi) for train_smi in train if train_smi != candidate]
        nearest.append(max(sims) if sims else 1.0)
    ge_090 = float(np.mean([sim >= 0.90 for sim in nearest])) if nearest else 0.0
    return {
        "sampled_nearest_train_tanimoto": float(np.mean(nearest)) if nearest else 0.0,
        "sampled_nearest_train_tanimoto_ge_0_90": ge_090,
        "sampled_novelty_at_tanimoto_0_90": float(1.0 - ge_090),
    }


def scaffold_prompt(smiles: str) -> str | None:
    can = canonicalize_smiles(smiles)
    if can is None:
        return None
    if chem_mod.RDKIT_AVAILABLE:
        try:  # pragma: no cover - RDKit path is exercised on server.
            from rdkit.Chem.Scaffolds import MurckoScaffold

            mol = chem_mod.Chem.MolFromSmiles(can)
            if mol is None:
                return None
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            if scaffold:
                return canonicalize_smiles(scaffold)
        except Exception:
            pass
    desc = molecular_descriptors(can)
    if not desc.valid:
        return None
    return can if float(desc.descriptors.get("ring_count", 0.0)) > 0 else None


def fragment_prompt(smiles: str) -> str | None:
    can = canonicalize_smiles(smiles)
    if can is None:
        return None
    if chem_mod.RDKIT_AVAILABLE:
        try:  # pragma: no cover - RDKit path is exercised on server.
            mol = chem_mod.Chem.MolFromSmiles(can)
            if mol is None:
                return None
            ring_info = mol.GetRingInfo()
            atom_rings = list(ring_info.AtomRings())
            if atom_rings:
                atoms = sorted(atom_rings[0])
                frag = chem_mod.Chem.MolFragmentToSmiles(mol, atomsToUse=atoms, canonical=True)
                return canonicalize_smiles(frag)
        except Exception:
            pass
    scaffold = scaffold_prompt(can)
    if scaffold:
        return scaffold
    desc = molecular_descriptors(can)
    if desc.valid and float(desc.descriptors.get("C", 0.0)) >= 3:
        return "CCC"
    return None


@lru_cache(maxsize=500000)
def contains_prompt(prompt_smiles: str, candidate_smiles: str) -> bool:
    prompt = canonicalize_smiles(prompt_smiles)
    candidate = canonicalize_smiles(candidate_smiles)
    if prompt is None or candidate is None:
        return False
    if prompt == candidate:
        return True
    if chem_mod.RDKIT_AVAILABLE:
        try:  # pragma: no cover - RDKit path is exercised on server.
            prompt_mol = chem_mod.Chem.MolFromSmiles(prompt)
            candidate_mol = chem_mod.Chem.MolFromSmiles(candidate)
            return bool(candidate_mol is not None and prompt_mol is not None and candidate_mol.HasSubstructMatch(prompt_mol))
        except Exception:
            return False
    return prompt in candidate or tanimoto(prompt, candidate) >= 0.5


def _prompt_image_features(prompt_smiles: str) -> dict[str, float]:
    return image_array_features(descriptor_image(prompt_smiles))


def _condition_frame(eval_df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if eval_df.empty:
        raise ValueError("Structure prompt benchmark requires a non-empty evaluation dataframe.")
    return eval_df.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)


def _clipped_property_target(prop: str, value: float) -> float:
    if prop == "QED":
        return float(np.clip(value, 0.0, 1.0))
    if prop == "TPSA":
        return float(np.clip(value, 0.0, 180.0))
    if prop == "LogP":
        return float(np.clip(value, -3.0, 7.0))
    return float(value)


def _mean_bool(series) -> float:
    return float(pd.Series(series).astype(bool).mean()) if len(series) else 0.0


def _stable_noise(smiles: str, seed: int) -> float:
    digest = hashlib.sha256(f"structure-prompt:{seed}:{smiles}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


def _light_retrieval_config(config: RetrievalDecoderConfig | None) -> RetrievalDecoderConfig | None:
    if config is None:
        return None
    return replace(
        config,
        neighbors=min(config.neighbors, 64),
        edit_neighbors=min(config.edit_neighbors, 4),
        prompt_match_bonus=max(config.prompt_match_bonus, 2.0),
        prompt_miss_penalty=max(config.prompt_miss_penalty, 7.0),
    )


def _light_mmp_config(config: MMPTransformConfig | None) -> MMPTransformConfig | None:
    if config is None:
        return None
    return replace(
        config,
        target_neighbors=min(config.target_neighbors, 96),
        delta_neighbors=min(config.delta_neighbors, 64),
        source_neighbors=min(config.source_neighbors, 64),
        fragment_neighbors=min(config.fragment_neighbors, 16),
        attachment_limit=min(config.attachment_limit, 3),
        fragment_growth_steps=min(config.fragment_growth_steps, 2),
        fragment_growth_beam_size=min(config.fragment_growth_beam_size, 12),
        fragment_second_step_neighbors=min(config.fragment_second_step_neighbors, 6),
        prompt_match_bonus=max(config.prompt_match_bonus, 3.0),
        prompt_miss_penalty=max(config.prompt_miss_penalty, 10.0),
    )
