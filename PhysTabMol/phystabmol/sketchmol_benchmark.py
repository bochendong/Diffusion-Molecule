"""SketchMol-aligned benchmark suite for PhysTabMol."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import arrays_from_dataframe
from .evaluate import evaluate_smiles
from .features import IMAGE_FEATURE_COLUMNS
from .retrieval_decoder import RetrievalCandidateIndex, RetrievalDecoderConfig, decode_retrieval_table_row
from .schema import TARGET_COLUMNS

SINGLE_PROPERTY_TARGETS = {
    "LogP": [0.2, 2.5, 3.4, 4.1, 6.1],
    "QED": [0.32, 0.50, 0.62, 0.75, 0.90],
    "MW": [191, 313, 366, 410, 510],
    "TPSA": [17, 50, 68, 85, 125],
    "HBD": [0, 1, 2, 3],
    "HBA": [1, 3, 4, 6, 8],
    "RB": [1, 4, 5, 6, 9],
}

OOD_TARGETS = {
    "LogP": [8, 9, 10],
    "TPSA": [160, 170, 180],
    "HBA": [11, 12, 13],
    "RB": [11, 12, 13],
    "MW": [600, 650, 700],
}

STRICT_SUCCESS_TOLERANCE = {
    "LogP": 1.0,
    "QED": 0.10,
    "MW": 35.0,
    "TPSA": 20.0,
    "HBD": 1.0,
    "HBA": 1.0,
    "RB": 1.0,
}

SKETCHMOL_SUCCESS_TOLERANCE = {
    "LogP": 1.8,
    "QED": 0.20,
    "MW": 90.0,
    "TPSA": 30.0,
    "HBD": 1.0,
    "HBA": 2.0,
    "RB": 2.0,
    "SA": 0.5,
}

SUCCESS_TOLERANCE = STRICT_SUCCESS_TOLERANCE

OPTIMIZATION_TASKS = {
    "LogP": 2.5,
    "QED": 0.3,
    "TPSA": -45.0,
}


@dataclass
class SketchMolBenchmarkConfig:
    single_conditions: int = 125
    samples_per_condition: int = 100
    decode_top_k: int = 1
    multi_conditions: int = 200
    optimization_conditions: int = 100
    seed: int = 7


def run_sketchmol_benchmark(
    diffusion,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    aligner,
    args,
    compose_conditions_fn,
    output_dir: str | Path,
    config: SketchMolBenchmarkConfig,
    retrieval_index: RetrievalCandidateIndex | None = None,
    retrieval_config: RetrievalDecoderConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decoded_parts = []
    summary_parts = []

    decoded, summary = _single_property(
        diffusion, train_df, eval_df, aligner, args, compose_conditions_fn, config, "single_property", SINGLE_PROPERTY_TARGETS, retrieval_index, retrieval_config
    )
    decoded_parts.append(decoded)
    summary_parts.append(summary)

    decoded, summary = _single_property(
        diffusion, train_df, eval_df, aligner, args, compose_conditions_fn, config, "ood_property", OOD_TARGETS, retrieval_index, retrieval_config
    )
    decoded_parts.append(decoded)
    summary_parts.append(summary)

    decoded, summary = _multi_property(diffusion, train_df, eval_df, aligner, args, compose_conditions_fn, config, retrieval_index, retrieval_config)
    decoded_parts.append(decoded)
    summary_parts.append(summary)

    decoded, summary = _optimization(diffusion, train_df, eval_df, aligner, args, compose_conditions_fn, config, retrieval_index, retrieval_config)
    decoded_parts.append(decoded)
    summary_parts.append(summary)

    all_decoded = pd.concat(decoded_parts, ignore_index=True) if decoded_parts else pd.DataFrame()
    all_summary = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    all_decoded.to_csv(output_dir / "sketchmol_benchmark_decoded.csv", index=False)
    all_summary.to_csv(output_dir / "sketchmol_benchmark_summary.csv", index=False)

    dist = _distribution_matching(all_decoded, train_df)
    with open(output_dir / "sketchmol_distribution_matching.json", "w", encoding="utf-8") as f:
        json.dump(dist, f, indent=2, sort_keys=True)
    return all_decoded, all_summary


def _single_property(diffusion, train_df, eval_df, aligner, args, compose_conditions_fn, config, task_name, targets_by_property, retrieval_index, retrieval_config):
    decoded_rows = []
    summary_rows = []
    for prop, values in targets_by_property.items():
        for value in values:
            cond_df = _condition_frame(eval_df, config.single_conditions, config.seed)
            _set_neutral_image_features(cond_df, train_df)
            for col in TARGET_COLUMNS:
                cond_df[col] = float(train_df[col].median())
            cond_df[prop] = float(value)
            _assign_condition_masks(cond_df, [prop])
            decoded = _generate_decode(
                diffusion,
                cond_df,
                aligner,
                args,
                compose_conditions_fn,
                config.samples_per_condition,
                config.decode_top_k,
                retrieval_index=retrieval_index,
                retrieval_config=retrieval_config,
            )
            decoded["benchmark_task"] = task_name
            decoded["constraint_properties"] = prop
            decoded["target_json"] = json.dumps({prop: value})
            decoded_rows.append(decoded)
            summary_rows.append(_summarize(decoded, train_df, task_name, prop, {prop: value}))
    return _concat(decoded_rows), pd.DataFrame(summary_rows)


def _multi_property(diffusion, train_df, eval_df, aligner, args, compose_conditions_fn, config, retrieval_index, retrieval_config):
    rng = np.random.default_rng(config.seed)
    props = list(SINGLE_PROPERTY_TARGETS)
    decoded_rows = []
    summary_rows = []
    for n_props in range(2, 8):
        cond_df = _condition_frame(eval_df, config.multi_conditions, config.seed + n_props)
        _set_neutral_image_features(cond_df, train_df)
        selected_sets = []
        for _ in range(len(cond_df)):
            selected = sorted(rng.choice(props, size=n_props, replace=False).tolist())
            selected_sets.append(selected)
        _assign_condition_masks(cond_df, selected_sets)
        decoded = _generate_decode(
            diffusion,
            cond_df,
            aligner,
            args,
            compose_conditions_fn,
            1,
            config.decode_top_k,
            retrieval_index=retrieval_index,
            retrieval_config=retrieval_config,
        )
        decoded["benchmark_task"] = "multi_property"
        decoded["constraint_properties"] = [",".join(selected_sets[int(row.condition_idx)]) for row in decoded.itertuples()]
        decoded["target_json"] = [
            json.dumps({p: float(cond_df.iloc[int(row.condition_idx)][p]) for p in selected_sets[int(row.condition_idx)]})
            for row in decoded.itertuples()
        ]
        decoded_rows.append(decoded)
        summary_rows.append(_summarize(decoded, train_df, "multi_property", f"{n_props}_properties", None))
    return _concat(decoded_rows), pd.DataFrame(summary_rows)


def _optimization(diffusion, train_df, eval_df, aligner, args, compose_conditions_fn, config, retrieval_index, retrieval_config):
    decoded_rows = []
    summary_rows = []
    for prop, delta in OPTIMIZATION_TASKS.items():
        cond_df = _condition_frame(eval_df, config.optimization_conditions, config.seed + int(abs(delta) * 10))
        _set_neutral_image_features(cond_df, train_df)
        before = cond_df[prop].astype(float).to_numpy()
        cond_df[prop] = np.clip(cond_df[prop].astype(float) + delta, 0.0 if prop in {"QED", "TPSA"} else -10.0, 700.0)
        _assign_condition_masks(cond_df, [prop])
        decoded = _generate_decode(
            diffusion,
            cond_df,
            aligner,
            args,
            compose_conditions_fn,
            1,
            config.decode_top_k,
            retrieval_index=retrieval_index,
            retrieval_config=retrieval_config,
        )
        decoded["benchmark_task"] = "property_optimization"
        decoded["constraint_properties"] = prop
        decoded["target_json"] = [
            json.dumps({prop: float(cond_df.iloc[int(row.condition_idx)][prop]), "delta": delta})
            for row in decoded.itertuples()
        ]
        decoded["before_property"] = [float(before[int(row.condition_idx)]) for row in decoded.itertuples()]
        decoded["requested_delta"] = float(delta)
        decoded_rows.append(decoded)
        summary = _summarize(decoded, train_df, "property_optimization", prop, None)
        actual_col = f"actual_{prop}"
        if actual_col in decoded:
            achieved = decoded[actual_col].astype(float).to_numpy() - decoded["before_property"].astype(float).to_numpy()
            summary["mean_achieved_delta"] = float(np.mean(achieved))
            summary["requested_delta"] = float(delta)
        summary_rows.append(summary)
    return _concat(decoded_rows), pd.DataFrame(summary_rows)


def _generate_decode(
    diffusion,
    cond_df,
    aligner,
    args,
    compose_conditions_fn,
    samples_per_condition,
    top_k,
    retrieval_index: RetrievalCandidateIndex | None = None,
    retrieval_config: RetrievalDecoderConfig | None = None,
):
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
        sampled_rows = diffusion.sample_batch(conditions, samples_per_condition=samples_per_condition)
    else:
        sampled_rows = []
        for condition_idx, condition in enumerate(conditions):
            for sample_idx, row in enumerate(diffusion.sample(condition[None, :], n=samples_per_condition)):
                sampled_rows.append((condition_idx, sample_idx, row))
    for condition_idx, sample_idx, row in sampled_rows:
        candidates = decode_retrieval_table_row(
            row,
            top_k=top_k,
            seed=int(condition_idx) * 1009 + int(sample_idx),
            mode=getattr(args, "decoder_mode", "physics"),
            index=retrieval_index,
            config=retrieval_config,
            include_dynamic=bool(getattr(args, "dynamic_decoder", False)),
        )
        for rank, candidate in enumerate(candidates, start=1):
            out = {
                "condition_idx": condition_idx,
                "sample_idx": sample_idx,
                "rank": rank,
                "smiles": candidate.smiles,
                "valid": candidate.valid,
                "decoder_score": candidate.score,
                "candidate_source": candidate.source,
            }
            out.update({f"target_{k}": v for k, v in row.items()})
            out.update({f"actual_{k}": v for k, v in candidate.descriptors.items() if isinstance(v, (int, float))})
            rows.append(out)
    return pd.DataFrame(rows)


def _summarize(decoded, train_df, task, label, targets):
    train_smiles = train_df["smiles"].astype(str).tolist()
    metrics = evaluate_smiles(decoded["smiles"].astype(str).tolist(), train_smiles=train_smiles)
    constrained = _constraint_list(label)
    if targets:
        constrained = list(targets)
    if "target_json" in decoded:
        success = _success_rate_from_target_json(decoded, STRICT_SUCCESS_TOLERANCE)
        sketchmol_success = _success_rate_from_target_json(decoded, SKETCHMOL_SUCCESS_TOLERANCE)
    else:
        success = _success_rate(decoded, constrained, STRICT_SUCCESS_TOLERANCE)
        sketchmol_success = _success_rate(decoded, constrained, SKETCHMOL_SUCCESS_TOLERANCE)
    row = {
        "benchmark_task": task,
        "benchmark_label": label,
        "success_rate_in_valid_mols": success,
        "success_rate_strict_in_valid_mols": success,
        "success_rate_sketchmol_tolerance_in_valid_mols": sketchmol_success,
    }
    row.update(metrics)
    for prop in constrained:
        requested_mae = _mae_from_target_json(decoded, prop)
        if not np.isnan(requested_mae):
            row[f"{prop}_mae"] = requested_mae
            plan_mae = _plan_mae_from_target_json(decoded, prop)
            if not np.isnan(plan_mae):
                row[f"{prop}_generated_table_mae"] = plan_mae
        else:
            target_col = f"target_{prop}"
            actual_col = f"actual_{prop}"
            if target_col in decoded and actual_col in decoded:
                row[f"{prop}_mae"] = float((decoded[actual_col] - decoded[target_col]).abs().mean())
    return row


def _success_rate_from_target_json(decoded, tolerance):
    valid = decoded[decoded["valid"].astype(bool)].copy()
    if valid.empty:
        return float("nan")
    ok = []
    for _, row in valid.iterrows():
        try:
            targets = json.loads(row["target_json"])
        except Exception:
            continue
        row_ok = True
        for prop, target in targets.items():
            if prop not in tolerance:
                continue
            actual_col = f"actual_{prop}"
            if actual_col not in valid:
                continue
            row_ok = row_ok and abs(float(row[actual_col]) - float(target)) <= tolerance.get(prop, 1.0)
        ok.append(row_ok)
    return float(np.mean(ok)) if ok else float("nan")


def _success_rate(decoded, props, tolerance):
    valid = decoded[decoded["valid"].astype(bool)].copy()
    if valid.empty or not props:
        return float("nan")
    ok = np.ones(len(valid), dtype=bool)
    for prop in props:
        target_col = f"target_{prop}"
        actual_col = f"actual_{prop}"
        if target_col not in valid or actual_col not in valid:
            continue
        tol = tolerance.get(prop, 1.0)
        ok &= (valid[actual_col].astype(float) - valid[target_col].astype(float)).abs().to_numpy() <= tol
    return float(ok.mean())


def _assign_condition_masks(cond_df, constrained):
    mask_cols = [f"condition_mask_{prop}" for prop in TARGET_COLUMNS]
    for col in mask_cols:
        cond_df[col] = 0.0
    if not constrained:
        return cond_df
    if isinstance(constrained[0], str):
        constrained_sets = [set(constrained) for _ in range(len(cond_df))]
    else:
        constrained_sets = [set(items) for items in constrained]
    for row_idx, props in enumerate(constrained_sets):
        for prop in props:
            if prop in TARGET_COLUMNS:
                cond_df.at[row_idx, f"condition_mask_{prop}"] = 1.0
    return cond_df


def _set_neutral_image_features(cond_df, train_df):
    for col in IMAGE_FEATURE_COLUMNS:
        if col in cond_df and col in train_df:
            cond_df[col] = float(train_df[col].median())
    return cond_df


def _mae_from_target_json(decoded, prop):
    if "target_json" not in decoded or f"actual_{prop}" not in decoded:
        return float("nan")
    valid = decoded[decoded["valid"].astype(bool)].copy()
    if valid.empty:
        return float("nan")
    errors = []
    for _, row in valid.iterrows():
        try:
            targets = json.loads(row["target_json"])
        except Exception:
            continue
        if prop not in targets:
            continue
        errors.append(abs(float(row[f"actual_{prop}"]) - float(targets[prop])))
    return float(np.mean(errors)) if errors else float("nan")


def _plan_mae_from_target_json(decoded, prop):
    if "target_json" not in decoded or f"target_{prop}" not in decoded:
        return float("nan")
    errors = []
    for _, row in decoded.iterrows():
        try:
            targets = json.loads(row["target_json"])
        except Exception:
            continue
        if prop not in targets:
            continue
        errors.append(abs(float(row[f"target_{prop}"]) - float(targets[prop])))
    return float(np.mean(errors)) if errors else float("nan")


def _distribution_matching(decoded, train_df):
    out = {}
    for prop in ["LogP", "QED", "MW", "TPSA"]:
        actual_col = f"actual_{prop}"
        if actual_col in decoded and prop in train_df:
            out[f"{prop}_wasserstein_approx"] = _wasserstein_1d(decoded[actual_col].dropna().to_numpy(float), train_df[prop].dropna().to_numpy(float))
    return out


def _wasserstein_1d(a, b, n_quantiles=200):
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    qs = np.linspace(0.0, 1.0, n_quantiles)
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def _condition_frame(eval_df, n, seed):
    if eval_df.empty:
        raise ValueError("Benchmark requires a non-empty evaluation dataframe.")
    return eval_df.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)


def _constraint_list(label):
    if isinstance(label, str) and "," in label:
        return [x for x in label.split(",") if x]
    if isinstance(label, str) and label in SUCCESS_TOLERANCE:
        return [label]
    return []


def _concat(parts):
    parts = [p for p in parts if p is not None and not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
