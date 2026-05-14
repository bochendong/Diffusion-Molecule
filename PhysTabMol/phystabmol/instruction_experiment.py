"""Train and evaluate an instruction-guided tabular diffusion edit planner."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .decoder import decode_table_row
from .diffusion import TabularDiffusion
from .features import table_row_from_smiles
from .instruction_evaluate import evaluate_instruction_candidates
from .instruction_features import (
    INSTRUCTION_SPEC_FEATURE_END,
    INSTRUCTION_SPEC_FEATURE_START,
    condition_from_source_and_spec,
    target_table_from_smiles,
)
from .instruction_mmp_decoder import MMPDecoderConfig, MMPTransformationIndex, decode_mmp_transform
from .instruction_multimodal import MULTIMODAL_CONTEXT_MODES, multimodal_context_from_row, multimodal_feature_names
from .instruction_schema import EDIT_RULES, PROPERTY_GOALS, normalize_spec, threshold
from .instruction_source_decoder import SourceAwareCandidateIndex, SourceAwareDecoderConfig, decode_source_aware
from .instruction_verifier import verify_instruction
from .io import make_run_dir, save_json, save_text, set_seed
from .mmp_transform_decoder import MMPTransformConfig, MMPTransformIndex, decode_mmp_table_row
from .schema import TABLE_COLUMNS


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_dir = make_run_dir(args.output_dir, args.run_name)
    save_json(vars(args), run_dir / "config.json")

    print(f"Loading dataset {args.dataset}...", flush=True)
    dataset = pd.read_csv(args.dataset)
    if args.limit:
        dataset = dataset.head(args.limit).reset_index(drop=True)
    train_df = dataset[dataset.get("split", "train") == "train"].reset_index(drop=True) if "split" in dataset else dataset
    eval_df = dataset[dataset["split"].isin(args.eval_splits)].reset_index(drop=True) if "split" in dataset else dataset
    if args.eval_limit:
        eval_df = eval_df.head(args.eval_limit).reset_index(drop=True)
    if train_df.empty or eval_df.empty:
        raise ValueError("Instruction dataset must contain non-empty train and eval splits.")

    print(
        f"Building feature arrays: train={len(train_df)} eval={len(eval_df)} "
        f"(RDKit per row; may take several minutes)...",
        flush=True,
    )
    train_x, train_y = _build_arrays(
        train_df,
        multimodal_context=args.multimodal_context,
        allow_target_reference=args.allow_target_reference,
        disable_instruction_features=args.disable_instruction_features,
    )
    eval_x, _ = _build_arrays(
        eval_df,
        multimodal_context=args.multimodal_context,
        allow_target_reference=args.allow_target_reference,
        disable_instruction_features=args.disable_instruction_features,
    )
    train_df.to_csv(run_dir / "tables" / "instruction_train.csv", index=False)
    eval_df.to_csv(run_dir / "tables" / "instruction_eval.csv", index=False)

    fit_name = "latent VAE diffusion" if args.latent_vae else "diffusion"
    print(f"Fitting {fit_name} (backend may be sklearn MLP on ~{len(train_df) * args.noise_repeats} synthetic steps)...", flush=True)
    diffusion, backend = _fit_diffusion(args, train_y, train_x)
    print(f"Fit done ({backend}).", flush=True)
    model_path = run_dir / "models" / ("instruction_diffusion.pt" if backend.startswith("torch") else "instruction_diffusion.pkl")
    diffusion.save(model_path)
    mmp_index = None
    if args.mmp_decoder:
        print("Building MMP-style transformation index from verified train pairs...", flush=True)
        mmp_index = MMPTransformationIndex.from_dataframe(train_df)
        print(f"MMP transformation index size={len(mmp_index.transforms)} verified pair transformations.", flush=True)
    source_index = None
    if args.source_aware_decoder:
        print("Building source-aware retrieval index from the train split...", flush=True)
        source_index = SourceAwareCandidateIndex.from_dataframe(train_df)
        print(f"Source-aware index size={len(source_index.candidates)} molecules.", flush=True)
    fragment_index = None
    if args.fragment_growth_decoder:
        fragment_library = Path(args.fragment_transform_library)
        if fragment_library.exists():
            print(f"Loading fragment-growth transform library from {fragment_library}...", flush=True)
            fragment_index = MMPTransformIndex.from_library_csv(fragment_library)
            print(
                f"Fragment-growth index size={len(fragment_index.pairs)} pairs, "
                f"{len(fragment_index.fragments)} fragments.",
                flush=True,
            )
        else:
            print(
                f"Fragment-growth transform library not found at {fragment_library}; "
                "continuing without fragment growth.",
                flush=True,
            )

    n_gen = len(eval_df) * args.samples_per_instruction
    print(
        f"Sampling {len(eval_df)} instructions x {args.samples_per_instruction} = {n_gen} plans "
        f"({args.timesteps} denoise steps each)...",
        flush=True,
    )
    generated_df = _sample_edit_plans(diffusion, eval_x, eval_df, args.samples_per_instruction)
    generated_df.to_csv(run_dir / "tables" / "generated_edit_plans.csv", index=False)
    print(f"Decoding {len(generated_df)} rows to SMILES (RDKit; often the slowest step)...", flush=True)
    decoded_df = _decode_and_attach(
        generated_df,
        eval_df,
        top_k=args.decode_top_k,
        dynamic_decoder=args.dynamic_decoder,
        mmp_index=mmp_index,
        mmp_config=MMPDecoderConfig(
            pool_size=args.mmp_pool_size,
            source_neighbors=args.mmp_source_neighbors,
            delta_neighbors=args.mmp_delta_neighbors,
            tag_neighbors=args.mmp_tag_neighbors,
            reference_neighbors=args.mmp_reference_neighbors,
            verify_candidates=args.mmp_verify_candidates,
        ),
        source_index=source_index,
        source_aware_config=SourceAwareDecoderConfig(
            pool_size=args.source_aware_pool_size,
            plan_neighbors=args.source_aware_plan_neighbors,
            source_neighbors=args.source_aware_source_neighbors,
            reference_neighbors=args.source_aware_reference_neighbors,
            verify_candidates=args.source_aware_verify_candidates,
        ),
        fragment_index=fragment_index,
        fragment_config=MMPTransformConfig(
            target_neighbors=args.fragment_pair_neighbors,
            delta_neighbors=args.fragment_pair_neighbors,
            source_neighbors=args.fragment_pair_neighbors,
            fragment_neighbors=args.fragment_neighbors,
            attachment_limit=args.fragment_attachment_limit,
            exact_train_penalty=args.fragment_exact_penalty,
            fragment_bonus=args.fragment_bonus,
            prompt_match_bonus=args.fragment_prompt_match_bonus,
            prompt_miss_penalty=args.fragment_prompt_miss_penalty,
            fragment_exact_penalty=args.fragment_exact_penalty,
            fragment_growth_steps=args.fragment_growth_steps,
            fragment_growth_beam_size=args.fragment_growth_beam_size,
            fragment_second_step_neighbors=args.fragment_second_step_neighbors,
            fragment_growth_mw_gap=args.fragment_growth_mw_gap,
        ),
        use_instruction_guided_plan=not args.disable_instruction_guided_plan,
    )
    decoded_df.to_csv(run_dir / "tables" / "decoded_instruction_candidates.csv", index=False)

    train_smiles = set(train_df["source_smiles"].astype(str)) | set(train_df["target_smiles"].astype(str))
    metrics, detailed = evaluate_instruction_candidates(decoded_df, train_smiles=train_smiles)
    detailed.to_csv(run_dir / "tables" / "verified_instruction_candidates.csv", index=False)
    metrics.update(
        {
            "backend": backend,
            "train_size": float(len(train_df)),
            "eval_size": float(len(eval_df)),
            "condition_dim": float(train_x.shape[1]),
            "multimodal_feature_dim": float(len(multimodal_feature_names(args.multimodal_context))),
            "mmp_decoder": bool(args.mmp_decoder),
            "source_aware_decoder": bool(args.source_aware_decoder),
            "fragment_growth_decoder": bool(fragment_index is not None),
            "fragment_transform_pairs": float(len(fragment_index.pairs)) if fragment_index is not None else 0.0,
            "fragment_transform_fragments": float(len(fragment_index.fragments)) if fragment_index is not None else 0.0,
            "instruction_guided_plan": bool(not args.disable_instruction_guided_plan),
            "latent_vae": bool(args.latent_vae),
            "vae_latent_dim": float(args.vae_latent_dim) if args.latent_vae else 0.0,
            "run_dir": str(run_dir),
        }
    )
    metrics.update(_candidate_source_metrics(decoded_df))
    save_json(metrics, run_dir / "metrics.json")
    save_json(_environment(), run_dir / "environment.json")
    save_text(_summary(metrics, args), run_dir / "summary.txt")
    print(_summary(metrics, args))
    print(f"run_dir={run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Instruction-guided molecular editing with tabular diffusion.")
    parser.add_argument("--dataset", default="data/instruction_editing.csv")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--run-name", default="instruction_editing")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval-splits", nargs="+", default=["valid", "test"])
    parser.add_argument("--eval-limit", type=int, default=2000)
    parser.add_argument("--backend", choices=["auto", "torch", "sklearn"], default="auto")
    parser.add_argument("--latent-vae", action="store_true", help="Train a table/molecule VAE and run diffusion in its latent space.")
    parser.add_argument("--samples-per-instruction", type=int, default=8)
    parser.add_argument("--decode-top-k", type=int, default=2)
    parser.add_argument("--dynamic-decoder", action="store_true")
    parser.add_argument("--disable-mmp-decoder", dest="mmp_decoder", action="store_false")
    parser.set_defaults(mmp_decoder=True)
    parser.add_argument("--mmp-pool-size", type=int, default=768)
    parser.add_argument("--mmp-source-neighbors", type=int, default=256)
    parser.add_argument("--mmp-delta-neighbors", type=int, default=256)
    parser.add_argument("--mmp-tag-neighbors", type=int, default=384)
    parser.add_argument("--mmp-reference-neighbors", type=int, default=128)
    parser.add_argument("--mmp-verify-candidates", type=int, default=512)
    parser.add_argument("--disable-source-aware-decoder", dest="source_aware_decoder", action="store_false")
    parser.set_defaults(source_aware_decoder=True)
    parser.add_argument("--source-aware-pool-size", type=int, default=256)
    parser.add_argument("--source-aware-plan-neighbors", type=int, default=128)
    parser.add_argument("--source-aware-source-neighbors", type=int, default=128)
    parser.add_argument("--source-aware-reference-neighbors", type=int, default=64)
    parser.add_argument("--source-aware-verify-candidates", type=int, default=192)
    parser.add_argument("--disable-fragment-growth-decoder", dest="fragment_growth_decoder", action="store_false")
    parser.set_defaults(fragment_growth_decoder=True)
    parser.add_argument("--fragment-transform-library", default="data/mmp_transform_library.csv")
    parser.add_argument(
        "--fragment-pair-neighbors",
        type=int,
        default=0,
        help="Number of mined pair targets to retrieve. Default 0 keeps the main decoder source-fragment based.",
    )
    parser.add_argument("--fragment-neighbors", type=int, default=16)
    parser.add_argument("--fragment-attachment-limit", type=int, default=3)
    parser.add_argument("--fragment-growth-steps", type=int, default=2)
    parser.add_argument("--fragment-growth-beam-size", type=int, default=12)
    parser.add_argument("--fragment-second-step-neighbors", type=int, default=6)
    parser.add_argument("--fragment-growth-mw-gap", type=float, default=25.0)
    parser.add_argument("--fragment-bonus", type=float, default=0.24)
    parser.add_argument("--fragment-exact-penalty", type=float, default=0.30)
    parser.add_argument("--fragment-prompt-match-bonus", type=float, default=3.0)
    parser.add_argument("--fragment-prompt-miss-penalty", type=float, default=10.0)
    parser.add_argument(
        "--disable-instruction-guided-plan",
        action="store_true",
        help="Ablation: decode raw diffusion plans without deterministic spec-guided clipping/count hints.",
    )
    parser.add_argument(
        "--disable-instruction-features",
        action="store_true",
        help="Ablation: keep source/target hints but zero goal/edit/constraint features.",
    )
    parser.add_argument("--multimodal-context", choices=MULTIMODAL_CONTEXT_MODES, default="none")
    parser.add_argument(
        "--allow-target-reference",
        action="store_true",
        help="Use target_smiles as reference_smiles when a dataset lacks reference_smiles. This is an oracle-reference setting.",
    )
    parser.add_argument("--timesteps", type=int, default=80)
    parser.add_argument("--noise-repeats", type=int, default=8)
    parser.add_argument("--torch-epochs", type=int, default=80)
    parser.add_argument("--torch-batch-size", type=int, default=1024)
    parser.add_argument("--torch-hidden-dim", type=int, default=1024)
    parser.add_argument("--torch-layers", type=int, default=6)
    parser.add_argument("--torch-lr", type=float, default=2e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--vae-latent-dim", type=int, default=16)
    parser.add_argument("--vae-hidden-dim", type=int, default=512)
    parser.add_argument("--vae-layers", type=int, default=3)
    parser.add_argument("--vae-epochs", type=int, default=60)
    parser.add_argument("--vae-batch-size", type=int, default=1024)
    parser.add_argument("--vae-lr", type=float, default=1e-3)
    parser.add_argument("--vae-beta", type=float, default=1e-3)
    parser.add_argument("--source-anchor-weight", type=float, default=0.35)
    parser.add_argument("--source-count-anchor-weight", type=float, default=0.15)
    parser.add_argument("--source-anchor-neighbors", type=int, default=32)
    parser.add_argument("--sklearn-hidden", type=int, nargs=2, default=(160, 160))
    parser.add_argument("--target-anchor", type=float, default=1.0)
    parser.add_argument("--anchor-neighbors", type=int, default=128)
    parser.add_argument("--count-anchor-weight", type=float, default=0.8)
    return parser.parse_args()


def _build_arrays(
    df: pd.DataFrame,
    multimodal_context: str = "none",
    allow_target_reference: bool = False,
    disable_instruction_features: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    conditions = []
    targets = []
    for _, row in df.iterrows():
        condition = condition_from_source_and_spec(row["source_smiles"], row["instruction_spec_json"])
        if disable_instruction_features:
            condition[INSTRUCTION_SPEC_FEATURE_START:INSTRUCTION_SPEC_FEATURE_END] = 0.0
        multimodal = multimodal_context_from_row(
            row,
            mode=multimodal_context,
            allow_target_reference=allow_target_reference,
        )
        if len(multimodal):
            condition = np.concatenate([condition, multimodal])
        conditions.append(condition)
        targets.append(target_table_from_smiles(row["target_smiles"]))
    return np.asarray(conditions, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def _fit_diffusion(args, train_y: np.ndarray, train_x: np.ndarray):
    target_start = len(TABLE_COLUMNS)
    if args.latent_vae:
        if args.backend not in {"auto", "torch"}:
            raise ValueError("--latent-vae requires --backend auto or --backend torch.")
        try:
            from .torch_latent_vae import TORCH_AVAILABLE, TorchLatentVAEDiffusion

            if not TORCH_AVAILABLE:
                raise RuntimeError("Requested --latent-vae but PyTorch is not installed.")
            model = TorchLatentVAEDiffusion(
                timesteps=args.timesteps,
                noise_repeats=args.noise_repeats,
                hidden_dim=args.torch_hidden_dim,
                layers=args.torch_layers,
                epochs=args.torch_epochs,
                batch_size=args.torch_batch_size,
                lr=args.torch_lr,
                seed=args.seed,
                device=args.device,
                target_condition_start=target_start,
                target_anchor=args.target_anchor,
                anchor_neighbors=args.anchor_neighbors,
                count_anchor_weight=args.count_anchor_weight,
                vae_latent_dim=args.vae_latent_dim,
                vae_hidden_dim=args.vae_hidden_dim,
                vae_layers=args.vae_layers,
                vae_epochs=args.vae_epochs,
                vae_batch_size=args.vae_batch_size,
                vae_lr=args.vae_lr,
                vae_beta=args.vae_beta,
                source_anchor_weight=args.source_anchor_weight,
                source_count_anchor_weight=args.source_count_anchor_weight,
                source_anchor_neighbors=args.source_anchor_neighbors,
            ).fit(train_y, train_x)
            return model, "torch_latent_vae"
        except Exception:
            raise
    if args.backend in {"auto", "torch"}:
        try:
            from .torch_diffusion import TORCH_AVAILABLE, TorchTabularDiffusion

            if TORCH_AVAILABLE:
                model = TorchTabularDiffusion(
                    timesteps=args.timesteps,
                    noise_repeats=args.noise_repeats,
                    hidden_dim=args.torch_hidden_dim,
                    layers=args.torch_layers,
                    epochs=args.torch_epochs,
                    batch_size=args.torch_batch_size,
                    lr=args.torch_lr,
                    seed=args.seed,
                    device=args.device,
                    target_condition_start=target_start,
                    target_anchor=args.target_anchor,
                    anchor_neighbors=args.anchor_neighbors,
                    count_anchor_weight=args.count_anchor_weight,
                ).fit(train_y, train_x)
                return model, "torch"
            if args.backend == "torch":
                raise RuntimeError("Requested --backend torch but PyTorch is not installed.")
        except Exception:
            if args.backend == "torch":
                raise

    model = TabularDiffusion(
        timesteps=args.timesteps,
        noise_repeats=args.noise_repeats,
        hidden=tuple(args.sklearn_hidden),
        seed=args.seed,
        target_condition_start=target_start,
        target_anchor=args.target_anchor,
        anchor_neighbors=args.anchor_neighbors,
        count_anchor_weight=args.count_anchor_weight,
    ).fit(train_y, train_x)
    return model, "sklearn"


def _sample_edit_plans(diffusion, conditions: np.ndarray, eval_df: pd.DataFrame, samples_per_instruction: int) -> pd.DataFrame:
    rows = []
    for instruction_idx, condition in enumerate(conditions):
        for sample_idx, row in enumerate(diffusion.sample(condition[None, :], n=samples_per_instruction)):
            out = {
                "instruction_id": instruction_idx,
                "pair_id": eval_df.iloc[instruction_idx].get("pair_id", instruction_idx),
                "sample_idx": sample_idx,
            }
            out.update(row)
            rows.append(out)
    return pd.DataFrame(rows)


def _decode_and_attach(
    generated_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    top_k: int,
    dynamic_decoder: bool = False,
    mmp_index: MMPTransformationIndex | None = None,
    mmp_config: MMPDecoderConfig | None = None,
    source_index: SourceAwareCandidateIndex | None = None,
    source_aware_config: SourceAwareDecoderConfig | None = None,
    fragment_index: MMPTransformIndex | None = None,
    fragment_config: MMPTransformConfig | None = None,
    use_instruction_guided_plan: bool = True,
) -> pd.DataFrame:
    rows = []
    for _, source in generated_df.iterrows():
        instruction_idx = int(source["instruction_id"])
        instruction = eval_df.iloc[instruction_idx]
        table_row = {col: float(source[col]) for col in TABLE_COLUMNS}
        decode_row = _instruction_guided_table_row(table_row, instruction) if use_instruction_guided_plan else table_row
        decode_seed = instruction_idx * 1009 + int(source["sample_idx"])
        verified_candidates = []
        if fragment_index is not None:
            verified_candidates.extend(
                decode_mmp_table_row(
                    decode_row,
                    top_k=max(top_k * 8, top_k),
                    seed=decode_seed,
                    index=fragment_index,
                    config=fragment_config,
                    prompt_smiles=str(instruction["source_smiles"]),
                )
            )
        if mmp_index is not None:
            verified_candidates.extend(
                decode_mmp_transform(
                    decode_row,
                    instruction,
                    mmp_index,
                    top_k=max(top_k * 2, top_k),
                    seed=decode_seed,
                    config=mmp_config,
                )
            )
        if source_index is not None:
            verified_candidates.extend(
                decode_source_aware(
                    decode_row,
                    instruction,
                    source_index,
                    top_k=max(top_k * 2, top_k),
                    seed=decode_seed,
                    config=source_aware_config,
                )
            )
        candidates = _rank_instruction_candidates(_dedupe_candidates(verified_candidates), instruction, top_k=top_k)
        if len(candidates) < top_k:
            candidates.extend(
                decode_table_row(
                    decode_row,
                    top_k=top_k - len(candidates),
                    seed=decode_seed,
                    include_dynamic=dynamic_decoder,
                )
            )
        ranked = _rank_instruction_candidates(_dedupe_candidates(candidates), instruction, top_k=top_k)
        for rank, candidate in enumerate(ranked, start=1):
            out = {
                "instruction_id": instruction_idx,
                "pair_id": source["pair_id"],
                "sample_idx": int(source["sample_idx"]),
                "rank": rank,
                "source_smiles": instruction["source_smiles"],
                "target_smiles": instruction["target_smiles"],
                "candidate_smiles": candidate.smiles,
                "instruction_text": instruction["instruction_text"],
                "instruction_spec_json": instruction["instruction_spec_json"],
                "decoder_score": candidate.score,
                "decoder_valid": candidate.valid,
                "candidate_source": candidate.source,
            }
            out.update({f"sampled_plan_{key}": value for key, value in table_row.items()})
            out.update({f"plan_{key}": value for key, value in decode_row.items()})
            out.update({f"actual_{key}": value for key, value in candidate.descriptors.items() if isinstance(value, (int, float))})
            rows.append(out)
    return pd.DataFrame(rows)


def _instruction_guided_table_row(table_row: dict[str, float], instruction: pd.Series) -> dict[str, float]:
    """Clip a sampled plan toward the executable instruction without using the target molecule."""

    out = {col: float(table_row.get(col, 0.0)) for col in TABLE_COLUMNS}
    try:
        source_row = table_row_from_smiles(str(instruction["source_smiles"]))
        spec = normalize_spec(str(instruction["instruction_spec_json"]))
    except Exception:
        return out

    for goal in spec["goals"]:
        rule = PROPERTY_GOALS.get(goal)
        if rule is None:
            continue
        col = str(rule["column"])
        required = threshold(spec, str(rule["threshold"]), float(rule["default"]))
        target_value = float(source_row[col]) + float(rule["direction"]) * required
        if float(rule["direction"]) > 0:
            out[col] = max(float(out.get(col, 0.0)), target_value)
        else:
            out[col] = min(float(out.get(col, 0.0)), target_value)

    if "keep_mw_similar" in spec["constraints"]:
        max_delta = threshold(spec, "delta_mw_abs_max")
        out["MW"] = float(np.clip(out.get("MW", source_row["MW"]), source_row["MW"] - max_delta, source_row["MW"] + max_delta))

    for edit in spec["edits"]:
        rule = EDIT_RULES.get(edit)
        if rule is None:
            continue
        required = threshold(spec, str(rule["threshold"]), float(rule["default"]))
        direction = float(rule["direction"])
        if "columns" in rule:
            columns = [str(col) for col in rule["columns"]]
            if direction > 0:
                primary = "N" if "N" in columns else columns[0]
                out[primary] = max(float(out.get(primary, 0.0)), float(source_row[primary]) + required)
            else:
                remaining = required
                for col in sorted(columns, key=lambda name: float(source_row.get(name, 0.0)), reverse=True):
                    target_value = max(0.0, float(source_row[col]) - remaining)
                    out[col] = min(float(out.get(col, 0.0)), target_value)
                    remaining = max(0.0, remaining - max(0.0, float(source_row[col]) - target_value))
        else:
            col = str(rule["column"])
            target_value = float(source_row.get(col, 0.0)) + direction * required
            if direction > 0:
                out[col] = max(float(out.get(col, 0.0)), target_value)
            else:
                out[col] = min(float(out.get(col, 0.0)), max(0.0, target_value))
        _apply_edit_count_hints(out, source_row, edit, required)

    return out


def _apply_edit_count_hints(out: dict[str, float], source_row: dict[str, float], edit: str, required: float) -> None:
    if edit == "add_halogen":
        out["fg_halogen"] = max(float(out.get("fg_halogen", 0.0)), float(source_row["fg_halogen"]) + required)
        out["F"] = max(float(out.get("F", 0.0)), float(source_row["F"]) + required)
    elif edit == "remove_halogen":
        out["fg_halogen"] = min(float(out.get("fg_halogen", 0.0)), max(0.0, float(source_row["fg_halogen"]) - required))
        for col in ("F", "Cl", "Br", "I"):
            out[col] = min(float(out.get(col, 0.0)), float(source_row[col]))
    elif edit == "add_amide":
        out["N"] = max(float(out.get("N", 0.0)), float(source_row["N"]) + required)
        out["O"] = max(float(out.get("O", 0.0)), float(source_row["O"]) + required)
    elif edit == "add_ester":
        out["O"] = max(float(out.get("O", 0.0)), float(source_row["O"]) + 2.0 * required)
    elif edit == "add_amine":
        out["N"] = max(float(out.get("N", 0.0)), float(source_row["N"]) + required)
    elif edit == "add_alcohol":
        out["O"] = max(float(out.get("O", 0.0)), float(source_row["O"]) + required)


def _rank_instruction_candidates(candidates, instruction: pd.Series, top_k: int):
    if not candidates:
        return []
    source_smiles = str(instruction["source_smiles"])
    spec_json = str(instruction["instruction_spec_json"])
    ranked = []
    for candidate in candidates:
        result = verify_instruction(source_smiles, candidate.smiles, spec_json)
        ranked.append((_instruction_rank_key(result, candidate), candidate))
    ranked.sort(key=lambda item: item[0])
    return [candidate for _, candidate in ranked[:top_k]]


def _instruction_rank_key(result: dict[str, object], candidate) -> tuple[float, ...]:
    passed_groups = sum(
        1.0
        for key in ("goal_success", "edit_success", "constraint_success")
        if result.get(key)
    )
    return (
        0.0 if result.get("overall_success") else 1.0,
        -passed_groups,
        0.0 if result.get("goal_success") else 1.0,
        0.0 if result.get("edit_success") else 1.0,
        0.0 if result.get("constraint_success") else 1.0,
        0.0 if result.get("valid") else 1.0,
        -float(result.get("similarity_to_source", 0.0) or 0.0),
        float(candidate.score),
    )


def _dedupe_candidates(candidates):
    best = {}
    order = []
    seen = set()
    for candidate in candidates:
        if candidate.smiles not in seen:
            best[candidate.smiles] = candidate
            order.append(candidate.smiles)
            seen.add(candidate.smiles)
        elif candidate.score < best[candidate.smiles].score:
            best[candidate.smiles] = candidate
    return [best[smiles] for smiles in order]


def _candidate_source_metrics(decoded_df: pd.DataFrame) -> dict[str, float]:
    if "candidate_source" not in decoded_df.columns or decoded_df.empty:
        return {}
    sources = decoded_df["candidate_source"].fillna("").astype(str)
    metrics = {
        "fragment_growth_candidate_fraction": float(sources.str.contains("fragment_grow_decoder").mean()),
        "two_step_fragment_candidate_fraction": float((sources == "mmp_two_step_fragment_grow_decoder").mean()),
        "mmp_pair_target_candidate_fraction": float((sources == "mmp_transform_target_decoder").mean()),
    }
    for source, value in sources.value_counts(normalize=True).head(8).items():
        key = "".join(ch if ch.isalnum() else "_" for ch in source.strip().lower()).strip("_")
        if key:
            metrics[f"candidate_source_fraction_{key}"] = float(value)
    return metrics


def _summary(metrics: dict[str, float], args: argparse.Namespace) -> str:
    keys = [
        "backend",
        "train_size",
        "eval_size",
        "n",
        "validity",
        "goal_success_rate",
        "constraint_success_rate",
        "edit_success_rate",
        "overall_instruction_success_rate",
        "overall_success_at_1_by_instruction",
        "overall_success_at_5_by_instruction",
        "overall_success_at_10_by_instruction",
        "sketchmol_local_edit_success_rate",
        "local_edit_success_at_5_by_instruction",
        "similarity_to_source",
        "target_similarity",
        "druglike_rate",
        "novelty",
        "exact_train_hit_rate",
        "sampled_nearest_train_tanimoto",
        "sampled_novelty_at_tanimoto_0_90",
        "fragment_growth_candidate_fraction",
        "two_step_fragment_candidate_fraction",
    ]
    lines = ["PhysTabMol instruction-editing experiment complete", f"dataset={args.dataset}"]
    lines.append(f"multimodal_context={args.multimodal_context}")
    lines.append(f"mmp_decoder={args.mmp_decoder}")
    lines.append(f"source_aware_decoder={args.source_aware_decoder}")
    lines.append(f"fragment_growth_decoder={args.fragment_growth_decoder}")
    lines.append(f"instruction_guided_plan={not args.disable_instruction_guided_plan}")
    lines.append(f"latent_vae={args.latent_vae}")
    if args.latent_vae:
        lines.append(f"vae_latent_dim={args.vae_latent_dim}")
    for key in keys:
        if key in metrics:
            value = metrics[key]
            lines.append(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}")
    return "\n".join(lines) + "\n"


def _environment() -> dict[str, object]:
    env = {"python": sys.version, "platform": platform.platform()}
    try:
        import rdkit

        env["rdkit"] = rdkit.__version__
    except Exception:
        env["rdkit"] = None
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["cuda_device_name"] = torch.cuda.get_device_name(0)
            env["cuda_max_memory_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 2)
    except Exception:
        env["torch"] = None
        env["cuda_available"] = False
    return env


if __name__ == "__main__":
    main()
