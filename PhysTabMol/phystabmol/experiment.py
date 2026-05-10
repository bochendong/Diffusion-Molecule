"""Server-oriented PhysTabMol training and evaluation entrypoint."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .contrastive import ContrastiveAligner
from .context import INTENT_DELTAS, InContextConditioner
from .dataset import arrays_from_dataframe, load_experiment_dataframe, train_test_split_df
from .diffusion import TabularDiffusion
from .evaluate import evaluate_decoded_table
from .features import IMAGE_FEATURE_COLUMNS, extract_image_features
from .geometry3d import add_3d_metrics
from .io import make_run_dir, save_json, save_text, set_seed
from .mmp_transform_decoder import MMPTransformConfig, MMPTransformIndex
from .pretrained_understanding import PretrainedImageUnderstanding
from .retrieval_decoder import RetrievalCandidateIndex, RetrievalDecoderConfig, decode_retrieval_table_row
from .schema import TARGET_COLUMNS
from .sketchmol_benchmark import SketchMolBenchmarkConfig, run_sketchmol_benchmark
from .structure_prompt_benchmark import StructurePromptBenchmarkConfig, run_structure_prompt_benchmark
from .understanding import UnderstandingStream, understanding_matrix


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_dir = make_run_dir(args.output_dir, args.run_name)
    save_json(vars(args), run_dir / "config.json")

    df = load_experiment_dataframe(
        data_path=args.data,
        smiles_column=args.smiles_column,
        image_column=args.image_column,
        limit=args.limit,
    )
    train_df, test_df = train_test_split_df(df, test_fraction=args.test_fraction, seed=args.seed)
    if args.property_mask_conditioning:
        setattr(args, "_property_mask_unset_values", {col: float(train_df[col].median()) for col in TARGET_COLUMNS})
    train_df.to_csv(run_dir / "tables" / "train_table.csv", index=False)
    test_df.to_csv(run_dir / "tables" / "test_table.csv", index=False)

    eval_df = test_df if not test_df.empty else train_df
    train_image_x, _, train_base_condition_x, train_table_y = arrays_from_dataframe(train_df)
    eval_image_x, _, eval_base_condition_x, _ = arrays_from_dataframe(eval_df)

    aligner = ContrastiveAligner(
        embedding_dim=args.embedding_dim,
        temperature=args.contrastive_temperature,
        lr=args.contrastive_lr,
        epochs=args.contrastive_epochs,
        batch_size=args.contrastive_batch_size,
        max_pairs=args.contrastive_max_pairs if args.contrastive_max_pairs > 0 else None,
        retrieval_eval_size=args.contrastive_retrieval_samples,
        seed=args.seed,
    ).fit(train_image_x, train_table_y)
    aligner.save(run_dir / "models" / "contrastive_aligner.pkl")

    train_condition, train_understanding_df = _compose_conditions(
        train_df,
        train_base_condition_x,
        train_image_x,
        aligner,
        args,
        intent="default",
        reference_smiles=None,
        use_in_context=False,
        property_mask_mode="train",
    )
    eval_condition, eval_understanding_df = _compose_conditions(
        eval_df,
        eval_base_condition_x,
        eval_image_x,
        aligner,
        args,
        intent=args.intent,
        reference_smiles=args.reference_smiles,
        use_in_context=bool(args.reference_smiles or args.reference_image or args.intent != "default"),
        property_mask_mode="full",
    )
    train_understanding_df.to_csv(run_dir / "tables" / "understanding_stream_train.csv", index=False)
    eval_understanding_df.to_csv(run_dir / "tables" / "understanding_stream_eval.csv", index=False)

    diffusion, backend = _fit_diffusion(args, train_table_y, train_condition)
    model_path = run_dir / "models" / ("diffusion.pt" if backend == "torch" else "diffusion.pkl")
    diffusion.save(model_path)
    retrieval_index = _build_retrieval_index(train_df, args)
    retrieval_config = _retrieval_config(args)
    mmp_config = _mmp_transform_config(args)
    mmp_index = _build_mmp_transform_index(train_df, args, mmp_config)

    generated_df = _sample_tables(diffusion, eval_condition, args.samples_per_condition)
    generated_df.to_csv(run_dir / "tables" / "generated_table_rows.csv", index=False)

    decoded_df = _decode_generated(
        generated_df,
        top_k=args.decode_top_k,
        dynamic_decoder=args.dynamic_decoder,
        decoder_mode=args.decoder_mode,
        retrieval_index=retrieval_index,
        retrieval_config=retrieval_config,
        mmp_index=mmp_index,
        mmp_config=mmp_config,
    )
    if args.enable_3d:
        sdf_path = run_dir / "tables" / "decoded_candidates_3d.sdf" if args.save_3d_sdf else None
        decoded_df = add_3d_metrics(decoded_df, sdf_path=sdf_path, max_sdf=args.max_3d_sdf)
    decoded_df.to_csv(run_dir / "tables" / "decoded_candidates.csv", index=False)

    metrics = evaluate_decoded_table(decoded_df, train_smiles=train_df["smiles"].tolist())
    if args.enable_3d and "3d_embed_success" in decoded_df:
        metrics["3d_embed_success_rate"] = float(decoded_df["3d_embed_success"].mean())
    metrics.update(
        {
            "backend": backend,
            "train_size": float(len(train_df)),
            "test_size": float(len(test_df)),
            "decoder_mode": args.decoder_mode,
            "retrieval_candidates": float(len(retrieval_index.candidates)) if retrieval_index is not None else 0.0,
            "mmp_transform_pairs": float(len(mmp_index.pairs)) if mmp_index is not None else 0.0,
            "mmp_transform_fragments": float(len(mmp_index.fragments)) if mmp_index is not None else 0.0,
            "property_mask_conditioning": bool(args.property_mask_conditioning),
            "property_mask_dim": float(len(TARGET_COLUMNS)) if args.property_mask_conditioning else 0.0,
            "contrastive_train_retrieval_accuracy": aligner.retrieval_accuracy(train_image_x, train_table_y),
            "run_dir": str(run_dir),
        }
    )
    if args.run_sketchmol_benchmark:
        benchmark_dir = run_dir / "tables" / "sketchmol_benchmark"
        benchmark_decoded, benchmark_summary = run_sketchmol_benchmark(
            diffusion=diffusion,
            train_df=train_df,
            eval_df=eval_df,
            aligner=aligner,
            args=args,
            compose_conditions_fn=_compose_conditions,
            output_dir=benchmark_dir,
            config=SketchMolBenchmarkConfig(
                single_conditions=args.benchmark_single_conditions,
                samples_per_condition=args.benchmark_samples_per_condition,
                decode_top_k=args.benchmark_decode_top_k,
                multi_conditions=args.benchmark_multi_conditions,
                optimization_conditions=args.benchmark_optimization_conditions,
                seed=args.seed,
            ),
            retrieval_index=retrieval_index,
            retrieval_config=retrieval_config,
            mmp_index=mmp_index,
            mmp_config=mmp_config,
        )
        if args.enable_3d and not benchmark_decoded.empty:
            benchmark_3d = add_3d_metrics(
                benchmark_decoded,
                sdf_path=benchmark_dir / "sketchmol_benchmark_3d.sdf" if args.save_3d_sdf else None,
                max_sdf=args.max_3d_sdf,
            )
            benchmark_3d.to_csv(benchmark_dir / "sketchmol_benchmark_decoded_3d.csv", index=False)
        if not benchmark_summary.empty:
            metrics["sketchmol_benchmark_mean_success"] = float(benchmark_summary["success_rate_in_valid_mols"].dropna().mean())
            if "success_rate_sketchmol_tolerance_in_valid_mols" in benchmark_summary:
                metrics["sketchmol_benchmark_mean_success_sketchmol_tolerance"] = float(
                    benchmark_summary["success_rate_sketchmol_tolerance_in_valid_mols"].dropna().mean()
                )
    if args.run_structure_prompt_benchmark:
        structure_dir = run_dir / "tables" / "structure_prompt_benchmark"
        _, structure_summary = run_structure_prompt_benchmark(
            diffusion=diffusion,
            train_df=train_df,
            eval_df=eval_df,
            aligner=aligner,
            args=args,
            compose_conditions_fn=_compose_conditions,
            output_dir=structure_dir,
            config=StructurePromptBenchmarkConfig(
                conditions_per_task=args.structure_prompt_conditions,
                samples_per_prompt=args.structure_prompt_samples,
                decode_top_k=args.structure_prompt_decode_top_k,
                seed=args.seed,
            ),
            retrieval_index=retrieval_index,
            retrieval_config=retrieval_config,
            mmp_index=mmp_index,
            mmp_config=mmp_config,
        )
        if not structure_summary.empty:
            metrics["structure_prompt_mean_joint_success_strict"] = float(structure_summary["joint_success_strict"].dropna().mean())
            metrics["structure_prompt_mean_joint_success_sketchmol_tolerance"] = float(
                structure_summary["joint_success_sketchmol_tolerance"].dropna().mean()
            )
    save_json(metrics, run_dir / "metrics.json")
    save_text(_summary(metrics, args), run_dir / "summary.txt")
    save_json(_environment(), run_dir / "environment.json")

    print(_summary(metrics, args))
    print(f"run_dir={run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate PhysTabMol on a server.")
    parser.add_argument("--data", type=str, default=None, help="CSV dataset. If omitted, uses built-in smoke-test molecules.")
    parser.add_argument("--smiles-column", type=str, default="smiles")
    parser.add_argument("--image-column", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default="phystabmol")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--backend", choices=["auto", "torch", "sklearn"], default="auto")
    parser.add_argument("--samples-per-condition", type=int, default=16)
    parser.add_argument("--decode-top-k", type=int, default=5)
    parser.add_argument("--dynamic-decoder", action="store_true")
    parser.add_argument(
        "--decoder-mode",
        choices=["physics", "retrieval", "hybrid", "mmp", "hybrid_mmp"],
        default="physics",
        help="SMILES decoder to use after tabular diffusion.",
    )
    parser.add_argument("--retrieval-neighbors", type=int, default=256)
    parser.add_argument("--retrieval-edit-neighbors", type=int, default=12)
    parser.add_argument("--retrieval-max-candidates", type=int, default=100000)
    parser.add_argument("--retrieval-exact-penalty", type=float, default=0.08)
    parser.add_argument("--retrieval-edit-bonus", type=float, default=0.08)
    parser.add_argument("--retrieval-prompt-match-bonus", type=float, default=1.5)
    parser.add_argument("--retrieval-prompt-miss-penalty", type=float, default=4.0)
    parser.add_argument("--mmp-transform-library", type=str, default=None)
    parser.add_argument("--mmp-max-pairs", type=int, default=80000)
    parser.add_argument("--mmp-pairs-per-source", type=int, default=6)
    parser.add_argument("--mmp-min-pair-similarity", type=float, default=0.25)
    parser.add_argument("--mmp-max-pair-similarity", type=float, default=0.98)
    parser.add_argument("--mmp-target-neighbors", type=int, default=384)
    parser.add_argument("--mmp-delta-neighbors", type=int, default=256)
    parser.add_argument("--mmp-source-neighbors", type=int, default=128)
    parser.add_argument("--mmp-fragment-neighbors", type=int, default=12)
    parser.add_argument("--mmp-attachment-limit", type=int, default=4)
    parser.add_argument("--mmp-max-fragments", type=int, default=12000)
    parser.add_argument("--mmp-max-fragment-atoms", type=int, default=8)
    parser.add_argument("--mmp-exact-penalty", type=float, default=0.30)
    parser.add_argument("--mmp-transform-bonus", type=float, default=0.10)
    parser.add_argument("--mmp-fragment-bonus", type=float, default=0.16)
    parser.add_argument("--mmp-prompt-match-bonus", type=float, default=2.5)
    parser.add_argument("--mmp-prompt-miss-penalty", type=float, default=8.0)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--contrastive-epochs", type=int, default=600)
    parser.add_argument("--contrastive-batch-size", type=int, default=512)
    parser.add_argument("--contrastive-max-pairs", type=int, default=20000)
    parser.add_argument("--contrastive-retrieval-samples", type=int, default=2048)
    parser.add_argument("--contrastive-lr", type=float, default=0.04)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--noise-repeats", type=int, default=16)
    parser.add_argument("--torch-epochs", type=int, default=200)
    parser.add_argument("--torch-batch-size", type=int, default=1024)
    parser.add_argument("--torch-hidden-dim", type=int, default=1024)
    parser.add_argument("--torch-layers", type=int, default=6)
    parser.add_argument("--torch-lr", type=float, default=2e-4)
    parser.add_argument("--sample-chunk-size", type=int, default=8192)
    parser.add_argument("--target-anchor", type=float, default=1.0)
    parser.add_argument("--anchor-neighbors", type=int, default=128)
    parser.add_argument("--count-anchor-weight", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--sklearn-hidden", type=int, nargs=2, default=(160, 160))
    parser.add_argument(
        "--property-mask-conditioning",
        action="store_true",
        help="Append property-present mask bits and randomly hide target properties during training, matching SketchMol None-token style conditioning.",
    )
    parser.add_argument("--property-mask-min-properties", type=int, default=1)
    parser.add_argument("--property-mask-max-properties", type=int, default=7)
    parser.add_argument("--reference-image", type=str, default=None)
    parser.add_argument("--reference-smiles", type=str, default=None)
    parser.add_argument("--intent", choices=sorted(INTENT_DELTAS), default="default")
    parser.add_argument("--disable-understanding-stream", action="store_true")
    parser.add_argument("--understanding-backbone", choices=["handcrafted", "clip"], default="handcrafted")
    parser.add_argument("--understanding-model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--understanding-batch-size", type=int, default=64)
    parser.add_argument("--run-sketchmol-benchmark", action="store_true")
    parser.add_argument("--benchmark-single-conditions", type=int, default=125)
    parser.add_argument("--benchmark-samples-per-condition", type=int, default=100)
    parser.add_argument("--benchmark-decode-top-k", type=int, default=1)
    parser.add_argument("--benchmark-multi-conditions", type=int, default=200)
    parser.add_argument("--benchmark-optimization-conditions", type=int, default=100)
    parser.add_argument("--run-structure-prompt-benchmark", action="store_true")
    parser.add_argument("--structure-prompt-conditions", type=int, default=200)
    parser.add_argument("--structure-prompt-samples", type=int, default=8)
    parser.add_argument("--structure-prompt-decode-top-k", type=int, default=2)
    parser.add_argument("--enable-3d", action="store_true")
    parser.add_argument("--save-3d-sdf", action="store_true")
    parser.add_argument("--max-3d-sdf", type=int, default=500)
    return parser.parse_args()


def _fit_diffusion(args, train_table_y: np.ndarray, train_condition: np.ndarray):
    backend = args.backend
    if backend in {"auto", "torch"}:
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
                    target_anchor=args.target_anchor,
                    anchor_neighbors=args.anchor_neighbors,
                    count_anchor_weight=args.count_anchor_weight,
                    sample_chunk_size=args.sample_chunk_size,
                ).fit(train_table_y, train_condition)
                return model, "torch"
            if backend == "torch":
                raise RuntimeError("Requested --backend torch but PyTorch is not available.")
        except Exception:
            if backend == "torch":
                raise

    model = TabularDiffusion(
        timesteps=args.timesteps,
        noise_repeats=args.noise_repeats,
        hidden=tuple(args.sklearn_hidden),
        seed=args.seed,
        target_anchor=args.target_anchor,
        anchor_neighbors=args.anchor_neighbors,
        count_anchor_weight=args.count_anchor_weight,
    ).fit(train_table_y, train_condition)
    return model, "sklearn"


def _sample_tables(diffusion, conditions: np.ndarray, samples_per_condition: int) -> pd.DataFrame:
    rows = []
    if hasattr(diffusion, "sample_batch"):
        for condition_idx, sample_idx, row in diffusion.sample_batch(conditions, samples_per_condition=samples_per_condition):
            out = {"condition_idx": condition_idx, "sample_idx": sample_idx}
            out.update(row)
            rows.append(out)
        return pd.DataFrame(rows)
    for condition_idx, condition in enumerate(conditions):
        samples = diffusion.sample(condition[None, :], n=samples_per_condition)
        for sample_idx, row in enumerate(samples):
            out = {"condition_idx": condition_idx, "sample_idx": sample_idx}
            out.update(row)
            rows.append(out)
    return pd.DataFrame(rows)


def _decode_generated(
    generated_df: pd.DataFrame,
    top_k: int,
    dynamic_decoder: bool = False,
    decoder_mode: str = "physics",
    retrieval_index: RetrievalCandidateIndex | None = None,
    retrieval_config: RetrievalDecoderConfig | None = None,
    mmp_index: MMPTransformIndex | None = None,
    mmp_config: MMPTransformConfig | None = None,
) -> pd.DataFrame:
    rows = []
    table_cols = [col for col in generated_df.columns if col not in {"condition_idx", "sample_idx"}]
    for _, source in generated_df.iterrows():
        row = {col: float(source[col]) for col in table_cols}
        decode_seed = int(source["condition_idx"]) * 1009 + int(source["sample_idx"])
        for rank, candidate in enumerate(
            decode_retrieval_table_row(
                row,
                top_k=top_k,
                seed=decode_seed,
                mode=decoder_mode,
                index=retrieval_index,
                config=retrieval_config,
                include_dynamic=dynamic_decoder,
                mmp_index=mmp_index,
                mmp_config=mmp_config,
            ),
            start=1,
        ):
            out = {
                "condition_idx": int(source["condition_idx"]),
                "sample_idx": int(source["sample_idx"]),
                "rank": rank,
                "smiles": candidate.smiles,
                "decoder_score": candidate.score,
                "valid": candidate.valid,
                "candidate_source": candidate.source,
            }
            out.update({f"target_{k}": v for k, v in row.items()})
            out.update({f"actual_{k}": v for k, v in candidate.descriptors.items() if isinstance(v, (int, float))})
            rows.append(out)
    return pd.DataFrame(rows)


def _build_retrieval_index(train_df: pd.DataFrame, args) -> RetrievalCandidateIndex | None:
    if args.decoder_mode not in {"retrieval", "hybrid", "hybrid_mmp"}:
        return None
    return RetrievalCandidateIndex.from_dataframe(train_df, max_candidates=args.retrieval_max_candidates)


def _retrieval_config(args) -> RetrievalDecoderConfig:
    return RetrievalDecoderConfig(
        neighbors=args.retrieval_neighbors,
        edit_neighbors=args.retrieval_edit_neighbors,
        max_candidates=args.retrieval_max_candidates,
        exact_train_penalty=args.retrieval_exact_penalty,
        edit_bonus=args.retrieval_edit_bonus,
        prompt_match_bonus=args.retrieval_prompt_match_bonus,
        prompt_miss_penalty=args.retrieval_prompt_miss_penalty,
    )


def _build_mmp_transform_index(train_df: pd.DataFrame, args, config: MMPTransformConfig) -> MMPTransformIndex | None:
    if args.decoder_mode not in {"mmp", "hybrid_mmp"}:
        return None
    if args.mmp_transform_library:
        path = Path(args.mmp_transform_library)
        if path.exists():
            return MMPTransformIndex.from_library_csv(path)
    return MMPTransformIndex.from_dataframe(train_df, config=config)


def _mmp_transform_config(args) -> MMPTransformConfig:
    return MMPTransformConfig(
        max_pairs=args.mmp_max_pairs,
        pairs_per_source=args.mmp_pairs_per_source,
        min_pair_similarity=args.mmp_min_pair_similarity,
        max_pair_similarity=args.mmp_max_pair_similarity,
        target_neighbors=args.mmp_target_neighbors,
        delta_neighbors=args.mmp_delta_neighbors,
        source_neighbors=args.mmp_source_neighbors,
        fragment_neighbors=args.mmp_fragment_neighbors,
        attachment_limit=args.mmp_attachment_limit,
        max_fragments=args.mmp_max_fragments,
        max_fragment_atoms=args.mmp_max_fragment_atoms,
        exact_train_penalty=args.mmp_exact_penalty,
        transform_bonus=args.mmp_transform_bonus,
        fragment_bonus=args.mmp_fragment_bonus,
        prompt_match_bonus=args.mmp_prompt_match_bonus,
        prompt_miss_penalty=args.mmp_prompt_miss_penalty,
    )


def _compose_conditions(
    df: pd.DataFrame,
    base_condition_x: np.ndarray,
    image_x: np.ndarray,
    aligner: ContrastiveAligner,
    args,
    intent: str,
    reference_smiles: str | None,
    use_in_context: bool,
    property_mask_mode: str = "full",
) -> tuple[np.ndarray, pd.DataFrame]:
    if use_in_context:
        base_condition_x, targets_df = _build_in_context_base(df, args, intent, reference_smiles)
    else:
        targets_df = df[TARGET_COLUMNS].reset_index(drop=True)

    base_condition_x = _apply_property_mask_conditioning(df, base_condition_x, args, property_mask_mode)

    stream = UnderstandingStream(enabled=not args.disable_understanding_stream)
    understanding_df = stream.describe_dataframe(
        df,
        targets_df=targets_df,
        intent=intent if not args.disable_understanding_stream else "default",
        reference_smiles=reference_smiles if not args.disable_understanding_stream else None,
    )
    condition_parts = [base_condition_x]
    if not args.disable_understanding_stream:
        condition_parts.append(understanding_matrix(understanding_df))
    condition_parts.append(aligner.transform_image(image_x))
    condition = np.concatenate(condition_parts, axis=1)
    if args.understanding_backbone != "handcrafted":
        encoder = _get_pretrained_understanding_encoder(args)
        condition = np.concatenate([condition, encoder.encode_dataframe(df, image_column=args.image_column)], axis=1)
    return condition, understanding_df


def _apply_property_mask_conditioning(
    df: pd.DataFrame,
    base_condition_x: np.ndarray,
    args,
    mode: str,
) -> np.ndarray:
    if not getattr(args, "property_mask_conditioning", False):
        return base_condition_x
    base = np.asarray(base_condition_x, dtype=float).copy()
    mask = _property_mask_matrix(df, args, mode=mode)
    unset_values = getattr(args, "_property_mask_unset_values", None)
    if unset_values is None:
        unset_values = {col: float(df[col].median()) if col in df else 0.0 for col in TARGET_COLUMNS}
    target_start = len(IMAGE_FEATURE_COLUMNS)
    for idx, col in enumerate(TARGET_COLUMNS):
        base[:, target_start + idx] = np.where(
            mask[:, idx] > 0.5,
            base[:, target_start + idx],
            float(unset_values.get(col, 0.0)),
        )
    return np.concatenate([base, mask], axis=1)


def _property_mask_matrix(df: pd.DataFrame, args, mode: str) -> np.ndarray:
    mask_cols = [f"condition_mask_{col}" for col in TARGET_COLUMNS]
    if all(col in df.columns for col in mask_cols):
        return df[mask_cols].to_numpy(dtype=float)
    if mode == "train":
        min_props = max(1, min(len(TARGET_COLUMNS), int(args.property_mask_min_properties)))
        max_props = max(min_props, min(len(TARGET_COLUMNS), int(args.property_mask_max_properties)))
        rng = np.random.default_rng(int(args.seed) + 4049)
        mask = np.zeros((len(df), len(TARGET_COLUMNS)), dtype=float)
        for row_idx in range(len(df)):
            n_props = int(rng.integers(min_props, max_props + 1))
            chosen = rng.choice(len(TARGET_COLUMNS), size=n_props, replace=False)
            mask[row_idx, chosen] = 1.0
        return mask
    return np.ones((len(df), len(TARGET_COLUMNS)), dtype=float)


def _build_in_context_base(
    df: pd.DataFrame,
    args,
    intent: str,
    reference_smiles: str | None,
) -> tuple[np.ndarray, pd.DataFrame]:
    conditioner = InContextConditioner()
    conditions = []
    targets = []
    reference_features = extract_image_features(args.reference_image) if args.reference_image else None
    for _, row in df.iterrows():
        query_features = {col: float(row[col]) for col in IMAGE_FEATURE_COLUMNS}
        default_targets = {col: float(row[col]) for col in TARGET_COLUMNS}
        base, readable_targets = conditioner.build(
            query_image_features=query_features,
            default_targets=default_targets,
            reference_image_features=reference_features,
            reference_smiles=reference_smiles,
            intent=intent,
        )
        conditions.append(base[0])
        targets.append(readable_targets)
    return np.asarray(conditions, dtype=float), pd.DataFrame(targets)


def _with_alignment(condition_x: np.ndarray, image_embed: np.ndarray) -> np.ndarray:
    return np.concatenate([condition_x, image_embed], axis=1)


def _get_pretrained_understanding_encoder(args):
    cached = getattr(args, "_pretrained_understanding_encoder", None)
    if cached is not None:
        return cached
    if args.understanding_backbone == "clip":
        encoder = PretrainedImageUnderstanding(
            model_name=args.understanding_model,
            device=args.device,
            batch_size=args.understanding_batch_size,
        )
        setattr(args, "_pretrained_understanding_encoder", encoder)
        return encoder
    raise ValueError(f"Unsupported understanding backbone: {args.understanding_backbone}")


def _summary(metrics: dict, args: argparse.Namespace) -> str:
    keys = [
        "backend",
        "train_size",
        "test_size",
        "retrieval_candidates",
        "mmp_transform_pairs",
        "mmp_transform_fragments",
        "n",
        "validity",
        "uniqueness",
        "novelty",
        "druglike_rate",
        "mean_pairwise_tanimoto",
        "MW_mae",
        "LogP_mae",
        "QED_mae",
        "TPSA_mae",
        "sketchmol_benchmark_mean_success_sketchmol_tolerance",
        "structure_prompt_mean_joint_success_sketchmol_tolerance",
    ]
    lines = [
        "PhysTabMol experiment complete",
        f"intent={args.intent}",
        f"reference_smiles={args.reference_smiles}",
        f"understanding_stream={not args.disable_understanding_stream}",
        f"understanding_backbone={args.understanding_backbone}",
        f"property_mask_conditioning={args.property_mask_conditioning}",
        f"decoder_mode={args.decoder_mode}",
    ]
    for key in keys:
        if key in metrics:
            value = metrics[key]
            lines.append(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}")
    return "\n".join(lines) + "\n"


def _environment() -> dict:
    env = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            env["cuda_device_name"] = torch.cuda.get_device_name(0)
            env["cuda_max_memory_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 2)
            env["cuda_max_memory_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 1024**2, 2)
    except Exception:
        env["torch"] = None
        env["cuda_available"] = False
    try:
        import rdkit

        env["rdkit"] = rdkit.__version__
    except Exception:
        env["rdkit"] = None
    return env


if __name__ == "__main__":
    main()
