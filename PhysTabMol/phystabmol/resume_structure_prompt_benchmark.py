"""Resume only the structure-prompt benchmark for an existing run."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import pandas as pd

from .contrastive import ContrastiveAligner
from .diffusion import TabularDiffusion
from .experiment import _build_mmp_transform_index, _build_retrieval_index, _mmp_transform_config, _retrieval_config
from .io import save_json, save_text
from .schema import TARGET_COLUMNS
from .structure_prompt_benchmark import StructurePromptBenchmarkConfig, run_structure_prompt_benchmark


def main() -> None:
    cli = parse_args()
    run_dir = Path(cli.run_dir)
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config: {config_path}")
    args = Namespace(**json.loads(config_path.read_text()))
    _apply_overrides(args, cli)

    train_df = pd.read_csv(run_dir / "tables" / "train_table.csv")
    eval_df = pd.read_csv(run_dir / "tables" / "test_table.csv")
    if eval_df.empty:
        eval_df = train_df
    if getattr(args, "property_mask_conditioning", False):
        setattr(args, "_property_mask_unset_values", {col: float(train_df[col].median()) for col in TARGET_COLUMNS})

    aligner = ContrastiveAligner.load(run_dir / "models" / "contrastive_aligner.pkl")
    diffusion = _load_diffusion(run_dir, device=cli.device)
    retrieval_index = _build_retrieval_index(train_df, args)
    retrieval_config = _retrieval_config(args)
    mmp_config = _mmp_transform_config(args)
    mmp_index = _build_mmp_transform_index(train_df, args, mmp_config)

    output_dir = run_dir / "tables" / "structure_prompt_benchmark"
    _, structure_summary = run_structure_prompt_benchmark(
        diffusion=diffusion,
        train_df=train_df,
        eval_df=eval_df,
        aligner=aligner,
        args=args,
        compose_conditions_fn=_compose_conditions_proxy,
        output_dir=output_dir,
        config=StructurePromptBenchmarkConfig(
            conditions_per_task=int(args.structure_prompt_conditions),
            samples_per_prompt=int(args.structure_prompt_samples),
            decode_top_k=int(args.structure_prompt_decode_top_k),
            seed=int(args.seed),
        ),
        retrieval_index=retrieval_index,
        retrieval_config=retrieval_config,
        mmp_index=mmp_index,
        mmp_config=mmp_config,
    )

    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics.update(_metrics_from_existing_sketchmol(run_dir))
    if not structure_summary.empty:
        metrics["structure_prompt_mean_joint_success_strict"] = float(structure_summary["joint_success_strict"].dropna().mean())
        metrics["structure_prompt_mean_joint_success_sketchmol_tolerance"] = float(
            structure_summary["joint_success_sketchmol_tolerance"].dropna().mean()
        )
        for col in [
            "exact_train_hit_rate",
            "direct_train_decoder_fraction",
            "source_aware_edit_decoder_fraction",
            "mmp_fragment_decoder_fraction",
            "mmp_two_step_fragment_decoder_fraction",
            "sampled_nearest_train_tanimoto",
            "sampled_novelty_at_tanimoto_0_90",
        ]:
            if col in structure_summary:
                metrics[f"structure_prompt_mean_{col}"] = float(structure_summary[col].dropna().mean())
    metrics.update(
        {
            "postprocess_structure_prompt": True,
            "postprocess_run_dir": str(run_dir),
            "decoder_mode": getattr(args, "decoder_mode", None),
            "retrieval_candidates": float(len(retrieval_index.candidates)) if retrieval_index is not None else 0.0,
            "mmp_transform_pairs": float(len(mmp_index.pairs)) if mmp_index is not None else 0.0,
            "mmp_transform_fragments": float(len(mmp_index.fragments)) if mmp_index is not None else 0.0,
        }
    )
    save_json(metrics, metrics_path)
    save_text(_summary(metrics, run_dir), run_dir / "summary.txt")
    save_json(vars(cli), run_dir / "structure_prompt_postprocess_config.json")
    print(_summary(metrics, run_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume structure prompt benchmark for an existing PhysTabMol run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--structure-prompt-conditions", type=int, default=0, help="0 keeps the original run config.")
    parser.add_argument("--structure-prompt-samples", type=int, default=0, help="0 keeps the original run config.")
    parser.add_argument("--structure-prompt-decode-top-k", type=int, default=0, help="0 keeps the original run config.")
    parser.add_argument("--device", default=None, help="Override model device, e.g. cuda or cpu.")
    return parser.parse_args()


def _apply_overrides(args: Namespace, cli: argparse.Namespace) -> None:
    defaults = {
        "decoder_mode": "hybrid_mmp",
        "retrieval_neighbors": 256,
        "retrieval_edit_neighbors": 12,
        "retrieval_max_candidates": 100000,
        "retrieval_exact_penalty": 0.25,
        "retrieval_edit_bonus": 0.15,
        "retrieval_prompt_match_bonus": 1.5,
        "retrieval_prompt_miss_penalty": 4.0,
        "mmp_transform_library": None,
        "mmp_max_pairs": 80000,
        "mmp_pairs_per_source": 6,
        "mmp_min_pair_similarity": 0.25,
        "mmp_max_pair_similarity": 0.98,
        "mmp_target_neighbors": 384,
        "mmp_delta_neighbors": 256,
        "mmp_source_neighbors": 128,
        "mmp_fragment_neighbors": 12,
        "mmp_attachment_limit": 4,
        "mmp_max_fragments": 12000,
        "mmp_max_fragment_atoms": 8,
        "mmp_exact_penalty": 0.30,
        "mmp_transform_bonus": 0.10,
        "mmp_fragment_bonus": 0.16,
        "mmp_fragment_exact_penalty": 0.18,
        "mmp_fragment_growth_steps": 2,
        "mmp_fragment_growth_beam_size": 12,
        "mmp_fragment_second_step_neighbors": 6,
        "mmp_fragment_growth_mw_gap": 25.0,
        "mmp_prompt_match_bonus": 2.5,
        "mmp_prompt_miss_penalty": 8.0,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    if cli.structure_prompt_conditions:
        args.structure_prompt_conditions = cli.structure_prompt_conditions
    if cli.structure_prompt_samples:
        args.structure_prompt_samples = cli.structure_prompt_samples
    if cli.structure_prompt_decode_top_k:
        args.structure_prompt_decode_top_k = cli.structure_prompt_decode_top_k


def _load_diffusion(run_dir: Path, device: str | None = None):
    torch_path = run_dir / "models" / "diffusion.pt"
    sklearn_path = run_dir / "models" / "diffusion.pkl"
    if torch_path.exists():
        from .torch_diffusion import TorchTabularDiffusion

        return TorchTabularDiffusion.load(torch_path, device=device)
    if sklearn_path.exists():
        return TabularDiffusion.load(sklearn_path)
    raise FileNotFoundError(f"Missing diffusion model under {run_dir / 'models'}")


def _compose_conditions_proxy(*args, **kwargs):
    from .experiment import _compose_conditions

    return _compose_conditions(*args, **kwargs)


def _metrics_from_existing_sketchmol(run_dir: Path) -> dict:
    path = run_dir / "tables" / "sketchmol_benchmark" / "sketchmol_benchmark_summary.csv"
    if not path.exists():
        return {}
    summary = pd.read_csv(path)
    out = {}
    if "success_rate_in_valid_mols" in summary:
        out["sketchmol_benchmark_mean_success"] = float(summary["success_rate_in_valid_mols"].dropna().mean())
    if "success_rate_sketchmol_tolerance_in_valid_mols" in summary:
        out["sketchmol_benchmark_mean_success_sketchmol_tolerance"] = float(
            summary["success_rate_sketchmol_tolerance_in_valid_mols"].dropna().mean()
        )
    return out


def _summary(metrics: dict, run_dir: Path) -> str:
    keys = [
        "decoder_mode",
        "retrieval_candidates",
        "mmp_transform_pairs",
        "mmp_transform_fragments",
        "sketchmol_benchmark_mean_success_sketchmol_tolerance",
        "structure_prompt_mean_joint_success_strict",
        "structure_prompt_mean_joint_success_sketchmol_tolerance",
        "structure_prompt_mean_exact_train_hit_rate",
        "structure_prompt_mean_direct_train_decoder_fraction",
        "structure_prompt_mean_mmp_two_step_fragment_decoder_fraction",
        "structure_prompt_mean_sampled_novelty_at_tanimoto_0_90",
    ]
    lines = ["PhysTabMol structure-prompt postprocess complete", f"run_dir={run_dir}"]
    for key in keys:
        if key in metrics:
            value = metrics[key]
            lines.append(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
