"""Generate trajectory logs from SketchMol-style assets.

This module does not reimplement SketchMol. It defines the reproducible bridge
between SketchMol outputs and the memory-aware diffusion experiments:

1. image_path.csv after MolScribe/RDKit evaluation -> trajectory JSONL
2. opt_examples before/after tables -> small bootstrapped multi-step JSONL

The second path is a development bootstrap only; paper experiments should use
the first path with real iterative sampling/inpainting outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .data import SketchMolOptPairConfig, load_sketchmol_opt_rows
from .schema import (
    PROPERTY_NAMES,
    TrajectoryStep,
    compute_properties,
    finite_or_zero,
    property_delta,
    reward_from_delta,
    write_csv,
    write_jsonl,
)


def _task_name_from_row(row: dict[str, Any], default: str = "sketchmol") -> str:
    return str(row.get("task_name") or row.get("task") or row.get("condition_name") or default)


def _condition_from_row(row: dict[str, Any]) -> dict[str, Any]:
    condition: dict[str, Any] = {}
    for key, value in row.items():
        key_lower = key.lower()
        if key_lower.endswith("_setting") and str(value) not in ("", "nan", "None"):
            condition[key_lower.replace("_setting", "")] = finite_or_zero(value)
        elif key_lower in {"target", "target_property", "preset", "preset_str"} and str(value):
            condition[key_lower] = value
    if "molscribe_score" in row:
        condition["molscribe_score"] = finite_or_zero(row.get("molscribe_score"))
    return condition


def steps_from_smiles_sequence(
    trajectory_id: str,
    smiles_sequence: list[str],
    task_name: str,
    source: str,
    conditions: list[dict[str, Any]] | None = None,
    image_paths: list[str] | None = None,
    molscribe_scores: list[float | None] | None = None,
    edit_type: str = "",
) -> list[TrajectoryStep]:
    """Convert a SMILES sequence into trajectory log steps."""

    conditions = conditions or [{} for _ in smiles_sequence]
    image_paths = image_paths or ["" for _ in smiles_sequence]
    molscribe_scores = molscribe_scores or [None for _ in smiles_sequence]
    steps: list[TrajectoryStep] = []
    previous_props: dict[str, float] = {}
    previous_smiles = ""
    for index, smiles in enumerate(smiles_sequence):
        props, valid, failure = compute_properties(smiles)
        delta = property_delta(previous_props, props) if previous_props else {name: 0.0 for name in PROPERTY_NAMES}
        condition = dict(conditions[index] if index < len(conditions) else {})
        condition["current_tpsa"] = props.get("tpsa", 0.0)
        condition["previous_tpsa"] = previous_props.get("tpsa", props.get("tpsa", 0.0))
        reward = reward_from_delta(delta, task_name=task_name, condition=condition) if index > 0 else 0.0
        steps.append(
            TrajectoryStep(
                trajectory_id=trajectory_id,
                step=index,
                smiles=smiles,
                parent_smiles=previous_smiles,
                image_path=image_paths[index] if index < len(image_paths) else "",
                source=source,
                task_name=task_name,
                condition=condition,
                properties=props,
                delta_properties=delta,
                reward=reward,
                validity=valid,
                molscribe_score=molscribe_scores[index] if index < len(molscribe_scores) else None,
                failure_reason=failure,
                selected_next_action="continue" if index < len(smiles_sequence) - 1 else "terminal",
                edit_type=edit_type or task_name,
            )
        )
        previous_props = props
        previous_smiles = smiles
    return steps


def bootstrap_from_opt_examples(
    opt_examples_dir: str | Path = SketchMolOptPairConfig.opt_examples_dir,
    output_jsonl: str | Path = "outputs/trajectories/sketchmol_opt_bootstrap.jsonl",
    output_csv: str | Path | None = None,
    max_pairs_per_task: int | None = None,
    steps_per_trajectory: int = 5,
) -> list[TrajectoryStep]:
    """Build development trajectories by chaining SketchMol before/after pairs per task."""

    if steps_per_trajectory < 2:
        raise ValueError("steps_per_trajectory must be at least 2.")
    rows = load_sketchmol_opt_rows(opt_examples_dir)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task = _task_name_from_row(row)
        if max_pairs_per_task is None or len(by_task[task]) < max_pairs_per_task:
            by_task[task].append(row)

    all_steps: list[TrajectoryStep] = []
    for task_name, task_rows in sorted(by_task.items()):
        for chunk_index in range(0, len(task_rows), max(1, steps_per_trajectory - 1)):
            chunk = task_rows[chunk_index : chunk_index + steps_per_trajectory - 1]
            if not chunk:
                continue
            sequence = [str(chunk[0]["Before_opt_smiles"])] + [str(row["After_opt_smiles"]) for row in chunk]
            conditions = []
            for seq_index, smiles in enumerate(sequence):
                row = chunk[max(0, seq_index - 1)]
                condition = {"task_name": task_name}
                before_score = row.get("Before_opt_act_score")
                after_score = row.get("After_opt_act_score")
                if before_score not in (None, "") and after_score not in (None, ""):
                    condition["activity_before"] = finite_or_zero(before_score)
                    condition["activity_after"] = finite_or_zero(after_score)
                    condition["activity_delta"] = finite_or_zero(after_score) - finite_or_zero(before_score)
                condition["smiles"] = smiles
                conditions.append(condition)
            trajectory_id = f"{task_name}_bootstrap_{chunk_index // max(1, steps_per_trajectory - 1):04d}"
            all_steps.extend(
                steps_from_smiles_sequence(
                    trajectory_id=trajectory_id,
                    smiles_sequence=sequence,
                    task_name=task_name,
                    source="sketchmol_opt_examples_bootstrap",
                    conditions=conditions,
                    edit_type=task_name,
                )
            )
    write_jsonl(output_jsonl, all_steps)
    if output_csv is not None:
        write_csv(output_csv, all_steps)
    return all_steps


def _load_rdkit_fingerprint_tools() -> tuple[Any, Any, Any]:
    try:
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import AllChem

        RDLogger.DisableLog("rdApp.warning")
        return Chem, AllChem, DataStructs
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("Agentic trajectory generation requires RDKit.") from exc


def _fingerprint(smiles: str, radius: int = 2, bits: int = 2048) -> Any | None:
    Chem, AllChem, _ = _load_rdkit_fingerprint_tools()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bits)


def _tanimoto(a: Any | None, b: Any | None) -> float:
    if a is None or b is None:
        return 0.0
    _, _, DataStructs = _load_rdkit_fingerprint_tools()
    return float(DataStructs.TanimotoSimilarity(a, b))


def _task_reward(current_props: dict[str, float], candidate_props: dict[str, float], task_name: str) -> float:
    delta = property_delta(current_props, candidate_props)
    return reward_from_delta(delta, task_name=task_name)


def generate_agentic_opt_trajectories(
    opt_examples_dir: str | Path = SketchMolOptPairConfig.opt_examples_dir,
    output_jsonl: str | Path = "outputs/trajectories/sketchmol_agentic_opt.jsonl",
    output_csv: str | Path | None = "outputs/trajectories/sketchmol_agentic_opt.csv",
    trajectories_per_task: int = 24,
    steps_per_trajectory: int = 6,
    top_k: int = 8,
    similarity_weight: float = 0.15,
    novelty_weight: float = 0.05,
    seed: int = 7,
) -> list[TrajectoryStep]:
    """Generate path-dependent trajectories from SketchMol optimization molecules.

    The agent starts from a real SketchMol before molecule, repeatedly searches a
    task-specific candidate pool, and moves to candidates that trade off property
    improvement, similarity to the current molecule, and novelty relative to the
    already visited path.
    """

    if steps_per_trajectory < 2:
        raise ValueError("steps_per_trajectory must be at least 2.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    rng = random.Random(seed)
    rows = load_sketchmol_opt_rows(opt_examples_dir)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[_task_name_from_row(row)].append(row)

    all_steps: list[TrajectoryStep] = []
    for task_name, task_rows in sorted(by_task.items()):
        candidate_smiles = sorted(
            {
                str(row.get("Before_opt_smiles") or "")
                for row in task_rows
                if row.get("Before_opt_smiles")
            }
            | {
                str(row.get("After_opt_smiles") or "")
                for row in task_rows
                if row.get("After_opt_smiles")
            }
        )
        candidate_props: dict[str, dict[str, float]] = {}
        candidate_fps: dict[str, Any] = {}
        for smiles in candidate_smiles:
            props, valid, _ = compute_properties(smiles)
            fp = _fingerprint(smiles)
            if valid and fp is not None:
                candidate_props[smiles] = props
                candidate_fps[smiles] = fp
        candidates = sorted(candidate_props)
        if len(candidates) < steps_per_trajectory:
            continue

        starts = [str(row["Before_opt_smiles"]) for row in task_rows if str(row.get("Before_opt_smiles") or "") in candidate_props]
        rng.shuffle(starts)
        for traj_idx, start_smiles in enumerate(starts[:trajectories_per_task]):
            sequence = [start_smiles]
            used = {start_smiles}
            for step_idx in range(1, steps_per_trajectory):
                current = sequence[-1]
                current_props = candidate_props[current]
                current_fp = candidate_fps[current]
                scored = []
                for candidate in candidates:
                    if candidate in used:
                        continue
                    reward = _task_reward(current_props, candidate_props[candidate], task_name)
                    similarity = _tanimoto(current_fp, candidate_fps[candidate])
                    novelty = 1.0 - max(_tanimoto(candidate_fps[candidate], candidate_fps[old]) for old in used)
                    # Add a tiny deterministic jitter so equal-scoring candidates do not collapse.
                    jitter = rng.random() * 1e-6
                    score = reward + similarity_weight * similarity + novelty_weight * novelty + jitter
                    scored.append((score, reward, similarity, novelty, candidate))
                if not scored:
                    break
                scored.sort(reverse=True)
                selected = rng.choice(scored[: min(top_k, len(scored))])
                sequence.append(selected[-1])
                used.add(selected[-1])

            conditions = []
            prev_props = {}
            for smiles in sequence:
                props = candidate_props[smiles]
                delta = property_delta(prev_props, props) if prev_props else {name: 0.0 for name in PROPERTY_NAMES}
                conditions.append(
                    {
                        "task_name": task_name,
                        "agent": "rdkit_property_similarity",
                        "reward_delta": reward_from_delta(delta, task_name=task_name),
                        "smiles": smiles,
                    }
                )
                prev_props = props
            trajectory_id = f"{task_name}_agentic_{traj_idx:04d}"
            all_steps.extend(
                steps_from_smiles_sequence(
                    trajectory_id=trajectory_id,
                    smiles_sequence=sequence,
                    task_name=task_name,
                    source="sketchmol_opt_examples_agentic_rdkit",
                    conditions=conditions,
                    edit_type=f"{task_name}_agentic",
                )
            )

    write_jsonl(output_jsonl, all_steps)
    if output_csv is not None:
        write_csv(output_csv, all_steps)
    return all_steps


def trajectories_from_sketchmol_csv(
    input_csv: str | Path,
    output_jsonl: str | Path,
    output_csv: str | Path | None = None,
    trajectory_id_column: str = "trajectory_id",
    step_column: str = "step",
    smiles_column: str = "SMILES",
    image_column: str = "image_path",
    task_name: str = "sketchmol",
) -> list[TrajectoryStep]:
    """Convert a SketchMol image_path.csv after MolScribe into trajectory JSONL."""

    with Path(input_csv).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        trajectory_id = str(row.get(trajectory_id_column) or f"{task_name}_{row_index:06d}")
        row["_row_index"] = row_index
        grouped[trajectory_id].append(row)

    all_steps: list[TrajectoryStep] = []
    for trajectory_id, group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: int(float(row.get(step_column) or row.get("_row_index") or 0)))
        smiles_sequence = [str(row.get(smiles_column) or "") for row in group]
        conditions = [_condition_from_row(row) for row in group]
        image_paths = [str(row.get(image_column) or "") for row in group]
        molscribe_scores = [
            None if row.get("molscribe_score") in (None, "") else finite_or_zero(row.get("molscribe_score")) for row in group
        ]
        all_steps.extend(
            steps_from_smiles_sequence(
                trajectory_id=trajectory_id,
                smiles_sequence=smiles_sequence,
                task_name=task_name,
                source=str(input_csv),
                conditions=conditions,
                image_paths=image_paths,
                molscribe_scores=molscribe_scores,
                edit_type=task_name,
            )
        )
    write_jsonl(output_jsonl, all_steps)
    if output_csv is not None:
        write_csv(output_csv, all_steps)
    return all_steps


def write_sketchmol_command_manifest(
    output_path: str | Path,
    sketchmol_root: str | Path,
    checkpoint: str,
    molscribe_model: str,
    preset: str,
    validation_dataset: str = "",
    mode: str = "inpaint",
    conditional_count: int = 15,
) -> dict[str, Any]:
    """Write the exact commands needed for a real SketchMol trajectory round."""

    root = Path(sketchmol_root)
    if mode not in {"sample", "inpaint"}:
        raise ValueError("mode must be 'sample' or 'inpaint'.")
    script = "scripts/inpaint_continuousV2.py" if mode == "inpaint" else "scripts/sample_diffusion_condition_continuousV2.py"
    generation_cmd = [
        "python",
        str(root / script),
        "-r",
        checkpoint,
        "-p",
        preset,
        "--conditional_count",
        str(conditional_count),
    ]
    if validation_dataset:
        generation_cmd += ["--validation_dataset", validation_dataset]
    manifest = {
        "mode": mode,
        "sketchmol_root": str(root),
        "generation_command": generation_cmd,
        "ocr_command_template": [
            "python",
            str(root / "evaluate/predict_csv.py"),
            "--model_path",
            molscribe_model,
            "--image_path",
            "<generated image_path.csv>",
        ],
        "trajectory_conversion": {
            "command": "python -m latent_edit_trajectory_attention.trajectory_generator from-sketchmol-csv",
            "input": "<image_path.csv after MolScribe>",
        },
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build SketchMol trajectory logs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    boot = subparsers.add_parser("bootstrap-opt", help="Bootstrap trajectories from SketchMol opt_examples.")
    boot.add_argument("--opt-examples-dir", default=SketchMolOptPairConfig.opt_examples_dir)
    boot.add_argument("--output-jsonl", default="outputs/trajectories/sketchmol_opt_bootstrap.jsonl")
    boot.add_argument("--output-csv", default="outputs/trajectories/sketchmol_opt_bootstrap.csv")
    boot.add_argument("--max-pairs-per-task", type=int, default=None)
    boot.add_argument("--steps-per-trajectory", type=int, default=5)

    agentic = subparsers.add_parser("agentic-opt", help="Generate iterative RDKit-agent trajectories from opt_examples.")
    agentic.add_argument("--opt-examples-dir", default=SketchMolOptPairConfig.opt_examples_dir)
    agentic.add_argument("--output-jsonl", default="outputs/trajectories/sketchmol_agentic_opt.jsonl")
    agentic.add_argument("--output-csv", default="outputs/trajectories/sketchmol_agentic_opt.csv")
    agentic.add_argument("--trajectories-per-task", type=int, default=24)
    agentic.add_argument("--steps-per-trajectory", type=int, default=6)
    agentic.add_argument("--top-k", type=int, default=8)
    agentic.add_argument("--similarity-weight", type=float, default=0.15)
    agentic.add_argument("--novelty-weight", type=float, default=0.05)
    agentic.add_argument("--seed", type=int, default=7)

    csv_parser = subparsers.add_parser("from-sketchmol-csv", help="Convert SketchMol image_path.csv to trajectory JSONL.")
    csv_parser.add_argument("--input-csv", required=True)
    csv_parser.add_argument("--output-jsonl", required=True)
    csv_parser.add_argument("--output-csv", default=None)
    csv_parser.add_argument("--trajectory-id-column", default="trajectory_id")
    csv_parser.add_argument("--step-column", default="step")
    csv_parser.add_argument("--smiles-column", default="SMILES")
    csv_parser.add_argument("--image-column", default="image_path")
    csv_parser.add_argument("--task-name", default="sketchmol")

    manifest = subparsers.add_parser("write-command-manifest", help="Write SketchMol/MolScribe command manifest.")
    manifest.add_argument("--output-path", default="outputs/trajectories/sketchmol_command_manifest.json")
    manifest.add_argument("--sketchmol-root", default="/home/bdong/scratch/projects/Diffusion-Molecule/Research/Molecule Generation/SketchMol/SketchMol-v1-main")
    manifest.add_argument("--checkpoint", required=True)
    manifest.add_argument("--molscribe-model", required=True)
    manifest.add_argument("--preset", required=True)
    manifest.add_argument("--validation-dataset", default="")
    manifest.add_argument("--mode", choices=["sample", "inpaint"], default="inpaint")
    manifest.add_argument("--conditional-count", type=int, default=15)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "bootstrap-opt":
        steps = bootstrap_from_opt_examples(
            opt_examples_dir=args.opt_examples_dir,
            output_jsonl=args.output_jsonl,
            output_csv=args.output_csv,
            max_pairs_per_task=args.max_pairs_per_task,
            steps_per_trajectory=args.steps_per_trajectory,
        )
        print(json.dumps({"steps": len(steps), "output_jsonl": args.output_jsonl}, indent=2))
    elif args.command == "agentic-opt":
        steps = generate_agentic_opt_trajectories(
            opt_examples_dir=args.opt_examples_dir,
            output_jsonl=args.output_jsonl,
            output_csv=args.output_csv,
            trajectories_per_task=args.trajectories_per_task,
            steps_per_trajectory=args.steps_per_trajectory,
            top_k=args.top_k,
            similarity_weight=args.similarity_weight,
            novelty_weight=args.novelty_weight,
            seed=args.seed,
        )
        print(json.dumps({"steps": len(steps), "output_jsonl": args.output_jsonl}, indent=2))
    elif args.command == "from-sketchmol-csv":
        steps = trajectories_from_sketchmol_csv(
            input_csv=args.input_csv,
            output_jsonl=args.output_jsonl,
            output_csv=args.output_csv,
            trajectory_id_column=args.trajectory_id_column,
            step_column=args.step_column,
            smiles_column=args.smiles_column,
            image_column=args.image_column,
            task_name=args.task_name,
        )
        print(json.dumps({"steps": len(steps), "output_jsonl": args.output_jsonl}, indent=2))
    elif args.command == "write-command-manifest":
        manifest = write_sketchmol_command_manifest(
            output_path=args.output_path,
            sketchmol_root=args.sketchmol_root,
            checkpoint=args.checkpoint,
            molscribe_model=args.molscribe_model,
            preset=args.preset,
            validation_dataset=args.validation_dataset,
            mode=args.mode,
            conditional_count=args.conditional_count,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

