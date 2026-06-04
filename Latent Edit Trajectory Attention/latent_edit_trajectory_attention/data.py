"""Data utilities for latent edit trajectory attention.

The first implementation provides synthetic trajectories so the model can be
tested without committing to a molecular encoder or a trajectory dataset.
Real data loaders should produce the same batch keys.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import TrajectoryDiffusionConfig, _load_torch
from .schema import PROPERTY_NAMES, TrajectoryStep, read_jsonl


@dataclass(frozen=True)
class SyntheticTrajectoryConfig:
    """Controls synthetic latent trajectory generation."""

    examples: int = 256
    history_length: int = 6
    seed: int = 7
    reward_scale: float = 0.2
    edit_scale: float = 0.1

    def validate(self) -> None:
        if self.examples <= 0:
            raise ValueError("examples must be positive.")
        if self.history_length <= 0:
            raise ValueError("history_length must be positive.")
        if self.reward_scale < 0:
            raise ValueError("reward_scale cannot be negative.")
        if self.edit_scale < 0:
            raise ValueError("edit_scale cannot be negative.")


class SyntheticTrajectoryDataset:
    """Factory for a torch Dataset of simple latent optimization trajectories."""

    def __new__(
        cls,
        model_config: TrajectoryDiffusionConfig,
        data_config: SyntheticTrajectoryConfig | None = None,
    ) -> Any:
        torch = _load_torch()
        data_cfg = data_config or SyntheticTrajectoryConfig()
        model_config.validate()
        data_cfg.validate()

        class _Dataset(torch.utils.data.Dataset):
            def __init__(self) -> None:
                generator = torch.Generator().manual_seed(data_cfg.seed)
                n = data_cfg.examples
                h = min(data_cfg.history_length, model_config.max_history)
                latent_dim = model_config.latent_dim
                property_dim = model_config.property_dim
                target_dim = max(1, model_config.target_dim)

                base = torch.randn(n, latent_dim, generator=generator)
                target = torch.randn(n, target_dim, generator=generator)
                target_projection = torch.randn(target_dim, latent_dim, generator=generator) / target_dim**0.5
                edit_projection = torch.randn(model_config.edit_type_count, latent_dim, generator=generator) * data_cfg.edit_scale

                z_steps = []
                property_steps = []
                edit_steps = []
                current = base
                for step in range(h):
                    edit_ids = torch.randint(
                        low=0,
                        high=model_config.edit_type_count,
                        size=(n,),
                        generator=generator,
                    )
                    target_push = target @ target_projection
                    edit_push = edit_projection[edit_ids]
                    drift = (0.15 + step / max(1, h) * 0.05) * target_push + edit_push
                    noise = torch.randn(n, latent_dim, generator=generator) * 0.03
                    current = current + drift + noise
                    z_steps.append(current)
                    edit_steps.append(edit_ids)

                    if property_dim > 0:
                        raw_delta = current[:, :property_dim] - base[:, :property_dim]
                        prop_delta = torch.tanh(raw_delta) * data_cfg.reward_scale
                    else:
                        prop_delta = torch.empty(n, 0)
                    property_steps.append(prop_delta)

                self.z_history = torch.stack(z_steps, dim=1)
                if property_dim > 0:
                    self.property_delta = torch.stack(property_steps, dim=1)
                else:
                    self.property_delta = torch.empty(n, h, 0)
                self.edit_type_ids = torch.stack(edit_steps, dim=1)
                self.history_mask = torch.ones(n, h, dtype=torch.bool)

                final_target_push = target @ target_projection
                recent_direction = self.z_history[:, -1, :] - self.z_history[:, 0, :]
                self.next_z = self.z_history[:, -1, :] + 0.20 * final_target_push + 0.10 * recent_direction
                self.target = target

            def __len__(self) -> int:
                return int(self.z_history.shape[0])

            def __getitem__(self, idx: int) -> dict[str, Any]:
                return {
                    "z_history": self.z_history[idx],
                    "property_delta": self.property_delta[idx],
                    "edit_type_ids": self.edit_type_ids[idx],
                    "history_mask": self.history_mask[idx],
                    "next_z": self.next_z[idx],
                    "target": self.target[idx],
                }

        return _Dataset()


@dataclass(frozen=True)
class SketchMolOptPairConfig:
    """Configuration for SketchMol optimization-pair data."""

    opt_examples_dir: str = (
        "/home/bdong/scratch/projects/Diffusion-Molecule/Research/Molecule Generation/"
        "SketchMol/SketchMol-v1-main/opt_examples"
    )
    fingerprint_radius: int = 2
    max_examples: int | None = None

    def validate(self) -> None:
        if self.fingerprint_radius < 0:
            raise ValueError("fingerprint_radius cannot be negative.")
        if self.max_examples is not None and self.max_examples <= 0:
            raise ValueError("max_examples must be positive when provided.")


def _load_rdkit() -> tuple[Any, Any, Any, Any]:
    try:
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import AllChem, Descriptors, QED

        RDLogger.DisableLog("rdApp.warning")
        return Chem, AllChem, DataStructs, (Descriptors, QED)
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("SketchMol opt-pair loading requires RDKit.") from exc


def _smiles_to_fingerprint(smiles: str, bits: int, radius: int) -> list[float] | None:
    Chem, AllChem, _, _ = _load_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bits)
    return [float(char) for char in fp.ToBitString()]


def _smiles_properties(smiles: str) -> list[float] | None:
    Chem, _, _, descriptor_modules = _load_rdkit()
    descriptors, qed = descriptor_modules
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return [
        float(descriptors.MolLogP(mol)),
        float(qed.qed(mol)),
        float(descriptors.TPSA(mol)),
        float(descriptors.MolWt(mol)),
    ]


def _task_name_from_path(path: Path) -> str:
    return path.name.replace("_opt_examples.csv", "")


def load_sketchmol_opt_rows(opt_examples_dir: str | Path, max_examples: int | None = None) -> list[dict[str, Any]]:
    """Load before/after optimization rows from the original SketchMol examples."""

    root = Path(opt_examples_dir)
    if not root.exists():
        raise FileNotFoundError(f"SketchMol opt_examples directory not found: {root}")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*_opt_examples.csv")):
        task_name = _task_name_from_path(path)
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                before = (row.get("Before_opt_smiles") or "").strip()
                after = (row.get("After_opt_smiles") or "").strip()
                if not before or not after:
                    continue
                row = dict(row)
                row["task_name"] = task_name
                row["source_file"] = str(path)
                rows.append(row)
                if max_examples is not None and len(rows) >= max_examples:
                    return rows
    return rows


class SketchMolOptPairDataset:
    """Factory for SketchMol before/after SMILES optimization pairs."""

    def __new__(
        cls,
        model_config: TrajectoryDiffusionConfig,
        data_config: SketchMolOptPairConfig | None = None,
    ) -> Any:
        torch = _load_torch()
        data_cfg = data_config or SketchMolOptPairConfig()
        model_config.validate()
        data_cfg.validate()

        class _Dataset(torch.utils.data.Dataset):
            def __init__(self) -> None:
                raw_rows = load_sketchmol_opt_rows(data_cfg.opt_examples_dir, max_examples=data_cfg.max_examples)
                task_names = sorted({str(row["task_name"]) for row in raw_rows})
                self.task_to_id = {task: idx for idx, task in enumerate(task_names)}
                self.rows: list[dict[str, Any]] = []
                z_history = []
                next_z = []
                property_delta = []
                edit_type_ids = []
                target = []

                for row in raw_rows:
                    before = str(row["Before_opt_smiles"])
                    after = str(row["After_opt_smiles"])
                    before_fp = _smiles_to_fingerprint(before, model_config.latent_dim, data_cfg.fingerprint_radius)
                    after_fp = _smiles_to_fingerprint(after, model_config.latent_dim, data_cfg.fingerprint_radius)
                    before_props = _smiles_properties(before)
                    after_props = _smiles_properties(after)
                    if before_fp is None or after_fp is None or before_props is None or after_props is None:
                        continue

                    task_id = self.task_to_id[str(row["task_name"])]
                    prop_delta_values = [after_props[idx] - before_props[idx] for idx in range(len(before_props))]
                    before_score = row.get("Before_opt_act_score")
                    after_score = row.get("After_opt_act_score")
                    if before_score not in (None, "") and after_score not in (None, ""):
                        prop_delta_values.insert(0, float(after_score) - float(before_score))
                    prop_delta_values = prop_delta_values[: model_config.property_dim]
                    while len(prop_delta_values) < model_config.property_dim:
                        prop_delta_values.append(0.0)

                    target_values = [0.0] * max(1, model_config.target_dim)
                    if target_values:
                        target_values[0] = float(task_id) / max(1, len(self.task_to_id) - 1)
                    if len(target_values) > 1 and after_score not in (None, ""):
                        target_values[1] = float(after_score)

                    z_history.append(torch.tensor([before_fp], dtype=torch.float32))
                    next_z.append(torch.tensor(after_fp, dtype=torch.float32))
                    property_delta.append(torch.tensor([prop_delta_values], dtype=torch.float32))
                    edit_type_ids.append(torch.tensor([task_id % model_config.edit_type_count], dtype=torch.long))
                    target.append(torch.tensor(target_values, dtype=torch.float32))
                    self.rows.append(row)

                if not self.rows:
                    raise ValueError(f"No valid SketchMol opt pairs found under {data_cfg.opt_examples_dir}.")

                self.z_history = torch.stack(z_history, dim=0)
                self.property_delta = torch.stack(property_delta, dim=0)
                self.edit_type_ids = torch.stack(edit_type_ids, dim=0)
                self.history_mask = torch.ones(len(self.rows), 1, dtype=torch.bool)
                self.next_z = torch.stack(next_z, dim=0)
                self.target = torch.stack(target, dim=0)

            def __len__(self) -> int:
                return len(self.rows)

            def __getitem__(self, idx: int) -> dict[str, Any]:
                row = self.rows[idx]
                return {
                    "z_history": self.z_history[idx],
                    "property_delta": self.property_delta[idx],
                    "edit_type_ids": self.edit_type_ids[idx],
                    "history_mask": self.history_mask[idx],
                    "next_z": self.next_z[idx],
                    "target": self.target[idx],
                    "task_name": row["task_name"],
                    "before_smiles": row["Before_opt_smiles"],
                    "after_smiles": row["After_opt_smiles"],
                }

        return _Dataset()


@dataclass(frozen=True)
class SketchMolTrajectoryConfig:
    """Configuration for JSONL trajectory data."""

    trajectory_path: str = "outputs/trajectories/sketchmol_opt_bootstrap.jsonl"
    fingerprint_radius: int = 2
    max_examples: int | None = None
    min_history: int = 1

    def validate(self) -> None:
        if self.fingerprint_radius < 0:
            raise ValueError("fingerprint_radius cannot be negative.")
        if self.max_examples is not None and self.max_examples <= 0:
            raise ValueError("max_examples must be positive when provided.")
        if self.min_history <= 0:
            raise ValueError("min_history must be positive.")


def _property_vector(step: TrajectoryStep, property_dim: int) -> list[float]:
    values = [float(step.reward)]
    values.extend(float(step.delta_properties.get(name, 0.0)) for name in PROPERTY_NAMES)
    values = values[:property_dim]
    while len(values) < property_dim:
        values.append(0.0)
    return values


class SketchMolTrajectoryDataset:
    """Factory for multi-step SketchMol trajectory logs."""

    def __new__(
        cls,
        model_config: TrajectoryDiffusionConfig,
        data_config: SketchMolTrajectoryConfig | None = None,
    ) -> Any:
        torch = _load_torch()
        data_cfg = data_config or SketchMolTrajectoryConfig()
        model_config.validate()
        data_cfg.validate()

        class _Dataset(torch.utils.data.Dataset):
            def __init__(self) -> None:
                all_steps = read_jsonl(data_cfg.trajectory_path)
                by_trajectory: dict[str, list[TrajectoryStep]] = {}
                for step in all_steps:
                    if not step.validity:
                        continue
                    by_trajectory.setdefault(step.trajectory_id, []).append(step)
                for trajectory_id in by_trajectory:
                    by_trajectory[trajectory_id] = sorted(by_trajectory[trajectory_id], key=lambda item: item.step)

                task_names = sorted({step.task_name for steps in by_trajectory.values() for step in steps})
                self.task_to_id = {task: idx for idx, task in enumerate(task_names)}
                self.examples: list[dict[str, Any]] = []

                for trajectory_id, steps in sorted(by_trajectory.items()):
                    if len(steps) <= data_cfg.min_history:
                        continue
                    fingerprints: list[list[float]] = []
                    valid_steps: list[TrajectoryStep] = []
                    for step in steps:
                        fp = _smiles_to_fingerprint(step.smiles, model_config.latent_dim, data_cfg.fingerprint_radius)
                        if fp is None:
                            continue
                        fingerprints.append(fp)
                        valid_steps.append(step)
                    if len(valid_steps) <= data_cfg.min_history:
                        continue

                    for target_index in range(data_cfg.min_history, len(valid_steps)):
                        history_start = max(0, target_index - model_config.max_history)
                        history_steps = valid_steps[history_start:target_index]
                        history_fps = fingerprints[history_start:target_index]
                        target_step = valid_steps[target_index]
                        next_fp = fingerprints[target_index]
                        task_id = self.task_to_id.get(target_step.task_name, 0)
                        target_values = [0.0] * max(1, model_config.target_dim)
                        target_values[0] = float(task_id) / max(1, len(self.task_to_id) - 1)
                        if len(target_values) > 1:
                            target_values[1] = float(target_step.reward)
                        if len(target_values) > 2:
                            target_values[2] = float(target_step.properties.get("qed", 0.0))
                        if len(target_values) > 3:
                            target_values[3] = float(target_step.properties.get("logp", 0.0))

                        self.examples.append(
                            {
                                "trajectory_id": trajectory_id,
                                "target_step": target_step.step,
                                "task_name": target_step.task_name,
                                "z_history": torch.tensor(history_fps, dtype=torch.float32),
                                "property_delta": torch.tensor(
                                    [_property_vector(step, model_config.property_dim) for step in history_steps],
                                    dtype=torch.float32,
                                ),
                                "edit_type_ids": torch.tensor(
                                    [task_id % model_config.edit_type_count for _ in history_steps],
                                    dtype=torch.long,
                                ),
                                "history_mask": torch.ones(len(history_steps), dtype=torch.bool),
                                "next_z": torch.tensor(next_fp, dtype=torch.float32),
                                "target": torch.tensor(target_values, dtype=torch.float32),
                                "reward": target_step.reward,
                                "validity": target_step.validity,
                            }
                        )
                        if data_cfg.max_examples is not None and len(self.examples) >= data_cfg.max_examples:
                            break
                    if data_cfg.max_examples is not None and len(self.examples) >= data_cfg.max_examples:
                        break

                if not self.examples:
                    raise ValueError(f"No valid trajectory examples found in {data_cfg.trajectory_path}.")

            def __len__(self) -> int:
                return len(self.examples)

            def __getitem__(self, idx: int) -> dict[str, Any]:
                return dict(self.examples[idx])

        return _Dataset()


def collate_trajectory_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length trajectory histories for DataLoader."""

    torch = _load_torch()
    max_len = max(int(item["z_history"].shape[0]) for item in batch)
    latent_dim = int(batch[0]["z_history"].shape[-1])
    property_dim = int(batch[0]["property_delta"].shape[-1])
    z_history = torch.zeros(len(batch), max_len, latent_dim, dtype=torch.float32)
    property_delta = torch.zeros(len(batch), max_len, property_dim, dtype=torch.float32)
    edit_type_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    history_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for row_idx, item in enumerate(batch):
        length = int(item["z_history"].shape[0])
        z_history[row_idx, :length] = item["z_history"]
        property_delta[row_idx, :length] = item["property_delta"]
        edit_type_ids[row_idx, :length] = item["edit_type_ids"]
        history_mask[row_idx, :length] = True
    return {
        "z_history": z_history,
        "property_delta": property_delta,
        "edit_type_ids": edit_type_ids,
        "history_mask": history_mask,
        "next_z": torch.stack([item["next_z"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "trajectory_id": [item.get("trajectory_id", "") for item in batch],
        "target_step": [item.get("target_step", 0) for item in batch],
        "task_name": [item.get("task_name", "") for item in batch],
        "reward": torch.tensor([float(item.get("reward", 0.0)) for item in batch], dtype=torch.float32),
        "validity": torch.tensor([bool(item.get("validity", False)) for item in batch], dtype=torch.bool),
    }


def move_batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    """Move tensor values in a batch to a torch device."""

    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}

