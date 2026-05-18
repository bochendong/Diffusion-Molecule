"""Build shared arrays/tables for staged training."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

from .data import build_smoke_requests, load_smiles_csv
from .schema import GenerationRequest
from .understanding import UnderstandingConfig, UnderstandingStream


def load_smiles_and_pairs(data: str | None, limit: int = 0):
    smiles = load_smiles_csv(data, limit=limit)
    pairs = build_smoke_requests(smiles)
    return smiles, pairs


def build_condition_table(
    pairs: list[tuple[GenerationRequest, str]],
    condition_dim: int = 256,
    render_missing_images: bool = False,
    render_dir: str = "outputs/rendered_inputs",
):
    stream = UnderstandingStream(
        UnderstandingConfig(
            condition_dim=condition_dim,
            render_missing_images=render_missing_images,
            render_dir=render_dir,
        )
    )
    bundles = [stream.encode(req) for req, _ in pairs]
    raw_conditions = np.asarray([bundle.branches["multimodal"].vector for bundle in bundles], dtype=np.float32)
    target_smiles = [target for _, target in pairs]
    rows = []
    for idx, ((request, target), bundle) in enumerate(zip(pairs, bundles)):
        rows.append(
            {
                "request_id": idx,
                "task_type": request.task_type.value,
                "source_smiles": request.source_smiles or "",
                "target_smiles": target,
                "instruction": request.instruction,
                "objective_json": json.dumps(bundle.objective.to_dict(), sort_keys=True),
                "notes": "|".join(bundle.notes),
            }
        )
    return raw_conditions, target_smiles, bundles, rows

