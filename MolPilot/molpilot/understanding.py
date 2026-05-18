"""Multimodal understanding stream.

This is the molecular analogue of UniVideo's MLLM stream: it grounds the user
request, builds condition branches, and leaves a continuous vector for the
generator. The first version uses deterministic featurizers so the pipeline can
be tested without a large MLLM checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chem import render_molecule_image
from .features import concat_and_project, image_feature_vector, molecule_feature_vector, spec_feature_vector, stable_hash_vector
from .schema import ConditionBranch, ConditionBundle, GenerationRequest, ObjectiveSpec, TaskType


@dataclass
class UnderstandingConfig:
    condition_dim: int = 256
    text_dim: int = 96
    molecule_dim: int = 96
    image_dim: int = 32
    spec_dim: int = 32
    render_missing_images: bool = True
    render_dir: str = "outputs/rendered_inputs"


class UnderstandingStream:
    def __init__(self, config: UnderstandingConfig | None = None):
        self.config = config or UnderstandingConfig()

    def encode(self, request: GenerationRequest) -> ConditionBundle:
        objective = ground_instruction(request.instruction)
        rendered = None
        source_image = request.source_image
        notes: list[str] = []
        if not source_image and request.source_smiles and self.config.render_missing_images:
            safe_name = str(abs(hash(request.source_smiles)))[:12] + ".png"
            rendered = render_molecule_image(request.source_smiles, Path(self.config.render_dir) / safe_name)
            source_image = rendered
            if rendered is None:
                notes.append("RDKit rendering unavailable; image branch is zero or path-hash fallback.")

        text_vec = stable_hash_vector(_task_instruction(request), self.config.text_dim, salt="text")
        mol_vec = molecule_feature_vector(request.source_smiles, self.config.molecule_dim)
        img_vec = image_feature_vector(source_image, self.config.image_dim)
        spec_vec = spec_feature_vector(objective.goals, objective.constraints, objective.proxy_goals, self.config.spec_dim)

        uncond = np.zeros(self.config.condition_dim, dtype=np.float32)
        text_spec = concat_and_project([text_vec, spec_vec], self.config.condition_dim)
        multimodal = concat_and_project([text_vec, spec_vec, mol_vec, img_vec, _task_type_vec(request.task_type)], self.config.condition_dim)

        branches = {
            "uncond": ConditionBranch("uncond", uncond.tolist(), [0] * self.config.condition_dim),
            "text_spec": ConditionBranch("text_spec", text_spec.tolist(), [1] * self.config.condition_dim),
            "multimodal": ConditionBranch("multimodal", multimodal.tolist(), [1] * self.config.condition_dim),
        }
        if objective.unverifiable_goals:
            notes.append("Prompt contains unverifiable disease-level goals; exclude from hard verified main table.")
        return ConditionBundle(request=request, objective=objective, branches=branches, rendered_image=rendered, notes=notes)


def ground_instruction(instruction: str) -> ObjectiveSpec:
    text = str(instruction).lower()
    goals: list[str] = []
    constraints: list[str] = []
    proxy_goals: list[str] = []
    unverifiable: list[str] = []
    thresholds = {}
    target_hint = None
    disease_hint = None

    if any(word in text for word in ("solubility", "soluble", "aqueous")):
        goals.append("decrease_logp")
        proxy_goals.append("improve_solubility_proxy")
        constraints.extend(["keep_mw_similar", "preserve_scaffold"])
    if any(word in text for word in ("logp too high", "too lipophilic", "lower logp", "reduce logp")):
        goals.append("decrease_logp")
    if any(word in text for word in ("increase logp", "more lipophilic")):
        goals.append("increase_logp")
    if any(word in text for word in ("tpsa too high", "reduce tpsa", "lower tpsa", "permeability", "bbb", "brain", "cns")):
        goals.append("reduce_tpsa")
        constraints.append("cns_like")
        if "bbb" in text or "brain" in text:
            proxy_goals.append("improve_bbb_proxy")
    if any(word in text for word in ("qed", "drug-like", "druglike", "drug likeness")):
        goals.append("improve_qed")
        constraints.append("keep_druglike")
    if any(word in text for word in ("too large", "lower mw", "reduce mw", "molecular weight too high")):
        goals.append("decrease_mw")
    if any(word in text for word in ("too flexible", "reduce rotatable", "lower rb")):
        goals.append("decrease_rb")
    if any(word in text for word in ("preserve scaffold", "keep scaffold", "same core", "preserve core", "do not change core")):
        constraints.append("preserve_scaffold")
    if any(word in text for word in ("similar", "minimal change", "small edit")):
        constraints.append("keep_similarity")
    if any(word in text for word in ("herg", "cardiotoxic")):
        proxy_goals.append("reduce_herg_proxy")
    if any(word in text for word in ("potency", "activity", "binding", "ic50", "ec50")):
        proxy_goals.append("improve_activity_proxy")
    if any(word in text for word in ("cancer", "alzheimer", "parkinson", "diabetes", "covid", "lung cancer", "disease")):
        unverifiable.append("treat_disease")
        disease_hint = _first_matching(text, ["lung cancer", "cancer", "alzheimer", "parkinson", "diabetes", "covid"])

    target_hint = _first_matching(text, ["egfr", "alk", "kras", "bace1", "jak2", "herg", "ep4", "akt1", "rock1"])
    if target_hint and "improve_activity_proxy" not in proxy_goals:
        proxy_goals.append("improve_activity_proxy")

    return ObjectiveSpec(
        goals=_dedupe(goals),
        constraints=_dedupe(constraints),
        proxy_goals=_dedupe(proxy_goals),
        unverifiable_goals=_dedupe(unverifiable),
        thresholds={**ObjectiveSpec().thresholds, **thresholds},
        target_hint=target_hint,
        disease_hint=disease_hint,
    )


def _task_instruction(request: GenerationRequest) -> str:
    if request.task_type == TaskType.EDIT:
        prefix = "Edit the source molecule according to the medicinal chemistry instruction:"
    elif request.task_type == TaskType.INPAINT:
        prefix = "Complete the masked molecular region while preserving the known scaffold:"
    else:
        prefix = "Generate a molecule from scratch that matches the requested profile:"
    parts = [prefix, request.instruction]
    if request.source_smiles:
        parts.append(f"source_smiles={request.source_smiles}")
    if request.mask_smarts:
        parts.append(f"mask_smarts={request.mask_smarts}")
    return " ".join(parts)


def _task_type_vec(task_type: TaskType, dim: int = 8) -> np.ndarray:
    return stable_hash_vector(task_type.value, dim, salt="task")


def _dedupe(values: list[str]) -> list[str]:
    out = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _first_matching(text: str, options: list[str]) -> str | None:
    for option in options:
        if option in text:
            return option
    return None

