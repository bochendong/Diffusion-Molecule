"""Shared schemas for MolPilot.

The schema separates three objects:

- raw user request, which may contain unverifiable disease-level text
- executable objective spec, which can be checked by deterministic code
- condition bundle, which is passed to a generator
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskType(str, Enum):
    EDIT = "edit"
    INPAINT = "inpaint"
    DE_NOVO = "de_novo"
    REPAIR = "repair"


HARD_VERIFIABLE_GOALS = (
    "decrease_logp",
    "increase_logp",
    "improve_qed",
    "reduce_tpsa",
    "increase_tpsa",
    "decrease_mw",
    "increase_mw",
    "decrease_rb",
    "lower_sa",
)

HARD_VERIFIABLE_CONSTRAINTS = (
    "preserve_scaffold",
    "keep_similarity",
    "keep_mw_similar",
    "keep_druglike",
    "cns_like",
)

PROXY_GOALS = (
    "improve_solubility_proxy",
    "improve_bbb_proxy",
    "reduce_herg_proxy",
    "improve_activity_proxy",
)

UNVERIFIED_GOALS = (
    "treat_disease",
    "improve_clinical_efficacy",
    "cure_condition",
)


DEFAULT_THRESHOLDS = {
    "delta_logp_min": 0.5,
    "delta_qed_min": 0.05,
    "delta_tpsa_min": 15.0,
    "delta_mw_min": 25.0,
    "delta_mw_abs_max": 80.0,
    "delta_rb_min": 1.0,
    "delta_sa_min": 0.35,
    "similarity_min": 0.60,
    "cns_mw_max": 450.0,
    "cns_tpsa_max": 90.0,
    "cns_hbd_max": 1.0,
}


@dataclass
class ObjectiveSpec:
    goals: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    proxy_goals: list[str] = field(default_factory=list)
    unverifiable_goals: list[str] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    target_hint: str | None = None
    disease_hint: str | None = None

    @property
    def hard_verifiable(self) -> bool:
        return not self.unverifiable_goals

    def to_dict(self) -> dict[str, Any]:
        return {
            "goals": list(self.goals),
            "constraints": list(self.constraints),
            "proxy_goals": list(self.proxy_goals),
            "unverifiable_goals": list(self.unverifiable_goals),
            "thresholds": dict(self.thresholds),
            "target_hint": self.target_hint,
            "disease_hint": self.disease_hint,
            "hard_verifiable": self.hard_verifiable,
        }


@dataclass
class GenerationRequest:
    task_type: TaskType
    instruction: str
    source_smiles: str | None = None
    source_image: str | None = None
    mask_smarts: str | None = None
    mask_atoms: list[int] = field(default_factory=list)
    reference_smiles: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConditionBranch:
    name: str
    vector: list[float]
    attention_mask: list[int] = field(default_factory=list)


@dataclass
class ConditionBundle:
    request: GenerationRequest
    objective: ObjectiveSpec
    branches: dict[str, ConditionBranch]
    rendered_image: str | None = None
    notes: list[str] = field(default_factory=list)
