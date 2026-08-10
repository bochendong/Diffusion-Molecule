#!/usr/bin/env python3
"""Shared protocol, executor, and reward for the common-LLM tool policy.

This module deliberately separates *action support* from *action quality*.
GraphEditDSL candidates come only from the universal grammar and RDKit
execution.  Property predictors are consulted after execution to produce an
environment observation and a trajectory reward; they never rank or truncate
the action set.
"""

from __future__ import annotations

import atexit
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
POLICY_DIR = SCRIPT_DIR.parent / "unified_smiles_generator"
for import_dir in (SCRIPT_DIR, POLICY_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

import evaluate_common_llm_constrained_actions as constrained  # noqa: E402


@lru_cache(maxsize=1)
def graph_policy_module():
    """Load the torch/RDKit executor stack only when molecular execution starts."""
    import umtp_graph_action_policy

    return umtp_graph_action_policy

POLICY_PROTOCOL = "unified_constraint_common_llm_tool_policy_v1"
ADMET_PROPERTY_KEYS = {
    "bbbp": "bbbp",
    "hia": "hia",
    "mutagenicity": "mutagenicity",
}
SYSTEM_PROMPT = (
    "You are one unified molecular design policy. Read the constraint IR and the current environment "
    "observation, then return exactly one JSON tool call. For edit tasks call graph_edit_dsl with one "
    "executable action; after at least one edit you may call stop. Use feedback from earlier steps to "
    "repair unmet constraints while preserving source similarity. Do not add prose or markdown."
)


@dataclass(frozen=True)
class ExecutableToolAction:
    payload: dict[str, object]
    next_smiles: str
    terminal: bool
    action_key: tuple[object, ...]


@dataclass(frozen=True)
class ConstraintOutcome:
    property: str
    objective: str
    source_value: float | None
    candidate_value: float | None
    required_change: float | None
    normalized_margin: float
    success: bool


@dataclass(frozen=True)
class PolicyFeedback:
    valid: bool
    source_similarity: float
    source_similarity_success: bool
    property_success_fraction: float
    property_all_success: bool
    strict_success: bool
    mean_satisfaction: float
    reward: float
    outcomes: tuple[ConstraintOutcome, ...]

    def observation(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "source_similarity": finite_or_none(self.source_similarity),
            "source_similarity_success": self.source_similarity_success,
            "property_success_fraction": self.property_success_fraction,
            "property_all_success": self.property_all_success,
            "strict_success": self.strict_success,
            "mean_satisfaction": self.mean_satisfaction,
            "reward": self.reward,
            "constraints": [asdict(item) for item in self.outcomes],
        }


class AdmetAIOracleClient:
    """One persistent official ADMET-AI process with a molecule cache."""

    def __init__(self, python_bin: str):
        environment = dict(os.environ)
        # The common-LLM overlay contains a different torch stack.  The ADMET
        # venv gets its dependencies from the loaded cluster modules instead.
        environment.pop("PYTHONPATH", None)
        environment["CUDA_VISIBLE_DEVICES"] = ""
        server = SCRIPT_DIR / "admet_ai_jsonl_server.py"
        self.process = subprocess.Popen(
            [str(python_bin), "-u", str(server)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=environment,
        )
        self.cache: dict[str, dict[str, float]] = {}
        self.request_count = 0
        atexit.register(self.close)

    def predict(self, smiles: str) -> dict[str, float]:
        unified = graph_policy_module().unified
        canonical = unified.safe_canonical_smiles(smiles)
        if not canonical:
            return {}
        if canonical in self.cache:
            return self.cache[canonical]
        if self.process.poll() is not None or self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("ADMET-AI oracle bridge exited before prediction")
        self.request_count += 1
        request = {"request_id": self.request_count, "smiles": [canonical]}
        self.process.stdin.write(json.dumps(request, sort_keys=True) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("ADMET-AI oracle bridge returned no response")
        response = json.loads(line)
        if response.get("error"):
            raise RuntimeError(f"ADMET-AI oracle bridge failed: {response['error']}")
        predictions = response.get("predictions", [])
        if not isinstance(predictions, list) or not predictions:
            raise RuntimeError("ADMET-AI oracle bridge returned no predictions")
        row = predictions[0]
        if not isinstance(row, Mapping):
            raise RuntimeError("ADMET-AI oracle bridge returned a malformed prediction")
        values = {
            str(key): float(value)
            for key, value in row.items()
            if key != "smiles" and value is not None
        }
        self.cache[canonical] = values
        return values

    def close(self) -> None:
        if self.process.poll() is None:
            if self.process.stdin is not None:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()


@lru_cache(maxsize=1)
def admet_client() -> AdmetAIOracleClient | None:
    python_bin = str(os.environ.get("SUCC_ADMET_PYTHON_BIN", "") or "").strip()
    return AdmetAIOracleClient(python_bin) if python_bin else None


def score_property_value(smiles: str, prop: str) -> float | None:
    unified = graph_policy_module().unified
    canonical_prop = str(unified.canonical_prop(prop) or prop).strip().lower()
    client = admet_client()
    if canonical_prop in ADMET_PROPERTY_KEYS and client is not None:
        return client.predict(smiles).get(ADMET_PROPERTY_KEYS[canonical_prop])
    return unified.score_property(smiles, prop)


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def constraint_ir(record: Mapping[str, object]) -> dict[str, object]:
    ir, _expected = constrained.constraint_payload(record)
    return ir


def policy_prompt_messages(
    ir: Mapping[str, object],
    *,
    current_smiles: str,
    original_source_smiles: str,
    previous_steps: Sequence[Mapping[str, object]],
    step_index: int,
    max_steps: int,
) -> list[dict[str, str]]:
    """Build a target-free state prompt for one policy decision."""
    payload = {
        "protocol": POLICY_PROTOCOL,
        "constraint_ir": dict(ir),
        "environment": {
            "original_source_smiles": original_source_smiles,
            "current_smiles": current_smiles,
            "step_index": int(step_index),
            "max_steps": int(max_steps),
            "previous_steps": list(previous_steps),
        },
        "available_tools": [
            {
                "action_type": "graph_edit_dsl",
                "value": {
                    "op": "one GraphEditDSL operation",
                    "site": "optional atom index",
                    "bond": "optional atom-index pair",
                    "atom": "optional atom symbol",
                    "fragment": "optional fragment SMILES",
                    "bond_order": "optional bond order",
                },
            },
            *(
                [{"action_type": "stop", "value": {"reason": "constraints_satisfied_or_no_safe_edit"}}]
                if int(step_index) > 0
                else []
            ),
        ],
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, sort_keys=True, separators=(",", ":"))},
    ]


def graph_action_payload(action: object) -> dict[str, object]:
    value = asdict(action)
    # policy_score is executor metadata, never part of the model-visible action.
    value.pop("policy_score", None)
    return {"action_type": "graph_edit_dsl", "value": value}


def action_identity(action: object) -> tuple[object, ...]:
    return graph_policy_module().action_key(action)


def executable_grammar_actions(
    current_smiles: str,
    *,
    site_limit: int,
    max_actions: int,
    include_stop: bool,
) -> list[ExecutableToolAction]:
    """Enumerate a balanced, property-agnostic slice of the typed grammar."""
    graph_policy = graph_policy_module()
    unified = graph_policy.unified
    graph = graph_policy.graph
    raw = graph_policy.universal_actions(current_smiles, site_limit=int(site_limit))
    # RDKit will reject some syntactically valid local edits. Draw a larger,
    # still property-agnostic grammar prefix so ``max_actions`` counts actual
    # executable choices rather than failed execution attempts.
    raw = graph_policy.balanced_action_cap(raw, max(int(max_actions) * 4, int(max_actions)))
    canonical_current = unified.safe_canonical_smiles(current_smiles)
    seen = {canonical_current} if canonical_current else set()
    output: list[ExecutableToolAction] = []
    for action in raw:
        generated = graph.execute_graph_edit_action(current_smiles, action)
        canonical = unified.safe_canonical_smiles(generated)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        output.append(
            ExecutableToolAction(
                payload=graph_action_payload(action),
                next_smiles=canonical,
                terminal=False,
                action_key=action_identity(action),
            )
        )
        if len(output) >= int(max_actions):
            break
    if include_stop:
        output.append(
            ExecutableToolAction(
                payload={"action_type": "stop", "value": {"reason": "constraints_satisfied_or_no_safe_edit"}},
                next_smiles=canonical_current,
                terminal=True,
                action_key=("stop",),
            )
        )
    return output


def _property_outcome(
    constraint: Mapping[str, object],
    *,
    original_source_smiles: str,
    candidate_smiles: str,
) -> ConstraintOutcome:
    unified = graph_policy_module().unified
    prop = str(constraint.get("property", "") or "").strip()
    objective = str(constraint.get("objective", "improve") or "improve").strip().lower()
    direction = int(constraint.get("direction", 0) or 0)
    threshold_value = constraint.get("threshold")
    threshold = float(threshold_value) if threshold_value is not None else None
    target_value = constraint.get("target")
    target = float(target_value) if target_value is not None else None
    source_value_raw = constraint.get("source_value")
    source_value = float(source_value_raw) if source_value_raw is not None else None
    if source_value is None and original_source_smiles:
        source_value = score_property_value(original_source_smiles, prop)
    candidate_value = score_property_value(candidate_smiles, prop)
    normalizer = max(float(unified.PROPERTY_NORMALIZERS.get(unified.canonical_prop(prop), 1.0)), 1e-8)

    required_change: float | None = None
    margin = -1.0
    success = False
    if candidate_value is not None and objective == "target" and target is not None:
        tolerance = threshold
        if tolerance is None or tolerance <= 0:
            tolerance = float(unified.STRICT_TOLERANCE.get(unified.canonical_prop(prop), normalizer))
        distance = abs(float(candidate_value) - target)
        margin = 1.0 - distance / max(float(tolerance), 1e-8)
        required_change = float(tolerance)
        success = distance <= float(tolerance)
    elif candidate_value is not None and source_value is not None and direction:
        signed_change = float(direction) * (float(candidate_value) - float(source_value))
        required_change = max(0.0, float(threshold or 0.0))
        margin = (signed_change - required_change) / normalizer
        success = signed_change >= required_change if required_change > 0 else signed_change > 0
    return ConstraintOutcome(
        property=prop,
        objective=objective,
        source_value=None if source_value is None else float(source_value),
        candidate_value=None if candidate_value is None else float(candidate_value),
        required_change=required_change,
        normalized_margin=float(max(-4.0, min(4.0, margin))),
        success=bool(success),
    )


def score_policy_state(
    ir: Mapping[str, object],
    *,
    original_source_smiles: str,
    candidate_smiles: str,
    source_similarity_threshold: float,
    step_count: int,
) -> PolicyFeedback:
    unified = graph_policy_module().unified
    canonical = unified.safe_canonical_smiles(candidate_smiles)
    if not canonical:
        return PolicyFeedback(False, math.nan, False, 0.0, False, False, 0.0, -1.0, ())
    constraints = ir.get("constraints", [])
    constraint_rows = constraints if isinstance(constraints, list) else []
    outcomes = tuple(
        _property_outcome(item, original_source_smiles=original_source_smiles, candidate_smiles=canonical)
        for item in constraint_rows
        if isinstance(item, Mapping) and bool(item.get("hard", True))
    )
    success_fraction = sum(int(item.success) for item in outcomes) / max(len(outcomes), 1)
    all_success = bool(outcomes) and all(item.success for item in outcomes)
    satisfaction = (
        sum(0.5 * (math.tanh(item.normalized_margin) + 1.0) for item in outcomes) / len(outcomes)
        if outcomes
        else 0.0
    )
    similarity = unified.morgan_tanimoto(original_source_smiles, canonical)
    similarity_success = bool(
        math.isfinite(similarity) and similarity >= float(source_similarity_threshold)
    )
    similarity_scale = max(1.0 - float(source_similarity_threshold), 1e-6)
    similarity_component = (
        max(-1.0, min(1.0, (float(similarity) - float(source_similarity_threshold)) / similarity_scale))
        if math.isfinite(similarity)
        else -1.0
    )
    strict = bool(all_success and similarity_success)
    canonical_source = unified.safe_canonical_smiles(original_source_smiles)
    copy_penalty = 0.25 if canonical_source and canonical == canonical_source and not strict else 0.0
    reward = (
        0.5
        + 1.5 * success_fraction
        + 0.5 * satisfaction
        + 0.5 * similarity_component
        + (0.75 if all_success else 0.0)
        + (1.25 if strict else 0.0)
        - copy_penalty
        - 0.03 * max(0, int(step_count))
    )
    return PolicyFeedback(
        valid=True,
        source_similarity=float(similarity),
        source_similarity_success=similarity_success,
        property_success_fraction=float(success_fraction),
        property_all_success=all_success,
        strict_success=strict,
        mean_satisfaction=float(satisfaction),
        reward=float(reward),
        outcomes=outcomes,
    )


def group_relative_advantages(rewards: Sequence[float], *, clip: float = 3.0) -> list[float]:
    if not rewards:
        return []
    average = sum(float(item) for item in rewards) / len(rewards)
    variance = sum((float(item) - average) ** 2 for item in rewards) / len(rewards)
    scale = max(math.sqrt(variance), 1e-6)
    return [max(-clip, min(clip, (float(item) - average) / scale)) for item in rewards]
