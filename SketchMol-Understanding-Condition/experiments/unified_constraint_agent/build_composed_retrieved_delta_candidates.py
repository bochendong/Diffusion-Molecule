#!/usr/bin/env python3
"""Expand RetrievedDelta support with property-observed two-step compositions.

The builder learns side-chain substitutions and their normalized property
effects from paired training rows.  At inference it applies compatible
train-only substitutions for at most two steps while preserving the evaluation
source as the similarity anchor.  Evaluation targets and evaluation oracle
values are never read.  The frozen v5 n=20 pool is retained as an immutable
prefix, so this support experiment can only add reachability before a later
fixed-n planner is trained.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
PROJECT_SCRIPTS = PROJECT_DIR / "scripts"
for path in (SCRIPT_DIR, PROJECT_DIR, PROJECT_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_retrieved_delta_edit_candidates as base  # noqa: E402


@dataclass(frozen=True)
class EffectTransform:
    task_key: str
    source_variable: str
    target_variable: str
    frequency: int
    train_condition_id: str
    effects: tuple[tuple[str, float], ...]

    @property
    def effect_map(self) -> dict[str, float]:
        return dict(self.effects)


@dataclass(frozen=True)
class CandidateState:
    smiles: str
    source: str
    source_tanimoto: float
    admet_prior_score: float
    steps: tuple[EffectTransform, ...] = ()
    query_variables: tuple[str, ...] = ()
    retrieval_similarities: tuple[float, ...] = ()
    exact_matches: tuple[bool, ...] = ()
    anchor_rank: int = 0
    prior_effects: tuple[tuple[str, float], ...] = ()

    @property
    def predicted_effects(self) -> dict[str, float]:
        totals: dict[str, float] = defaultdict(float)
        for prop, value in self.prior_effects:
            totals[prop] += float(value)
        for step in self.steps:
            for prop, value in step.effects:
                totals[prop] += float(value)
        return dict(totals)

    @property
    def mean_retrieval_similarity(self) -> float:
        return sum(self.retrieval_similarities) / max(len(self.retrieval_similarities), 1)

    @property
    def actual_step_count(self) -> int:
        return len(self.steps) + int(self.anchor_rank > 0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--eval-csv", required=True, type=Path)
    parser.add_argument("--anchor-candidates-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--enumerated-output-csv", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--candidate-budget", type=int, default=20)
    parser.add_argument("--enumerated-limit", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--beam-size", type=int, default=24)
    parser.add_argument("--max-transforms-per-fragment", type=int, default=16)
    parser.add_argument("--max-compatible-transforms", type=int, default=512)
    parser.add_argument("--min-retrieval-similarity", type=float, default=0.15)
    parser.add_argument("--min-source-tanimoto", type=float, default=0.4)
    parser.add_argument("--min-core-heavy-atoms", type=int, default=5)
    parser.add_argument("--max-variable-heavy-atoms", type=int, default=30)
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_number(value: object) -> float | None:
    try:
        parsed = float(str(value or "").strip())
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def property_names(row: Mapping[str, object]) -> tuple[str, ...]:
    raw = str(
        row.get("external_task_properties", "")
        or row.get("condition_properties", "")
        or base.task_key(row)
    )
    return tuple(
        sorted(
            {
                base.graph.revise.canonical_prop(value)
                for value in raw.replace("+", ",").split(",")
                if value.strip()
            }
        )
    )


def json_mapping(value: object) -> dict[str, object]:
    try:
        payload = json.loads(str(value or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def direction_sign(value: object) -> float:
    text = str(value or "increase").strip().lower()
    return -1.0 if text in {"decrease", "down", "lower", "-1"} else 1.0


def observed_property_effects(row: Mapping[str, object]) -> dict[str, float]:
    """Return threshold-normalized desired-direction effects from one train pair."""
    directions = json_mapping(row.get("external_property_directions_json"))
    thresholds = json_mapping(row.get("external_property_thresholds_json"))
    output = {}
    for prop in property_names(row):
        source = parse_number(row.get(f"external_source_{prop}"))
        target = parse_number(row.get(f"external_target_{prop}"))
        if source is None or target is None:
            source = parse_number(row.get(f"source_{prop}"))
            target = parse_number(row.get(f"target_{prop}"))
        if source is None or target is None:
            continue
        threshold = parse_number(thresholds.get(prop))
        if threshold is None or abs(threshold) < 1e-8:
            threshold = float(base.graph.revise.DEFAULT_THRESHOLDS.get(prop, 0.1) or 0.1)
        effect = direction_sign(directions.get(prop)) * (target - source) / abs(float(threshold))
        output[prop] = max(-4.0, min(4.0, float(effect)))
    return output


def build_effect_transform_index(
    rows: Sequence[Mapping[str, object]],
    *,
    min_core_heavy_atoms: int,
    max_variable_heavy_atoms: int,
) -> tuple[list[EffectTransform], dict[str, object]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    effect_sums: dict[tuple[str, str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    effect_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    first_condition: dict[tuple[str, str, str], str] = {}
    rows_with_transform = 0
    for row in rows:
        source_by_core: dict[str, set[str]] = defaultdict(set)
        target_by_core: dict[str, set[str]] = defaultdict(set)
        for split in base.fragment_splits(
            str(row.get("source_smiles", "") or ""),
            int(min_core_heavy_atoms),
            int(max_variable_heavy_atoms),
        ):
            source_by_core[split.core].add(split.variable)
        for split in base.fragment_splits(
            str(row.get("target_smiles", "") or ""),
            int(min_core_heavy_atoms),
            int(max_variable_heavy_atoms),
        ):
            target_by_core[split.core].add(split.variable)
        effects = observed_property_effects(row)
        added = False
        for core in sorted(set(source_by_core) & set(target_by_core)):
            for source_variable in sorted(source_by_core[core]):
                for target_variable in sorted(target_by_core[core]):
                    if source_variable == target_variable:
                        continue
                    key = (base.task_key(row), source_variable, target_variable)
                    counts[key] += 1
                    first_condition.setdefault(key, base.row_key(row))
                    for prop, value in effects.items():
                        effect_sums[key][prop] += float(value)
                        effect_counts[key][prop] += 1
                    added = True
        rows_with_transform += int(added)

    transforms = []
    for (task, source_variable, target_variable), frequency in counts.items():
        effects = tuple(
            sorted(
                (prop, effect_sums[(task, source_variable, target_variable)][prop] / count)
                for prop, count in effect_counts[(task, source_variable, target_variable)].items()
                if count > 0
            )
        )
        transforms.append(
            EffectTransform(
                task_key=task,
                source_variable=source_variable,
                target_variable=target_variable,
                frequency=int(frequency),
                train_condition_id=first_condition[(task, source_variable, target_variable)],
                effects=effects,
            )
        )
    transforms.sort(key=lambda item: (-item.frequency, item.task_key, item.source_variable, item.target_variable))
    return transforms, {
        "training_rows": len(rows),
        "training_rows_with_transform": rows_with_transform,
        "unique_transforms": len(transforms),
        "transforms_with_property_effect": sum(bool(item.effects) for item in transforms),
        "transform_observations": sum(counts.values()),
    }


def compatible_transforms(
    transforms: Sequence[EffectTransform],
    *,
    query_task: str,
    query_properties: Sequence[str],
) -> list[EffectTransform]:
    query = set(query_properties)
    output: list[tuple[tuple[object, ...], EffectTransform]] = []
    for transform in transforms:
        effects = transform.effect_map
        positive_overlap = sum(effects.get(prop, 0.0) > 0.0 for prop in query)
        if transform.task_key == query_task or positive_overlap:
            values = [float(effects.get(prop, 0.0)) for prop in query_properties]
            output.append(
                (
                    (
                        transform.task_key == query_task,
                        positive_overlap,
                        sum(prop in effects for prop in query),
                        sum(values) / max(len(values), 1),
                        transform.frequency,
                        transform.source_variable,
                        transform.target_variable,
                    ),
                    transform,
                )
            )
    output.sort(key=lambda item: item[0], reverse=True)
    return [transform for _key, transform in output]


def transform_rank_key(
    transform: EffectTransform,
    *,
    query_variable: str,
    query_task: str,
    query_properties: Sequence[str],
) -> tuple[object, ...]:
    similarity = base.variable_similarity(query_variable, transform.source_variable)
    effects = transform.effect_map
    values = [float(effects.get(prop, 0.0)) for prop in query_properties]
    positive = sum(value > 0.0 for value in values)
    covered = sum(prop in effects for prop in query_properties)
    return (
        query_variable == transform.source_variable,
        similarity,
        positive,
        covered,
        transform.task_key == query_task,
        sum(values) / max(len(values), 1),
        transform.frequency,
        transform.target_variable,
    )


def state_rank_key(
    state: CandidateState,
    *,
    query_properties: Sequence[str],
    min_source_tanimoto: float,
) -> tuple[object, ...]:
    effects = state.predicted_effects
    values = [float(effects.get(prop, 0.0)) for prop in query_properties]
    threshold_hits = sum(value >= 1.0 for value in values)
    positive = sum(value > 0.0 for value in values)
    minimum = min(values) if values else 0.0
    mean = sum(values) / max(len(values), 1)
    return (
        state.source_tanimoto >= float(min_source_tanimoto),
        threshold_hits,
        positive,
        minimum,
        mean,
        state.admet_prior_score,
        state.source_tanimoto,
        state.mean_retrieval_similarity,
        -state.actual_step_count,
        state.smiles,
    )


def best_by_smiles(
    states: Sequence[CandidateState],
    *,
    query_properties: Sequence[str],
    min_source_tanimoto: float,
) -> list[CandidateState]:
    best: dict[str, CandidateState] = {}
    for state in states:
        previous = best.get(state.smiles)
        if previous is None or state_rank_key(
            state,
            query_properties=query_properties,
            min_source_tanimoto=min_source_tanimoto,
        ) > state_rank_key(
            previous,
            query_properties=query_properties,
            min_source_tanimoto=min_source_tanimoto,
        ):
            best[state.smiles] = state
    return sorted(
        best.values(),
        key=lambda item: state_rank_key(
            item,
            query_properties=query_properties,
            min_source_tanimoto=min_source_tanimoto,
        ),
        reverse=True,
    )


def anchor_states(
    row: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    transforms: Sequence[EffectTransform],
) -> list[CandidateState]:
    lookup = {
        (item.task_key, item.source_variable, item.target_variable): item
        for item in transforms
    }
    output = []
    for index, anchor in enumerate(rows, start=1):
        smiles = base.canonical_smiles(str(anchor.get("generated_smiles", "") or ""))
        if not smiles:
            continue
        source_tanimoto = parse_number(anchor.get("delta_source_tanimoto"))
        if source_tanimoto is None:
            source_tanimoto = float(base.graph.revise.morgan_tanimoto(str(row.get("source_smiles", "")), smiles))
        prior = parse_number(anchor.get("delta_admet_prior_score"))
        if prior is None:
            prior = float(base.graph.admet_prior_score(row, smiles))
        transform = lookup.get(
            (
                base.task_key(row),
                str(anchor.get("delta_source_variable", "") or ""),
                str(anchor.get("delta_target_variable", "") or ""),
            )
        )
        output.append(
            CandidateState(
                smiles=smiles,
                source="v5_retrieved_delta_anchor",
                source_tanimoto=float(source_tanimoto),
                admet_prior_score=float(prior),
                anchor_rank=index,
                prior_effects=transform.effects if transform is not None else (),
            )
        )
    return output


def expand_state(
    row: Mapping[str, object],
    state: CandidateState,
    transforms: Sequence[EffectTransform],
    *,
    query_task: str,
    query_properties: Sequence[str],
    max_transforms_per_fragment: int,
    min_retrieval_similarity: float,
    min_core_heavy_atoms: int,
    max_variable_heavy_atoms: int,
) -> list[CandidateState]:
    original_source = str(row.get("source_smiles", "") or "")
    current = state.smiles or original_source
    output = []
    used = {(step.task_key, step.source_variable, step.target_variable) for step in state.steps}
    for split in base.fragment_splits(current, min_core_heavy_atoms, max_variable_heavy_atoms):
        ranked = []
        for transform in transforms:
            identity = (transform.task_key, transform.source_variable, transform.target_variable)
            if identity in used:
                continue
            similarity = base.variable_similarity(split.variable, transform.source_variable)
            if similarity < float(min_retrieval_similarity):
                continue
            ranked.append(
                (
                    transform_rank_key(
                        transform,
                        query_variable=split.variable,
                        query_task=query_task,
                        query_properties=query_properties,
                    ),
                    similarity,
                    transform,
                )
            )
        ranked.sort(key=lambda item: item[0], reverse=True)
        for _rank_key, similarity, transform in ranked[: max(1, int(max_transforms_per_fragment))]:
            generated = base.join_fragments(split.core, transform.target_variable)
            canonical = base.canonical_smiles(generated)
            if not canonical or canonical == base.canonical_smiles(current):
                continue
            source_tanimoto = float(base.graph.revise.morgan_tanimoto(original_source, canonical))
            if not math.isfinite(source_tanimoto):
                continue
            output.append(
                CandidateState(
                    smiles=canonical,
                    source="composed_retrieved_delta_edit",
                    source_tanimoto=source_tanimoto,
                    admet_prior_score=float(base.graph.admet_prior_score(row, canonical)),
                    steps=(*state.steps, transform),
                    query_variables=(*state.query_variables, split.variable),
                    retrieval_similarities=(*state.retrieval_similarities, float(similarity)),
                    exact_matches=(*state.exact_matches, split.variable == transform.source_variable),
                    anchor_rank=state.anchor_rank,
                    prior_effects=state.prior_effects,
                )
            )
    return output


def output_row(
    row: Mapping[str, object],
    state: CandidateState,
    *,
    rank: int,
    selected: bool,
) -> dict[str, object]:
    query_props = property_names(row)
    effects = state.predicted_effects
    values = [float(effects.get(prop, 0.0)) for prop in query_props]
    trace = []
    if state.anchor_rank > 0:
        trace.append(
            {
                "source": "v5_retrieved_delta_anchor",
                "anchor_rank": int(state.anchor_rank),
                "effects": dict(state.prior_effects),
            }
        )
    trace.extend(
        {
            "task_key": step.task_key,
            "source_variable": step.source_variable,
            "target_variable": step.target_variable,
            "effects": step.effect_map,
        }
        for step in state.steps
    )
    return {
        **dict(row),
        "generated_smiles": state.smiles,
        "method": "composed_retrieved_delta_support",
        "candidate_rank": int(rank),
        "candidate_selected": "True" if selected else "False",
        "graph_edit_candidate_source": state.source,
        "delta_source_tanimoto": round(state.source_tanimoto, 6),
        "delta_admet_prior_score": round(state.admet_prior_score, 6),
        "delta_retrieval_similarity": round(state.mean_retrieval_similarity, 6),
        "composed_delta_step_count": state.actual_step_count,
        "composed_delta_anchor_rank": int(state.anchor_rank),
        "composed_delta_exact_match_count": sum(state.exact_matches),
        "composed_delta_predicted_effects_json": json.dumps(effects, sort_keys=True),
        "composed_delta_predicted_positive_count": sum(value > 0.0 for value in values),
        "composed_delta_predicted_threshold_count": sum(value >= 1.0 for value in values),
        "composed_delta_predicted_min_effect": round(min(values) if values else 0.0, 6),
        "composed_delta_action_trace_json": json.dumps(trace, sort_keys=True),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.candidate_budget) != 20:
        raise ValueError("The paper-facing candidate budget remains fixed at n=20")
    if int(args.max_steps) not in {1, 2}:
        raise ValueError("This bounded support builder permits one or two delta steps")
    if int(args.enumerated_limit) < 96:
        raise ValueError("The support audit requires an enumerated limit of at least 96")

    train_rows = read_rows(args.train_csv)
    eval_rows = read_rows(args.eval_csv)
    anchors_by_condition: dict[str, list[dict[str, str]]] = defaultdict(list)
    for anchor in read_rows(args.anchor_candidates_csv):
        anchors_by_condition[base.row_key(anchor)].append(anchor)
    for key in anchors_by_condition:
        anchors_by_condition[key].sort(
            key=lambda row: parse_number(row.get("candidate_rank")) or math.inf
        )

    transforms, transform_manifest = build_effect_transform_index(
        train_rows,
        min_core_heavy_atoms=int(args.min_core_heavy_atoms),
        max_variable_heavy_atoms=int(args.max_variable_heavy_atoms),
    )
    final_rows: list[dict[str, object]] = []
    enumerated_rows: list[dict[str, object]] = []
    internal_counts = []
    composed_counts = []
    for index, row in enumerate(eval_rows, start=1):
        key = base.row_key(row)
        query_props = property_names(row)
        query_task = base.task_key(row)
        anchors = anchor_states(row, anchors_by_condition.get(key, []), transforms)
        if len(anchors) != int(args.candidate_budget):
            raise ValueError(f"Condition {key} has {len(anchors)} v5 anchors; expected 20")
        compatible = compatible_transforms(
            transforms,
            query_task=query_task,
            query_properties=query_props,
        )[: max(1, int(args.max_compatible_transforms))]
        root = CandidateState(
            smiles=base.canonical_smiles(str(row.get("source_smiles", "") or "")),
            source="source_root",
            source_tanimoto=1.0,
            admet_prior_score=float(base.graph.admet_prior_score(row, str(row.get("source_smiles", "") or ""))),
        )
        first = expand_state(
            row,
            root,
            compatible,
            query_task=query_task,
            query_properties=query_props,
            max_transforms_per_fragment=int(args.max_transforms_per_fragment),
            min_retrieval_similarity=float(args.min_retrieval_similarity),
            min_core_heavy_atoms=int(args.min_core_heavy_atoms),
            max_variable_heavy_atoms=int(args.max_variable_heavy_atoms),
        )
        first_ranked = best_by_smiles(
            [*anchors, *first],
            query_properties=query_props,
            min_source_tanimoto=float(args.min_source_tanimoto),
        )
        composed = list(first)
        if int(args.max_steps) >= 2:
            beam = first_ranked[: max(1, int(args.beam_size))]
            for state in beam:
                composed.extend(
                    expand_state(
                        row,
                        state,
                        compatible,
                        query_task=query_task,
                        query_properties=query_props,
                        max_transforms_per_fragment=int(args.max_transforms_per_fragment),
                        min_retrieval_similarity=float(args.min_retrieval_similarity),
                        min_core_heavy_atoms=int(args.min_core_heavy_atoms),
                        max_variable_heavy_atoms=int(args.max_variable_heavy_atoms),
                    )
                )
        ranked_expansion = best_by_smiles(
            composed,
            query_properties=query_props,
            min_source_tanimoto=float(args.min_source_tanimoto),
        )
        anchor_smiles = {state.smiles for state in anchors}
        novel = [state for state in ranked_expansion if state.smiles not in anchor_smiles]
        diagnostic = [*anchors, *novel][: int(args.enumerated_limit)]
        if len(diagnostic) < 96:
            raise ValueError(f"Condition {key} has only {len(diagnostic)} diagnostic candidates")

        final_rows.extend(
            output_row(row, state, rank=rank, selected=rank == 1)
            for rank, state in enumerate(anchors, start=1)
        )
        enumerated_rows.extend(
            output_row(row, state, rank=rank, selected=False)
            for rank, state in enumerate(diagnostic, start=1)
        )
        internal_counts.append(len({state.smiles for state in [*anchors, *composed]}))
        composed_counts.append(len(novel))
        if index % 10 == 0 or index == len(eval_rows):
            print(f"[composed-delta] {index}/{len(eval_rows)} conditions", flush=True)

    write_rows(args.output_csv, final_rows)
    write_rows(args.enumerated_output_csv, enumerated_rows)
    manifest = {
        "protocol": "composed_retrieved_delta_candidate_builder_v1",
        "data_role": "train_only_to_disjoint_train_audit",
        "evaluation_target_access": False,
        "oracle_used_for_selection": False,
        "candidate_budget": int(args.candidate_budget),
        "paper_facing_output_rows": len(final_rows),
        "evaluation_conditions": len(eval_rows),
        "immutable_anchor_budget": int(args.candidate_budget),
        "immutable_anchor_source": str(args.anchor_candidates_csv),
        "max_steps": int(args.max_steps),
        "beam_size": int(args.beam_size),
        "max_transforms_per_fragment": int(args.max_transforms_per_fragment),
        "max_compatible_transforms": int(args.max_compatible_transforms),
        "enumerated_limit": int(args.enumerated_limit),
        "enumerated_output_rows": len(enumerated_rows),
        "mean_internal_unique_candidates": sum(internal_counts) / max(len(internal_counts), 1),
        "mean_novel_composed_candidates": sum(composed_counts) / max(len(composed_counts), 1),
        "min_internal_unique_candidates": min(internal_counts),
        "max_internal_unique_candidates": max(internal_counts),
        "retrieval": {
            "min_similarity": float(args.min_retrieval_similarity),
            "min_source_tanimoto": float(args.min_source_tanimoto),
            "min_core_heavy_atoms": int(args.min_core_heavy_atoms),
            "max_variable_heavy_atoms": int(args.max_variable_heavy_atoms),
            "cross_task_positive_effect_retrieval": True,
        },
        "transform_index": transform_manifest,
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
