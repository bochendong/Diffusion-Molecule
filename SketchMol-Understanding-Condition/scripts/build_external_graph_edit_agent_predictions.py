#!/usr/bin/env python3
"""Build external benchmark predictions with a GraphEditDSL agent.

This is the main source-conditioned edit direction: an LLM/planner-facing DSL
produces executable graph-edit actions, RDKit executes the actions on the source
molecule, and the verifier selects among valid edited molecules.  The initial
planner is heuristic but emits the same JSONL DSL records that a learned/LLM
planner can be trained to produce later.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[0]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_external_agentic_revise_predictions as revise  # noqa: E402


@dataclass(frozen=True)
class GraphEditAction:
    op: str
    site: int | None = None
    bond: tuple[int, int] | None = None
    atom: str = ""
    fragment: str = ""
    bond_order: str = ""
    prop: str = ""
    direction: str = ""
    reason: str = ""
    policy_score: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-csv", required=True, type=Path)
    parser.add_argument("--prediction-csv", required=True, type=Path)
    parser.add_argument("--plan-jsonl", type=Path, default=None)
    parser.add_argument("--direct-prediction-csv", type=Path, default=None)
    parser.add_argument("--direct-smiles-column", default="generated_smiles")
    parser.add_argument("--planner-mode", choices=("heuristic_graph_dsl", "policy_graph_dsl"), default="heuristic_graph_dsl")
    parser.add_argument("--selection-mode", choices=("score", "similarity_first"), default="similarity_first")
    parser.add_argument("--min-source-tanimoto", type=float, default=0.4)
    parser.add_argument("--planner-steps", type=int, default=1)
    parser.add_argument("--beam-size", type=int, default=64)
    parser.add_argument("--site-limit", type=int, default=24)
    parser.add_argument("--max-plans-per-property", type=int, default=80)
    parser.add_argument("--max-candidates-per-parent", type=int, default=256)
    parser.add_argument("--max-candidates-per-row", type=int, default=4096)
    parser.add_argument("--similarity-first-min-local-success-fraction", type=float, default=1.0)
    parser.add_argument("--property-weight", type=float, default=100.0)
    parser.add_argument("--distance-weight", type=float, default=10.0)
    parser.add_argument("--similarity-weight", type=float, default=30.0)
    parser.add_argument("--similarity-bonus", type=float, default=80.0)
    parser.add_argument("--copy-penalty", type=float, default=8.0)
    parser.add_argument("--method", default="external_graph_edit_agent")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rng = random.Random(int(args.seed))
    rows = revise.read_rows(args.rows_csv)
    direct_rows = revise.read_direct_predictions(args.direct_prediction_csv) if args.direct_prediction_csv else {}
    output_rows = []
    plan_records = []
    for index, row in enumerate(rows):
        direct_row = direct_rows.get(revise.row_key(row), {})
        output_row, records = predict_row(row, direct_row=direct_row, args=args, rng=rng)
        output_rows.append(output_row)
        plan_records.extend(records)
        if (index + 1) % 100 == 0 or index + 1 == len(rows):
            print(f"[graph-edit-agent] wrote {index + 1}/{len(rows)} rows", flush=True)
    revise.write_rows(args.prediction_csv, output_rows)
    if args.plan_jsonl:
        args.plan_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.plan_jsonl.open("w", encoding="utf-8") as handle:
            for record in plan_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary = summarize_graph_agent_rows(output_rows)
    args.prediction_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def predict_row(
    row: Mapping[str, str],
    *,
    direct_row: Mapping[str, str],
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    source_smiles = str(row.get("source_smiles", "") or "").strip()
    direct_smiles = str(direct_row.get(str(args.direct_smiles_column), "") or "").strip()
    candidates: list[revise.Candidate] = []
    if source_smiles:
        candidates.append(revise.Candidate(source_smiles, "dsl:source_copy", "source_copy"))
    if direct_smiles:
        candidates.append(revise.Candidate(direct_smiles, "dsl:direct_model_proposal", "direct_model"))
    plan_records = []
    seen = {revise.safe_canonical(candidate.smiles) for candidate in candidates if revise.safe_canonical(candidate.smiles)}
    total_plan_count = 0
    planner_mode = str(args.planner_mode)
    planner_steps = max(1, int(args.planner_steps))
    for step in range(1, planner_steps + 1):
        if planner_mode == "heuristic_graph_dsl" and planner_steps == 1:
            parent_smiles = [revise.safe_canonical(source_smiles) or source_smiles]
        else:
            ranked = revise.rank_candidates(row, candidates, args=args)
            parent_smiles = [item.canonical_smiles for item in ranked[: max(1, int(args.beam_size))]]
            if step == 1 and source_smiles:
                parent_smiles.insert(0, revise.safe_canonical(source_smiles) or source_smiles)
        parent_smiles = dedupe_strings(parent_smiles)
        if planner_mode == "heuristic_graph_dsl":
            rng.shuffle(parent_smiles)
        for parent in parent_smiles:
            if len(seen) >= int(args.max_candidates_per_row):
                break
            actions = plan_graph_edit_actions(
                row,
                source_smiles=parent,
                site_limit=int(args.site_limit),
                max_plans_per_property=int(args.max_plans_per_property),
                planner_mode=planner_mode,
            )
            if planner_mode == "heuristic_graph_dsl":
                rng.shuffle(actions)
            total_plan_count += len(actions)
            executed_for_parent = 0
            for action in actions:
                if len(seen) >= int(args.max_candidates_per_row):
                    break
                if executed_for_parent >= max(1, int(args.max_candidates_per_parent)):
                    break
                executed_for_parent += 1
                generated = execute_graph_edit_action(parent, action)
                canonical = revise.safe_canonical(generated)
                plan_records.append(
                    {
                        "condition_id": str(row.get("condition_id") or ""),
                        "source_smiles": source_smiles,
                        "parent_smiles": parent,
                        "step": step,
                        "action": asdict(action),
                        "generated_smiles": canonical,
                        "valid": bool(canonical),
                    }
                )
                if not canonical or canonical in seen:
                    continue
                seen.add(canonical)
                candidates.append(revise.Candidate(canonical, f"step{step}:dsl:{action.to_json()}", "graph_edit_dsl"))
        if len(seen) >= int(args.max_candidates_per_row):
            break
    ranked = revise.rank_candidates(row, candidates, args=args)
    best = ranked[0] if ranked else revise.score_candidate(row, revise.Candidate(source_smiles, "dsl:source_copy", "source_copy"), args=args)
    out = dict(row)
    out["generated_smiles"] = best.canonical_smiles
    out["method"] = str(args.method)
    out["direct_generated_smiles"] = direct_smiles
    out["graph_edit_action_trace"] = best.candidate.action_trace
    out["graph_edit_candidate_source"] = best.candidate.source
    out["graph_edit_planner_mode"] = planner_mode
    out["graph_edit_planner_steps"] = planner_steps
    out["graph_edit_plan_count"] = total_plan_count
    out["graph_edit_executed_plan_count"] = len(plan_records)
    out["graph_edit_candidate_count"] = len(candidates)
    out["graph_edit_valid_candidate_count"] = len([item for item in ranked if item.canonical_smiles])
    out["graph_edit_best_score"] = revise.format_float(best.score, digits=6)
    out["graph_edit_source_tanimoto"] = "" if math.isnan(best.source_tanimoto) else revise.format_float(best.source_tanimoto, digits=6)
    out["graph_edit_source_similarity_success"] = "True" if best.source_similarity_success else "False"
    out["graph_edit_local_success_fraction"] = revise.format_float(best.local_success_fraction, digits=6)
    out["graph_edit_all_evaluated_local_success"] = "True" if best.all_evaluated_local_success else "False"
    out["graph_edit_evaluated_local_property_count"] = best.evaluated_local_property_count
    out["graph_edit_local_property_distance"] = revise.format_float(best.local_property_distance, digits=6)
    out["graph_edit_missing_local_properties"] = ",".join(best.missing_local_properties)
    return out, plan_records


def plan_graph_edit_actions(
    row: Mapping[str, str],
    *,
    source_smiles: str,
    site_limit: int,
    max_plans_per_property: int,
    planner_mode: str = "heuristic_graph_dsl",
) -> list[GraphEditAction]:
    if str(planner_mode) == "policy_graph_dsl":
        return plan_policy_graph_edit_actions(
            row,
            source_smiles=source_smiles,
            site_limit=site_limit,
            max_plans_per_property=max_plans_per_property,
        )
    props = revise.parse_list(row.get("external_task_properties") or row.get("condition_properties"))
    directions = revise.parse_json_dict(row.get("external_property_directions_json"), revise.DEFAULT_DIRECTION)
    sites = editable_atom_sites(source_smiles, site_limit=site_limit)
    bonds = editable_bond_sites(source_smiles, site_limit=site_limit)
    actions: list[GraphEditAction] = []
    for prop in props:
        prop = revise.canonical_prop(prop)
        direction = str(directions.get(prop, revise.DEFAULT_DIRECTION.get(prop, "increase"))).lower()
        templates = action_templates_for_property(prop, direction)
        prop_actions: list[GraphEditAction] = []
        for template in templates:
            for site in sites:
                if len(prop_actions) >= max(1, int(max_plans_per_property)):
                    break
                if template.op in {"add_atom", "add_fragment", "replace_atom", "delete_terminal_atom"}:
                    prop_actions.append(
                        GraphEditAction(
                            op=template.op,
                            site=site,
                            atom=template.atom,
                            fragment=template.fragment,
                            prop=prop,
                            direction=direction,
                            reason=template.reason,
                        )
                    )
            for bond in bonds:
                if len(prop_actions) >= max(1, int(max_plans_per_property)):
                    break
                if template.op == "change_bond_order":
                    prop_actions.append(
                        GraphEditAction(
                            op=template.op,
                            bond=bond,
                            bond_order=template.bond_order,
                            prop=prop,
                            direction=direction,
                            reason=template.reason,
                        )
                    )
        actions.extend(prop_actions)
    if not actions:
        for site in sites[: max(1, min(8, len(sites)))]:
            actions.extend(
                [
                    GraphEditAction("add_atom", site=site, atom="F", reason="generic_conservative_polarity"),
                    GraphEditAction("add_fragment", site=site, fragment="C", reason="generic_conservative_methyl"),
                ]
            )
    return dedupe_actions(actions)


def plan_policy_graph_edit_actions(
    row: Mapping[str, str],
    *,
    source_smiles: str,
    site_limit: int,
    max_plans_per_property: int,
) -> list[GraphEditAction]:
    props = revise.parse_list(row.get("external_task_properties") or row.get("condition_properties"))
    directions = revise.parse_json_dict(row.get("external_property_directions_json"), revise.DEFAULT_DIRECTION)
    sites = editable_atom_sites(source_smiles, site_limit=site_limit)
    bonds = editable_bond_sites(source_smiles, site_limit=site_limit)
    out: list[GraphEditAction] = []
    for prop in props:
        prop = revise.canonical_prop(prop)
        direction = str(directions.get(prop, revise.DEFAULT_DIRECTION.get(prop, "increase"))).lower()
        prop_actions: list[GraphEditAction] = []
        for template in policy_action_templates_for_property(prop, direction):
            if template.op in {"add_atom", "add_fragment", "replace_atom", "delete_terminal_atom"}:
                prop_actions.extend(
                    replace(template, site=site, prop=prop, direction=direction)
                    for site in sites
                )
            if template.op == "change_bond_order":
                prop_actions.extend(
                    replace(template, bond=bond, prop=prop, direction=direction)
                    for bond in bonds
                )
        prop_actions = score_policy_actions(row, source_smiles, dedupe_actions(prop_actions))
        out.extend(prop_actions[: max(1, int(max_plans_per_property))])
    if not out:
        out = score_policy_actions(
            row,
            source_smiles,
            [
                GraphEditAction("add_atom", site=site, atom="F", reason="policy_generic_small_substituent")
                for site in sites[: max(1, min(8, len(sites)))]
            ],
        )
    return dedupe_actions(out)


def action_templates_for_property(prop: str, direction: str) -> list[GraphEditAction]:
    if prop in {"plogp", "logp"} and direction == "increase":
        return [
            GraphEditAction("add_fragment", fragment="C", reason="increase_hydrophobicity"),
            GraphEditAction("add_fragment", fragment="CC", reason="increase_hydrophobicity"),
            GraphEditAction("add_atom", atom="Cl", reason="increase_hydrophobicity"),
            GraphEditAction("add_atom", atom="F", reason="increase_hydrophobicity"),
            GraphEditAction("add_atom", atom="Br", reason="increase_hydrophobicity"),
            GraphEditAction("add_fragment", fragment="c1ccccc1", reason="increase_aromatic_hydrophobicity"),
        ]
    if prop in {"plogp", "logp"}:
        return [
            GraphEditAction("add_atom", atom="O", reason="decrease_hydrophobicity"),
            GraphEditAction("add_atom", atom="N", reason="decrease_hydrophobicity"),
            GraphEditAction("add_fragment", fragment="C(=O)O", reason="decrease_hydrophobicity"),
            GraphEditAction("delete_terminal_atom", reason="remove_terminal_hydrophobe"),
        ]
    if prop == "qed" and direction == "increase":
        return [
            GraphEditAction("add_atom", atom="F", reason="qed_small_substituent"),
            GraphEditAction("add_atom", atom="Cl", reason="qed_small_substituent"),
            GraphEditAction("add_atom", atom="O", reason="qed_hbond_acceptor"),
            GraphEditAction("add_atom", atom="N", reason="qed_hbond_acceptor"),
            GraphEditAction("add_fragment", fragment="C", reason="qed_size_adjustment"),
        ]
    if prop in {"sas", "sa"} and direction == "decrease":
        return [
            GraphEditAction("delete_terminal_atom", reason="simplify_synthesis"),
            GraphEditAction("replace_atom", atom="C", reason="simplify_atom_type"),
            GraphEditAction("replace_atom", atom="N", reason="simplify_atom_type"),
            GraphEditAction("change_bond_order", bond_order="single", reason="simplify_bond_order"),
        ]
    return [
        GraphEditAction("add_atom", atom="F", reason=f"{prop}_{direction}_small_substituent"),
        GraphEditAction("add_atom", atom="O", reason=f"{prop}_{direction}_heteroatom"),
        GraphEditAction("add_fragment", fragment="C", reason=f"{prop}_{direction}_methyl"),
        GraphEditAction("delete_terminal_atom", reason=f"{prop}_{direction}_terminal_prune"),
    ]


def policy_action_templates_for_property(prop: str, direction: str) -> list[GraphEditAction]:
    hydrophobic = [
        GraphEditAction("add_fragment", fragment="C", reason="policy_logp_methyl"),
        GraphEditAction("add_fragment", fragment="CC", reason="policy_logp_ethyl"),
        GraphEditAction("add_fragment", fragment="c1ccccc1", reason="policy_logp_phenyl"),
        GraphEditAction("add_atom", atom="F", reason="policy_logp_halogen"),
        GraphEditAction("add_atom", atom="Cl", reason="policy_logp_halogen"),
        GraphEditAction("add_atom", atom="Br", reason="policy_logp_halogen"),
    ]
    polar = [
        GraphEditAction("add_atom", atom="O", reason="policy_polar_heteroatom"),
        GraphEditAction("add_atom", atom="N", reason="policy_polar_heteroatom"),
        GraphEditAction("add_atom", atom="S", reason="policy_polar_heteroatom"),
        GraphEditAction("add_fragment", fragment="O", reason="policy_hydroxyl"),
        GraphEditAction("add_fragment", fragment="N", reason="policy_amino"),
        GraphEditAction("add_fragment", fragment="C#N", reason="policy_cyano"),
        GraphEditAction("add_fragment", fragment="C(=O)O", reason="policy_carboxyl"),
        GraphEditAction("add_fragment", fragment="C(=O)N", reason="policy_amide"),
    ]
    simplify = [
        GraphEditAction("delete_terminal_atom", reason="policy_terminal_prune"),
        GraphEditAction("replace_atom", atom="C", reason="policy_simplify_atom_type"),
        GraphEditAction("replace_atom", atom="N", reason="policy_simplify_atom_type"),
        GraphEditAction("replace_atom", atom="O", reason="policy_simplify_atom_type"),
        GraphEditAction("change_bond_order", bond_order="single", reason="policy_simplify_bond"),
    ]
    if prop in {"plogp", "logp"} and direction == "increase":
        return hydrophobic + [
            GraphEditAction("change_bond_order", bond_order="double", reason="policy_logp_unsaturate"),
        ]
    if prop in {"plogp", "logp"}:
        return polar + simplify
    if prop == "qed" and direction == "increase":
        return [
            GraphEditAction("add_atom", atom="F", reason="policy_qed_small_halogen"),
            GraphEditAction("add_atom", atom="Cl", reason="policy_qed_small_halogen"),
            GraphEditAction("add_atom", atom="O", reason="policy_qed_hbond"),
            GraphEditAction("add_atom", atom="N", reason="policy_qed_hbond"),
            GraphEditAction("add_fragment", fragment="C", reason="policy_qed_small_size"),
            GraphEditAction("add_fragment", fragment="C#N", reason="policy_qed_cyano"),
        ] + simplify[:2]
    if prop in {"sas", "sa"} and direction == "decrease":
        return simplify + [
            GraphEditAction("add_atom", atom="F", reason="policy_sa_small_substituent"),
            GraphEditAction("add_fragment", fragment="C", reason="policy_sa_small_alkyl"),
        ]
    return [
        GraphEditAction("add_atom", atom="F", reason=f"policy_{prop}_{direction}_small_halogen"),
        GraphEditAction("add_atom", atom="Cl", reason=f"policy_{prop}_{direction}_halogen"),
        GraphEditAction("add_atom", atom="O", reason=f"policy_{prop}_{direction}_heteroatom"),
        GraphEditAction("add_atom", atom="N", reason=f"policy_{prop}_{direction}_heteroatom"),
        GraphEditAction("add_atom", atom="S", reason=f"policy_{prop}_{direction}_heteroatom"),
        GraphEditAction("add_fragment", fragment="C", reason=f"policy_{prop}_{direction}_methyl"),
        GraphEditAction("add_fragment", fragment="CC", reason=f"policy_{prop}_{direction}_ethyl"),
        GraphEditAction("add_fragment", fragment="C#N", reason=f"policy_{prop}_{direction}_cyano"),
        GraphEditAction("add_fragment", fragment="C(=O)N", reason=f"policy_{prop}_{direction}_amide"),
        GraphEditAction("delete_terminal_atom", reason=f"policy_{prop}_{direction}_terminal_prune"),
    ]


def score_policy_actions(
    row: Mapping[str, str],
    source_smiles: str,
    actions: Sequence[GraphEditAction],
) -> list[GraphEditAction]:
    try:
        from rdkit import Chem
    except ImportError:
        return [replace(action, policy_score=policy_template_score(action)) for action in actions]
    mol = Chem.MolFromSmiles(str(source_smiles or ""))
    if mol is None:
        return [replace(action, policy_score=policy_template_score(action)) for action in actions]
    task_props = set(revise.parse_list(row.get("external_task_properties") or row.get("condition_properties")))
    scored = []
    for action in actions:
        score = policy_template_score(action)
        score += site_policy_score(mol, action)
        if action.prop in task_props:
            score += 0.5
        if "policy" in action.reason:
            score += 0.25
        scored.append(replace(action, policy_score=round(float(score), 6)))
    return sorted(scored, key=lambda action: (action.policy_score, action.reason, action.op), reverse=True)


def policy_template_score(action: GraphEditAction) -> float:
    score = 1.0
    if action.op == "add_fragment":
        fragment_score = {
            "C": 1.3,
            "CC": 1.15,
            "O": 1.0,
            "N": 1.0,
            "C#N": 1.1,
            "C(=O)O": 0.9,
            "C(=O)N": 1.0,
            "c1ccccc1": 0.65,
        }
        score += fragment_score.get(action.fragment, 0.75)
    if action.op == "add_atom":
        atom_score = {"F": 1.35, "Cl": 1.15, "O": 1.05, "N": 1.05, "C": 0.95, "Br": 0.75, "S": 0.7}
        score += atom_score.get(action.atom, 0.5)
    if action.op == "delete_terminal_atom":
        score += 1.25
    if action.op == "replace_atom":
        score += 0.8
    if action.op == "change_bond_order":
        score += 0.45 if action.bond_order == "single" else 0.2
    if action.prop in {"sas", "sa"} and action.direction == "decrease":
        score += 0.8 if action.op in {"delete_terminal_atom", "replace_atom", "change_bond_order"} else 0.1
    if action.prop in {"plogp", "logp"} and action.direction == "increase":
        score += 0.8 if action.atom in {"F", "Cl", "Br"} or action.fragment in {"C", "CC"} else 0.2
    if action.prop == "qed" and action.direction == "increase":
        score += 0.55 if action.atom in {"F", "Cl", "O", "N"} or action.fragment in {"C", "C#N"} else 0.0
    return score


def site_policy_score(mol, action: GraphEditAction) -> float:
    score = 0.0
    if action.site is not None and 0 <= int(action.site) < mol.GetNumAtoms():
        atom = mol.GetAtomWithIdx(int(action.site))
        if not atom.IsInRing():
            score += 0.7
        if atom.GetDegree() <= 2:
            score += 0.5
        if atom.GetDegree() == 1:
            score += 0.35
        if atom.GetSymbol() == "C":
            score += 0.25
        if atom.GetIsAromatic():
            score -= 0.4
        if action.op == "delete_terminal_atom" and atom.GetDegree() != 1:
            score -= 3.0
    if action.bond:
        begin, end = action.bond
        bond = mol.GetBondBetweenAtoms(int(begin), int(end))
        if bond is not None:
            if not bond.IsInRing():
                score += 0.45
            if bond.GetIsAromatic():
                score -= 1.0
    return score


def editable_atom_sites(source_smiles: str, *, site_limit: int) -> list[int]:
    try:
        from rdkit import Chem
    except ImportError:
        return [0]
    mol = Chem.MolFromSmiles(str(source_smiles or ""))
    if mol is None:
        return [0]
    atoms = list(mol.GetAtoms())
    atoms.sort(key=lambda atom: (atom.IsInRing(), atom.GetDegree(), atom.GetIdx()))
    return [atom.GetIdx() for atom in atoms[: max(1, int(site_limit))]]


def editable_bond_sites(source_smiles: str, *, site_limit: int) -> list[tuple[int, int]]:
    try:
        from rdkit import Chem
    except ImportError:
        return []
    mol = Chem.MolFromSmiles(str(source_smiles or ""))
    if mol is None:
        return []
    out = []
    for bond in mol.GetBonds():
        if bond.GetIsAromatic() or bond.IsInRing():
            continue
        out.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        if len(out) >= max(1, int(site_limit)):
            break
    return out


def execute_graph_edit_action(source_smiles: str, action: GraphEditAction) -> str:
    try:
        from rdkit import Chem
    except ImportError:
        return ""
    mol = Chem.MolFromSmiles(str(source_smiles or ""))
    if mol is None:
        return ""
    try:
        if action.op == "add_atom":
            rw = Chem.RWMol(mol)
            if action.site is None or action.site >= mol.GetNumAtoms():
                return ""
            new_index = rw.AddAtom(Chem.Atom(action.atom or "C"))
            rw.AddBond(int(action.site), new_index, Chem.BondType.SINGLE)
            return sanitize_to_smiles(rw.GetMol())
        if action.op == "add_fragment":
            if action.site is None or action.site >= mol.GetNumAtoms():
                return ""
            fragment = Chem.MolFromSmiles(action.fragment or "C")
            if fragment is None or fragment.GetNumAtoms() == 0:
                return ""
            combo = Chem.CombineMols(mol, fragment)
            rw = Chem.RWMol(combo)
            rw.AddBond(int(action.site), mol.GetNumAtoms(), Chem.BondType.SINGLE)
            return sanitize_to_smiles(rw.GetMol())
        if action.op == "replace_atom":
            rw = Chem.RWMol(mol)
            if action.site is None or action.site >= mol.GetNumAtoms():
                return ""
            atom = rw.GetAtomWithIdx(int(action.site))
            if atom.GetIsAromatic():
                return ""
            atom.SetAtomicNum(Chem.Atom(action.atom or "C").GetAtomicNum())
            return sanitize_to_smiles(rw.GetMol())
        if action.op == "delete_terminal_atom":
            rw = Chem.RWMol(mol)
            if action.site is None or action.site >= mol.GetNumAtoms() or mol.GetNumAtoms() <= 1:
                return ""
            atom = mol.GetAtomWithIdx(int(action.site))
            if atom.GetDegree() != 1:
                return ""
            rw.RemoveAtom(int(action.site))
            return sanitize_to_smiles(rw.GetMol())
        if action.op == "change_bond_order" and action.bond:
            begin, end = action.bond
            bond = mol.GetBondBetweenAtoms(int(begin), int(end))
            if bond is None or bond.GetIsAromatic() or bond.IsInRing():
                return ""
            order = {
                "single": Chem.BondType.SINGLE,
                "double": Chem.BondType.DOUBLE,
                "triple": Chem.BondType.TRIPLE,
            }.get(action.bond_order or "single", Chem.BondType.SINGLE)
            rw = Chem.RWMol(mol)
            rw.RemoveBond(int(begin), int(end))
            rw.AddBond(int(begin), int(end), order)
            return sanitize_to_smiles(rw.GetMol())
    except Exception:
        return ""
    return ""


def sanitize_to_smiles(mol) -> str:
    try:
        from rdkit import Chem

        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return ""


def dedupe_actions(actions: Sequence[GraphEditAction]) -> list[GraphEditAction]:
    out = []
    seen = set()
    for action in actions:
        key = action_identity(action)
        if key in seen:
            continue
        seen.add(key)
        out.append(action)
    return out


def action_identity(action: GraphEditAction) -> tuple[object, ...]:
    return (
        action.op,
        action.site,
        action.bond,
        action.atom,
        action.fragment,
        action.bond_order,
        action.prop,
        action.direction,
        action.reason,
    )


def dedupe_strings(values: Sequence[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        value = str(value or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def summarize_graph_agent_rows(rows: list[Mapping[str, object]]) -> dict[str, object]:
    return {
        "rows": len(rows),
        "valid_generated": sum(1 for row in rows if str(row.get("generated_smiles") or "").strip()),
        "source_similarity_success_rate": revise.mean_bool(rows, "graph_edit_source_similarity_success"),
        "all_evaluated_local_success_rate": revise.mean_bool(rows, "graph_edit_all_evaluated_local_success"),
        "mean_local_success_fraction": revise.mean_float(rows, "graph_edit_local_success_fraction"),
        "mean_source_tanimoto": revise.mean_float(rows, "graph_edit_source_tanimoto"),
        "mean_plan_count": revise.mean_float(rows, "graph_edit_plan_count"),
        "mean_executed_plan_count": revise.mean_float(rows, "graph_edit_executed_plan_count"),
        "mean_candidate_count": revise.mean_float(rows, "graph_edit_candidate_count"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
