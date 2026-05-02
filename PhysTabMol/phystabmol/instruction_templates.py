"""Deterministic instruction text templates and consistency checks."""

from __future__ import annotations

import re
from typing import Any

from .instruction_schema import CONSTRAINTS, EDIT_NAMES, GOAL_NAMES, normalize_spec


GOAL_PHRASES = {
    "increase_logp": ("increase LogP", "make it more lipophilic", "raise the hydrophobic character"),
    "decrease_logp": ("decrease LogP", "make it less lipophilic", "reduce the hydrophobic character"),
    "improve_qed": ("improve QED", "make the molecule more drug-like by QED", "raise the QED score"),
    "reduce_tpsa": ("reduce TPSA", "lower polar surface area", "make it less polar by TPSA"),
    "increase_tpsa": ("increase TPSA", "raise polar surface area", "make it more polar by TPSA"),
    "increase_mw": ("increase molecular weight", "make the molecule heavier", "raise MW"),
    "decrease_mw": ("decrease molecular weight", "make the molecule lighter", "lower MW"),
    "increase_hba": ("increase HBA", "add hydrogen-bond acceptor capacity", "raise acceptor count"),
    "increase_hbd": ("increase HBD", "add hydrogen-bond donor capacity", "raise donor count"),
    "decrease_rb": ("reduce rotatable bonds", "make it less flexible", "lower RB"),
    "lower_sa": ("lower synthetic accessibility score", "make it easier to synthesize", "reduce SA"),
}

CONSTRAINT_PHRASES = {
    "preserve_scaffold": ("preserve the main scaffold", "keep the core scaffold", "do not change the core"),
    "keep_mw_similar": ("keep MW similar", "avoid a large molecular-weight shift", "keep size close to the source"),
    "keep_similarity": ("stay similar to the source molecule", "avoid a drastic structural change", "keep high source similarity"),
    "keep_druglike": ("keep the product drug-like", "stay inside basic drug-likeness filters", "avoid violating drug-like filters"),
}

EDIT_PHRASES = {
    "add_halogen": ("add a halogen", "introduce a halogen substituent", "halogenate the molecule"),
    "remove_halogen": ("remove a halogen", "dehalogenate the molecule", "reduce halogen substitution"),
    "add_heteroatom": ("add a heteroatom", "introduce an N/O/S atom", "increase heteroatom content"),
    "reduce_heteroatom": ("reduce heteroatom content", "remove an N/O/S atom", "make it less heteroatom-rich"),
    "increase_hba": ("increase HBA", "add acceptor functionality", "raise hydrogen-bond acceptors"),
    "increase_hbd": ("increase HBD", "add donor functionality", "raise hydrogen-bond donors"),
    "add_ester": ("add an ester", "introduce ester functionality", "install an ester group"),
    "remove_ester": ("remove an ester", "reduce ester functionality", "avoid the ester group"),
    "add_amide": ("add an amide", "introduce amide functionality", "install an amide group"),
    "remove_amide": ("remove an amide", "reduce amide functionality", "avoid the amide group"),
    "add_amine": ("add an amine", "introduce amine functionality", "install an amine group"),
    "remove_amine": ("remove an amine", "reduce amine functionality", "avoid the amine group"),
    "add_alcohol": ("add an alcohol", "introduce alcohol functionality", "install an alcohol group"),
    "remove_alcohol": ("remove an alcohol", "reduce alcohol functionality", "avoid the alcohol group"),
}

TAG_PATTERNS: dict[str, tuple[str, ...]] = {
    "increase_logp": (r"increase logp", r"raise logp", r"more lipophilic", r"more hydrophobic"),
    "decrease_logp": (r"decrease logp", r"lower logp", r"less lipophilic", r"less hydrophobic"),
    "improve_qed": (r"improve qed", r"raise qed", r"higher qed", r"more drug-like by qed"),
    "reduce_tpsa": (r"reduce tpsa", r"lower tpsa", r"lower polar surface", r"less polar"),
    "increase_tpsa": (r"increase tpsa", r"raise tpsa", r"higher polar surface", r"more polar"),
    "increase_mw": (r"increase molecular weight", r"raise mw", r"heavier"),
    "decrease_mw": (r"decrease molecular weight", r"lower mw", r"lighter"),
    "increase_hba": (r"increase hba", r"raise hba", r"acceptor"),
    "increase_hbd": (r"increase hbd", r"raise hbd", r"donor"),
    "decrease_rb": (r"reduce rotatable", r"lower rb", r"less flexible"),
    "lower_sa": (r"lower synthetic accessibility", r"reduce sa", r"easier to synthesize"),
    "preserve_scaffold": (r"preserve .*scaffold", r"keep .*scaffold", r"do not change .*core", r"keep .*core"),
    "keep_mw_similar": (r"keep mw similar", r"molecular-weight shift", r"size close"),
    "keep_similarity": (r"stay similar", r"high source similarity", r"avoid .*drastic"),
    "keep_druglike": (r"drug-like", r"druglikeness", r"drug-likeness"),
    "add_halogen": (r"add .*halogen", r"introduce .*halogen", r"halogenate"),
    "remove_halogen": (r"remove .*halogen", r"dehalogenate", r"reduce halogen"),
    "add_heteroatom": (r"add .*heteroatom", r"introduce .*n/o/s", r"increase heteroatom"),
    "reduce_heteroatom": (r"reduce heteroatom", r"remove .*n/o/s", r"less heteroatom"),
    "add_ester": (r"add .*ester", r"introduce .*ester", r"install .*ester"),
    "remove_ester": (r"remove .*ester", r"reduce ester", r"avoid .*ester"),
    "add_amide": (r"add .*amide", r"introduce .*amide", r"install .*amide"),
    "remove_amide": (r"remove .*amide", r"reduce amide", r"avoid .*amide"),
    "add_amine": (r"add .*amine", r"introduce .*amine", r"install .*amine"),
    "remove_amine": (r"remove .*amine", r"reduce amine", r"avoid .*amine"),
    "add_alcohol": (r"add .*alcohol", r"introduce .*alcohol", r"install .*alcohol"),
    "remove_alcohol": (r"remove .*alcohol", r"reduce alcohol", r"avoid .*alcohol"),
}


def generate_instruction_texts(spec: str | dict[str, Any], max_variants: int = 5) -> list[dict[str, str]]:
    normalized = normalize_spec(spec)
    goals = _phrases(normalized["goals"], GOAL_PHRASES)
    edits = _phrases(normalized["edits"], EDIT_PHRASES)
    constraints = _phrases(normalized["constraints"], CONSTRAINT_PHRASES)

    primary = _join_clause(goals + edits) or "make the requested verified molecular edit"
    guardrails = _join_clause(constraints)
    suffix = f" while you {guardrails}" if guardrails else ""
    variants = [
        ("template_direct", f"{primary.capitalize()}{suffix}."),
        ("template_source", f"Starting from the source molecule, {primary}{suffix}."),
        ("template_candidate", f"Generate an edited molecule that will {primary}{suffix}."),
        ("template_concise", f"{primary}; {guardrails}." if guardrails else f"{primary}."),
        ("template_planner", f"Plan a small structural edit to {primary}{suffix}."),
    ]
    clean = []
    seen = set()
    for template_id, text in variants:
        text = _clean_text(text)
        if text not in seen and instruction_text_is_consistent(text, normalized)["consistent"]:
            clean.append({"template_id": template_id, "instruction_text": text})
            seen.add(text)
        if len(clean) >= max_variants:
            break
    return clean


def instruction_text_is_consistent(text: str, spec: str | dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_spec(spec)
    allowed = set(normalized["goals"]) | set(normalized["constraints"]) | set(normalized["edits"])
    mentioned = set(extract_instruction_tags(text))
    extra = sorted(mentioned - allowed)
    return {"consistent": not extra, "mentioned_tags": sorted(mentioned), "extra_tags": extra}


def extract_instruction_tags(text: str) -> list[str]:
    lowered = text.lower()
    tags = []
    for tag, patterns in TAG_PATTERNS.items():
        if any(re.search(pattern, lowered) for pattern in patterns):
            tags.append(tag)
    return tags


def _phrases(tags: list[str], bank: dict[str, tuple[str, ...]]) -> list[str]:
    out = []
    for idx, tag in enumerate(tags):
        choices = bank.get(tag)
        if choices:
            out.append(choices[idx % len(choices)])
    return out


def _join_clause(parts: list[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1].upper() + text[1:]
