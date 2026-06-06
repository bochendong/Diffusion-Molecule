"""Build dual-stream examples from pure SMILES rows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .data import SmilesPair
from .edit_corruption import edit_operations, levenshtein_distance, make_edit_example, operation_counts
from .hierarchy import adaptive_polymerization_levels, hierarchy_alignment
from .tokenization import detokenize_smiles, tokenize_smiles


@dataclass(frozen=True)
class DualStreamExample:
    """One row consumed by the pure-SMILES dual-stream experiment."""

    sample_id: str
    mode: str
    split: str
    source_smiles: str
    corrupted_smiles: str
    target_smiles: str
    source_tokens: tuple[str, ...]
    corrupted_tokens: tuple[str, ...]
    target_tokens: tuple[str, ...]
    edit: dict[str, float | int | str] = field(default_factory=dict)
    alignment: dict[str, float] = field(default_factory=dict)
    hierarchy: dict[str, list[str]] = field(default_factory=dict)
    instruction: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["source_tokens"] = list(self.source_tokens)
        payload["corrupted_tokens"] = list(self.corrupted_tokens)
        payload["target_tokens"] = list(self.target_tokens)
        return payload


def build_dual_stream_example(pair: SmilesPair, *, seed: int = 0) -> DualStreamExample:
    """Create a dual-stream row from one source-target or self-supervised pair."""

    source_tokens = tokenize_smiles(pair.source_smiles)
    target_tokens = tokenize_smiles(pair.target_smiles)

    if detokenize_smiles(source_tokens) == detokenize_smiles(target_tokens):
        edit_example = make_edit_example(pair.target_smiles, seed=seed)
        mode = "self_supervised_corruption"
        corrupted_tokens = list(edit_example.corrupted_tokens)
        corrupted_smiles = edit_example.corrupted_smiles
        ops = list(edit_example.operations)
        policy = edit_example.policy
    else:
        mode = "pair_edit"
        corrupted_tokens = list(source_tokens)
        corrupted_smiles = detokenize_smiles(corrupted_tokens)
        ops = edit_operations(corrupted_tokens, target_tokens)
        policy = "source_to_target"

    counts = operation_counts(ops)
    edit_payload: dict[str, float | int | str] = {
        "policy": policy,
        "levenshtein": levenshtein_distance(corrupted_tokens, target_tokens),
        "source_length": len(corrupted_tokens),
        "target_length": len(target_tokens),
        "length_delta": len(target_tokens) - len(corrupted_tokens),
        **counts,
    }
    alignment_payload = hierarchy_alignment(corrupted_tokens, target_tokens)
    hierarchy_payload = {
        f"input_{key}": value for key, value in adaptive_polymerization_levels(corrupted_tokens).items()
    }
    hierarchy_payload.update(
        {f"target_{key}": value for key, value in adaptive_polymerization_levels(target_tokens).items()}
    )

    return DualStreamExample(
        sample_id=pair.sample_id,
        mode=mode,
        split=pair.split,
        source_smiles=detokenize_smiles(source_tokens),
        corrupted_smiles=corrupted_smiles,
        target_smiles=detokenize_smiles(target_tokens),
        source_tokens=tuple(source_tokens),
        corrupted_tokens=tuple(corrupted_tokens),
        target_tokens=tuple(target_tokens),
        edit=edit_payload,
        alignment=alignment_payload,
        hierarchy=hierarchy_payload,
        instruction=pair.instruction,
        metadata=pair.metadata,
    )


def summarize_examples(examples: list[DualStreamExample]) -> dict[str, object]:
    if not examples:
        return {"rows": 0}
    modes: dict[str, int] = {}
    splits: dict[str, int] = {}
    total_edit_distance = 0.0
    total_fragment_jaccard = 0.0
    for example in examples:
        modes[example.mode] = modes.get(example.mode, 0) + 1
        splits[example.split] = splits.get(example.split, 0) + 1
        total_edit_distance += float(example.edit.get("levenshtein", 0.0))
        total_fragment_jaccard += float(example.alignment.get("fragment_jaccard", 0.0))
    rows = len(examples)
    return {
        "rows": rows,
        "modes": modes,
        "splits": splits,
        "mean_edit_distance": total_edit_distance / rows,
        "mean_fragment_jaccard": total_fragment_jaccard / rows,
    }

