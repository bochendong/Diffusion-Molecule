from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "experiments" / "unified_latent_table1"
SCRIPT = EXPERIMENT_DIR / "elementwise_additive_semantic_set_compiler_v11.py"
GATE = EXPERIMENT_DIR / "gate_elementwise_additive_semantic_set_compiler_v11.py"
MANIFEST = EXPERIMENT_DIR / "elementwise_additive_semantic_set_compiler_v11_preregistration.json"
RUNNER = EXPERIMENT_DIR / "run_elementwise_additive_semantic_set_compiler_v11.sh"
SUBMITTER = EXPERIMENT_DIR / "submit_elementwise_additive_semantic_set_compiler_v11.sh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_is_locked_final_fresh_representation_attempt() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["implementation_sha256"] == sha256(SCRIPT)
    assert manifest["gate_implementation_sha256"] == sha256(GATE)
    assert manifest["fit_property_cardinalities"] == [1]
    assert manifest["probe_property_cardinalities"] == [5, 6, 7]
    assert manifest["v10_probe_reuse"] is False
    assert manifest["final_representation_attempt"] is True
    assert manifest["threshold_search"] is False


def test_compiler_is_shared_elementwise_then_unordered_sum() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    body = script.split("class ElementwiseAdditiveCompiler", 1)[1].split(
        "def singleton_training_tensors", 1
    )[0]
    assert "self.element_outputs(embeddings)" in body
    assert "(element_coefficients * active).sum(dim=1)" in body
    assert "(token_delta * active).sum(dim=1)" in body
    assert "self.numeric_basis[0].unsqueeze(0)" in body
    assert "position" not in body
    assert "cardinality.sqrt" not in body


def test_training_is_singleton_only_and_probe_uses_new_templates() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["fit_templates"] == ["canonical", "train_paraphrase", "schema"]
    assert set(manifest["probe_templates"]) == {
        "probe_enhance_suppress",
        "probe_promote_reduce",
    }
    script = SCRIPT.read_text(encoding="utf-8")
    assert 'raise ValueError("V11 training accepts singleton examples only")' in script
    assert 'if {len(row) for row in probe_rows} != {5, 6, 7}' in script


def test_target_oracle_generation_and_ranking_are_unavailable() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for key in (
        "generation_target_access",
        "molecule_target_path_accepted",
        "property_oracle_access",
        "molecule_generation",
        "molecular_candidate_ranking",
        "oracle_selection",
        "official_test_access",
    ):
        assert manifest[key] is False
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'add_argument("--target' not in text
    assert 'add_argument("--oracle' not in text
    assert "sealed_evaluation_targets" not in RUNNER.read_text(encoding="utf-8")


def test_science_gate_is_separate_and_final() -> None:
    gate = GATE.read_text(encoding="utf-8")
    assert "advance_frozen_elementwise_compiler_to_three_exact_n20_pilots" in gate
    assert "final_stop_llm_core_generation_mechanism_claim" in gate
    assert "no_more_representation_retries_and_no_molecule_pilots" in gate
    assert "return 0" in gate
    submitter = SUBMITTER.read_text(encoding="utf-8")
    assert "uca-elemset-v11-exec" in submitter
    assert "uca-elemset-v11-scigate" in submitter
    assert "afterok:$execution_job_id" in submitter
