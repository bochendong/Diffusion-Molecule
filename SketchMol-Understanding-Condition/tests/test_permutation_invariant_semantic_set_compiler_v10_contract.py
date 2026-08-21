from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "experiments" / "unified_latent_table1"
SCRIPT = EXPERIMENT_DIR / "permutation_invariant_semantic_set_compiler_v10.py"
GATE = EXPERIMENT_DIR / "gate_permutation_invariant_semantic_set_compiler_v10.py"
MANIFEST = EXPERIMENT_DIR / "permutation_invariant_semantic_set_compiler_v10_preregistration.json"
RUNNER = EXPERIMENT_DIR / "run_permutation_invariant_semantic_set_compiler_v10.sh"
SUBMITTER = EXPERIMENT_DIR / "submit_permutation_invariant_semantic_set_compiler_v10.sh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_locks_implementation_and_target_free_contract() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["implementation_sha256"] == sha256(SCRIPT)
    assert manifest["gate_implementation_sha256"] == sha256(GATE)
    assert manifest["representation"] == "permutation_invariant_signed_property_set"
    assert manifest["fit_property_cardinalities"] == [1, 2]
    assert manifest["probe_property_cardinalities"] == [3, 4]
    assert manifest["molecule_target_path_accepted"] is False
    assert manifest["property_oracle_access"] is False
    assert manifest["molecule_generation"] is False
    assert manifest["molecular_candidate_ranking"] is False
    assert manifest["threshold_search"] is False


def test_deepsets_is_exactly_permutation_invariant() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    class_body = script.split("class SemanticSetCompiler", 1)[1].split(
        "@torch.no_grad()", 1
    )[0]
    assert "self.phi(element_embeddings.float())" in class_body
    assert "(encoded * mask).sum(dim=1)" in class_body
    assert "cardinality.sqrt()" in class_body
    assert "position" not in class_body
    assert "self.rho(pooled)" in class_body


def test_fit_and_probe_property_sets_are_cardinality_disjoint() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert set(manifest["fit_property_cardinalities"]).isdisjoint(
        manifest["probe_property_cardinalities"]
    )
    assert manifest["probe_property_sets_unseen_in_fit"] is True
    script = SCRIPT.read_text(encoding="utf-8")
    assert "if fit_sets & probe_sets" in script
    assert 'raise ValueError("V10 unseen property-set leakage")' in script


def test_cli_and_runner_offer_no_target_or_oracle_path() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert 'add_argument("--target' not in script
    assert 'add_argument("--oracle' not in script
    assert "sealed_evaluation_targets" not in runner
    assert "SUCC_V10_STAGE=execute" in SUBMITTER.read_text(encoding="utf-8")
    assert "SUCC_V10_STAGE=gate" in SUBMITTER.read_text(encoding="utf-8")


def test_science_stop_is_separate_from_execution_status() -> None:
    gate = GATE.read_text(encoding="utf-8")
    assert "stop_llm_core_generation_claim_before_molecule_pilots" in gate
    assert "advance_semantic_set_compiler_to_three_exact_n20_pilots" in gate
    assert "return 0" in gate
    submitter = SUBMITTER.read_text(encoding="utf-8")
    assert "uca-setcomp-v10-exec" in submitter
    assert "uca-setcomp-v10-scigate" in submitter
    assert "afterok:$execution_job_id" in submitter
