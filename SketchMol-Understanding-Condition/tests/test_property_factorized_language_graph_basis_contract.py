from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "unified_latent_table1"
IMPLEMENTATION = EXPERIMENT / "property_factorized_language_graph_basis_v1.py"
PREREGISTRATION = EXPERIMENT / "property_factorized_language_graph_basis_v1_preregistration.json"
RUNNER = EXPERIMENT / "run_property_factorized_language_graph_basis_v1.sh"
SUBMITTER = EXPERIMENT / "submit_property_factorized_language_graph_basis_v1.sh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preregistration_locks_factorized_hidden_space_kill_test() -> None:
    manifest = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    assert manifest["protocol"] == "train_only_property_factorized_language_graph_basis_v1"
    assert manifest["single_mechanism_change"] == "property_factorized_language_coefficients"
    assert manifest["common_llm_prompt_contains_source"] is False
    assert manifest["molecule_generation"] is False
    assert manifest["molecular_candidate_ranking"] is False
    assert manifest["oracle_selection"] is False
    assert manifest["generation_target_access"] is False
    assert manifest["fit_probe_split"] == "canonical_source_group_exact_condition_budget"
    assert manifest["implementation_sha256"] == sha256(IMPLEMENTATION)


def test_llm_only_composes_property_basis_coefficients() -> None:
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    assert "class LanguagePropertyComposer" in source
    assert "fit_property_token_basis" in source
    assert "compose_tokens" in source
    assert "aligned_vs_shuffled" in source
    assert "source_smiles" not in source
    assert "def run_freeze" not in source
    assert "generated_smiles" not in source


def test_runner_has_no_generation_or_oracle_surface() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    assert "evaluation-target" not in runner
    assert "oracle" not in runner.lower()
    assert "candidate" not in runner.lower()
    submitter = SUBMITTER.read_text(encoding="utf-8")
    assert "nvidia_h100_80gb_hbm3_2g.20gb:1" in submitter
    assert "--time=00:30:00" in submitter
    assert "dependency" not in submitter.lower()
