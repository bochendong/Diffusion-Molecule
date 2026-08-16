from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "unified_latent_flow"
IMPLEMENTATION = EXPERIMENT / "valid_terminal_molecule_latent_jump.py"
PREREGISTRATION = EXPERIMENT / "valid_terminal_molecule_latent_jump_v1_preregistration.json"
RUNNER = EXPERIMENT / "run_valid_terminal_molecule_latent_jump.sh"
SUBMITTER = EXPERIMENT / "submit_valid_terminal_molecule_latent_jump.sh"


def test_preregistration_locks_valid_molecule_state_change() -> None:
    payload = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    assert payload["implementation_sha256"] == hashlib.sha256(
        IMPLEMENTATION.read_bytes()
    ).hexdigest()
    assert payload["single_mechanism_change_after_b41"] is True
    assert payload["frozen_b41_checkpoint"] is True
    assert payload["b41_training"] is False
    assert payload["source_conditioned_continuous_latent_particles"] is True
    assert payload["direct_atom_bond_graph_events"] is True
    assert payload["exact_molecule_materialization_is_stop_support"] is True
    assert payload["particle_pool_size"] == 20
    assert payload["exact_raw_attempts_per_condition"] == 20
    assert payload["molecular_candidate_ranking"] is False
    assert payload["oracle_selection"] is False
    assert payload["retry_or_resampling"] is False
    assert payload["posthoc_molecule_repair"] is False
    assert payload["generation_target_access"] is False
    assert payload["generation_property_oracle_access"] is False


def test_materialization_only_masks_stop() -> None:
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    support = source[
        source.index("class ExactMoleculeStopSupport") : source.index("def gate_result(")
    ]
    assert "materializable_terminal_states(" in support
    assert "legal[:, 0] &= materializable" in support
    assert "legal[:, 1:]" not in support
    assert "property" not in support.lower()
    assert "target" not in support.lower()
    assert "retry" not in support.lower()


def test_frozen_candidates_are_hashed_before_evaluation() -> None:
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    load = source.index('model.load_state_dict(dict(b41_checkpoint["model_state"])')
    frozen = source.index("frozen = b41.freeze_candidates(", load)
    write = source.index("base.write_candidate_rows(frozen_path, frozen)", frozen)
    digest = source.index("frozen_sha256 = belief.file_sha256(frozen_path)", write)
    evaluate = source.index(
        "b41.evaluate_frozen_candidates(frozen, development_pairs)", digest
    )
    assert "model.eval().requires_grad_(False)" in source[load:frozen]
    assert load < frozen < write < digest < evaluate


def test_runner_and_submitter_use_one_bounded_mig_signal() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    submitter = SUBMITTER.read_text(encoding="utf-8")
    assert "viability_preserving_interacting_particle_transport_v41/seed_1991" in runner
    assert "--device auto" in runner
    assert "--account=def-hup-ab" in submitter
    assert "00:45:00" in submitter
    assert "gpu:nvidia_h100_80gb_hbm3_1g.10gb:1" in submitter
    assert "--array" not in submitter
