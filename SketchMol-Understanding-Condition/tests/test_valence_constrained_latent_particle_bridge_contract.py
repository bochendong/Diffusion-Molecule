import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FLOW = ROOT / "experiments" / "unified_latent_flow"
SCRIPT = FLOW / "valence_constrained_latent_particle_bridge.py"
PREREG = FLOW / "valence_constrained_latent_particle_bridge_v40_preregistration.json"
RUN = FLOW / "run_valence_constrained_latent_particle_bridge.sh"
SUBMIT = FLOW / "submit_valence_constrained_latent_particle_bridge.sh"


def test_b40_preregisters_one_frozen_exact_n20_particle_experiment():
    payload = json.loads(PREREG.read_text(encoding="utf-8"))
    assert payload["protocol"] == (
        "train_only_valence_constrained_latent_particle_bridge_v40"
    )
    assert payload["status"] == "preregistered_before_first_run"
    assert payload["frozen_b39_checkpoint"] is True
    assert payload["particle_pool_size"] == 20
    assert payload["exact_raw_attempts_per_condition"] == 20
    assert payload["orthogonal_latent_particles"] is True
    assert payload["molecular_candidate_ranking"] is False
    assert payload["oracle_selection"] is False
    assert payload["retry_or_resampling"] is False
    assert payload["posthoc_molecule_repair"] is False
    assert payload["implementation_sha256"] == hashlib.sha256(
        SCRIPT.read_bytes()
    ).hexdigest()


def test_b40_uses_train_only_dynamic_support_before_sampling():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "hierarchical.build_train_atom_state_grammar(fit_pairs)" in source
    assert "def constrained_event_mask(" in source
    assert "source_atom_state_valence_caps" in source
    assert "train_observed_bond_support" in source
    assert "logits.masked_fill(~legal, -torch.inf)" in source
    assert source.index("constrained_event_mask(") < source.index(
        "torch.multinomial(\n                    probability"
    )
    assert "graph.canonical_smiles(smiles or \"\")" in source


def test_b40_draws_twenty_direct_orthogonal_particles_without_selection():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "def orthogonal_latent_particles(" in source
    assert "torch.linalg.qr" in source
    assert "particles[start : start + count]" in source
    assert "B40 expected {attempts} attempts" in source
    assert '"molecular_candidate_ranking": False' in source
    assert '"oracle_selection": False' in source
    assert "topk(" not in source
    assert "argsort(" not in source


def test_b40_freezes_rows_before_opening_internal_dev_evaluation():
    source = SCRIPT.read_text(encoding="utf-8")
    main = source[source.index("def main(") :]
    assert main.index("freeze_candidates(") < main.index(
        "evaluate_frozen_candidates(frozen, development_pairs)"
    )
    assert "frozen_train_only_dev_candidates.csv" in source
    assert "post_freeze_train_only_dev_target_access" in source
    assert '"generation_target_access": False' in source
    assert '"generation_property_oracle_access": False' in source


def test_b40_runner_locks_b39_evidence_and_uses_one_short_mig():
    run = RUN.read_text(encoding="utf-8")
    submit = SUBMIT.read_text(encoding="utf-8")
    assert "valence_constrained_latent_particle_bridge_v40_preregistration.json" in run
    assert "latent_cardinality_graph_jump_bridge_v39/seed_1987" in run
    assert "evaluated_train_only_dev_candidates.csv" in run
    assert "--b39-checkpoint" in run
    assert "--b39-summary" in run
    assert "--b39-evaluated-candidates" in run
    assert "--account=def-hup-ab" in submit
    assert "gpu:nvidia_h100_80gb_hbm3_1g.10gb:1" in submit
    assert "00:30:00" in submit
