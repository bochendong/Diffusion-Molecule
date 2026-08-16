import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FLOW = ROOT / "experiments" / "unified_latent_flow"
SCRIPT = FLOW / "set_closed_graph_transport.py"
PREREG = FLOW / "set_closed_graph_transport_v1_preregistration.json"
RUN = FLOW / "run_set_closed_graph_transport.sh"
SUBMIT = FLOW / "submit_set_closed_graph_transport.sh"


def test_set_transport_preregistration_locks_exact20_joint_training():
    payload = json.loads(PREREG.read_text(encoding="utf-8"))
    assert payload["protocol"] == "train_only_set_closed_graph_transport_v1"
    assert payload["status"] == "preregistered_before_first_run"
    assert payload["set_closed_representation_gate_required"] is True
    assert payload["atomic_transaction_commit"] is True
    assert payload["joint_set_training"] is True
    assert payload["single_particle_training_loss"] is False
    assert payload["set_particles"] == 20
    assert payload["particle_pool_size"] == 20
    assert payload["exact_raw_attempts_per_condition"] == 20
    assert payload["molecular_candidate_ranking"] is False
    assert payload["oracle_selection"] is False
    assert payload["retry_or_resampling"] is False
    assert payload["generation_target_access"] is False
    assert payload["official_test_access"] is False
    assert len(payload["locked_inputs"]) == 11


def test_set_transport_code_has_joint_coverage_loss_and_atomic_generation():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "def train_joint_set_kernel" in source
    assert "def target_mode_coverage_loss" in source
    assert "best_set_loss" in source
    assert "participation_loss" in source
    assert "set_diversity_cosine" in source
    assert "b41.freeze_candidates" in source
    assert '"atomic_transaction_commit": True' in source
    assert '"molecular_candidate_ranking": False' in source
    assert '"retry_or_resampling": False' in source
    assert '"oracle_selection": False' in source


def test_set_transport_runner_uses_locked_signal_and_small_mig():
    run = RUN.read_text(encoding="utf-8")
    submit = SUBMIT.read_text(encoding="utf-8")
    assert "set_closed_graph_rewrite_evidence_v1/seed_2001" in run
    assert "viability_preserving_interacting_particle_transport_v41/seed_1991" in run
    assert "--set-evidence-summary" in run
    assert "--set-evidence-records" in run
    assert "--b36-records" in run
    assert "gpu:nvidia_h100_80gb_hbm3_1g.10gb:1" in submit
    assert "01:30:00" in submit
    assert "--array" not in submit
