import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FLOW = ROOT / "experiments" / "unified_latent_flow"
SCRIPT = FLOW / "viability_preserving_interacting_particle_transport.py"
PREREG = (
    FLOW / "viability_preserving_interacting_particle_transport_v41_preregistration.json"
)
RUN = FLOW / "run_viability_preserving_interacting_particle_transport.sh"
SUBMIT = FLOW / "submit_viability_preserving_interacting_particle_transport.sh"


def test_b41_preregisters_support_training_and_exact_interacting_particles():
    payload = json.loads(PREREG.read_text(encoding="utf-8"))
    assert payload["protocol"] == (
        "train_only_viability_preserving_interacting_particle_transport_v41"
    )
    assert payload["status"] == (
        "amended_after_two_engineering_failures_before_training"
    )
    assert payload["engineering_amendment"] == {
        "failed_job_ids": [19881100, 19894820],
        "failure_stage": "support_replay_gate",
        "fit_pairs": 957,
        "complete_stop_legal_by_attempt": [921, 930],
        "recovered_aromatic_degree_false_rejections": 9,
        "remaining_aromatic_endpoint_mismatches": 27,
        "training_started": False,
        "scientific_result_observed": False,
    }
    assert payload["warm_start_from_frozen_b39"] is True
    assert payload["support_consistent_event_finetuning"] is True
    assert payload["representation_aware_aromatic_stop_rule"] is True
    assert payload["transport_weights_frozen_during_event_finetuning"] is True
    assert payload["interacting_particle_transport"] is True
    assert payload["particle_pool_size"] == 20
    assert payload["exact_raw_attempts_per_condition"] == 20
    assert payload["max_jumps"] == 64
    assert payload["molecular_candidate_ranking"] is False
    assert payload["retry_or_resampling"] is False
    assert payload["implementation_sha256"] == hashlib.sha256(
        SCRIPT.read_bytes()
    ).hexdigest()


def test_b41_finetunes_only_event_kernel_under_generation_support():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "model.requires_grad_(False)" in source
    assert "model.denoiser.requires_grad_(True)" in source
    assert "def viability_event_mask(" in source
    assert "def terminal_stop_support(" in source
    assert "representation_false_rejection_recovered" in source
    assert "def support_replay_gate(" in source
    assert "fit_complete_stop_legal_rate" in source
    assert "stop_margin_loss" in source
    assert "transport_velocity.parameters" not in source


def test_b41_stop_support_uses_state_and_bond_grammar_not_aromatic_label_degree():
    source = SCRIPT.read_text(encoding="utf-8")
    terminal = source[
        source.index("def terminal_stop_support(") : source.index(
            "def viability_event_mask("
        )
    ]
    assert "aromatic_endpoints_ok" in terminal
    assert "bond_support_ok" in terminal
    assert "atom_support_ok" in terminal
    assert "aromatic_degree" not in terminal
    assert '"stop_aromatic_endpoints_legal"' not in terminal
    assert '"representation_aromatic_endpoints_match"' in terminal
    assert '"passed": rate == 1.0' in source


def test_b41_particle_interaction_occurs_inside_flow_without_selection():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "def interacting_transport_particles(" in source
    assert "neighbours = torch.softmax(" in source
    assert "repulsion = normalized - neighbours @ normalized" in source
    assert "minimum_rms = math.sqrt(model.transport_dim)" in source
    assert source.index("interacting_transport_particles(") < source.index(
        "predicted_cardinality = torch.multinomial("
    )
    assert "topk(" not in source
    assert "argsort(" not in source
    assert "B41 expected {attempts} attempts" in source


def test_b41_freezes_checkpoint_and_candidates_before_dev_evaluation():
    source = SCRIPT.read_text(encoding="utf-8")
    main = source[source.index("def main(") :]
    assert main.index("torch.save(") < main.index("freeze_candidates(")
    assert main.index("freeze_candidates(") < main.index(
        "evaluate_frozen_candidates(frozen, development_pairs)"
    )
    assert '"generation_target_access": False' in source
    assert '"generation_property_oracle_access": False' in source
    assert '"oracle_selection": False' in source
    assert '"posthoc_molecule_repair": False' in source


def test_b41_runner_locks_b40_and_uses_one_short_mig():
    run = RUN.read_text(encoding="utf-8")
    submit = SUBMIT.read_text(encoding="utf-8")
    assert "viability_preserving_interacting_particle_transport_v41_preregistration.json" in run
    assert "valence_constrained_latent_particle_bridge_v40/seed_1989" in run
    assert "--b40-summary" in run
    assert "--b40-evaluated-candidates" in run
    assert "--account=def-hup-ab" in submit
    assert "gpu:nvidia_h100_80gb_hbm3_1g.10gb:1" in submit
    assert "00:45:00" in submit
