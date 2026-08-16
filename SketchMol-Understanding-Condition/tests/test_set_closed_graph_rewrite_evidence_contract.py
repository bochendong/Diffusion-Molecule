import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FLOW = ROOT / "experiments" / "unified_latent_flow"
SCRIPT = FLOW / "set_closed_graph_rewrite_evidence.py"
PREREG = FLOW / "set_closed_graph_rewrite_v1_preregistration.json"
RUN = FLOW / "run_set_closed_graph_rewrite_evidence.sh"
SUBMIT = FLOW / "submit_set_closed_graph_rewrite_evidence.sh"


def test_set_closed_preregistration_is_a_clean_architecture_reset():
    payload = json.loads(PREREG.read_text(encoding="utf-8"))
    assert payload["protocol"] == "train_only_set_closed_graph_rewrite_evidence_v1"
    assert payload["architecture_reset_after_b42"] is True
    assert payload["status"] == "preregistered_before_first_run"
    assert payload["model_training"] is False
    assert payload["molecular_candidate_generation"] is False
    assert payload["evaluation_target_access"] is False
    assert payload["b26_heldout_access"] is False
    assert payload["b33_fresh_source_access"] is False
    assert payload["moledit_table1_benchmark_access"] is False
    assert payload["official_test_access"] is False
    assert payload["molecular_candidate_ranking"] is False
    assert payload["oracle_selection"] is False
    assert payload["future_exact_raw_attempts_per_condition"] == 20
    assert payload["future_joint_set_decoder"] is True
    assert payload["future_independent_event_decoder"] is False
    assert len(payload["implementation_sha256"]) == 64
    assert len(payload["locked_inputs"]) == 6


def test_set_closed_code_uses_atomic_rewrites_and_source_group_meta_split():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "def record_tokens" in source
    assert "def stable_meta_assignment" in source
    assert "fit_grammar_supported" in source
    assert "atomic_valence_closed_transaction" in source
    assert "fit_meta_source_group_split" in source
    assert "train_set_closed_graph_transport_exact_n20" in source
    assert '"molecular_candidate_generation": False' in source
    assert '"molecular_candidate_ranking": False' in source
    assert '"oracle_selection": False' in source


def test_set_closed_runner_is_cpu_only_and_bounded():
    run = RUN.read_text(encoding="utf-8")
    submit = SUBMIT.read_text(encoding="utf-8")
    assert "set_closed_graph_rewrite_v1_preregistration.json" in run
    assert "valid_early_stop_delta_diffusion_v22/seed_1757" in run
    assert "--protocol-manifest" in run
    assert "--account=def-hup-ab_cpu" in submit
    assert "00:30:00" in submit
    assert "--cpus-per-task" in submit
    assert "--mem" in submit
    assert "--gres" not in submit
