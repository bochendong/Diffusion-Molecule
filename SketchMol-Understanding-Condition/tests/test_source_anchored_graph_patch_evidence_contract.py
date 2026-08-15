import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FLOW = ROOT / "experiments" / "unified_latent_flow"
SCRIPT = FLOW / "source_anchored_graph_patch_evidence.py"
PREREG = FLOW / "source_anchored_graph_patch_v36_preregistration.json"
RUN = FLOW / "run_source_anchored_graph_patch_evidence.sh"
SUBMIT = FLOW / "submit_source_anchored_graph_patch_evidence.sh"


def test_b36_preregistration_locks_train_only_representation_evidence():
    payload = json.loads(PREREG.read_text(encoding="utf-8"))
    assert payload["protocol"] == "train_only_source_anchored_graph_patch_evidence_v36"
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
    assert payload["max_patch_components"] == 3
    assert payload["max_nodes_per_patch"] == 12
    assert payload["max_boundary_anchors_per_patch"] == 2
    assert len(payload["implementation_sha256"]) == 64
    assert len(payload["locked_inputs"]) == 6


def test_b36_code_is_connected_patch_evidence_not_molecule_selection():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "def connected_components" in source
    assert "def component_signature" in source
    assert "outside_source_invariant" in source
    assert "strict_compact_patch_coverage" in source
    assert "train_source_anchored_set_graph_patch_flow_v37" in source
    assert "molecular_candidate_generation\": False" in source
    assert "molecular_candidate_ranking\": False" in source
    assert "oracle_selection\": False" in source
    assert "B36 locked input drift" in source
    assert "B36 implementation drift" in source


def test_b36_runner_is_cpu_only_and_uses_locked_inputs():
    run = RUN.read_text(encoding="utf-8")
    submit = SUBMIT.read_text(encoding="utf-8")
    assert "source_anchored_graph_patch_v36_preregistration.json" in run
    assert "valid_early_stop_delta_diffusion_v22/seed_1757" in run
    assert "--protocol-manifest" in run
    assert "--account=def-hup-ab_cpu" in submit
    assert "--cpus-per-task" in submit
    assert "--mem" in submit
    assert "--gres" not in submit
    assert "00:45:00" in submit
