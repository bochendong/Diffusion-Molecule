import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FLOW = ROOT / "experiments" / "unified_latent_flow"
SCRIPT = FLOW / "latent_cardinality_graph_jump_bridge.py"
PREREG = FLOW / "latent_cardinality_graph_jump_bridge_v39_preregistration.json"
RUN = FLOW / "run_latent_cardinality_graph_jump_bridge.sh"
SUBMIT = FLOW / "submit_latent_cardinality_graph_jump_bridge.sh"


def test_b39_preregisters_latent_cardinality_without_a_hard_event_budget():
    payload = json.loads(PREREG.read_text(encoding="utf-8"))
    assert payload["protocol"] == "train_only_latent_cardinality_graph_jump_bridge_v39"
    assert payload["status"] == "preregistered_before_first_run"
    assert payload["warm_start_from_frozen_b38"] is True
    assert payload["latent_cardinality_distribution"] is True
    assert payload["continuous_remaining_edit_mass"] is True
    assert payload["hard_event_budget"] is False
    assert payload["learned_stop_event"] is True
    assert payload["absolute_jump_clock_train_inference_match"] is True
    assert payload["train_only_completed_set_overrun_exposure"] is True
    assert payload["molecular_candidate_ranking"] is False
    assert payload["oracle_selection"] is False
    assert payload["exact_raw_attempts_per_condition"] == 20
    assert len(payload["locked_inputs"]) == 10
    assert payload["implementation_sha256"] == hashlib.sha256(
        SCRIPT.read_bytes()
    ).hexdigest()


def test_b39_consumes_latent_mass_and_calibrates_event_family_intensities():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "class CardinalityJumpEventField" in source
    assert "class LatentCardinalityGraphJumpBridge" in source
    assert "remaining_mass_bias" in source
    assert "def cardinality_logits" in source
    assert "expected_count - executed_count" in source
    assert "predicted_cardinality.float() - event_counts.float()" in source
    assert "jump_time = executed / float(preregistration[\"max_jumps\"])" in source
    assert "hard_event_budget\": False" in source


def test_b39_exposes_completed_sets_to_legal_overrun_then_targets_stop():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "def legal_event_indices_cpu" in source
    assert "completed_set_overrun_probability" in source
    assert "maximum_overrun_events" in source
    assert "target_next[0] = True" in source
    assert "overrun_exposure_rate" in source


def test_b39_warm_starts_locked_b38_and_freezes_exact_rows_before_evaluation():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "b38_checkpoint_sha256" in source
    assert "model.load_state_dict" in source
    main = source[source.index("def main(") :]
    assert main.index("torch.save(") < main.index("freeze_candidates(")
    assert "checkpoint_frozen_before_generation" in source
    assert "frozen_train_only_dev_candidates.csv" in source
    assert "B39 expected {attempts} attempts" in source
    assert "generation_target_access\": False" in source
    assert "molecular_candidate_ranking\": False" in source


def test_b39_runner_uses_one_short_mig_and_locked_b38_artifacts():
    run = RUN.read_text(encoding="utf-8")
    submit = SUBMIT.read_text(encoding="utf-8")
    assert "latent_cardinality_graph_jump_bridge_v39_preregistration.json" in run
    assert "source_clamped_latent_graph_jump_process_v38/seed_1985" in run
    assert "source_clamped_latent_graph_jump_process.pt" in run
    assert "--b38-checkpoint" in run
    assert "--b38-summary" in run
    assert "--account=def-hup-ab" in submit
    assert "gpu:nvidia_h100_80gb_hbm3_1g.10gb:1" in submit
    assert "00:45:00" in submit
