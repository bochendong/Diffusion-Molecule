import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FLOW = ROOT / "experiments" / "unified_latent_flow"
SCRIPT = FLOW / "source_clamped_latent_graph_jump_process.py"
PREREG = FLOW / "source_clamped_latent_graph_jump_process_v38_preregistration.json"
RUN = FLOW / "run_source_clamped_latent_graph_jump_process.sh"
SUBMIT = FLOW / "submit_source_clamped_latent_graph_jump_process.sh"


def test_b38_preregisters_a_single_event_process_not_a_region_patch():
    payload = json.loads(PREREG.read_text(encoding="utf-8"))
    assert payload["protocol"] == "train_only_source_clamped_latent_graph_jump_process_v38"
    assert payload["status"] == "preregistered_before_first_run"
    assert payload["explicit_region_mask"] is False
    assert payload["globally_normalized_single_event_jumps"] is True
    assert payload["learned_stop_event"] is True
    assert payload["orderless_event_set_objective"] is True
    assert payload["hard_patch_count"] is False
    assert payload["hard_anchor_limit"] is False
    assert payload["hard_edit_radius"] is False
    assert payload["molecular_candidate_ranking"] is False
    assert payload["oracle_selection"] is False
    assert payload["exact_raw_attempts_per_condition"] == 20
    assert payload["max_jumps"] == 32
    assert len(payload["locked_inputs"]) == 8
    assert payload["implementation_sha256"] == hashlib.sha256(
        SCRIPT.read_bytes()
    ).hexdigest()


def test_b38_uses_orderless_dependency_ready_event_sets_and_learned_stop():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "class GraphEvent" in source
    assert "class EventLayout" in source
    assert "class JumpEventField" in source
    assert "def event_dependencies" in source
    assert "def random_topological_prefix" in source
    assert "def orderless_jump_loss" in source
    assert "log_target_mass" in source
    assert "globally_normalized_single_event_jumps\": True" in source
    assert "explicit_region_mask\": False" in source
    assert "molecular_candidate_ranking\": False" in source


def test_b38_replay_gates_then_checkpoints_before_exact_raw_freeze():
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.index("preflight_event_replay(") < source.index("train_model(")
    main = source[source.index("def main(") :]
    assert main.index("torch.save(") < main.index("freeze_candidates(")
    assert "checkpoint_frozen_before_generation" in source
    assert "frozen_train_only_dev_candidates.csv" in source
    assert "B38 expected {attempts} attempts" in source
    assert "source_atom_count" in source
    assert "target_atom_count" in source


def test_b38_generation_is_target_blind_and_has_no_retry_or_ranking_pool():
    source = SCRIPT.read_text(encoding="utf-8")
    sample = source[source.index("def sample_from_source(") : source.index("def freeze_candidates(")]
    assert "target" not in sample
    assert "oracle" not in sample
    assert "retry" not in sample
    assert "topk" not in sample.lower()
    assert "torch.multinomial" in sample
    assert "execute_flat_event" in sample
    assert "max_jumps" in sample


def test_b38_runner_uses_locked_evidence_and_one_short_mig():
    run = RUN.read_text(encoding="utf-8")
    submit = SUBMIT.read_text(encoding="utf-8")
    assert "source_clamped_latent_graph_jump_process_v38_preregistration.json" in run
    assert "valid_early_stop_delta_diffusion_v22/seed_1757" in run
    assert "source_anchored_graph_patch_evidence_v36/seed_1981" in run
    assert "source_clamped_region_graph_diffusion_v37/seed_1983" in run
    assert "--protocol-manifest" in run
    assert "--account=def-hup-ab" in submit
    assert "gpu:nvidia_h100_80gb_hbm3_1g.10gb:1" in submit
    assert "00:45:00" in submit
