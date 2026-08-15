import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FLOW = ROOT / "experiments" / "unified_latent_flow"
SCRIPT = FLOW / "source_clamped_region_graph_diffusion.py"
PREREG = FLOW / "source_clamped_region_graph_diffusion_v37_preregistration.json"
RUN = FLOW / "run_source_clamped_region_graph_diffusion.sh"
SUBMIT = FLOW / "submit_source_clamped_region_graph_diffusion.sh"


def test_b37_preregisters_a_region_diffusion_not_a_wider_patch_grammar():
    payload = json.loads(PREREG.read_text(encoding="utf-8"))
    assert payload["protocol"] == "train_only_source_clamped_region_graph_diffusion_v37"
    assert payload["status"] == "preregistered_before_first_run"
    assert payload["region_mask_is_diffusion_state"] is True
    assert payload["source_exterior_clamped_each_reverse_step"] is True
    assert payload["region_incident_edges_jointly_denoised"] is True
    assert payload["hard_patch_count"] is False
    assert payload["hard_anchor_limit"] is False
    assert payload["hard_edit_radius"] is False
    assert payload["fragment_library"] is False
    assert payload["molecular_candidate_ranking"] is False
    assert payload["oracle_selection"] is False
    assert payload["exact_raw_attempts_per_condition"] == 20
    assert payload["fit_dev_source_group_overlap"] == 0
    assert payload["train_selection_seed"] == 1741
    assert payload["fingerprint_bits"] == 512
    assert payload["engineering_amendment"] == {
        "failed_job_id": 19863790,
        "failure_signature": "KeyError: train_selection_seed before data reconstruction",
        "scientific_configuration_changed": False,
    }
    assert len(payload["implementation_sha256"]) == 64
    assert len(payload["locked_inputs"]) == 7


def test_b37_clamps_exterior_and_models_boundary_edges_in_the_reverse_process():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "class RegionJointGraphDenoiser" in source
    assert "class SourceClampedRegionDiffusion" in source
    assert "def region_targets" in source
    assert "def corrupt_region_and_actions" in source
    assert "region[:, :, None] | region[:, None, :]" in source
    assert "sampled_region[:, :, None] | sampled_region[:, None, :]" in source
    assert "outside_source_invariant" in source
    assert "hard_patch_count\": False" in source
    assert "hard_anchor_limit\": False" in source
    assert "hard_edit_radius\": False" in source
    assert "molecular_candidate_ranking\": False" in source
    assert "generation_target_access\": False" in source


def test_b37_freezes_exact_raw_rows_before_internal_dev_evaluation():
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.index("freeze_candidates(") < source.index("evaluate_frozen_candidates(")
    assert "frozen_train_only_dev_candidates.csv" in source
    assert "frozen_candidates_sha256" in source
    assert "B37 expected {attempts} attempts" in source
    assert "development_source_limit" in source
    assert "fit_dev_source_overlap" in source


def test_b37_runner_uses_locked_evidence_and_one_bounded_mig():
    run = RUN.read_text(encoding="utf-8")
    submit = SUBMIT.read_text(encoding="utf-8")
    assert "source_clamped_region_graph_diffusion_v37_preregistration.json" in run
    assert "valid_early_stop_delta_diffusion_v22/seed_1757" in run
    assert "source_anchored_graph_patch_evidence_v36/seed_1981" in run
    assert "--protocol-manifest" in run
    assert "--account=def-hup-ab" in submit
    assert "gpu:nvidia_h100_80gb_hbm3_1g.10gb:1" in submit
    assert "01:30:00" in submit
