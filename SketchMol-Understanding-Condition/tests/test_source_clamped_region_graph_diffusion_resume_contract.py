import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FLOW = ROOT / "experiments" / "unified_latent_flow"
SCRIPT = FLOW / "resume_source_clamped_region_graph_diffusion_evaluation.py"
MODEL_SCRIPT = FLOW / "source_clamped_region_graph_diffusion.py"
PREREG = FLOW / "source_clamped_region_graph_diffusion_v37_preregistration.json"
MANIFEST = FLOW / "source_clamped_region_graph_diffusion_v37r1_resume_manifest.json"
RUN = FLOW / "run_source_clamped_region_graph_diffusion_evaluation_resume.sh"
SUBMIT = FLOW / "submit_source_clamped_region_graph_diffusion_evaluation_resume.sh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_b37r1_locks_the_failed_run_and_forbids_new_generation():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["protocol"] == "frozen_candidate_evaluation_resume_v37r1"
    assert payload["failed_job_id"] == 19864238
    assert payload["expected_conditions"] == 235
    assert payload["expected_candidate_rows"] == 4700
    assert payload["exact_raw_attempts_per_condition"] == 20
    assert payload["model_training"] is False
    assert payload["molecular_candidate_generation"] is False
    assert payload["candidate_modification"] is False
    assert payload["candidate_filtering"] is False
    assert payload["candidate_ranking"] is False
    assert payload["oracle_selection"] is False
    assert payload["post_freeze_evaluation_only"] is True
    assert payload["scientific_configuration_changed"] is False
    locked = payload["locked_files"]
    assert locked["original_preregistration_sha256"] == sha256(PREREG)
    assert locked["b37_implementation_sha256"] == sha256(MODEL_SCRIPT)
    assert locked["resume_implementation_sha256"] == sha256(SCRIPT)


def test_b37r1_validates_frozen_rows_before_opening_the_evaluator():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "def read_and_validate_frozen" in source
    assert "def evaluate_frozen" in source
    assert source.index("read_and_validate_frozen(") < source.index("evaluate_frozen(")
    assert "duplicate frozen attempt" in source
    assert "non-exact attempt counts" in source
    assert '"target_atom_count"' in source
    assert '"candidate_modification": False' in source
    assert '"molecular_candidate_generation_in_resume": False' in source
    assert '"model_checkpoint_available": False' in source


def test_b37r1_runner_is_cpu_only_and_reads_the_locked_csv():
    run = RUN.read_text(encoding="utf-8")
    submit = SUBMIT.read_text(encoding="utf-8")
    assert "frozen_train_only_dev_candidates.csv" in run
    assert "uca-region-diff-v37-19864238.log" in run
    assert "source_clamped_region_graph_diffusion_v37r1_resume_manifest.json" in run
    assert "--account=def-hup-ab_cpu" in submit
    assert "00:15:00" in submit
    assert "--gres" not in submit
