import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FLOW = ROOT / "experiments" / "unified_latent_flow"
SCRIPT = FLOW / "atomic_closed_transaction_latent_decoder.py"
PREREG = FLOW / "atomic_closed_transaction_latent_decoder_v1_preregistration.json"
RUN = FLOW / "run_atomic_closed_transaction_latent_decoder.sh"
SUBMIT = FLOW / "submit_atomic_closed_transaction_latent_decoder.sh"


def test_atomic_transaction_preregistration_is_exact20_and_target_blind():
    payload = json.loads(PREREG.read_text(encoding="utf-8"))
    assert payload["protocol"] == "train_only_atomic_closed_transaction_latent_decoder_v1"
    assert payload["status"] == "preregistered_before_first_run"
    assert payload["complete_transaction_decoding"] is True
    assert payload["fit_only_transaction_grammar"] is True
    assert payload["source_applicable_transaction_support"] is True
    assert payload["orthogonal_latent_particles"] is True
    assert payload["particle_pool_size"] == 20
    assert payload["exact_raw_attempts_per_condition"] == 20
    assert payload["only_sampled_transactions_committed"] is True
    assert payload["molecular_candidate_ranking"] is False
    assert payload["oracle_selection"] is False
    assert payload["retry_or_resampling"] is False
    assert payload["posthoc_molecule_repair"] is False
    assert payload["generation_target_access"] is False
    assert payload["generation_property_oracle_access"] is False
    assert payload["official_test_access"] is False
    assert len(payload["locked_inputs"]) == 13


def test_atomic_transaction_implementation_is_content_locked():
    payload = json.loads(PREREG.read_text(encoding="utf-8"))
    digest = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert payload["implementation_sha256"] == digest


def test_atomic_transaction_code_samples_actions_without_ranking_or_retry():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "class TransactionEnergyDecoder" in source
    assert "def applicable_transactions" in source
    assert "def set_transaction_loss" in source
    assert "def freeze_candidates" in source
    assert "torch.multinomial" in source
    assert "only_sampled_transactions_committed" in source
    assert '"molecular_candidate_ranking": False' in source
    assert '"oracle_selection": False' in source
    assert '"retry_or_resampling": False' in source
    assert "frozen_sha256 = belief.file_sha256(frozen_path)" in source


def test_atomic_transaction_runner_is_single_seed_cpu_only():
    run = RUN.read_text(encoding="utf-8")
    submit = SUBMIT.read_text(encoding="utf-8")
    assert "set_closed_graph_transport_v1/seed_2003" in run
    assert "set_closed_graph_rewrite_evidence_v1/seed_2001" in run
    assert "--device cpu" in run
    assert "--account=def-hup-ab_cpu" in submit
    assert "00:45:00" in submit
    assert "--gres" not in submit
    assert "--array" not in submit
