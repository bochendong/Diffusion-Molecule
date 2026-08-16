from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "unified_latent_flow"
IMPLEMENTATION = EXPERIMENT / "compositional_closed_transaction_vq_flow.py"
PREREGISTRATION = (
    EXPERIMENT / "compositional_closed_transaction_vq_flow_v1_preregistration.json"
)
RUNNER = EXPERIMENT / "run_compositional_closed_transaction_vq_flow.sh"
SUBMITTER = EXPERIMENT / "submit_compositional_closed_transaction_vq_flow.sh"


def test_preregistration_locks_implementation_and_exact20_contract() -> None:
    payload = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    digest = hashlib.sha256(IMPLEMENTATION.read_bytes()).hexdigest()
    assert payload["implementation_sha256"] == digest
    assert payload["reaction_context_radius"] == 0
    assert payload["condition_dim"] == 64
    assert payload["condition_slots"] == 18
    assert payload["particle_pool_size"] == 20
    assert payload["exact_raw_attempts_per_condition"] == 20
    assert payload["single_complete_transaction_per_attempt"] is True
    assert payload["support_is_decoder_action_space"] is True
    assert payload["molecular_candidate_ranking"] is False
    assert payload["oracle_selection"] is False
    assert payload["retry_or_resampling"] is False
    assert payload["second_edit"] is False
    assert payload["generation_target_access"] is False
    assert payload["generation_property_oracle_access"] is False


def test_implementation_freezes_before_oracle_evaluation() -> None:
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    freeze = source.index("frozen, development_support = freeze_candidates(")
    write = source.index("write_rows(frozen_path, frozen)", freeze)
    digest = source.index("frozen_sha256 = belief.file_sha256(frozen_path)", write)
    evaluate = source.index("evaluate_frozen_candidates(", digest)
    assert freeze < write < digest < evaluate
    assert "pair.target_smiles" not in source[
        source.index("def freeze_candidates(") : source.index("def write_rows(")
    ]
    assert "Compositional VQ condition-slot drift" in source


def test_generation_is_code_sampling_without_molecule_ranking() -> None:
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    freeze = source[
        source.index("def freeze_candidates(") : source.index("def write_rows(")
    ]
    assert "torch.multinomial" in freeze
    assert "reaction_code" in freeze
    assert "property_outcome" not in freeze
    assert "evaluate_frozen_candidates" not in freeze
    assert "sorted(actions" not in freeze
    assert "topk" not in freeze.lower()


def test_runner_and_submitter_use_bounded_cpu_job() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    submitter = SUBMITTER.read_text(encoding="utf-8")
    assert "--device cpu" in runner
    assert "source_disjoint_support_radius0_48.json" in runner
    assert "--account=def-hup-ab_cpu" in submitter
    assert "00:30:00" in submitter
    assert "--cpus-per-task" in submitter
    assert "--mem" in submitter
    assert "--gres" not in submitter
