from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "unified_latent_flow"
IMPLEMENTATION = EXPERIMENT / "contextual_source_conditioned_transition_operator.py"
PREREGISTRATION = EXPERIMENT / "contextual_source_conditioned_transition_operator_v1_preregistration.json"
RUNNER = EXPERIMENT / "run_contextual_source_conditioned_transition_operator.sh"
SUBMITTER = EXPERIMENT / "submit_contextual_source_conditioned_transition_operator.sh"


def test_preregistration_locks_one_contextual_mechanism_change() -> None:
    payload = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    assert payload["implementation_sha256"] == hashlib.sha256(IMPLEMENTATION.read_bytes()).hexdigest()
    assert payload["single_mechanism_change_after_universal_flow"] is True
    assert payload["frozen_universal_continuous_flow"] is True
    assert payload["flow_training"] is False
    assert payload["trainable_source_conditioned_effect_operator"] is True
    assert payload["source_independent_transaction_effect"] is False
    assert payload["universal_cross_task_transaction_grammar"] is True
    assert payload["task_partitioned_transaction_support"] is False
    assert payload["particle_pool_size"] == 20
    assert payload["exact_raw_attempts_per_condition"] == 20
    assert payload["molecular_candidate_ranking"] is False
    assert payload["oracle_selection"] is False
    assert payload["retry_or_resampling"] is False
    assert payload["generation_target_access"] is False
    assert payload["generation_property_oracle_access"] is False


def test_only_operator_is_optimized_and_flow_is_frozen() -> None:
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    check = source[source.index("def check_locked_inputs(") : source.index("def checkpoint_transactions(")]
    train = source[source.index("def train_effect_operator(") : source.index("@torch.no_grad()", source.index("def train_effect_operator("))]
    assert "flow.eval().requires_grad_(False)" in check
    assert "optimizer = torch.optim.AdamW" in train
    assert "model.parameters()" in train
    assert "flow.parameters()" not in train
    assert "universal_checkpoint" in source


def test_generation_has_no_target_or_oracle_and_freezes_first() -> None:
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    freeze = source[source.index("def freeze_candidates(") : source.index("def write_rows(")]
    assert "pair.target" not in freeze
    assert "score_property" not in freeze
    assert "instruction_success_and_distance" not in freeze
    assert "torch.multinomial" in freeze
    assert "topk" not in freeze.lower()
    assert "operator(source, action_graph)" in freeze
    call = source.index("frozen, development_support = freeze_candidates(")
    write = source.index("write_rows(frozen_path, frozen)", call)
    digest = source.index("frozen_sha256 = belief.file_sha256(frozen_path)", write)
    evaluate = source.index("vq.evaluate_frozen_transactions(frozen, development_pairs)", digest)
    assert call < write < digest < evaluate


def test_runner_and_submitter_use_one_bounded_cpu_signal() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    submitter = SUBMITTER.read_text(encoding="utf-8")
    assert "--device cpu" in runner
    assert "universal_continuous_graph_delta_flow_v1/seed_2015" in runner
    assert "--account=def-hup-ab_cpu" in submitter
    assert "00:45:00" in submitter
    assert "--gres" not in submitter
