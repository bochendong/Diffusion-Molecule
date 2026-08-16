from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "unified_latent_flow"
IMPLEMENTATION = EXPERIMENT / "universal_continuous_graph_delta_flow.py"
PREREGISTRATION = EXPERIMENT / "universal_continuous_graph_delta_flow_v1_preregistration.json"
RUNNER = EXPERIMENT / "run_universal_continuous_graph_delta_flow.sh"
SUBMITTER = EXPERIMENT / "submit_universal_continuous_graph_delta_flow.sh"


def test_preregistration_locks_continuous_architecture_reset() -> None:
    payload = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    assert payload["implementation_sha256"] == hashlib.sha256(IMPLEMENTATION.read_bytes()).hexdigest()
    assert payload["architecture_reset_after_vq"] is True
    assert payload["frozen_graph_autoencoder"] is True
    assert payload["continuous_graph_delta_latent"] is True
    assert payload["conditional_rectified_flow"] is True
    assert payload["discrete_vq_codebook"] is False
    assert payload["universal_cross_task_transaction_grammar"] is True
    assert payload["task_partitioned_transaction_support"] is False
    assert payload["development_target_latent_access"] is False
    assert payload["particle_pool_size"] == 20
    assert payload["exact_raw_attempts_per_condition"] == 20
    assert payload["molecular_candidate_ranking"] is False
    assert payload["oracle_selection"] is False
    assert payload["retry_or_resampling"] is False
    assert payload["generation_target_access"] is False
    assert payload["generation_property_oracle_access"] is False
    assert payload["gates"]["strict_any20"] == 0.90
    assert payload["gates"]["target_improvement_any20"] == 0.60


def test_development_encoder_and_freezer_cannot_read_targets_or_oracles() -> None:
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    encode = source[
        source.index("def encode_development_sources(") : source.index("def deterministic_pca(")
    ]
    freeze = source[
        source.index("def freeze_candidates(") : source.index("def write_rows(")
    ]
    assert "pair.target" not in encode
    assert "pair.target" not in freeze
    assert "score_property" not in freeze
    assert "instruction_success_and_distance" not in freeze
    assert "topk" not in freeze.lower()
    assert "torch.multinomial" in freeze
    assert "decoder_property_distance_weight" in freeze
    assert "generated_smiles" not in freeze[: freeze.index("rows.append(")]


def test_freeze_precedes_evaluation_and_universal_key_omits_task() -> None:
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    key = source[
        source.index("def universal_transaction_key(") : source.index("def build_universal_fit_grammar(")
    ]
    assert '"task"' not in key
    assert '"reaction_smarts"' in key
    freeze = source.index("frozen, development_support = freeze_candidates(")
    write = source.index("write_rows(frozen_path, frozen)", freeze)
    digest = source.index("frozen_sha256 = belief.file_sha256(frozen_path)", write)
    evaluate = source.index("vq.evaluate_frozen_transactions(frozen, development_pairs)", digest)
    assert freeze < write < digest < evaluate


def test_runner_and_submitter_use_one_bounded_cpu_signal() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    submitter = SUBMITTER.read_text(encoding="utf-8")
    assert "--device cpu" in runner
    assert "property_aligned_balanced_transaction_transport_v1/seed_2011" in runner
    assert "--account=def-hup-ab_cpu" in submitter
    assert "00:45:00" in submitter
    assert "--cpus-per-task" in submitter
    assert "--mem" in submitter
    assert "--gres" not in submitter
