from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "unified_latent_flow"
IMPLEMENTATION = EXPERIMENT / "property_aligned_balanced_transaction_transport.py"
PREREGISTRATION = (
    EXPERIMENT
    / "property_aligned_balanced_transaction_transport_v1_preregistration.json"
)
RUNNER = EXPERIMENT / "run_property_aligned_balanced_transaction_transport.sh"
SUBMITTER = EXPERIMENT / "submit_property_aligned_balanced_transaction_transport.sh"


def test_preregistration_locks_the_single_structural_change() -> None:
    payload = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    digest = hashlib.sha256(IMPLEMENTATION.read_bytes()).hexdigest()
    assert payload["implementation_sha256"] == digest
    assert payload["single_structural_change_after_vq"] is True
    assert payload["reaction_context_radius"] == 0
    assert payload["fit_only_property_delta_labels"] is True
    assert payload["property_aligned_codebook"] is True
    assert payload["joint_balanced_particle_transport"] is True
    assert payload["sinkhorn_code_assignment"] is True
    assert payload["within_code_sampling_without_replacement"] is True
    assert payload["balanced_min_codes"] == 8
    assert payload["particle_pool_size"] == 20
    assert payload["exact_raw_attempts_per_condition"] == 20
    assert payload["molecular_candidate_ranking"] is False
    assert payload["oracle_selection"] is False
    assert payload["retry_or_resampling"] is False
    assert payload["generation_target_access"] is False
    assert payload["generation_property_oracle_access"] is False
    assert payload["gates"]["strict_any20"] == 0.90
    assert payload["gates"]["two_property_strict_any20"] == 0.90
    assert payload["gates"]["target_improvement_any20"] == 0.60


def test_generation_freezes_before_evaluation_without_oracle_or_target() -> None:
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    freeze_start = source.index("def freeze_balanced_candidates(")
    freeze_end = source.index("def gate_result(", freeze_start)
    freeze = source[freeze_start:freeze_end]
    assert "pair.target_smiles" not in freeze
    assert "score_property" not in freeze
    assert "instruction_success_and_distance" not in freeze
    assert "sorted(actions" not in freeze
    assert "topk" not in freeze.lower()
    assert "balanced_code_capacities(" in freeze
    assert "sinkhorn_hard_assignment(" in freeze
    assert "torch.randperm(len(choices)" in freeze
    call = source.index("frozen, development_support = freeze_balanced_candidates(")
    write = source.index("vq.write_rows(frozen_path, frozen)", call)
    digest = source.index("frozen_sha256 = belief.file_sha256(frozen_path)", write)
    evaluate = source.index("vq.evaluate_frozen_transactions(frozen, development_pairs)", digest)
    assert call < write < digest < evaluate


def test_balanced_assignment_preserves_exact_capacity_and_diversity_floor() -> None:
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    capacity = source[
        source.index("def balanced_code_capacities(") : source.index(
            "def sinkhorn_hard_assignment("
        )
    ]
    sinkhorn = source[
        source.index("def sinkhorn_hard_assignment(") : source.index(
            "@torch.no_grad()", source.index("def sinkhorn_hard_assignment(")
        )
    ]
    assert "anchored = min" in capacity
    assert "sum(capacities.values()) != int(attempts)" in capacity
    assert "torch.multinomial" in capacity
    assert "Counter(assigned_codes) != Counter" in sinkhorn
    assert "plan = plan / plan.sum(dim=1" in sinkhorn
    assert "plan = plan / plan.sum(dim=0" in sinkhorn


def test_runner_and_submitter_use_one_bounded_cpu_signal() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    submitter = SUBMITTER.read_text(encoding="utf-8")
    assert "--device cpu" in runner
    assert "compositional_closed_transaction_vq_flow_v1/seed_2007" in runner
    assert "--account=def-hup-ab_cpu" in submitter
    assert "00:30:00" in submitter
    assert "--cpus-per-task" in submitter
    assert "--mem" in submitter
    assert "--gres" not in submitter
