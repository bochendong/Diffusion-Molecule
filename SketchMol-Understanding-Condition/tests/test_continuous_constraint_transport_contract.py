from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = (
    ROOT
    / "SketchMol-Understanding-Condition"
    / "experiments"
    / "unified_latent_flow"
)
FLOW_PATH = EXPERIMENT_DIR / "continuous_constraint_transport.py"
RUN_PATH = EXPERIMENT_DIR / "run_continuous_constraint_transport_pilot.sh"
SUBMIT_PATH = EXPERIMENT_DIR / "submit_continuous_constraint_transport_pilot.sh"


def test_continuous_transport_is_compositional_and_target_blind() -> None:
    source = FLOW_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert "class CompositionalTransportVelocity" in source
    assert "class ContinuousConstraintTransport" in source
    assert '"continuous_transport_latent": True' in source
    assert '"vq_codebook": False' in source
    assert '"posterior_train_only": True' in source
    assert '"conditional_flow_matching": True' in source
    assert '"set_compositional_unary_property_fields": True' in source
    assert '"symmetric_pairwise_property_fields": True' in source
    assert '"property_order_permutation_invariant": True' in source
    assert '"generation_target_access": False' in source
    assert '"property_oracle_generation_access": False' in source
    assert '"candidate_library": False' in source
    assert '"selector": False' in source
    assert '"finalizer": False' in source
    assert '"oracle_reranking": False' in source
    sample = source[source.index("def sample_from_source") : source.index("def evaluate")]
    assert "target_smiles" not in sample
    assert "target_example" not in sample
    assert "property_oracle" not in sample
    assert "codebook" not in sample


def test_continuous_transport_runner_is_bounded_exact_n20_and_lightweight() -> None:
    run_source = RUN_PATH.read_text(encoding="utf-8")
    submit_source = SUBMIT_PATH.read_text(encoding="utf-8")
    assert '--train-limit "${SUCC_CONTINUOUS_TRANSPORT_TRAIN_LIMIT:-1500}"' in run_source
    assert '--validation-limit "${SUCC_CONTINUOUS_TRANSPORT_VALIDATION_LIMIT:-20}"' in run_source
    assert '--property-counts "${SUCC_CONTINUOUS_TRANSPORT_PROPERTY_COUNTS:-2,3}"' in run_source
    assert "--num-attempts 20" in run_source
    assert "--gate-validity 0.95" in run_source
    assert "--gate-mean-unique-valid 10" in run_source
    assert "--cpus-per-task=1" in submit_source
    assert "--mem=4G" in submit_source
    assert "00:10:00" in submit_source
    assert "nvidia_h100_80gb_hbm3_1g.10gb:1" in submit_source
    assert "dongbochen1218@gmail.com" in submit_source
