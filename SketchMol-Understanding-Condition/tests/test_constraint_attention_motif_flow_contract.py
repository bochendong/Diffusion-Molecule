from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "experiments" / "unified_latent_flow"
MODEL_PATH = EXPERIMENT_DIR / "hierarchical_vq_motif_graph_flow.py"
RUN_PATH = EXPERIMENT_DIR / "run_constraint_attention_motif_graph_flow_pilot.sh"
SUBMIT_PATH = EXPERIMENT_DIR / "submit_constraint_attention_motif_graph_flow_pilot.sh"


def test_b12_is_target_blind_cross_attention_over_condition_tokens() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert "class SourceConstraintCrossAttention" in source
    assert "nn.MultiheadAttention" in source
    assert "self.condition_router(source_node, source_mask, condition)" in source
    assert 'CONSTRAINT_ATTENTION_PROTOCOL = "constraint_attention_motif_graph_flow_pilot_v12"' in source
    sample_source = source[source.index("def sample_from_source") : source.index("def evaluate")]
    assert "target_smiles" not in sample_source
    assert "property_oracle" not in sample_source


def test_b12_is_a_matched_b11_ablation_on_a_small_mig() -> None:
    run_source = RUN_PATH.read_text(encoding="utf-8")
    submit_source = SUBMIT_PATH.read_text(encoding="utf-8")
    assert '--train-limit "${SUCC_CONSTRAINT_ATTN_TRAIN_LIMIT:-1500}"' in run_source
    assert '--validation-limit "${SUCC_CONSTRAINT_ATTN_VALIDATION_LIMIT:-20}"' in run_source
    assert '--epochs "${SUCC_CONSTRAINT_ATTN_EPOCHS:-8}"' in run_source
    assert "--motif-attachment" in run_source
    assert "--condition-attention" in run_source
    assert "--num-attempts 20" in run_source
    assert 'SEED="${SUCC_CONSTRAINT_ATTN_SEED:-1741}"' in run_source
    assert "nvidia_h100_80gb_hbm3_1g.10gb:1" in submit_source
    assert 'SUCC_CONSTRAINT_ATTN_TIME:-00:10:00' in submit_source
    assert "dongbochen1218@gmail.com" in submit_source
