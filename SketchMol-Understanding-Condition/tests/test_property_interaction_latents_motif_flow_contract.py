from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "experiments" / "unified_latent_flow"
MODEL_PATH = EXPERIMENT_DIR / "hierarchical_vq_motif_graph_flow.py"
RUN_PATH = EXPERIMENT_DIR / "run_property_interaction_latents_motif_graph_flow_pilot.sh"
SUBMIT_PATH = EXPERIMENT_DIR / "submit_property_interaction_latents_motif_graph_flow_pilot.sh"


def test_b15_is_b13_plus_a_zero_initialized_symmetric_pair_residual() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert "class PropertyInteractionLatentComposer(PropertyLatentSlotComposer)" in source
    assert "b13_condition = super().forward(tokens)" in source
    assert "left + right" in source
    assert "left * right" in source
    assert "(left - right).abs()" in source
    assert "torch.triu(" in source
    assert "pair_count.sqrt()" in source
    assert "nn.init.zeros_(self.pair_residual[-1].weight)" in source
    assert "return b13_condition + pair_sum" in source
    assert "property_interaction_latents_motif_graph_flow_pilot_v15" in source


def test_b15_preserves_target_blind_generation_and_initialization_order() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    constructor = source[
        source.index("class HierarchicalVQGraphFlow") : source.index(
            "    @staticmethod\n    def source_pool"
        )
    ]
    assert constructor.index("self.decoder = decoder_class") < constructor.index(
        "self.property_interaction_router ="
    )
    assert '"initial_condition_exactly_b13": bool(' in source
    assert '"zero_initialized_pairwise_residuals": bool(' in source
    sample_source = source[source.index("def sample_from_source") : source.index("def evaluate")]
    assert "target_smiles" not in sample_source
    assert "property_oracle" not in sample_source


def test_b15_runner_is_a_matched_single_seed_small_mig_ablation() -> None:
    run_source = RUN_PATH.read_text(encoding="utf-8")
    submit_source = SUBMIT_PATH.read_text(encoding="utf-8")
    assert '--train-limit "${SUCC_PROPERTY_INTERACTIONS_TRAIN_LIMIT:-1500}"' in run_source
    assert '--validation-limit "${SUCC_PROPERTY_INTERACTIONS_VALIDATION_LIMIT:-20}"' in run_source
    assert '--epochs "${SUCC_PROPERTY_INTERACTIONS_EPOCHS:-8}"' in run_source
    assert "--motif-attachment" in run_source
    assert "--property-interaction-latents" in run_source
    assert "--property-latent-slots" not in run_source
    assert "--condition-attention" not in run_source
    assert "--num-attempts 20" in run_source
    assert 'SEED="${SUCC_PROPERTY_INTERACTIONS_SEED:-1741}"' in run_source
    assert "nvidia_h100_80gb_hbm3_1g.10gb:1" in submit_source
    assert "SUCC_PROPERTY_INTERACTIONS_TIME:-00:10:00" in submit_source
    assert "dongbochen1218@gmail.com" in submit_source
