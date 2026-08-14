from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "experiments" / "unified_latent_flow"
MODEL_PATH = EXPERIMENT_DIR / "hierarchical_vq_motif_graph_flow.py"
RUN_PATH = EXPERIMENT_DIR / "run_residual_property_slots_motif_graph_flow_pilot.sh"
SUBMIT_PATH = EXPERIMENT_DIR / "submit_residual_property_slots_motif_graph_flow_pilot.sh"


def test_b14_is_an_exact_b11_baseline_plus_zero_initialized_residuals() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert "class ResidualPropertyLatentSlotComposer" in source
    assert "def residual_property_latent_slot_tokens" in source
    assert "baseline = base.condition_vector" in source
    assert "return baseline + learned_sum + self.count_residual" in source
    assert "nn.init.zeros_(self.slot_residual[-1].weight)" in source
    assert "nn.init.zeros_(self.count_residual.weight)" in source
    assert 'RESIDUAL_PROPERTY_SLOTS_PROTOCOL = "residual_property_slots_motif_graph_flow_pilot_v14"' in source
    sample_source = source[source.index("def sample_from_source") : source.index("def evaluate")]
    assert "target_smiles" not in sample_source
    assert "property_oracle" not in sample_source


def test_b14_keeps_new_modules_after_the_unchanged_b11_decoder() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    constructor = source[source.index("class HierarchicalVQGraphFlow") : source.index("    @staticmethod\n    def source_pool")]
    assert constructor.index("self.decoder = decoder_class") < constructor.index(
        "self.residual_property_slot_router ="
    )
    assert '"exact_b11_condition_baseline": bool(' in source
    assert '"zero_initialized_property_residuals": bool(' in source


def test_b14_runner_is_matched_and_uses_a_small_mig() -> None:
    run_source = RUN_PATH.read_text(encoding="utf-8")
    submit_source = SUBMIT_PATH.read_text(encoding="utf-8")
    assert '--train-limit "${SUCC_RESIDUAL_SLOTS_TRAIN_LIMIT:-1500}"' in run_source
    assert '--validation-limit "${SUCC_RESIDUAL_SLOTS_VALIDATION_LIMIT:-20}"' in run_source
    assert '--epochs "${SUCC_RESIDUAL_SLOTS_EPOCHS:-8}"' in run_source
    assert "--motif-attachment" in run_source
    assert "--residual-property-latent-slots" in run_source
    assert "--condition-attention" not in run_source
    assert "--num-attempts 20" in run_source
    assert 'SEED="${SUCC_RESIDUAL_SLOTS_SEED:-1741}"' in run_source
    assert "nvidia_h100_80gb_hbm3_1g.10gb:1" in submit_source
    assert 'SUCC_RESIDUAL_SLOTS_TIME:-00:10:00' in submit_source
    assert "dongbochen1218@gmail.com" in submit_source
