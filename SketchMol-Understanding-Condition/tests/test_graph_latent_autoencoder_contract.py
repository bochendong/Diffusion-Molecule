from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "SketchMol-Understanding-Condition" / "experiments" / "unified_latent_flow"
TRAIN_PATH = EXPERIMENT_DIR / "train_graph_latent_autoencoder.py"
RUN_PATH = EXPERIMENT_DIR / "run_graph_latent_autoencoder_v1.sh"
SUBMIT_PATH = EXPERIMENT_DIR / "submit_graph_latent_autoencoder_v1.sh"


def test_graph_latent_stage_is_graph_native_and_has_no_selector() -> None:
    source = TRAIN_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert '"variable_length_atom_slots": True' in source
    assert '"explicit_bond_latent_slots": True' in source
    assert '"permutation_equivariant_encoder": True' in source
    assert '"one_shot_graph_decoder": True' in source
    assert '"raw_argmax_decoder": True' in source
    assert '"valence_projection_or_repair": False' in source
    assert '"condition_access": False' in source
    assert '"property_oracle_access": False' in source
    assert '"candidate_library": False' in source
    assert '"selector": False' in source
    assert '"benchmark_generation_target_access": False' in source
    assert '"stop_before_flow"' in source


def test_graph_latent_gate_covers_validity_and_structure() -> None:
    source = TRAIN_PATH.read_text(encoding="utf-8")
    for metric in (
        "clean_validity",
        "clean_connected",
        "clean_graph_tensor_exact",
        "clean_topology_exact",
        "clean_mean_tanimoto",
        "clean_scaffold_match",
        "noisy_validity",
        "noisy_mean_tanimoto",
    ):
        assert metric in source


def test_graph_latent_runner_is_bounded_and_uses_mig() -> None:
    run_source = RUN_PATH.read_text(encoding="utf-8")
    submit_source = SUBMIT_PATH.read_text(encoding="utf-8")
    assert '--epochs "${SUCC_GRAPH_LATENT_EPOCHS:-6}"' in run_source
    assert '--train-limit "${SUCC_GRAPH_LATENT_TRAIN_LIMIT:-15000}"' in run_source
    assert '--max-atoms "${SUCC_GRAPH_LATENT_MAX_ATOMS:-64}"' in run_source
    assert "01:00:00" in submit_source
    assert "nvidia_h100_80gb_hbm3_2g.20gb:1" in submit_source
    assert "dongbochen1218@gmail.com" in submit_source
