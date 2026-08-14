from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "SketchMol-Understanding-Condition" / "experiments" / "unified_latent_flow"
FLOW_PATH = EXPERIMENT_DIR / "categorical_graph_latent_flow.py"
RUN_PATH = EXPERIMENT_DIR / "run_categorical_graph_latent_flow_pilot.sh"
SUBMIT_PATH = EXPERIMENT_DIR / "submit_categorical_graph_latent_flow_pilot.sh"
SIZE_RUN_PATH = EXPERIMENT_DIR / "run_size_adaptive_graph_latent_flow_pilot.sh"
SIZE_SUBMIT_PATH = EXPERIMENT_DIR / "submit_size_adaptive_graph_latent_flow_pilot.sh"
BELIEF_FLOW_PATH = EXPERIMENT_DIR / "categorical_graph_belief_flow.py"
BELIEF_RUN_PATH = EXPERIMENT_DIR / "run_categorical_graph_belief_flow_pilot.sh"
BELIEF_SUBMIT_PATH = EXPERIMENT_DIR / "submit_categorical_graph_belief_flow_pilot.sh"
COUPLED_FLOW_PATH = EXPERIMENT_DIR / "coupled_local_graph_belief_flow.py"
COUPLED_RUN_PATH = EXPERIMENT_DIR / "run_coupled_local_graph_belief_flow_pilot.sh"
COUPLED_SUBMIT_PATH = EXPERIMENT_DIR / "submit_coupled_local_graph_belief_flow_pilot.sh"


def test_categorical_graph_flow_has_target_blind_direct_generation_contract() -> None:
    source = FLOW_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert '"generation_target_access": False' in source
    assert '"evaluation_target_access": True' in source
    assert '"candidate_library": False' in source
    assert '"selector": False' in source
    assert '"finalizer": False' in source
    assert '"oracle_reranking": False' in source
    assert '"valence_projection_or_repair": False' in source
    assert "class EquivariantGraphVelocity" in source
    assert "def sample_from_source" in source
    sample_source = source[source.index("def sample_from_source") : source.index("def finite_mean")]
    assert "target_smiles" not in sample_source
    assert "target_example" not in sample_source


def test_categorical_graph_flow_runner_is_small_and_exact_n20() -> None:
    run_source = RUN_PATH.read_text(encoding="utf-8")
    submit_source = SUBMIT_PATH.read_text(encoding="utf-8")
    assert '--train-limit "${SUCC_GRAPH_FLOW_TRAIN_LIMIT:-1500}"' in run_source
    assert '--validation-limit "${SUCC_GRAPH_FLOW_VALIDATION_LIMIT:-16}"' in run_source
    assert '--property-counts "${SUCC_GRAPH_FLOW_PROPERTY_COUNTS:-2,3}"' in run_source
    assert "--num-attempts 20" in run_source
    assert "nvidia_h100_80gb_hbm3_1g.10gb:1" in submit_source
    assert "00:20:00" in submit_source
    assert "dongbochen1218@gmail.com" in submit_source


def test_size_adaptive_flow_models_birth_death_without_target_access() -> None:
    flow_source = FLOW_PATH.read_text(encoding="utf-8")
    run_source = SIZE_RUN_PATH.read_text(encoding="utf-8")
    submit_source = SIZE_SUBMIT_PATH.read_text(encoding="utf-8")
    assert "def target_structure" in flow_source
    assert "def sample_target_masks" in flow_source
    assert '"size_adaptive_target_count_head": bool(args.size_adaptive)' in flow_source
    assert '"inactive_slot_velocity_mask": bool(args.size_adaptive)' in flow_source
    mask_source = flow_source[flow_source.index("def sample_target_masks") : flow_source.index("def sample_from_source")]
    assert "target_smiles" not in mask_source
    assert "--size-adaptive" in run_source
    assert '--epochs "${SUCC_GRAPH_FLOW_EPOCHS:-6}"' in run_source
    assert "nvidia_h100_80gb_hbm3_1g.10gb:1" in submit_source
    assert "00:20:00" in submit_source


def test_graph_belief_flow_is_native_categorical_and_target_blind() -> None:
    source = BELIEF_FLOW_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert "class CategoricalEndpointField" in source
    assert "def mixed_categorical_state" in source
    assert "def transition_to_endpoint" in source
    assert '"native_discrete_state_path": True' in source
    assert '"joint_atom_birth_death_categories": True' in source
    assert '"joint_bond_birth_death_categories": True' in source
    assert '"continuous_latent_regression_loss": False' in source
    assert '"separate_target_count_head": False' in source
    assert '"generation_target_access": False' in source
    assert '"property_oracle_generation_access": False' in source
    assert '"candidate_library": False' in source
    assert '"selector": False' in source
    assert '"finalizer": False' in source
    assert '"valence_projection_or_repair": False' in source
    sample_source = source[source.index("def sample_from_source") : source.index("def evaluate")]
    assert "target_smiles" not in sample_source
    assert "target_example" not in sample_source
    assert "property_oracle" not in sample_source


def test_graph_belief_flow_runner_is_bounded_and_exact_n20() -> None:
    run_source = BELIEF_RUN_PATH.read_text(encoding="utf-8")
    submit_source = BELIEF_SUBMIT_PATH.read_text(encoding="utf-8")
    assert '--train-limit "${SUCC_GRAPH_BELIEF_TRAIN_LIMIT:-1500}"' in run_source
    assert '--validation-limit "${SUCC_GRAPH_BELIEF_VALIDATION_LIMIT:-16}"' in run_source
    assert '--property-counts "${SUCC_GRAPH_BELIEF_PROPERTY_COUNTS:-2,3}"' in run_source
    assert '--epochs "${SUCC_GRAPH_BELIEF_EPOCHS:-8}"' in run_source
    assert "--num-attempts 20" in run_source
    assert "nvidia_h100_80gb_hbm3_1g.10gb:1" in submit_source
    assert "00:20:00" in submit_source
    assert "dongbochen1218@gmail.com" in submit_source


def test_coupled_local_flow_has_native_no_edit_and_target_blind_sampling() -> None:
    source = COUPLED_FLOW_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert "class CoupledLocalEndpointField" in source
    assert "def change_targets" in source
    assert "def sample_symmetric_bernoulli" in source
    assert '"native_no_edit_category": True' in source
    assert '"node_then_incident_edge_coupling": True' in source
    assert '"edge_distribution_conditioned_on_sampled_nodes": True' in source
    assert '"generation_target_access": False' in source
    assert '"property_oracle_generation_access": False' in source
    assert '"selector": False' in source
    assert '"finalizer": False' in source
    assert '"valence_projection_or_repair": False' in source
    sample_source = source[source.index("def sample_from_source") : source.index("def evaluate")]
    assert "target_smiles" not in sample_source
    assert "target_example" not in sample_source
    assert "property_oracle" not in sample_source


def test_coupled_local_runner_is_small_2p_exact_n20_and_mig() -> None:
    run_source = COUPLED_RUN_PATH.read_text(encoding="utf-8")
    submit_source = COUPLED_SUBMIT_PATH.read_text(encoding="utf-8")
    assert '--train-limit "${SUCC_COUPLED_BELIEF_TRAIN_LIMIT:-1500}"' in run_source
    assert '--validation-limit "${SUCC_COUPLED_BELIEF_VALIDATION_LIMIT:-12}"' in run_source
    assert "--property-counts 2" in run_source
    assert "--num-attempts 20" in run_source
    assert "nvidia_h100_80gb_hbm3_1g.10gb:1" in submit_source
    assert "00:20:00" in submit_source
    assert "dongbochen1218@gmail.com" in submit_source
