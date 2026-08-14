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
VQ_MOTIF_PATH = EXPERIMENT_DIR / "vq_motif_graph_belief_flow.py"
VQ_MOTIF_RUN_PATH = EXPERIMENT_DIR / "run_vq_motif_graph_belief_flow_pilot.sh"
VQ_MOTIF_SUBMIT_PATH = EXPERIMENT_DIR / "submit_vq_motif_graph_belief_flow_pilot.sh"
HIERARCHICAL_VQ_PATH = EXPERIMENT_DIR / "hierarchical_vq_motif_graph_flow.py"
HIERARCHICAL_VQ_RUN_PATH = EXPERIMENT_DIR / "run_hierarchical_vq_motif_graph_flow_pilot.sh"
HIERARCHICAL_VQ_SUBMIT_PATH = EXPERIMENT_DIR / "submit_hierarchical_vq_motif_graph_flow_pilot.sh"
ANCHORED_HIERARCHICAL_VQ_RUN_PATH = EXPERIMENT_DIR / "run_source_anchored_hierarchical_vq_graph_flow_pilot.sh"
ANCHORED_HIERARCHICAL_VQ_SUBMIT_PATH = EXPERIMENT_DIR / "submit_source_anchored_hierarchical_vq_graph_flow_pilot.sh"
REGION_HIERARCHICAL_VQ_RUN_PATH = EXPERIMENT_DIR / "run_connected_region_hierarchical_vq_graph_flow_pilot.sh"
REGION_HIERARCHICAL_VQ_SUBMIT_PATH = EXPERIMENT_DIR / "submit_connected_region_hierarchical_vq_graph_flow_pilot.sh"
DELTA_HIERARCHICAL_VQ_RUN_PATH = EXPERIMENT_DIR / "run_categorical_delta_hierarchical_vq_graph_flow_pilot.sh"
DELTA_HIERARCHICAL_VQ_SUBMIT_PATH = EXPERIMENT_DIR / "submit_categorical_delta_hierarchical_vq_graph_flow_pilot.sh"


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
    assert '--property-counts "${SUCC_COUPLED_BELIEF_PROPERTY_COUNTS:-2}"' in run_source
    assert "--num-attempts 20" in run_source
    assert "nvidia_h100_80gb_hbm3_1g.10gb:1" in submit_source
    assert "00:20:00" in submit_source
    assert "dongbochen1218@gmail.com" in submit_source


def test_vq_motif_flow_samples_one_latent_token_without_target_access() -> None:
    source = VQ_MOTIF_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert "class VQMotifGraphFlow" in source
    assert "def posterior_vector" in source
    assert "def quantize" in source
    assert "def prior_logits" in source
    assert '"single_discrete_motif_token_per_attempt": True' in source
    assert '"posterior_train_only": True' in source
    assert '"source_condition_prior": True' in source
    assert '"deterministic_category_decode_given_token": True' in source
    assert '"token_contrastive_reconstruction": True' in source
    assert '"independent_atom_or_bond_sampling": False' in source
    assert '"generation_target_access": False' in source
    assert '"candidate_library": False' in source
    assert '"selector": False' in source
    assert '"valence_projection_or_repair": False' in source
    sample_source = source[source.index("def sample_from_source") : source.index("def evaluate")]
    assert "target_smiles" not in sample_source
    assert "target_example" not in sample_source
    assert "property_oracle" not in sample_source
    assert "categorical_sample" not in sample_source


def test_vq_motif_runner_is_bounded_2p3p_exact_n20_and_mig() -> None:
    run_source = VQ_MOTIF_RUN_PATH.read_text(encoding="utf-8")
    submit_source = VQ_MOTIF_SUBMIT_PATH.read_text(encoding="utf-8")
    assert '--train-limit "${SUCC_VQ_MOTIF_TRAIN_LIMIT:-1500}"' in run_source
    assert '--validation-limit "${SUCC_VQ_MOTIF_VALIDATION_LIMIT:-20}"' in run_source
    assert '--property-counts "${SUCC_VQ_MOTIF_PROPERTY_COUNTS:-2,3}"' in run_source
    assert '--codebook-size "${SUCC_VQ_MOTIF_CODEBOOK_SIZE:-64}"' in run_source
    assert '--gate-min-active-codes "${SUCC_VQ_MOTIF_MIN_ACTIVE_CODES:-4}"' in run_source
    assert '--contrastive-loss-weight "${SUCC_VQ_MOTIF_CONTRASTIVE_WEIGHT:-0.25}"' in run_source
    assert "--num-attempts 20" in run_source
    assert "nvidia_h100_80gb_hbm3_1g.10gb:1" in submit_source
    assert "00:20:00" in submit_source
    assert "dongbochen1218@gmail.com" in submit_source


def test_hierarchical_vq_flow_is_two_level_and_target_blind() -> None:
    source = HIERARCHICAL_VQ_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert "class HierarchicalVQGraphFlow" in source
    assert "def constraint_prior_logits" in source
    assert "def motif_prior_logits" in source
    assert '"hierarchical_constraint_then_motif_tokens": True' in source
    assert '"constraint_posterior_train_only": True' in source
    assert '"motif_posterior_train_only": True' in source
    assert '"source_condition_constraint_prior": True' in source
    assert '"constraint_conditioned_motif_prior": True' in source
    assert '"separate_token_contrastive_reconstruction": True' in source
    assert '"deterministic_category_decode_given_tokens": True' in source
    assert '"generation_target_access": False' in source
    assert '"property_oracle_generation_access": False' in source
    assert '"independent_atom_or_bond_sampling": False' in source
    assert '"selector": False' in source
    assert '"finalizer": False' in source
    assert '"valence_projection_or_repair": False' in source
    sample_source = source[source.index("def sample_from_source") : source.index("def evaluate")]
    assert "target_smiles" not in sample_source
    assert "target_example" not in sample_source
    assert "property_oracle" not in sample_source
    assert "categorical_sample" not in sample_source


def test_hierarchical_vq_runner_is_bounded_2p3p_exact_n20_and_mig() -> None:
    run_source = HIERARCHICAL_VQ_RUN_PATH.read_text(encoding="utf-8")
    submit_source = HIERARCHICAL_VQ_SUBMIT_PATH.read_text(encoding="utf-8")
    assert '--train-limit "${SUCC_HIER_VQ_TRAIN_LIMIT:-1500}"' in run_source
    assert '--validation-limit "${SUCC_HIER_VQ_VALIDATION_LIMIT:-20}"' in run_source
    assert '--property-counts "${SUCC_HIER_VQ_PROPERTY_COUNTS:-2,3}"' in run_source
    assert '--constraint-codebook-size "${SUCC_HIER_VQ_CONSTRAINT_CODEBOOK_SIZE:-16}"' in run_source
    assert '--motif-codebook-size "${SUCC_HIER_VQ_MOTIF_CODEBOOK_SIZE:-64}"' in run_source
    assert '--gate-min-constraint-codes "${SUCC_HIER_VQ_MIN_CONSTRAINT_CODES:-3}"' in run_source
    assert '--gate-min-motif-codes "${SUCC_HIER_VQ_MIN_MOTIF_CODES:-4}"' in run_source
    assert "--num-attempts 20" in run_source
    assert "nvidia_h100_80gb_hbm3_1g.10gb:1" in submit_source
    assert "00:20:00" in submit_source
    assert "dongbochen1218@gmail.com" in submit_source


def test_source_anchored_hierarchical_decoder_is_learned_and_not_repair() -> None:
    source = HIERARCHICAL_VQ_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert "class SourceAnchoredEndpointField" in source
    assert "def change_targets" in source
    assert "def masked_binary_loss" in source
    assert '"source_anchored_residual_decoder": bool(args.source_anchored)' in source
    assert '"learned_atom_and_bond_edit_blocks": bool(' in source
    assert 'args.source_anchored and not args.connected_region' in source
    assert '"deterministic_edit_gates": bool(' in source
    assert '"posthoc_source_copy_heuristic": False' in source
    assert '"valence_projection_or_repair": False' in source
    sample_source = source[source.index("def sample_from_source") : source.index("def evaluate")]
    assert "target_smiles" not in sample_source
    assert "target_example" not in sample_source
    assert "property_oracle" not in sample_source
    assert "categorical_sample" not in sample_source
    assert ".gt(0)" in sample_source


def test_source_anchored_runner_is_matched_exact_n20_and_mig() -> None:
    run_source = ANCHORED_HIERARCHICAL_VQ_RUN_PATH.read_text(encoding="utf-8")
    submit_source = ANCHORED_HIERARCHICAL_VQ_SUBMIT_PATH.read_text(encoding="utf-8")
    assert '--train-limit "${SUCC_ANCHORED_HIER_VQ_TRAIN_LIMIT:-1500}"' in run_source
    assert '--validation-limit "${SUCC_ANCHORED_HIER_VQ_VALIDATION_LIMIT:-20}"' in run_source
    assert '--property-counts "${SUCC_ANCHORED_HIER_VQ_PROPERTY_COUNTS:-2,3}"' in run_source
    assert '--edit-gate-loss-weight "${SUCC_ANCHORED_HIER_VQ_GATE_WEIGHT:-0.50}"' in run_source
    assert "--source-anchored" in run_source
    assert "--num-attempts 20" in run_source
    assert 'SEED="${SUCC_ANCHORED_HIER_VQ_SEED:-1741}"' in run_source
    assert "nvidia_h100_80gb_hbm3_1g.10gb:1" in submit_source
    assert "00:20:00" in submit_source
    assert "dongbochen1218@gmail.com" in submit_source


def test_connected_region_decoder_is_structured_target_blind_generation() -> None:
    source = HIERARCHICAL_VQ_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert "def connected_region_target" in source
    assert "def connected_region_losses" in source
    assert "def project_connected_region" in source
    assert '"connected_region_decoder": bool(args.connected_region)' in source
    assert '"learned_region_size": bool(args.connected_region)' in source
    assert '"latent_scored_connected_projection": bool(args.connected_region)' in source
    assert '"whole_region_endpoint_subgraph": bool(' in source
    assert 'args.connected_region and not args.categorical_delta' in source
    assert '"source_boundary_preserved": bool(args.connected_region)' in source
    assert '"posthoc_source_copy_heuristic": False' in source
    assert '"valence_projection_or_repair": False' in source
    sample_source = source[source.index("def sample_from_source") : source.index("def evaluate")]
    assert "target_smiles" not in sample_source
    assert "target_example" not in sample_source
    assert "property_oracle" not in sample_source
    assert "categorical_sample" not in sample_source
    assert "project_connected_region" in sample_source


def test_connected_region_runner_is_matched_exact_n20_and_mig() -> None:
    run_source = REGION_HIERARCHICAL_VQ_RUN_PATH.read_text(encoding="utf-8")
    submit_source = REGION_HIERARCHICAL_VQ_SUBMIT_PATH.read_text(encoding="utf-8")
    assert '--train-limit "${SUCC_REGION_HIER_VQ_TRAIN_LIMIT:-1500}"' in run_source
    assert '--validation-limit "${SUCC_REGION_HIER_VQ_VALIDATION_LIMIT:-20}"' in run_source
    assert '--property-counts "${SUCC_REGION_HIER_VQ_PROPERTY_COUNTS:-2,3}"' in run_source
    assert '--edit-gate-loss-weight "${SUCC_REGION_HIER_VQ_REGION_WEIGHT:-0.50}"' in run_source
    assert "--source-anchored" in run_source
    assert "--connected-region" in run_source
    assert "--num-attempts 20" in run_source
    assert 'SEED="${SUCC_REGION_HIER_VQ_SEED:-1741}"' in run_source
    assert "nvidia_h100_80gb_hbm3_1g.10gb:1" in submit_source
    assert "00:20:00" in submit_source
    assert "dongbochen1218@gmail.com" in submit_source


def test_categorical_delta_decoder_has_sparse_legal_target_blind_grammar() -> None:
    source = HIERARCHICAL_VQ_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert "def categorical_delta_targets" in source
    assert "def categorical_delta_losses" in source
    assert "def apply_categorical_graph_delta" in source
    assert "selected_logits = logits[eligible].float()" in source
    assert "dtype=torch.float32" in source
    assert '["KEEP", "DELETE", "BIRTH", "REPLACE"]' in source
    assert '["KEEP", "DELETE", "SET"]' in source
    assert '"explicit_keep_category": bool(args.categorical_delta)' in source
    assert '"legal_operation_mask_from_source_occupancy": bool(args.categorical_delta)' in source
    assert '"region_internal_sparse_delta": bool(args.categorical_delta)' in source
    assert 'args.connected_region and not args.categorical_delta' in source
    sample_source = source[source.index("def sample_from_source") : source.index("def evaluate")]
    assert "target_smiles" not in sample_source
    assert "target_example" not in sample_source
    assert "property_oracle" not in sample_source
    assert "apply_categorical_graph_delta" in sample_source


def test_categorical_delta_runner_is_matched_exact_n20_and_mig() -> None:
    run_source = DELTA_HIERARCHICAL_VQ_RUN_PATH.read_text(encoding="utf-8")
    submit_source = DELTA_HIERARCHICAL_VQ_SUBMIT_PATH.read_text(encoding="utf-8")
    assert '--train-limit "${SUCC_DELTA_HIER_VQ_TRAIN_LIMIT:-1500}"' in run_source
    assert '--validation-limit "${SUCC_DELTA_HIER_VQ_VALIDATION_LIMIT:-20}"' in run_source
    assert '--property-counts "${SUCC_DELTA_HIER_VQ_PROPERTY_COUNTS:-2,3}"' in run_source
    assert '--delta-loss-weight "${SUCC_DELTA_HIER_VQ_DELTA_WEIGHT:-0.50}"' in run_source
    assert "--source-anchored" in run_source
    assert "--connected-region" in run_source
    assert "--categorical-delta" in run_source
    assert "--num-attempts 20" in run_source
    assert 'SEED="${SUCC_DELTA_HIER_VQ_SEED:-1741}"' in run_source
    assert "nvidia_h100_80gb_hbm3_1g.10gb:1" in submit_source
    assert "00:20:00" in submit_source
    assert "dongbochen1218@gmail.com" in submit_source
