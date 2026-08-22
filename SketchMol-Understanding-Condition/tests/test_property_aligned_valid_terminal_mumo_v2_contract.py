import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "unified_latent_external_transfer"
RUNNER = EXPERIMENT / "property_aligned_valid_terminal_mumo_v2.py"
PREREG = EXPERIMENT / "property_aligned_valid_terminal_mumo_v2_preregistration.json"


def load_runner():
    spec = importlib.util.spec_from_file_location("property_aligned_b", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_preregistration_locks_direct_signed_set_and_exact_n20():
    payload = json.loads(PREREG.read_text())
    assert payload["condition_count"] == 75
    assert payload["conditions_per_ood_task"] == 15
    assert payload["numeric_adapter"] is False
    assert payload["signed_property_set_direct_conditioning"] is True
    assert payload["train_only_graph_state_vocabulary_expansion"] is True
    assert payload["graph_slot_contract"] == "representation_checkpoint_equals_pair_tensor_slots"
    assert payload["terminal_dead_end_policy"] == "count_as_raw_failed_attempt_without_retry"
    assert payload["max_atoms"] == 64
    assert payload["condition_router_training"] is True
    assert payload["transport_training"] is True
    assert payload["event_kernel_training"] is True
    assert payload["candidate_pool_size"] == 20
    assert payload["exact_raw_attempts_per_condition"] == 20
    assert payload["molecular_candidate_ranking"] is False
    assert payload["oracle_selection"] is False
    assert payload["retry_or_resampling"] is False
    assert payload["generation_target_access"] is False
    assert payload["generation_property_oracle_access"] is False
    assert payload["support_audit_probe_target_access"] is False
    assert payload["official_test_access"] is False
    assert payload["engineering_amendment"]["failed_job_id"] == 20254004
    assert payload["engineering_amendment"]["model_training_started"] is False
    assert payload["engineering_amendment"]["probe_target_used_for_support_audit"] is False
    assert payload["engineering_amendment"]["probe_target_available_to_training_or_generation_process"] is False
    assert payload["graph_slot_amendment"]["failed_job_id"] == 20280538
    assert payload["graph_slot_amendment"]["optimizer_step_completed"] is False
    assert payload["terminal_dead_end_amendment"]["failed_job_ids"] == [20283293, 20284243]
    assert payload["terminal_dead_end_amendment"]["candidate_artifact_written"] is False


def test_signed_property_tokens_are_explicit_and_masked():
    module = load_runner()
    tokens = module.signed_property_tokens("BDMQ", 64)
    assert tokens.shape == (7, 64)
    assert np.count_nonzero(tokens[0]) == 0
    active = set(module.prior.task_spec("BDMQ").properties)
    for index, prop in enumerate(module.PROPERTIES):
        if prop in active:
            assert tokens[index + 1, index] == 1.0
            assert tokens[index + 1, 6] == module.SIGNS[prop]
            assert tokens[index + 1, 7] == 1.0
        else:
            assert np.count_nonzero(tokens[index + 1]) == 0
    assert tokens[1 + module.PROPERTIES.index("mutagenicity"), 6] == -1.0


def test_probe_selection_is_source_only_and_globally_unique(monkeypatch):
    module = load_runner()
    monkeypatch.setattr(module, "_canonical", lambda value: str(value or ""))
    rows = []
    for task in module.OOD_TASKS:
        for index in range(4):
            rows.append(
                {
                    "_uca_task_id": task,
                    "source_smiles": "C" * (index + 1) + "N" + task[0],
                    "target_smiles": "target-is-not-used-for-selection",
                }
            )
    # Use distinct valid canonical sources in the actual selector contract test.
    rows = []
    atoms = ["CCN", "CCO", "CCC", "CCF", "CCCl", "CCBr", "CCS"]
    for task_index, task in enumerate(module.OOD_TASKS):
        for index, source in enumerate(atoms):
            rows.append(
                {
                    "_uca_task_id": task,
                    "source_smiles": source + ("C" * task_index),
                    "target_smiles": "CC",
                }
            )
    selected = module._select_probe(rows, set(), per_task=1, seed=9)
    assert len(selected) == len(module.OOD_TASKS)
    assert len({module._canonical(row["source_smiles"]) for row in selected}) == len(selected)


def test_trainfreeze_generation_path_never_receives_sealed_targets():
    source = RUNNER.read_text()
    assert "--generation-conditions" in source
    assert "sealed_probe_targets" in source
    trainfreeze = source[source.index("def trainfreeze"): source.index("def gate")]
    assert "sealed_probe_targets" not in trainfreeze
    assert '"generation_target_access": False' in trainfreeze
    assert "A scientific STOP is a valid completed experiment" in source


def test_vocabulary_expansion_preserves_old_action_prefix(monkeypatch):
    module = load_runner()

    class FakeFullGraph:
        @staticmethod
        def build_joint_state_vocabulary(_pairs):
            return {
                "node_states": np.asarray([[0, 3], [6, 0], [8, 0]]),
                "edge_states": np.asarray([[0, 0], [1, 0], [2, 0]]),
            }

    monkeypatch.setitem(__import__("sys").modules, "discrete_graph_diffusion_decoder", FakeFullGraph)
    old = {
        "node_states": np.asarray([[0, 3], [6, 0]]),
        "edge_states": np.asarray([[0, 0], [1, 0]]),
        "blank_node_id": 0,
        "blank_edge_id": 0,
    }
    expanded = module._expanded_vocabulary(old, [object()])
    np.testing.assert_array_equal(expanded["node_states"][:2], old["node_states"])
    np.testing.assert_array_equal(expanded["edge_states"][:2], old["edge_states"])
    assert expanded["added_node_state_count"] == 1
    assert expanded["added_edge_state_count"] == 1


def test_pair_slot_contract_rejects_inconsistent_edge_axes():
    module = load_runner()

    class Example:
        atomic_number = np.zeros(64)
        bond = np.zeros((64, 63))

    class Pair:
        source = Example()
        target = Example()

    try:
        module._pair_slot_counts([Pair()])
    except ValueError as error:
        assert "inconsistent node and edge axes" in str(error)
    else:
        raise AssertionError("inconsistent graph axes must fail before training")


def test_dead_end_support_counts_raw_failure_without_retry():
    module = load_runner()
    torch = pytest.importorskip("torch")

    class FakeExactSupport:
        vocabulary = {"node_states": []}

        def __call__(self, *_args, **_kwargs):
            raise RuntimeError("B40 dynamic support reached a dead end: [0]")

        @staticmethod
        def manifest():
            return {"state_checks": 2}

    support = module._AbsorbingFailedAttemptSupport(
        FakeExactSupport(),
        lambda *_args: torch.tensor([[True, True]]),
        lambda *_args: torch.tensor([False]),
    )
    source = {"atomic_number": torch.ones((1, 2), dtype=torch.long)}
    legal, diagnostics = support(
        None,
        source,
        torch.zeros((1, 2), dtype=torch.long),
        torch.zeros((1, 2, 2), dtype=torch.long),
        torch.ones((1, 2), dtype=torch.bool),
        {},
        {},
    )
    assert legal.tolist() == [[True, False]]
    assert diagnostics["absorbing_failed_attempt"].tolist() == [True]
    assert support.manifest()["dead_end_absorptions"] == 1
    assert support.manifest()["dead_end_policy"] == "count_as_raw_failed_attempt_without_retry"


def test_slurm_dag_separates_oracle_and_science_gate():
    submit = (EXPERIMENT / "submit_property_aligned_valid_terminal_mumo_v2.sh").read_text()
    assert "afterok:$prepare_id" in submit
    assert "afterok:$train_id" in submit
    assert "afterok:$oracle_id" in submit
    assert "nvidia_h100_80gb_hbm3_2g.20gb:1" in submit
    run = (EXPERIMENT / "run_property_aligned_valid_terminal_mumo_v2.sh").read_text()
    assert "run_external_multiproperty_generated_oracle_pipeline.sh" in run
    assert '--representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt"' in run
    assert "property_aligned_valid_terminal_mumo_v2_deadend_safe" in run
    assert 'case "$STAGE" in' in run
