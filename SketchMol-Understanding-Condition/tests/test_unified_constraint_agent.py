from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = (
    ROOT
    / "SketchMol-Understanding-Condition"
    / "experiments"
    / "unified_constraint_agent"
)
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))


def load_module(name: str):
    path = MODULE_ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ir_module = load_module("molecular_constraint_ir")
audit = load_module("audit_candidate_pools")
trajectory = load_module("build_verifier_trajectories")
common_sft = load_module("build_common_llm_sft_dataset")
common_lora = load_module("train_common_llm_lora")
common_eval = load_module("evaluate_common_llm_pilot")
constrained_eval = load_module("evaluate_common_llm_constrained_actions")
preference_data = load_module("build_common_llm_action_preferences")
verifier_preference_data = load_module("build_common_llm_verifier_preferences")
preference_train = load_module("train_common_llm_preference")
plan_protocol = load_module("common_llm_plan_protocol")
plan_preference_data = load_module("build_common_llm_plan_preferences")
plan_gate = load_module("compare_common_llm_plan_rankers")
tool_policy = load_module("common_llm_tool_policy")
tool_policy_train = load_module("train_common_llm_tool_policy_grpo")
admet_server = load_module("admet_ai_jsonl_server")
hierarchical_support = load_module("audit_hierarchical_action_support")
support_split = load_module("select_disjoint_support_rows")
retrieved_delta = load_module("build_retrieved_delta_edit_candidates")
delta_plan_protocol = load_module("retrieved_delta_plan_protocol")
delta_preference = load_module("build_retrieved_delta_plan_preferences")
delta_ranker = load_module("rank_retrieved_delta_candidates")
delta_gate = load_module("finalize_retrieved_delta_planner_gate")
delta_ceiling_pool = load_module("materialize_retrieved_delta_ceiling_pool")
delta_ceiling_audit = load_module("audit_retrieved_delta_ceiling")
composed_delta = load_module("build_composed_retrieved_delta_candidates")
mumo_parallel = load_module("mumo_parallel_protocol")
mumo_verifier = load_module("train_mumo_property_verifier")
mumo_closed_loop = load_module("build_mumo_closed_loop_dev")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_mumo_parallel_json_array_streaming_crosses_small_chunks(tmp_path: Path) -> None:
    source = tmp_path / "rows.json"
    expected = [
        {"task": "BDP", "source_smiles": "CCO", "note": "x" * 31},
        {"task": "HMPQ", "source_smiles": "CCN", "nested": {"value": 0.4}},
    ]
    source.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")

    assert list(mumo_parallel.iter_json_array(source, chunk_size=7)) == expected


def test_mumo_parallel_partition_and_shard_are_deterministic() -> None:
    group = "BDP:CCO"

    assert mumo_parallel.stable_fraction(group, seed=1711) == mumo_parallel.stable_fraction(group, seed=1711)
    assert mumo_parallel.stable_shard(group, seed=1711, shard_count=32) == mumo_parallel.stable_shard(
        group,
        seed=1711,
        shard_count=32,
    )
    assert 0 <= mumo_parallel.stable_shard(group, seed=1711, shard_count=32) < 32


def test_mumo_pair_verifier_uses_source_target_and_delta_features() -> None:
    features = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    pair_features = mumo_verifier.pair_feature_matrix(features, [(0, 1)])
    labels = mumo_verifier.threshold_labels(
        np.asarray([0.1, 0.4], dtype=np.float32),
        [(0, 1)],
        direction=1.0,
        threshold=0.2,
    )

    assert pair_features.tolist() == [[1.0, 0.0, 0.0, 1.0, -1.0, 1.0]]
    assert labels.tolist() == [True]


def test_mumo_pair_threshold_calibration_is_fit_only_and_recall_constrained() -> None:
    threshold = mumo_verifier.calibrated_decision_threshold(
        np.asarray([True, True, True, False, False]),
        np.asarray([0.9, 0.6, 0.4, 0.3, 0.1]),
        target_recall=2 / 3,
        min_precision=0.8,
    )

    assert threshold == 0.5


def test_mumo_pair_threshold_calibration_never_raises_default_boundary() -> None:
    threshold = mumo_verifier.calibrated_decision_threshold(
        np.asarray([True, True, False, False]),
        np.asarray([0.9, 0.8, 0.7, 0.1]),
        target_recall=1.0,
        min_precision=2 / 3,
    )

    assert threshold == 0.5


def test_mumo_closed_loop_pair_score_rewards_all_constraints() -> None:
    class FakeClassifier:
        classes_ = np.asarray([False, True])

        def predict_proba(self, values):
            assert values.shape == (1, 6)
            return np.asarray([[0.1, 0.9]])

    models = {
        "bbbp": {"pair_classifier": FakeClassifier(), "pair_decision_threshold": 0.5},
        "drd2": {"pair_classifier": FakeClassifier(), "pair_decision_threshold": 0.5},
    }
    score, margins = mumo_closed_loop.score_candidate(
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 1.0], dtype=np.float32),
        properties=("bbbp", "drd2"),
        models=models,
        source_tanimoto=0.7,
        retrieval_similarity=0.8,
        frequency=4,
    )

    assert margins == {"bbbp": 0.4, "drd2": 0.4}
    assert score > 8.0


def test_mumo_closed_loop_condition_id_does_not_depend_on_shard_local_index() -> None:
    raw = {
        "_uca_task_id": "BDP",
        "_uca_source_group": "BDP:CCO",
        "_uca_pair_digest": "0123456789abcdef",
        "source_smiles": "CCO",
        "external_source_bbbp": 0.3,
        "external_source_drd2": 0.2,
        "external_source_plogp": 1.0,
    }

    first = mumo_closed_loop.condition_row(raw, 0)
    later = mumo_closed_loop.condition_row(raw, 71)

    assert first["condition_id"] == "mumo_dev_bdp_0123456789abcdef"
    assert first["condition_id"] == later["condition_id"]
    assert all("target" not in key for key in first)


def test_mumo_closed_loop_uses_exact_qed_margin_without_a_learned_verifier() -> None:
    descriptor_count = len(mumo_closed_loop.feature_builder.DESCRIPTOR_NAMES)
    source = np.zeros(2048 + descriptor_count, dtype=np.float32)
    candidate = np.zeros_like(source)
    qed_index = 2048 + mumo_closed_loop.feature_builder.DESCRIPTOR_NAMES.index("QED")
    source[qed_index] = 0.40
    candidate[qed_index] = 0.65

    score, margins = mumo_closed_loop.score_candidate(
        source,
        candidate,
        properties=("qed",),
        models={},
        source_tanimoto=0.7,
        retrieval_similarity=0.8,
        frequency=4,
    )

    assert margins["qed"] == pytest.approx(0.15)
    assert score > 4.0


def test_mumo_closed_loop_retrieves_only_similar_fit_analogs() -> None:
    descriptor_count = len(mumo_closed_loop.feature_builder.DESCRIPTOR_NAMES)
    source = np.zeros(4 + descriptor_count, dtype=np.float32)
    source[:4] = [1, 1, 0, 0]
    library = (
        ["near", "far", "same"],
        np.asarray([[1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0]], dtype=np.uint8),
        np.zeros((3, descriptor_count), dtype=np.float32),
    )

    retrieved = mumo_closed_loop.retrieve_fit_analogs(
        source,
        library,
        min_tanimoto=0.3,
        limit=2,
    )

    assert [(smiles, similarity) for smiles, similarity, _features in retrieved] == [
        ("same", 1.0),
        ("near", pytest.approx(1 / 3)),
    ]


def test_mumo_closed_loop_freezes_20_attempts_without_inventing_candidates() -> None:
    ranked = [{"generated_smiles": "A"}, {"generated_smiles": "B"}, {"generated_smiles": "C"}]

    frozen = mumo_closed_loop.freeze_attempts(ranked, budget=20)

    assert len(frozen) == 20
    assert {row["generated_smiles"] for row, _repeat, _unique_rank in frozen} == {"A", "B", "C"}
    assert [repeat for _row, repeat, _unique_rank in frozen[:4]] == [False, False, False, True]
    assert [rank for _row, _repeat, rank in frozen[:5]] == [1, 2, 3, 1, 2]


def test_mumo_closed_loop_empty_support_uses_explicit_source_noop() -> None:
    feature = np.asarray([1.0, 0.0, 0.5], dtype=np.float32)

    row, returned_feature = mumo_closed_loop.source_noop_candidate("C[N+](C)(C)C", feature)

    assert row["generated_smiles"] == "C[N+](C)(C)C"
    assert row["candidate_is_noop"] is True
    assert row["source_tanimoto"] == 1.0
    np.testing.assert_array_equal(returned_feature, feature)
    assert returned_feature is not feature


def test_constraint_ir_separates_design_and_edit_actions() -> None:
    design = ir_module.build_constraint_ir(
        {
            "condition_id": "design-1",
            "task_mode": "de_novo",
            "condition_properties": "MW,QED",
            "target_MW": "320",
            "target_QED": "0.7",
            "instruction": "Generate a molecule near the requested targets.",
        }
    )
    edit = ir_module.build_constraint_ir(
        {
            "condition_id": "edit-1",
            "source_smiles": "CCO",
            "external_task_properties": "QED,BBBP",
            "external_property_objectives_json": json.dumps({"QED": "improve", "BBBP": "maintain"}),
            "external_property_directions_json": json.dumps({"QED": 1, "BBBP": 1}),
        }
    )

    assert design.task_mode == "de_novo"
    assert design.action_space == "smiles"
    assert [item.objective for item in design.constraints] == ["target", "target"]
    assert edit.task_mode == "edit"
    assert edit.action_space == "graph_edit_dsl"
    assert [(item.property, item.objective, item.direction) for item in edit.constraints] == [
        ("QED", "improve", 1),
        ("BBBP", "maintain", 0),
    ]


def make_config(tmp_path: Path) -> Path:
    candidate_csv = tmp_path / "candidates.csv"
    rows = [
        {
            "condition_id": "a",
            "task_mode": "de_novo",
            "condition_properties": "MW,QED",
            "generated_smiles": "C",
            "generation_rank": "1",
            "candidate_rank": "2",
            "valid_smiles": "True",
            "unified_property_success_fraction": "0.5",
            "unified_property_distance": "0.3",
        },
        {
            "condition_id": "a",
            "task_mode": "de_novo",
            "condition_properties": "MW,QED",
            "generated_smiles": "CC",
            "generation_rank": "2",
            "candidate_rank": "1",
            "candidate_selected": "True",
            "valid_smiles": "True",
            "unified_property_success_fraction": "1.0",
            "unified_property_distance": "0.0",
        },
        {
            "condition_id": "b",
            "task_mode": "de_novo",
            "condition_properties": "MW,QED",
            "generated_smiles": "CCC",
            "generation_rank": "1",
            "candidate_rank": "1",
            "candidate_selected": "True",
            "valid_smiles": "True",
            "unified_property_success_fraction": "0.5",
            "unified_property_distance": "0.1",
        },
        {
            "condition_id": "b",
            "task_mode": "de_novo",
            "condition_properties": "MW,QED",
            "generated_smiles": "CCCC",
            "generation_rank": "2",
            "candidate_rank": "2",
            "valid_smiles": "True",
            "unified_property_success_fraction": "0.0",
            "unified_property_distance": "0.8",
        },
    ]
    write_csv(candidate_csv, rows)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "name": "teacher",
                        "suite": "denovo",
                        "candidate_csv": str(candidate_csv),
                        "budget": 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return config


def test_candidate_audit_decomposes_support_and_selection(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    spec = audit.load_specs(config)[0]
    rows, manifest = audit.audit_run(spec)
    overall = next(row for row in rows if row["split"] == "all" and row["property_count"] == "all")

    assert manifest["condition_groups"] == 2
    assert overall["raw_at_1"] == 0.0
    assert overall["any_hit_at_k"] == 0.5
    assert overall["selected_at_k"] == 0.5
    assert overall["selection_miss"] == 0.0


def test_table1_strict_success_requires_source_similarity() -> None:
    base = {
        "source_smiles": "CCO",
        "generated_smiles": "CCN",
        "valid_smiles": "True",
        "unified_property_success_fraction": "1.0",
    }

    assert not audit.strict_success({**base, "source_similarity_success": "False"})
    assert audit.strict_success({**base, "source_similarity_success": "True"})
    assert audit.strict_success({**base, "external_official_success": "True"})
    assert audit.strict_success({**base, "table1_strict_success": "True"})


def test_direct_smiles_fields_preserve_raw_order_and_finalizer_score() -> None:
    first = {
        "generated_smiles": "C",
        "direct_candidate_index": "0",
        "direct_candidate_score": "9.0",
        "direct_candidate_strict_fraction": "0.5",
        "direct_candidate_property_distance": "0.3",
    }
    second = {
        "generated_smiles": "CC",
        "direct_candidate_index": "1",
        "direct_candidate_score": "110.0",
        "direct_candidate_strict_fraction": "1.0",
        "direct_candidate_property_distance": "0.0",
    }

    assert audit.generation_rank(first, 4) < audit.generation_rank(second, 3)
    assert audit.selection_rank(second, 3) < audit.selection_rank(first, 4)
    assert audit.strict_success(second)
    assert audit.property_distance(first) == 0.3


def test_unified_finalizer_score_overrides_generation_rank() -> None:
    first = {"generation_rank": "1", "candidate_rank": "1", "unified_finalizer_score": "0.2"}
    second = {"generation_rank": "2", "candidate_rank": "2", "unified_finalizer_score": "0.9"}

    assert audit.selection_rank(second, 1) < audit.selection_rank(first, 0)


def test_trajectory_builder_requires_strict_positive_and_keeps_revision_case(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    spec = audit.load_specs(config)[0]
    rows, summary = trajectory.build_run(spec)

    assert summary == {
        "groups": 2,
        "strict_preferences": 1,
        "revision_cases": 1,
        "skipped_no_negative": 0,
    }
    preference = next(row for row in rows if row["trajectory_type"] == "strict_preference")
    revision = next(row for row in rows if row["trajectory_type"] == "revision_needed")
    assert preference["chosen_smiles"] == "CC"
    assert preference["rejected_smiles"] == "C"
    assert revision["chosen_smiles"] == ""
    assert revision["rejected_smiles"] == "CCC"


def test_common_llm_dataset_uses_train_actions_and_separate_action_spaces(tmp_path: Path) -> None:
    denovo_csv = tmp_path / "denovo.csv"
    table1_csv = tmp_path / "table1.csv"
    mumo_csv = tmp_path / "mumo.csv"
    write_csv(
        denovo_csv,
        [
            {
                "condition_id": "design-1",
                "split": "train",
                "condition_properties": "MW,QED",
                "target_MW": "300",
                "target_QED": "0.7",
                "target_smiles": "CCO",
            }
        ],
    )
    edit_base = {
        "split": "train",
        "source_smiles": "CCO",
        "instruction_tasks": json.dumps([{"property": "QED", "direction": "increase"}]),
        "policy_target_action_json": json.dumps({"op": "replace_atom", "site": 1, "value": "N"}),
    }
    write_csv(
        table1_csv,
        [{**edit_base, "condition_id": "edit-1", "policy_target_strict_success": "True"}],
    )
    write_csv(
        mumo_csv,
        [{**edit_base, "condition_id": "mumo-1", "policy_target_paired_teacher": "True"}],
    )

    output_dir = tmp_path / "sft"
    assert (
        common_sft.main(
            [
                "--denovo-train-csv",
                str(denovo_csv),
                "--table1-action-train-csv",
                str(table1_csv),
                "--mumo-action-train-csv",
                str(mumo_csv),
                "--output-dir",
                str(output_dir),
                "--max-denovo-rows",
                "1",
                "--table1-repeat",
                "2",
                "--mumo-repeat",
                "2",
                "--validation-fraction",
                "0.01",
                "--seed",
                "7",
            ]
        )
        == 0
    )
    manifest = json.loads((output_dir / "manifest.json").read_text())
    all_rows = [
        json.loads(line)
        for filename in ("train.jsonl", "validation.jsonl")
        for line in (output_dir / filename).read_text().splitlines()
        if line.strip()
    ]
    actions = {
        json.loads(row["messages"][-1]["content"])["action_type"]
        for row in all_rows
    }

    assert manifest["data_role"] == "train_only"
    assert manifest["unique_by_origin"] == {"denovo": 1, "mumo": 1, "table1": 1}
    assert actions == {"smiles", "graph_edit_dsl"}


def test_common_llm_token_ids_accept_new_transformers_batch_encoding_shape() -> None:
    assert common_lora.input_id_list({"input_ids": [[1, 2, 3]]}) == [1, 2, 3]


def test_common_llm_eval_extracts_json_from_markdown_wrapper() -> None:
    assert common_eval.parse_action_json('```json\n{"action_type":"smiles","value":"CCO"}\n```') == {
        "action_type": "smiles",
        "value": "CCO",
    }
    assert common_eval.parse_action_json("no action") is None


def test_constrained_action_eval_reconstructs_planner_contract() -> None:
    row = constrained_eval.planner_row_from_ir(
        {
            "condition_id": "edit-2",
            "source_smiles": "CCO",
            "constraints": [
                {
                    "property": "QED",
                    "direction": 1,
                    "threshold": 0.1,
                    "source_value": 0.4,
                    "target": None,
                },
                {
                    "property": "SA",
                    "direction": -1,
                    "threshold": None,
                    "source_value": 3.0,
                    "target": None,
                },
            ],
        }
    )

    assert row["source_smiles"] == "CCO"
    assert row["external_task_properties"] == "QED,SA"
    assert json.loads(row["external_property_directions_json"]) == {
        "QED": "increase",
        "SA": "decrease",
    }
    assert row["external_source_QED"] == "0.4"


def test_constrained_action_eval_normalizes_structural_bond_key() -> None:
    left = constrained_eval.structural_action_key(
        {"op": "change_bond_order", "bond": [1, 2], "bond_order": "double"}
    )
    right = constrained_eval.structural_action_key(
        {"op": "change_bond_order", "bond": (1, 2), "bond_order": "double"}
    )

    assert left == right


def test_constrained_action_eval_preserves_target_when_prompt_is_long() -> None:
    class FakeTokenizer:
        eos_token_id = 99

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize
            prompt = list(range(12))
            return {"input_ids": [prompt if add_generation_prompt else prompt + [20, 21, 22, 23, 24]]}

    encoded = constrained_eval.encoded_action(
        FakeTokenizer(),
        [{"role": "user", "content": "long prompt"}],
        {"action_type": "graph_edit_dsl", "value": {"op": "add_atom"}},
        max_length=10,
    )

    assert len(encoded["input_ids"]) == 10
    assert encoded["input_ids"][-6:] == [20, 21, 22, 23, 24, 99]
    assert encoded["labels"][-6:] == [20, 21, 22, 23, 24, 99]


def test_preference_data_selects_structurally_hard_negative() -> None:
    expected = {
        "op": "replace_atom",
        "site": 1,
        "atom": "N",
        "prop": "QED",
        "direction": "increase",
    }
    hard = {
        "op": "replace_atom",
        "site": 2,
        "atom": "N",
        "prop": "QED",
        "direction": "increase",
    }
    easy = {
        "op": "add_fragment",
        "site": 7,
        "fragment": "C",
        "prop": "MW",
        "direction": "decrease",
    }

    selected = preference_data.select_hard_negatives(expected, [easy, expected, hard], count=2)

    assert selected == [hard, easy]


def test_preference_training_defaults_to_one_small_stable_epoch() -> None:
    args = preference_train.parse_args(
        [
            "--train-jsonl",
            "train.jsonl",
            "--input-adapter-dir",
            "adapter",
            "--output-dir",
            "output",
        ]
    )

    assert args.epochs == 1
    assert args.learning_rate == 2e-5
    assert args.batch_size == 1
    assert args.gradient_accumulation == 8


def test_verifier_preference_uses_strict_positive_and_nearest_failure() -> None:
    strict = {
        "strict_success": True,
        "instruction_success_fraction": 1.0,
        "instruction_distance": 0.0,
        "source_similarity": 0.8,
    }
    hard_negative = {
        "strict_success": False,
        "source_similarity_success": True,
        "instruction_success_fraction": 0.5,
        "instruction_distance": 0.1,
        "source_similarity": 0.9,
    }
    easy_negative = {
        "strict_success": False,
        "source_similarity_success": False,
        "instruction_success_fraction": 0.0,
        "instruction_distance": 1.0,
        "source_similarity": 0.2,
    }

    chosen, rejected = verifier_preference_data.select_verifier_preference(
        [easy_negative, strict, hard_negative],
        negative_count=2,
    )

    assert chosen is strict
    assert rejected == [hard_negative, easy_negative]


def test_plan_protocol_hides_policy_score_and_target_molecule() -> None:
    row = {
        "condition_id": "plan-1",
        "benchmark_task": "external_multiproperty_mumo",
        "source_smiles": "CCO",
        "target_smiles": "SECRET_TARGET",
        "external_task_properties": "bbbp,qed",
        "external_property_directions_json": json.dumps({"bbbp": "increase", "qed": "increase"}),
        "external_property_thresholds_json": json.dumps({"bbbp": 0.2, "qed": 0.1}),
    }
    messages = plan_protocol.plan_prompt_messages(row)
    payload = plan_protocol.plan_payload(
        [
            {
                "op": "add_atom",
                "site": 1,
                "atom": "N",
                "prop": "bbbp",
                "direction": "increase",
                "policy_score": 999.0,
            }
        ]
    )

    assert "SECRET_TARGET" not in json.dumps(messages)
    assert payload["action_type"] == "graph_edit_plan"
    assert "policy_score" not in payload["value"]["steps"][0]


def test_two_step_plan_preferences_cover_both_tradeoff_negatives() -> None:
    base = {
        "condition_id": "mumo-train-1",
        "split": "train",
        "benchmark_task": "external_multiproperty_mumo",
        "source_smiles": "CCO",
        "target_smiles": "SECRET_TARGET",
        "external_task_id": "BDP",
        "external_task_split": "ind",
        "external_task_properties": "bbbp,drd2,plogp",
        "external_property_directions_json": json.dumps(
            {"bbbp": "increase", "drd2": "increase", "plogp": "increase"}
        ),
        "external_property_thresholds_json": json.dumps({"bbbp": 0.2, "drd2": 0.2, "plogp": 1.0}),
    }
    strict = {
        **base,
        "candidate_rank": "1",
        "generated_smiles": "CCN",
        "external_official_success": "True",
        "external_strict_success": "True",
        "external_source_similarity_success": "True",
        "external_source_tanimoto": "0.7",
        "external_property_success_json": json.dumps({"bbbp": True, "drd2": True, "plogp": True}),
    }
    similarity_trap = {
        **base,
        "candidate_rank": "2",
        "generated_smiles": "CCCCN",
        "external_official_success": "True",
        "external_strict_success": "False",
        "external_source_similarity_success": "False",
        "external_source_tanimoto": "0.2",
        "external_property_success_json": json.dumps({"bbbp": True, "drd2": True, "plogp": True}),
    }
    property_trap = {
        **base,
        "candidate_rank": "3",
        "generated_smiles": "CCF",
        "external_official_success": "False",
        "external_strict_success": "False",
        "external_source_similarity_success": "True",
        "external_source_tanimoto": "0.8",
        "external_property_success_json": json.dumps({"bbbp": True, "drd2": False, "plogp": True}),
    }
    plans = {
        ("mumo-train-1", "CCN"): [
            {"op": "replace_atom", "site": 1, "atom": "N", "policy_score": 9.0}
        ],
        ("mumo-train-1", "CCCCN"): [
            {"op": "add_atom", "site": 1, "atom": "C"},
            {"op": "add_atom", "site": 2, "atom": "N"},
        ],
        ("mumo-train-1", "CCF"): [
            {"op": "replace_atom", "site": 1, "atom": "F"}
        ],
    }

    pairs, outcomes = plan_preference_data.preference_pairs(
        [strict, similarity_trap, property_trap],
        plans,
        max_negatives=3,
    )

    assert {row["hard_negative_category"] for row in pairs} == {
        "property_success_similarity_failure",
        "similarity_success_property_failure",
    }
    assert outcomes["strict_preference_condition"] == 1
    assert all("SECRET_TARGET" not in json.dumps(row["prompt_messages"]) for row in pairs)
    assert all("policy_score" not in json.dumps(row["chosen"]) for row in pairs)


def test_preference_replay_is_deduplicated_and_task_balanced() -> None:
    rows = [
        {"example_id": "d1", "origin": "denovo"},
        {"example_id": "d1", "origin": "denovo", "repeat_index": 1},
        {"example_id": "d2", "origin": "denovo"},
        {"example_id": "t1", "origin": "table1"},
        {"example_id": "m1", "origin": "mumo"},
    ]
    selected = preference_train.task_balanced_replay_rows(
        rows,
        origins=("denovo", "table1", "mumo"),
        max_per_origin=1,
        seed=7,
    )

    assert len(selected) == 3
    assert {row["origin"] for row in selected} == {"denovo", "table1", "mumo"}


def test_plan_gate_rejects_property_gain_that_loses_source_similarity(tmp_path: Path) -> None:
    def summary(sr: float, strict: float, similarity: float) -> dict[str, object]:
        metrics = {
            "all": {
                "success_rate": sr,
                "strict_success_rate": strict,
                "source_similarity_success_rate": similarity,
            }
        }
        return {"selections": {"llm_at_1": metrics, "llm_verifier_at_5": metrics}}

    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "gate.json"
    report = tmp_path / "gate.md"
    baseline.write_text(json.dumps(summary(0.30, 0.10, 0.50)), encoding="utf-8")
    candidate.write_text(json.dumps(summary(0.35, 0.11, 0.40)), encoding="utf-8")

    assert (
        plan_gate.main(
            [
                "--baseline-summary",
                str(baseline),
                "--candidate-summary",
                str(candidate),
                "--output-json",
                str(output),
                "--output-report",
                str(report),
                "--fail-on-stop",
            ]
        )
        == 3
    )
    assert json.loads(output.read_text())["decision"] == "stop"


def test_tool_policy_prompt_is_target_free_and_stop_requires_feedback() -> None:
    ir = {
        "condition_id": "edit-policy-1",
        "source_smiles": "CCO",
        "constraints": [{"property": "QED", "direction": 1, "hard": True}],
    }
    first = tool_policy.policy_prompt_messages(
        ir,
        current_smiles="CCO",
        original_source_smiles="CCO",
        previous_steps=[],
        step_index=0,
        max_steps=2,
    )
    revised = tool_policy.policy_prompt_messages(
        ir,
        current_smiles="CCN",
        original_source_smiles="CCO",
        previous_steps=[{"observation": {"strict_success": False}}],
        step_index=1,
        max_steps=2,
    )

    assert "target_smiles" not in json.dumps(first)
    assert '"action_type":"stop"' not in first[-1]["content"]
    assert '"action_type":"stop"' in revised[-1]["content"]
    assert "strict_success" in revised[-1]["content"]


def test_tool_policy_group_advantages_and_gate_require_real_signal() -> None:
    advantages = tool_policy.group_relative_advantages([1.0, 2.0, 3.0])
    assert abs(sum(advantages)) < 1e-8
    assert advantages[0] < 0 < advantages[-1]

    def metrics(reward: float, strict: float, properties: float, similarity: float) -> dict[str, object]:
        return {
            "all": {
                "mean_best_reward": reward,
                "strict_any_rate": strict,
                "property_all_any_rate": properties,
                "source_similarity_any_rate": similarity,
            }
        }

    advance = tool_policy_train.gate_metrics(
        metrics(1.0, 0.1, 0.2, 0.8),
        metrics(1.04, 0.1, 0.2, 0.8),
    )
    unsafe = tool_policy_train.gate_metrics(
        metrics(1.0, 0.1, 0.2, 0.8),
        metrics(1.2, 0.1, 0.1, 0.7),
    )

    assert advance["decision"] == "advance"
    assert unsafe["decision"] == "stop"


def test_tool_policy_exact_action_distribution_is_zero_sum_and_reward_ordered() -> None:
    probabilities, advantages, weights = tool_policy_train.exact_action_distribution(
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        temperature=1.0,
    )

    assert all(abs(item - 1.0 / 3.0) < 1e-12 for item in probabilities)
    assert advantages[0] < advantages[1] < advantages[2]
    assert weights[0] < 0 < weights[2]
    assert abs(sum(weights)) < 1e-10


def test_tool_policy_pcgrad_projects_only_conflicting_task_gradients() -> None:
    left_coefficient, right_coefficient = tool_policy_train.pcgrad_projection_coefficients(
        -1.0,
        1.0,
        2.0,
    )

    assert left_coefficient == -0.5
    assert right_coefficient == -1.0
    assert tool_policy_train.pcgrad_projection_coefficients(0.5, 1.0, 2.0) == (0.0, 0.0)
    assert tool_policy_train.pcgrad_projection_coefficients(-1.0, 0.0, 2.0) == (0.0, 0.0)


def test_tool_policy_parser_accepts_paired_pcgrad_mode() -> None:
    args = tool_policy_train.parse_args(
        [
            "--train-jsonl",
            "train.jsonl",
            "--validation-jsonl",
            "validation.jsonl",
            "--input-adapter-dir",
            "adapter",
            "--output-dir",
            "output",
            "--policy-update",
            "paired_pcgrad_exact_action_value",
        ]
    )

    assert args.policy_update == "paired_pcgrad_exact_action_value"


def test_tool_policy_exact_gate_rejects_cross_task_retention_regression() -> None:
    def rollout_metrics() -> dict[str, object]:
        return {
            "all": {
                "mean_best_reward": 1.0,
                "strict_any_rate": 0.0,
                "property_all_any_rate": 0.0,
                "source_similarity_any_rate": 1.0,
            }
        }

    def action_metrics(expected_reward: float) -> dict[str, object]:
        return {
            "all": {
                "mean_expected_reward": expected_reward,
                "mean_top1_reward": 1.0,
                "mean_property_all_probability": 0.1,
                "mean_similarity_probability": 1.0,
            }
        }

    def retention(denovo: float) -> dict[str, object]:
        return {
            "by_origin": {
                "denovo": {"mean_canonical_action_log_probability": denovo},
                "table1": {"mean_canonical_action_log_probability": -0.2},
                "mumo": {"mean_canonical_action_log_probability": -0.2},
            }
        }

    gate = tool_policy_train.gate_metrics(
        rollout_metrics(),
        rollout_metrics(),
        baseline_action_values=action_metrics(1.0),
        candidate_action_values=action_metrics(1.02),
        baseline_retention=retention(-0.2),
        candidate_retention=retention(-0.27),
        retention_max_regression=0.05,
    )

    assert gate["action_expected_reward_gain"] > 0.01
    assert gate["canonical_action_retention_gain_by_origin"]["denovo"] < -0.05
    assert gate["decision"] == "stop"


def test_tool_policy_exact_gate_rejects_lucky_best_of_k_with_top1_regression() -> None:
    def rollout_metrics(reward: float) -> dict[str, object]:
        return {
            "all": {
                "mean_best_reward": reward,
                "strict_any_rate": 0.1,
                "property_all_any_rate": 0.1,
                "source_similarity_any_rate": 1.0,
            }
        }

    def action_metrics(expected: float, top1: float) -> dict[str, object]:
        return {
            "all": {
                "mean_expected_reward": expected,
                "mean_top1_reward": top1,
                "mean_property_all_probability": 0.1,
                "mean_similarity_probability": 1.0,
            }
        }

    gate = tool_policy_train.gate_metrics(
        rollout_metrics(1.0),
        rollout_metrics(1.05),
        baseline_action_values=action_metrics(1.0, 1.0),
        candidate_action_values=action_metrics(0.999, 0.8),
    )

    assert gate["mean_best_reward_gain"] > 0.03
    assert gate["action_top1_reward_gain"] < -0.1
    assert gate["decision"] == "stop"


def test_tool_policy_routes_official_admet_properties_to_sidecar(monkeypatch) -> None:
    class FakeUnified:
        @staticmethod
        def canonical_prop(prop: str) -> str:
            return prop

        @staticmethod
        def score_property(smiles: str, prop: str) -> float:
            assert prop == "QED"
            return 0.7

    class FakeClient:
        @staticmethod
        def predict(smiles: str) -> dict[str, float]:
            assert smiles == "CCO"
            return {"bbbp": 0.8, "mutagenicity": 0.2}

    monkeypatch.setattr(tool_policy, "graph_policy_module", lambda: type("P", (), {"unified": FakeUnified})())
    monkeypatch.setattr(tool_policy, "admet_client", lambda: FakeClient())

    assert tool_policy.score_property_value("CCO", "BBBP") == 0.8
    assert tool_policy.score_property_value("CCO", "Mutagenicity") == 0.2
    assert tool_policy.score_property_value("CCO", "QED") == 0.7
    assert admet_server.finite_float(float("nan")) is None


def test_tool_policy_gradient_groups_only_the_executed_action() -> None:
    prompt = [{"role": "user", "content": "state"}]
    payloads = [
        {"action_type": "graph_edit_dsl", "value": {"op": "add_atom", "site": 0, "atom": "N"}},
        {"action_type": "stop", "value": {"reason": "done"}},
    ]
    first = tool_policy_train.PolicyDecision(prompt, payloads, 0)
    second = tool_policy_train.PolicyDecision(prompt, payloads, 1)

    assert tool_policy_train._decision_signature(first) != tool_policy_train._decision_signature(second)


def test_hierarchical_support_gate_enforces_disjoint_sources_and_fixed_n20() -> None:
    proposer_rows = [
        {"external_task_id": "BDP", "external_source_row_index": "10"},
        {"external_task_id": "BDQ", "external_source_row_index": "11"},
    ]
    audit_rows = [
        {"external_task_id": "BDP", "external_source_row_index": "20"},
        {"external_task_id": "BDQ", "external_source_row_index": "21"},
    ]
    split = hierarchical_support.validate_disjoint_rows(proposer_rows, audit_rows)
    rows = []
    for condition, success in (("ind-1", True), ("ood-1", False)):
        for rank in range(20):
            rows.append(
                {
                    "condition_id": condition,
                    "external_task_split": "ind" if condition.startswith("ind") else "ood",
                    "generated_smiles": "CCO",
                    "external_valid": "True",
                    "external_official_success": str(success and rank == 3),
                    "external_strict_success": str(success and rank == 3),
                    "external_full_property_coverage": "True",
                    "graph_edit_candidate_source": "direct_model" if rank == 0 else "graph_edit_dsl",
                }
            )

    support = hierarchical_support.summarize_official_rows(rows, candidate_budget=20)

    assert split["source_overlap"] == 0
    assert support["candidate_rows"] == 40
    assert support["all"]["property_any_rate"] == 0.5
    assert support["all"]["strict_any_rate"] == 0.5
    assert support["all"]["full_oracle_condition_rate"] == 1.0
    assert support["all"]["direct_root_in_prefix_rate"] == 1.0


def test_hierarchical_support_marks_incomplete_oracle_groups() -> None:
    rows = []
    for rank in range(20):
        rows.append(
            {
                "condition_id": "missing-drd2",
                "external_task_split": "ind",
                "generated_smiles": "CCO",
                "external_valid": "True",
                "external_official_success": "False",
                "external_strict_success": "False",
                "external_full_property_coverage": str(rank != 0),
            }
        )

    support = hierarchical_support.summarize_official_rows(rows, candidate_budget=20)

    assert support["all"]["full_oracle_candidate_rate"] == 0.95
    assert support["all"]["full_oracle_condition_rate"] == 0.0


def test_hierarchical_support_split_backfills_each_task_after_overlap() -> None:
    proposer_rows = [
        {"external_task_id": "A", "external_source_row_index": "1"},
        {"external_task_id": "B", "external_source_row_index": "4"},
    ]
    candidates = [
        {"external_task_id": "A", "external_source_row_index": "1"},
        {"external_task_id": "A", "external_source_row_index": "2"},
        {"external_task_id": "A", "external_source_row_index": "3"},
        {"external_task_id": "B", "external_source_row_index": "4"},
        {"external_task_id": "B", "external_source_row_index": "5"},
        {"external_task_id": "B", "external_source_row_index": "6"},
    ]

    selected, manifest = support_split.select_disjoint_rows(
        proposer_rows,
        candidates,
        rows_per_task=2,
    )

    assert [row["external_source_row_index"] for row in selected] == ["2", "3", "5", "6"]
    assert manifest["selected_counts_by_task"] == {"A": 2, "B": 2}
    assert manifest["excluded_overlap_counts_by_task"] == {"A": 1, "B": 1}


def test_retrieved_delta_inference_does_not_read_evaluation_target(monkeypatch) -> None:
    class EvalRow(dict):
        def get(self, key, default=None):
            if key == "target_smiles":
                raise AssertionError("RetrievedDeltaEdit inference must not read the evaluation target")
            return super().get(key, default)

    row = EvalRow(
        condition_id="eval-1",
        external_task_key="QED+BBBP",
        source_smiles="source",
    )
    transform = retrieved_delta.DeltaTransform(
        task_key="QED+BBBP",
        source_variable="source-var",
        target_variable="target-var",
        frequency=3,
        train_condition_id="train-1",
    )
    monkeypatch.setattr(
        retrieved_delta,
        "fragment_splits",
        lambda *_args: (retrieved_delta.FragmentSplit("core", "query-var", 8, 2),),
    )
    monkeypatch.setattr(retrieved_delta, "variable_similarity", lambda *_args: 0.8)
    monkeypatch.setattr(retrieved_delta, "join_fragments", lambda *_args: "generated")
    monkeypatch.setattr(
        retrieved_delta,
        "candidate_for_smiles",
        lambda _row, _smiles, *, source, **kwargs: retrieved_delta.Candidate(
            smiles="generated",
            source=source,
            source_tanimoto=0.6,
            admet_prior_score=0.7,
            **kwargs,
        ),
    )
    monkeypatch.setattr(retrieved_delta, "canonical_smiles", lambda value: value)

    candidates, summary = retrieved_delta.retrieved_candidates(
        row,
        [transform],
        min_retrieval_similarity=0.15,
        max_transforms_per_query=8,
        min_core_heavy_atoms=5,
        max_variable_heavy_atoms=30,
    )

    assert [candidate.smiles for candidate in candidates] == ["generated"]
    assert summary["approximate_query_transform_matches"] == 1


def test_retrieved_delta_selection_preserves_fixed_n20_and_similarity_gate() -> None:
    candidates = [
        retrieved_delta.Candidate(
            smiles=f"fallback-{index}",
            source="v4_graph_fallback",
            source_tanimoto=0.5,
            admet_prior_score=0.1 + index / 100.0,
            fallback_rank=index + 1,
        )
        for index in range(20)
    ]
    candidates.append(
        retrieved_delta.Candidate(
            smiles="delta",
            source="retrieved_delta_edit",
            source_tanimoto=0.65,
            admet_prior_score=0.8,
            retrieval_similarity=0.9,
            transform_frequency=4,
        )
    )

    selected = retrieved_delta.select_candidates(
        candidates,
        candidate_budget=20,
        min_source_tanimoto=0.4,
    )

    assert len(selected) == 20
    assert selected[0].smiles == "delta"


def test_retrieved_delta_plan_prompt_and_action_hide_evaluation_target() -> None:
    class EvalRow(dict):
        def get(self, key, default=None):
            if key == "target_smiles":
                raise AssertionError("Planner must not read the evaluation target")
            return super().get(key, default)

    row = EvalRow(
        condition_id="eval-plan-1",
        source_smiles="CCO",
        external_task_properties="QED,BBBP",
        external_property_directions_json=json.dumps({"QED": 1, "BBBP": 1}),
        delta_query_variable="[*:1]C",
        delta_source_variable="[*:1]C",
        delta_target_variable="[*:1]N",
    )

    messages = delta_plan_protocol.prompt_messages(row)
    action = delta_plan_protocol.action_payload(row)

    assert "target_smiles" not in json.dumps(messages)
    assert action == {
        "action_type": "retrieved_delta_edit",
        "value": {
            "op": "replace_side_chain",
            "query_variable": "[*:1]C",
            "retrieved_source_variable": "[*:1]C",
            "target_variable": "[*:1]N",
        },
    }


def test_delta_preference_metadata_does_not_serialize_training_target() -> None:
    candidate = retrieved_delta.Candidate(
        smiles="TRAINING-TARGET",
        source="retrieved_delta_edit",
        source_tanimoto=0.7,
        admet_prior_score=0.3,
        retrieval_similarity=1.0,
        transform_frequency=2,
        exact_variable_match=True,
        train_condition_id="private-row",
    )

    metadata = delta_preference.safe_metadata(candidate)

    assert "TRAINING-TARGET" not in json.dumps(metadata)
    assert "private-row" not in json.dumps(metadata)


def test_delta_planner_ranks_internal_pool_but_emits_exactly_n20(monkeypatch) -> None:
    class EvalRow(dict):
        def get(self, key, default=None):
            if key == "target_smiles":
                raise AssertionError("Planner must not read the evaluation target")
            return super().get(key, default)

    rows = [
        EvalRow(
            condition_id="audit-1",
            source_smiles="CCO",
            external_task_properties="QED",
            external_property_directions_json=json.dumps({"QED": 1}),
            generated_smiles=f"candidate-{index}",
            graph_edit_candidate_source="retrieved_delta_edit",
            candidate_rank=str(index + 1),
            delta_query_variable=f"[*:1]C{index}",
            delta_source_variable=f"[*:1]C{index}",
            delta_target_variable=f"[*:1]N{index}",
            delta_source_tanimoto="0.6",
            delta_retrieval_similarity="0.8",
            delta_selection_score=str(index / 100.0),
        )
        for index in range(25)
    ]
    monkeypatch.setattr(delta_ranker.constrained, "encoded_action", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        delta_ranker.constrained,
        "score_encoded_actions",
        lambda _model, _tokenizer, encoded, *, batch_size: list(range(len(encoded))),
    )

    selected = delta_ranker.rank_condition(
        rows,
        model=object(),
        tokenizer=object(),
        candidate_budget=20,
        planner_candidate_limit=25,
        min_source_tanimoto=0.4,
        score_batch_size=8,
        max_length=512,
    )

    assert len(selected) == 20
    assert selected[0]["generated_smiles"] == "candidate-24"
    assert [row["candidate_rank"] for row in selected] == list(range(1, 21))


def test_delta_planner_gate_requires_gain_and_preserves_unified_format(tmp_path: Path) -> None:
    def support_summary(property_rate: float, strict_rate: float, *, builder: bool) -> dict[str, object]:
        scope = {
            "conditions": 50,
            "property_any_rate": property_rate,
            "strict_any_rate": strict_rate,
            "valid_any_rate": 1.0,
            "full_oracle_condition_rate": 1.0,
        }
        summary: dict[str, object] = {
            "support": {
                "all": scope,
                "by_split": {
                    "ind": {**scope, "conditions": 25},
                    "ood": {**scope, "conditions": 25},
                },
            },
            "split_audit": {"source_overlap": 0},
            "final_oracle_candidate_budget": 20,
        }
        if builder:
            summary["candidate_builder"] = {
                "protocol": "common_llm_retrieved_delta_planner_v1",
                "evaluation_target_access": False,
                "candidate_budget": 20,
            }
        return summary

    candidate_path = tmp_path / "candidate.json"
    baseline_path = tmp_path / "baseline.json"
    candidate_path.write_text(json.dumps(support_summary(0.46, 0.46, builder=True)))
    baseline_path.write_text(json.dumps(support_summary(0.40, 0.40, builder=False)))
    groups = {
        name: {"rows": rows, "json_parse_rate": rate, "action_type_rate": rate}
        for name, rows, rate in (
            ("all", 100, 0.98),
            ("denovo", 50, 1.0),
            ("table1", 25, 0.92),
            ("mumo", 25, 1.0),
        )
    }
    baseline_format = tmp_path / "baseline-format.json"
    candidate_format = tmp_path / "candidate-format.json"
    baseline_format.write_text(json.dumps({"groups": groups}))
    candidate_format.write_text(json.dumps({"groups": groups}))
    preference = tmp_path / "preference.json"
    preference.write_text(
        json.dumps(
            {
                "protocol": "common_llm_retrieved_delta_preference_v1",
                "prompt_target_access": False,
                "training_target_role": "positive_label_only",
                "source_group_overlap": 0,
            }
        )
    )
    training = tmp_path / "training.json"
    training.write_text(json.dumps({"adapter_nonfinite_parameters": 0}))
    output = tmp_path / "gate"

    assert delta_gate.main(
        [
            "--support-summary", str(candidate_path),
            "--baseline-support-summary", str(baseline_path),
            "--baseline-format-summary", str(baseline_format),
            "--candidate-format-summary", str(candidate_format),
            "--preference-manifest", str(preference),
            "--training-summary", str(training),
            "--output-dir", str(output),
        ]
    ) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["decision"] == "advance"
    assert summary["anti_forgetting"]["passed"] is True


def test_retrieved_delta_ceiling_pool_is_oracle_blind_and_variable_k(tmp_path: Path) -> None:
    candidates = tmp_path / "enumerated.csv"
    write_csv(
        candidates,
        [
            {
                "condition_id": condition,
                "candidate_rank": rank,
                "generated_smiles": f"{condition}-{rank}",
                "target_smiles": "must-not-drive-selection",
            }
            for condition, ranks in (("b", (3, 1, 2)), ("a", (2, 1)))
            for rank in ranks
        ],
    )
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text(
        json.dumps({"evaluation_target_access": False, "evaluation_conditions": 2})
    )
    output = tmp_path / "prefix.csv"
    manifest = tmp_path / "manifest.json"

    assert delta_ceiling_pool.main(
        [
            "--enumerated-candidates-csv", str(candidates),
            "--source-manifest-json", str(source_manifest),
            "--output-csv", str(output),
            "--manifest-json", str(manifest),
            "--candidate-limit", "20",
            "--paper-candidate-budget", "20",
            "--expected-conditions", "2",
        ]
    ) == 0
    rows = delta_ceiling_pool.read_rows(output)
    payload = json.loads(manifest.read_text())

    assert [(row["condition_id"], row["candidate_rank"]) for row in rows] == [
        ("a", "1"), ("a", "2"), ("b", "1"), ("b", "2"), ("b", "3")
    ]
    assert all(row["candidate_selected"] == "False" for row in rows)
    assert payload["evaluation_target_access"] is False
    assert payload["oracle_used_for_selection"] is False
    assert payload["paper_facing_candidate_budget"] == 20
    assert payload["short_condition_count"] == 2


def test_retrieved_delta_ceiling_audit_separates_property_and_strict_support(tmp_path: Path) -> None:
    detail = tmp_path / "detail.csv"
    write_csv(
        detail,
        [
            {
                "condition_id": "ind-1",
                "external_task_split": "ind",
                "external_task_key": "BDP",
                "external_valid": "True",
                "external_full_property_coverage": "True",
                "external_official_success": "True",
                "external_strict_success": "True",
            },
            {
                "condition_id": "ind-1",
                "external_task_split": "ind",
                "external_task_key": "BDP",
                "external_valid": "True",
                "external_full_property_coverage": "True",
                "external_official_success": "False",
                "external_strict_success": "False",
            },
            {
                "condition_id": "ood-1",
                "external_task_split": "ood",
                "external_task_key": "BDMQ",
                "external_valid": "True",
                "external_full_property_coverage": "True",
                "external_official_success": "True",
                "external_strict_success": "False",
            },
        ],
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "evaluation_target_access": False,
                "diagnostic_only": True,
                "paper_facing_candidate_budget": 20,
                "diagnostic_candidate_limit": 96,
            }
        )
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"support": {"all": {"property_any_rate": 0.4, "strict_any_rate": 0.4}}})
    )
    output = tmp_path / "audit"

    assert delta_ceiling_audit.main(
        [
            "--official-detail-csv", str(detail),
            "--candidate-manifest-json", str(manifest),
            "--baseline-support-summary", str(baseline),
            "--output-dir", str(output),
            "--target-strict-ceiling", "0.5",
            "--expected-conditions", "2",
        ]
    ) == 0
    summary = json.loads((output / "summary.json").read_text())

    assert summary["support_ceiling"]["all"]["property_any_rate"] == 1.0
    assert summary["support_ceiling"]["all"]["strict_any_rate"] == 0.5
    assert summary["support_ceiling"]["all"]["mean_candidates"] == 1.5
    assert summary["decision"] == "generator_expansion_required"
    assert abs(summary["comparison_to_v5"]["strict_ceiling_gain"] - 0.1) < 1e-12


def test_composed_delta_normalizes_observed_train_property_effects() -> None:
    effects = composed_delta.observed_property_effects(
        {
            "external_task_properties": "bbbp,mutagenicity",
            "external_property_directions_json": json.dumps(
                {"bbbp": "increase", "mutagenicity": "decrease"}
            ),
            "external_property_thresholds_json": json.dumps(
                {"bbbp": 0.1, "mutagenicity": 0.1}
            ),
            "external_source_bbbp": "0.2",
            "external_target_bbbp": "0.4",
            "external_source_mutagenicity": "0.7",
            "external_target_mutagenicity": "0.4",
        }
    )

    assert abs(effects["bbbp"] - 2.0) < 1e-12
    assert abs(effects["mutagenicity"] - 3.0) < 1e-12


def test_composed_delta_cross_task_retrieval_requires_positive_query_effect() -> None:
    transforms = [
        composed_delta.EffectTransform("exact", "a", "b", 1, "x", (("qed", -1.0),)),
        composed_delta.EffectTransform("other", "c", "d", 1, "y", (("qed", 2.0),)),
        composed_delta.EffectTransform("irrelevant", "e", "f", 1, "z", (("hia", 2.0),)),
    ]

    compatible = composed_delta.compatible_transforms(
        transforms,
        query_task="exact",
        query_properties=("qed",),
    )

    assert [item.task_key for item in compatible] == ["exact", "other"]


def test_composed_delta_state_accumulates_anchor_and_revision_effects() -> None:
    step = composed_delta.EffectTransform(
        "task", "a", "b", 1, "train", (("qed", 1.2), ("plogp", 0.5))
    )
    state = composed_delta.CandidateState(
        smiles="CC",
        source="composed",
        source_tanimoto=0.7,
        admet_prior_score=0.8,
        steps=(step,),
        anchor_rank=3,
        prior_effects=(("plogp", 0.7),),
    )

    assert state.actual_step_count == 2
    assert state.predicted_effects == {"plogp": 1.2, "qed": 1.2}
    key = composed_delta.state_rank_key(
        state,
        query_properties=("plogp", "qed"),
        min_source_tanimoto=0.4,
    )
    assert key[1:3] == (2, 2)
