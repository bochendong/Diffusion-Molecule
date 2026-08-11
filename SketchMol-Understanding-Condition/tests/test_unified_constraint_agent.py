from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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
