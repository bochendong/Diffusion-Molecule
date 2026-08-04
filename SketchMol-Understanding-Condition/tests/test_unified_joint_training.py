from __future__ import annotations

import importlib.util
import csv
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
UNIFIED_PATH = (
    ROOT
    / "SketchMol-Understanding-Condition"
    / "experiments"
    / "unified_smiles_generator"
    / "unified_smiles_generator.py"
)
PREPARE_PATH = UNIFIED_PATH.with_name("prepare_unified_joint_rows.py")
COLLECT_PATH = UNIFIED_PATH.with_name("collect_unified_joint_v2_results.py")
RUNNER_PATH = UNIFIED_PATH.with_name("unified_benchmark_runner.py")
SELECT_PATH = UNIFIED_PATH.with_name("select_unified_joint_checkpoint.py")
DISTILL_PATH = UNIFIED_PATH.with_name("build_transformation_search_distillation_rows.py")
SEARCH_POOL_PATH = UNIFIED_PATH.with_name("prepare_transformation_search_pool.py")
RL_PILOT_COLLECT_PATH = UNIFIED_PATH.with_name("collect_umtp_v1_rl_pilot.py")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


unified = load_module(UNIFIED_PATH, "unified_joint_training_module")
prepare = load_module(PREPARE_PATH, "prepare_unified_joint_rows_module")
collect = load_module(COLLECT_PATH, "collect_unified_joint_v2_results_module")
runner = load_module(RUNNER_PATH, "unified_joint_benchmark_runner_module")
selector = load_module(SELECT_PATH, "select_unified_joint_checkpoint_module")
distill = load_module(DISTILL_PATH, "build_transformation_search_distillation_rows_module")
search_pool = load_module(SEARCH_POOL_PATH, "prepare_transformation_search_pool_module")
rl_pilot_collect = load_module(RL_PILOT_COLLECT_PATH, "collect_umtp_v1_rl_pilot_module")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dataset_item(mode: str, group: str, index: int) -> dict[str, object]:
    if mode == "de_novo":
        row = {"property_count": group, "target_smiles": "CCO"}
    else:
        row = {
            "instruction_tasks": f'[{ {"property": group, "direction": "increase"} }]'.replace("'", '"'),
            "source_smiles": "CC",
            "target_smiles": "CCC",
        }
    return {"task_mode": mode, "row": row, "index": index}


def test_task_balanced_order_balances_modes_and_groups() -> None:
    dataset = []
    dataset.extend(dataset_item("de_novo", "2", idx) for idx in range(8))
    dataset.extend(dataset_item("de_novo", "3", 20 + idx) for idx in range(2))
    dataset.extend(dataset_item("edit", "MW", 40 + idx) for idx in range(3))
    dataset.extend(dataset_item("edit", "QED", 60 + idx) for idx in range(1))

    order = unified.build_epoch_order(
        dataset,
        sampling_mode="task_balanced",
        samples_per_epoch=40,
        seed=7,
    )
    selected = [dataset[index] for index in order]
    mode_counts = unified.task_mode_counts(selected)
    group_counts = unified.training_group_counts(selected)

    assert mode_counts == {"de_novo": 20, "edit": 20}
    assert group_counts == {
        "de_novo:2p": 10,
        "de_novo:3p": 10,
        "edit:MW:+1": 10,
        "edit:QED:+1": 10,
    }


def test_transformation_layout_preserves_legacy_denovo_goal_tokens() -> None:
    row = {
        "condition_id": "denovo-1",
        "task_mode": "de_novo",
        "condition_properties": "MW,QED",
        "property_count": "2",
        "target_MW": "350",
        "target_QED": "0.8",
    }
    store = unified.FeatureStore(None)
    legacy = unified.condition_array_for_row(
        row,
        store,
        32,
        max_source_tokens=16,
        condition_layout="direct_compat",
    )
    transformation = unified.condition_array_for_row(
        row,
        store,
        32,
        max_source_tokens=16,
        condition_layout="transformation",
    )
    assert np.array_equal(legacy, transformation)


def test_de_novo_distillation_masks_edit_rows() -> None:
    config = {
        "vocab_size": 7,
        "condition_dim": 4,
        "d_model": 8,
        "num_layers": 1,
        "num_heads": 2,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "pad_id": 0,
        "max_length": 16,
    }
    student = unified.ConditionedSmilesDecoder(**config)
    teacher = unified.ConditionedSmilesDecoder(**config)
    teacher.load_state_dict(student.state_dict())
    rows = [
        {
            "condition": np.ones((2, 4), dtype=np.float32),
            "decoder_input_ids": np.asarray([1, 4], dtype=np.int64),
            "target_ids": np.asarray([4, 2], dtype=np.int64),
            "task_mode": "de_novo",
        },
        {
            "condition": np.ones((2, 4), dtype=np.float32),
            "decoder_input_ids": np.asarray([1, 5], dtype=np.int64),
            "target_ids": np.asarray([5, 2], dtype=np.int64),
            "task_mode": "edit",
        },
    ]
    batch = unified.collate_batch(rows, pad_id=0)
    student_logits = student(
        batch["condition"],
        batch["decoder_input_ids"],
        condition_mask=batch["condition_mask"],
    )
    loss, token_count = unified.de_novo_distillation_loss(
        student_logits,
        teacher,
        batch,
        pad_id=0,
        temperature=1.0,
    )
    assert token_count == 2
    assert float(loss.item()) < 1e-6


def test_source_aware_decoder_preserves_de_novo_warmstart_and_reads_source() -> None:
    base_config = {
        "vocab_size": 11,
        "condition_dim": 4,
        "d_model": 8,
        "num_layers": 1,
        "num_heads": 2,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "pad_id": 0,
        "max_length": 16,
    }
    base = unified.ConditionedSmilesDecoder(**base_config).eval()
    source_aware = unified.ConditionedSmilesDecoder(
        **base_config,
        source_aware=True,
        source_encoder_layers=1,
    ).eval()
    incompatible = source_aware.load_state_dict(base.state_dict(), strict=False)
    assert not incompatible.unexpected_keys

    denovo_condition = torch.tensor([[[0.25, 0.1, 0.0, 0.0], [1.0, 0.2, 1.0, 0.0]]])
    condition_mask = torch.ones((1, 2), dtype=torch.bool)
    decoder_ids = torch.tensor([[1, 4, 5]])
    base_logits = base(denovo_condition, decoder_ids, condition_mask=condition_mask)
    source_aware_logits = source_aware(denovo_condition, decoder_ids, condition_mask=condition_mask)
    assert torch.allclose(base_logits, source_aware_logits, atol=1e-6)

    edit_a = torch.tensor([[[0.25, 0.1, 0.0, 0.0], [2.0, 0.1, 1.0, 0.0]]])
    edit_b = torch.tensor([[[0.25, 0.1, 0.0, 0.0], [2.0, 0.9, 0.0, 1.0]]])
    edit_a_logits = source_aware(edit_a, decoder_ids, condition_mask=condition_mask)
    edit_b_logits = source_aware(edit_b, decoder_ids, condition_mask=condition_mask)
    assert not torch.allclose(edit_a_logits, edit_b_logits)


def test_adaptive_distill_weight_is_a_bounded_dual_update() -> None:
    increased = unified.update_adaptive_distill_weight(
        0.3,
        observed_kl=0.08,
        target_kl=0.02,
        dual_lr=0.5,
        min_weight=0.0,
        max_weight=1.0,
    )
    decreased = unified.update_adaptive_distill_weight(
        increased,
        observed_kl=0.0,
        target_kl=0.02,
        dual_lr=1.0,
        min_weight=0.0,
        max_weight=1.0,
    )
    assert increased > 0.3
    assert 0.0 <= decreased < increased


def test_prepare_rows_drops_train_eval_overlap() -> None:
    eval_row = {
        "sample_id": "eval-1",
        "task_mode": "de_novo",
        "property_count": "2",
        "target_smiles": "CCO",
    }
    train_rows = [
        {
            "sample_id": "train-overlap",
            "task_mode": "de_novo",
            "property_count": "2",
            "target_smiles": "CCO",
        },
        {
            "sample_id": "train-clean",
            "task_mode": "de_novo",
            "property_count": "2",
            "target_smiles": "CCN",
        },
    ]
    kept, dropped = prepare.remove_train_eval_overlap(train_rows, [eval_row], policy="drop_train")
    assert [row["sample_id"] for row in kept] == ["train-clean"]
    assert len(dropped) == 1


def test_search_distillation_keeps_only_authorized_feasible_edit(tmp_path: Path) -> None:
    source_csv = tmp_path / "source.csv"
    candidate_csv = tmp_path / "candidates.csv"
    write_csv(
        source_csv,
        [
            {
                "condition_id": "edit-1",
                "task_mode": "edit",
                "source_smiles": "CC",
                "target_smiles": "CCC",
                "instruction_tasks": '[{"property":"QED","direction":"increase"}]',
            }
        ],
    )
    write_csv(
        candidate_csv,
        [
            {
                "condition_id": "edit-1",
                "generated_smiles": "CCC",
                "valid_smiles": "True",
                "unified_property_success_fraction": "1.0",
                "source_tanimoto": "0.70",
                "unified_finalizer_score": "8.0",
                "generation_rank": "2",
            },
            {
                "condition_id": "edit-1",
                "generated_smiles": "CCCC",
                "valid_smiles": "True",
                "unified_property_success_fraction": "1.0",
                "source_tanimoto": "0.40",
                "unified_finalizer_score": "9.0",
                "generation_rank": "1",
            },
        ],
    )
    args = Namespace(
        source_rows_csv=source_csv,
        candidate_csv=candidate_csv,
        output_csv=tmp_path / "distill.csv",
        manifest_json=tmp_path / "manifest.json",
        min_property_success=1.0,
        min_edit_similarity=0.65,
        winners_per_condition=1,
        source_replay_ratio=0.0,
        seed=73,
    )
    rows, manifest = distill.build_rows(args)
    assert len(rows) == 1
    assert rows[0]["target_smiles"] == "CCC"
    assert rows[0]["distillation_origin"] == "verifier_search"
    assert manifest["rejection_counts"] == {"similarity": 1}


def test_transformation_search_pool_balances_de_novo_and_edit_groups() -> None:
    denovo = {"task_mode": "de_novo", "property_count": "2", "target_smiles": "CC"}
    edit = {
        "task_mode": "edit",
        "source_smiles": "CC",
        "target_smiles": "CCC",
        "instruction_tasks": '[{"property":"MW","direction":"increase"}]',
    }
    assert search_pool.group_key(denovo) == "de_novo:2p"
    assert search_pool.group_key(edit) == "edit:MW:+1"


def test_transformation_search_pool_can_select_validation_mode(tmp_path: Path) -> None:
    input_csv = tmp_path / "validation.csv"
    output_csv = tmp_path / "edit.csv"
    manifest_json = tmp_path / "edit.manifest.json"
    write_csv(
        input_csv,
        [
            {"condition_id": "d1", "task_mode": "de_novo", "property_count": "2", "target_smiles": "CC"},
            {
                "condition_id": "e1",
                "task_mode": "edit",
                "source_smiles": "CC",
                "target_smiles": "CCC",
                "instruction_tasks": '[{"property":"MW","direction":"increase"}]',
            },
        ],
    )
    search_pool.main(
        [
            "--input-csv",
            str(input_csv),
            "--output-csv",
            str(output_csv),
            "--manifest-json",
            str(manifest_json),
            "--rows-per-group",
            "1",
            "--task-mode",
            "edit",
        ]
    )
    rows = list(csv.DictReader(output_csv.open(newline="", encoding="utf-8")))
    assert [row["condition_id"] for row in rows] == ["e1"]


def test_rl_pilot_decision_requires_edit_gain_and_ood_retention() -> None:
    records = []
    for variant, relaxed, strict, retention in (
        ("baseline", 0.10, 0.00, 0.40),
        ("rl", 0.16, 0.02, 0.39),
    ):
        for budget in (1, 8):
            for selection in ("raw", "finalizer"):
                records.append(
                    {
                        "variant": variant,
                        "task": "table1",
                        "budget": budget,
                        "selection": selection,
                        "acc_all_0_65": strict,
                        "acc_all_0_15": relaxed,
                    }
                )
                records.append(
                    {
                        "variant": variant,
                        "task": "retention",
                        "budget": budget,
                        "selection": selection,
                        "strict_success_rate": retention,
                    }
                )
    decision = rl_pilot_collect.pilot_decision(records)
    assert decision["decision"] == "go"
    assert decision["edit_gain"] is True
    assert decision["retention_ok"] is True


def test_collector_reads_stage_metadata() -> None:
    stage_root = Path("/tmp/eval/u2")
    path = stage_root / "table1" / "at128" / "raw" / "moledit_table1" / "n128" / "metrics" / "moledit_table_summary.csv"
    assert collect.metadata(path, stage_root) == {
        "task": "table1",
        "budget": "128",
        "selection": "raw",
        "source_summary": str(path),
    }


def test_prefix_budgets_share_generation_pool_and_raw_finalizer(tmp_path: Path) -> None:
    references = {"row-1": {"condition_id": "row-1", "target_smiles": "CCO"}}
    candidates = [
        {
            "condition_id": "row-1",
            "generated_smiles": f"C{rank}",
            "generation_rank": str(rank),
            "candidate_rank": str(9 - rank),
            "unified_finalizer_score": str(score),
            "candidate_pool_id": "pool-fixed",
        }
        for rank, score in ((1, 0.1), (2, 0.8), (3, 0.3), (4, 0.9), (5, 0.2), (6, 0.4), (7, 0.5), (8, 0.6))
    ]
    selected = {}
    for budget in (1, 4, 8):
        for mode in ("raw", "finalizer"):
            output = tmp_path / f"n{budget}_{mode}.csv"
            runner.select_candidate_prefixes(
                references=references,
                candidate_rows=candidates,
                output_csv=output,
                budget=budget,
                method_name="smoke",
                selection_mode=mode,
            )
            row = next(csv.DictReader(output.open(newline="", encoding="utf-8")))
            selected[(budget, mode)] = row
            assert row["candidate_pool_id"] == "pool-fixed"
            assert row["candidate_budget"] == str(budget)
    assert selected[(1, "raw")]["unified_selected_generation_rank"] == "1"
    assert selected[(8, "raw")]["unified_selected_generation_rank"] == "1"
    assert selected[(1, "finalizer")]["unified_selected_generation_rank"] == "1"
    assert selected[(4, "finalizer")]["unified_selected_generation_rank"] == "4"
    assert selected[(8, "finalizer")]["unified_selected_generation_rank"] == "4"
    assert selected[(4, "raw")]["oracle_call_type"] == "none"
    assert selected[(4, "finalizer")]["oracle_call_type"] == "rdkit_tdc_property_score"


def test_mixed_three_row_eight_candidate_protocol_smoke(tmp_path: Path) -> None:
    references = {
        row_id: {
            "condition_id": row_id,
            "task_mode": mode,
            "source_smiles": "CC" if mode == "edit" else "",
            "target_smiles": "CCC",
        }
        for row_id, mode in (("denovo-a", "de_novo"), ("edit-b", "edit"), ("denovo-c", "de_novo"))
    }
    candidates = []
    for row_id in references:
        for rank in range(1, 9):
            candidates.append(
                {
                    **references[row_id],
                    "generated_smiles": f"{row_id}-{rank}",
                    "generation_rank": rank,
                    "unified_finalizer_score": 9 - rank,
                    "candidate_pool_id": f"pool-{row_id}",
                }
            )
    for budget in (1, 4, 8):
        output = tmp_path / f"mixed_n{budget}.csv"
        summary = runner.select_candidate_prefixes(
            references=references,
            candidate_rows=candidates,
            output_csv=output,
            budget=budget,
            method_name="mixed-smoke",
            selection_mode="finalizer",
        )
        rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))
        assert summary["selected_rows"] == 3
        assert len(rows) == 3
        assert {row["candidate_budget"] for row in rows} == {str(budget)}


def test_candidate_pool_id_is_stable_and_source_copy_defaults_off(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"stable checkpoint")
    args = Namespace(
        checkpoint=checkpoint,
        resume_checkpoint=None,
        output_dir=tmp_path,
        seed=101,
        decoding_mode="sample",
        condition_layout="direct_compat",
        condition_feature_variant="full",
        condition_feature_array="query_tokens",
        num_samples=8,
        beam_size=4,
        max_new_tokens=32,
        temperature=0.85,
        top_k=40,
        top_p=0.95,
        source_similarity_threshold=0.4,
        include_source_copy_candidate=False,
    )
    row = {"condition_id": "row-1"}
    assert unified.candidate_pool_id_for_row(row, args) == unified.candidate_pool_id_for_row(row, args)
    assert unified.sampling_seed_for_row(row, 101) == unified.sampling_seed_for_row(row, 101)
    pool = [
        unified.Candidate("CC", 0.1, {}, 1),
        unified.Candidate("CCC", 0.2, {}, 2),
    ]
    assert unified.candidate_pool_hash(pool) == unified.candidate_pool_hash(list(reversed(pool)))
    parsed = unified.parse_args(
        ["sample", "--checkpoint", str(checkpoint), "--eval-csv", "x.csv", "--output-dir", "x"]
    )
    assert parsed.include_source_copy_candidate is False


def benchmark_rows(value: float) -> list[dict[str, object]]:
    return [
        {"property_count": group, "strict_success_rate": value}
        for group in range(2, 8)
    ]


def table_rows(value: float) -> list[dict[str, object]]:
    return [
        {"task_key": f"task-{index}", "Acc_all(0.15)": value}
        for index in range(10)
    ]


def write_validation_metrics(root: Path, denovo: float, table1: float) -> None:
    denovo_path, table_path = selector.metric_paths(root)
    write_csv(denovo_path, benchmark_rows(denovo))
    write_csv(table_path, table_rows(table1))


def test_validation_gate_uses_denovo_constraint_then_table1(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidates = tmp_path / "validation"
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    write_validation_metrics(baseline, 0.80, 0.10)
    for epoch, denovo, table1 in ((1, 0.79, 0.40), (2, 0.77, 0.90), (3, 0.80, 0.50)):
        checkpoint = checkpoints / f"checkpoint_epoch_{epoch:03d}.pt"
        checkpoint.write_bytes(str(epoch).encode())
        write_validation_metrics(candidates / checkpoint.stem, denovo, table1)
    result = selector.select_checkpoint(
        Namespace(
            baseline_root=baseline,
            candidate_root=candidates,
            checkpoint_dir=checkpoints,
            max_denovo_drop=0.02,
        )
    )
    assert result["status"] == "selected"
    assert result["selected_epoch"] == 3
    failed_epoch = next(row for row in result["checkpoints"] if row["epoch"] == 2)
    assert failed_epoch["status"] == "forgetting_gate_failed"


def test_prepare_joint_v2_writes_disjoint_validation_and_fixed_tests(tmp_path: Path) -> None:
    train_rows = []
    eval_rows = []
    for group in range(2, 8):
        train_rows.append(
            {
                "sample_id": f"denovo_train_{group}",
                "task_mode": "de_novo",
                "benchmark_task": "denovo_2p7p_property_design",
                "property_count": group,
                "target_smiles": f"TRAIN_D_{group}",
            }
        )
        eval_rows.append(
            {
                "sample_id": f"denovo_test_{group}",
                "task_mode": "de_novo",
                "benchmark_task": "denovo_2p7p_property_design",
                "property_count": group,
                "target_smiles": f"TEST_D_{group}",
            }
        )
    for task in range(10):
        instruction = f'[{ {"property": f"P{task}", "direction": "increase"} }]'.replace("'", '"')
        for index in range(2):
            train_rows.append(
                {
                    "sample_id": f"edit_train_{task}_{index}",
                    "task_mode": "edit",
                    "benchmark_task": "moledit_table1",
                    "instruction_tasks": instruction,
                    "source_smiles": f"TRAIN_S_{task}_{index}",
                    "target_smiles": f"TRAIN_T_{task}_{index}",
                }
            )
        eval_rows.append(
            {
                "sample_id": f"edit_test_{task}",
                "task_mode": "edit",
                "benchmark_task": "moledit_table1",
                "instruction_tasks": instruction,
                "source_smiles": f"TEST_S_{task}",
                "target_smiles": f"TEST_T_{task}",
            }
        )
    candidate_rows = [
        {"mol_id": f"val-{index}", "target_smiles": f"VALIDATION_D_{index}"}
        for index in range(6)
    ]
    ood_rows = [
        {
            "sample_id": f"ood-{index}",
            "task_mode": "de_novo",
            "benchmark_task": "denovo_ood",
            "ood_spec_id": f"spec-{index}",
            "target_smiles": f"OOD_T_{index}",
        }
        for index in range(10)
    ]
    source_train = tmp_path / "source_train.csv"
    source_eval = tmp_path / "source_eval.csv"
    candidates = tmp_path / "candidates.csv"
    ood = tmp_path / "ood.csv"
    write_csv(source_train, train_rows)
    write_csv(source_eval, eval_rows)
    write_csv(candidates, candidate_rows)
    write_csv(ood, ood_rows)
    output = {name: tmp_path / f"{name}.csv" for name in ("train", "validation", "dv", "tv", "dt", "tt", "ot")}
    manifest = tmp_path / "manifest.json"
    assert (
        prepare.main(
            [
                "--source-train-csv", str(source_train),
                "--source-eval-csv", str(source_eval),
                "--source-candidate-csv", str(candidates),
                "--ood-eval-csv", str(ood),
                "--train-output-csv", str(output["train"]),
                "--validation-output-csv", str(output["validation"]),
                "--denovo-validation-output-csv", str(output["dv"]),
                "--table1-validation-output-csv", str(output["tv"]),
                "--denovo-test-output-csv", str(output["dt"]),
                "--table1-test-output-csv", str(output["tt"]),
                "--ood-test-output-csv", str(output["ot"]),
                "--manifest-json", str(manifest),
                "--denovo-train-per-count", "1",
                "--denovo-validation-per-count", "1",
                "--denovo-test-per-count", "1",
                "--edit-train-per-task", "1",
                "--edit-validation-per-task", "1",
                "--edit-test-per-task", "1",
                "--ood-test-per-spec", "1",
            ]
        )
        == 0
    )
    payload = __import__("json").loads(manifest.read_text(encoding="utf-8"))
    assert payload["denovo_test"]["rows"] == 6
    assert payload["table1_test"]["rows"] == 10
    assert payload["ood_test"]["rows"] == 10
    assert payload["validation"]["rows"] == 16
    assert all(audit["train_test_target_overlap"] == 0 for audit in payload["split_audits"].values())


def test_collector_computes_mean_std_and_paired_delta() -> None:
    rows = []
    for stage, train_seed, values in (
        ("u0", "base", (0.50, 0.60)),
        ("u1", "7", (0.55, 0.65)),
        ("u1", "17", (0.45, 0.75)),
        ("u2", "7", (0.60, 0.70)),
    ):
        for eval_seed, value in zip(("101", "202"), values):
            rows.append(
                {
                    "stage": stage,
                    "train_seed": train_seed,
                    "eval_seed": eval_seed,
                    "benchmark": "2p7p",
                    "budget": "20",
                    "selection": "raw",
                    "group": "2",
                    "metric": "strict_success_rate",
                    "value": value,
                    "gate_status": "selected" if stage != "u0" else "baseline",
                }
            )
    aggregates = collect.aggregate_rows(rows)
    u1 = next(row for row in aggregates if row["stage"] == "u1")
    assert u1["n"] == 4
    assert u1["std"] > 0
    paired = collect.paired_delta_rows(rows)
    u1_delta = next(row for row in paired if row["stage"] == "u1")
    assert u1_delta["n_pairs"] == 4
    assert abs(u1_delta["mean_paired_delta_vs_u0"] - 0.05) < 1e-9
