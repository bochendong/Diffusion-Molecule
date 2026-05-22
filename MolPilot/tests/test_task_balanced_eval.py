import unittest

from molpilot.evaluate import _origin_breakdown, _task_breakdown
from molpilot.sample import _select_pairs_by_task
from molpilot.schema import GenerationRequest, TaskType


class TaskBalancedEvalTests(unittest.TestCase):
    def test_select_pairs_caps_each_task(self):
        pairs = []
        for task in (TaskType.EDIT, TaskType.INPAINT, TaskType.DE_NOVO):
            for idx in range(5):
                pairs.append((GenerationRequest(task_type=task, source_smiles="CCO", instruction=f"{task.value} {idx}"), "CCN"))
        for idx in range(3):
            pairs.append((GenerationRequest(task_type=TaskType.REPAIR, source_smiles="CC(", instruction=f"repair {idx}"), "CCO"))
        selected = _select_pairs_by_task(pairs, max_per_task=2, tasks="edit,inpaint,de_novo", seed=1)
        counts = {}
        for request, _ in selected:
            counts[request.task_type.value] = counts.get(request.task_type.value, 0) + 1
        self.assertEqual(counts, {"edit": 2, "inpaint": 2, "de_novo": 2})
        selected = _select_pairs_by_task(pairs, max_per_task=2, tasks="repair", seed=1)
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(request.task_type == TaskType.REPAIR for request, _ in selected))

    def test_task_breakdown_reports_request_topk(self):
        rows = [
            {"request_id": "0", "rank": "0", "task_type": "edit", "overall_success": "False", "goal_success": "True", "constraint_success": "False"},
            {"request_id": "0", "rank": "1", "task_type": "edit", "overall_success": "True", "goal_success": "True", "constraint_success": "True"},
            {"request_id": "1", "rank": "0", "task_type": "de_novo", "overall_success": "True", "goal_success": "True", "constraint_success": "True"},
        ]
        metrics = _task_breakdown(rows)
        self.assertEqual(metrics["task_edit_requests"], 1.0)
        self.assertEqual(metrics["task_edit_request_overall_at_1"], 0.0)
        self.assertEqual(metrics["task_edit_request_overall_at_5"], 1.0)
        self.assertEqual(metrics["task_de_novo_request_overall_at_1"], 1.0)
        self.assertEqual(metrics["macro_task_request_overall_at_1"], 0.5)

    def test_origin_breakdown_splits_merged_candidate_origins(self):
        rows = [
            {"candidate_origin": "diffusion", "overall_success": "False"},
            {"candidate_origin": "source_guided_0.25+source_neighborhood", "overall_success": "True"},
            {"candidate_origin": "graph_edit_methyl+graph_edit_fluoro", "overall_success": "True"},
            {"candidate_origin": "scaffold_library", "overall_success": "False"},
            {"candidate_origin": "condition_direct", "overall_success": "True"},
        ]
        metrics = _origin_breakdown(rows)
        self.assertEqual(metrics["origin_diffusion_rows"], 1.0)
        self.assertEqual(metrics["origin_source_guided_0_25_overall_success"], 1.0)
        self.assertEqual(metrics["origin_source_neighborhood_overall_success"], 1.0)
        self.assertEqual(metrics["origin_graph_edit_methyl_overall_success"], 1.0)
        self.assertEqual(metrics["origin_family_graph_edit_rows"], 1.0)
        self.assertEqual(metrics["origin_family_scaffold_library_overall_success"], 0.0)
        self.assertEqual(metrics["origin_family_condition_direct_overall_success"], 1.0)


if __name__ == "__main__":
    unittest.main()
