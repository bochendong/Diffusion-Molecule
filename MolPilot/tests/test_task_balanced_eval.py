import unittest

from molpilot.evaluate import _task_breakdown
from molpilot.sample import _select_pairs_by_task
from molpilot.schema import GenerationRequest, TaskType


class TaskBalancedEvalTests(unittest.TestCase):
    def test_select_pairs_caps_each_task(self):
        pairs = []
        for task in (TaskType.EDIT, TaskType.INPAINT, TaskType.DE_NOVO):
            for idx in range(5):
                pairs.append((GenerationRequest(task_type=task, source_smiles="CCO", instruction=f"{task.value} {idx}"), "CCN"))
        selected = _select_pairs_by_task(pairs, max_per_task=2, tasks="edit,inpaint,de_novo", seed=1)
        counts = {}
        for request, _ in selected:
            counts[request.task_type.value] = counts.get(request.task_type.value, 0) + 1
        self.assertEqual(counts, {"edit": 2, "inpaint": 2, "de_novo": 2})

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


if __name__ == "__main__":
    unittest.main()
