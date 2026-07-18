import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_dfop_tasks import EvaluationJob, _postprocess_malay
from scripts.summarize_merge_results import main as summarize_main
from core.dfop.task_presets import get_task_preset


class MalayPostprocessTests(unittest.TestCase):
    def test_prediction_csv_is_converted_to_accuracy_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = root / "MalayMMLU_result_model_True_0shot.csv"
            prediction.write_text(
                "input,golds,preds\nq1,0,0\nq2,1,2\nq3,2.0,2\n",
                encoding="utf-8",
            )
            job = EvaluationJob("malay", "target", root, [], root, root)
            _postprocess_malay(job)
            payload = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
            metrics = payload["results"]["MalayMMLU"]
            self.assertEqual(metrics["correct"], 2)
            self.assertEqual(metrics["total"], 3)
            self.assertAlmostEqual(metrics["acc,none"], 2 / 3)


class MergeSummaryTests(unittest.TestCase):
    def test_complete_medical_result_generates_macro_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "medical" / "target" / "primary"
            output.mkdir(parents=True)
            results = {
                benchmark: {"acc,none": 0.5}
                for benchmark in get_task_preset("medical").primary_eval_tasks
            }
            (output / "results_test.json").write_text(
                json.dumps({"results": results}), encoding="utf-8"
            )
            return_code = summarize_main(
                [
                    "--eval-root",
                    str(root),
                    "--tasks",
                    "medical",
                    "--variants",
                    "target",
                ]
            )
            self.assertEqual(return_code, 0)
            summary = (root / "merge_only_summary.md").read_text(encoding="utf-8")
            self.assertIn("| medical | 50.00 |", summary)


if __name__ == "__main__":
    unittest.main()
