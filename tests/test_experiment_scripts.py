import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluate_dfop_tasks import EvaluationJob, _postprocess_malay
from scripts.run_dfop_tasks import main as run_dfop_tasks_main
from scripts.summarize_merge_results import main as summarize_main
from core.dfop.task_presets import dfop_fusion_run_name, dfop_sft_run_name, get_task_preset


class DFOPPathNamingTests(unittest.TestCase):
    def test_fusion_and_sft_names_include_balanced_route(self):
        self.assertEqual(
            dfop_fusion_run_name("medical", "attn", "universal", 128),
            "medical_attn_universal_r128_balanced_beta0.05",
        )
        self.assertEqual(
            dfop_fusion_run_name("thai", "attn", "universal", 128),
            "thai_attn_universal_r128_balanced_beta0.05",
        )
        self.assertEqual(
            dfop_sft_run_name("malay", "attn", "universal", 128),
            "malay_attn_universal_r128_balanced",
        )


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

    def test_index_gold_matches_letter_pred(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = root / "MalayMMLU_result_model_True_0shot.csv"
            prediction.write_text(
                "input,golds,preds\nq1,1,B\nq2,0,A\nq3,2,C\n",
                encoding="utf-8",
            )
            job = EvaluationJob("malay", "target", root, [], root, root)
            _postprocess_malay(job)
            payload = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["results"]["MalayMMLU"]["correct"], 3)


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


class DFOPTaskLauncherTests(unittest.TestCase):
    def test_launcher_writes_serializable_manifest_and_starts_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models_root = root / "models"
            preset = get_task_preset("medical")
            (models_root / preset.target_local_dir).mkdir(parents=True)
            (models_root / preset.source_local_dir).mkdir(parents=True)
            results_root = root / "results"

            class SuccessfulProcess:
                def poll(self):
                    return 0

            with patch(
                "scripts.run_dfop_tasks.subprocess.Popen",
                return_value=SuccessfulProcess(),
            ) as popen:
                return_code = run_dfop_tasks_main(
                    [
                        "--tasks",
                        "medical",
                        "--gpus",
                        "0",
                        "--models-root",
                        str(models_root),
                        "--results-root",
                        str(results_root),
                        "--dry-run",
                    ]
                )

            self.assertEqual(return_code, 0)
            self.assertEqual(popen.call_count, 1)
            manifest = json.loads((results_root / "launch_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["tasks"], ["medical"])
            self.assertEqual(manifest["commands"][0][0], popen.call_args.args[0][0])


if __name__ == "__main__":
    unittest.main()
