import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.aci.presets import aci_run_name, aci_sft_run_name, get_task_preset
from scripts.evaluate_aci_tasks import EvaluationJob, _postprocess_malay
from scripts.run_aci_tasks import main as run_aci_tasks_main
from scripts.summarize_merge_results import main as summarize_main


class ACIPathNamingTests(unittest.TestCase):
    def test_names_contain_task_and_beta_only(self):
        self.assertEqual(aci_run_name("medical"), "medical_aci_beta0.03")
        self.assertEqual(aci_run_name("thai", 0.02), "thai_aci_beta0.02")
        self.assertEqual(aci_sft_run_name("malay"), "malay_aci_beta0.1_sft")

    def test_three_presets_define_reference_and_shared_source(self):
        medical = get_task_preset("medical")
        thai = get_task_preset("thai")
        malay = get_task_preset("malay")
        self.assertNotEqual(medical.reference_hf_id, medical.target_hf_id)
        self.assertEqual(thai.reference_hf_id, malay.reference_hf_id)
        self.assertEqual(medical.source_hf_id, thai.source_hf_id)


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


class ACITaskLauncherTests(unittest.TestCase):
    def test_launcher_writes_manifest_and_passes_three_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models_root = root / "models"
            preset = get_task_preset("medical")
            for name in (
                preset.target_local_dir,
                preset.reference_local_dir,
                preset.source_local_dir,
            ):
                (models_root / name).mkdir(parents=True, exist_ok=True)
            results_root = root / "results"

            class SuccessfulProcess:
                def poll(self):
                    return 0

            with patch(
                "scripts.run_aci_tasks.subprocess.Popen",
                return_value=SuccessfulProcess(),
            ) as popen:
                return_code = run_aci_tasks_main(
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
                        "--beta",
                        "medical=0.04",
                    ]
                )

            self.assertEqual(return_code, 0)
            self.assertEqual(popen.call_count, 1)
            manifest = json.loads(
                (results_root / "launch_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["method"], "aci")
            self.assertEqual(manifest["betas"]["medical"], 0.04)
            command = manifest["commands"][0]
            self.assertIn("--reference-model", command)
            self.assertIn("--source-model", command)
            self.assertNotIn("--route-solver", command)
            self.assertEqual(command[0], popen.call_args.args[0][0])


if __name__ == "__main__":
    unittest.main()
