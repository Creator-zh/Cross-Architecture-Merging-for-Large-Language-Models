import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.aci.presets import (
    ACI_ABLATION_PRESETS,
    aci_run_name,
    aci_sft_run_name,
    get_task_preset,
)
from scripts.evaluate_aci_ablations import summarize as summarize_ablations
from scripts.evaluate_aci_tasks import EvaluationJob, _postprocess_malay
from scripts.run_aci_ablations import main as run_aci_ablations_main
from scripts.run_aci_tasks import main as run_aci_tasks_main
from scripts.summarize_merge_results import main as summarize_main


class ACIPathNamingTests(unittest.TestCase):
    def test_names_contain_task_and_beta_only(self):
        self.assertEqual(aci_run_name("medical"), "medical_aci_beta0.03")
        self.assertEqual(aci_run_name("thai", 0.02), "thai_aci_beta0.02")
        self.assertEqual(
            aci_run_name("thai", 0.01, "attention"),
            "thai_aci_attention_beta0.01",
        )
        self.assertEqual(
            aci_run_name("malay", 0.10, "ffn"),
            "malay_aci_ffn_beta0.1",
        )
        self.assertEqual(aci_sft_run_name("malay"), "malay_aci_beta0.1_sft")

    def test_fixed_ablation_matrix_has_eight_unique_runs(self):
        self.assertEqual(len(ACI_ABLATION_PRESETS), 8)
        names = {
            aci_run_name(item.task, item.beta, item.fusion_mode)
            for item in ACI_ABLATION_PRESETS
        }
        self.assertEqual(len(names), 8)
        thai = [item for item in ACI_ABLATION_PRESETS if item.task == "thai"]
        self.assertEqual({item.beta for item in thai}, {0.01, 0.10})
        self.assertEqual({item.fusion_mode for item in thai}, {"attention", "ffn"})

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

    def test_ablation_summary_compares_variants_with_same_batch_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for task in ("medical", "thai", "malay"):
                for variant, score in self._ablation_variants(task):
                    output = root / task / variant / "primary"
                    output.mkdir(parents=True)
                    results = {}
                    for benchmark in get_task_preset(task).primary_eval_tasks:
                        metric = "f1,none" if benchmark == "xquad_th" else "acc,none"
                        results[benchmark] = {
                            metric: score,
                            "sample_len": 10,
                        }
                    (output / "results_test.json").write_text(
                        json.dumps({"results": results}),
                        encoding="utf-8",
                    )
            _, markdown = summarize_ablations(root, "primary")
            text = markdown.read_text(encoding="utf-8")
            self.assertIn("| medical | aci_attention_beta0.03 | 0.03 | 60.00 | +10.00 |", text)
            self.assertIn("| thai | aci_ffn_beta0.1 | 0.1 | 40.00 | -10.00 |", text)
            self.assertNotIn("DFOP", text)

    @staticmethod
    def _ablation_variants(task):
        variants = [("target", 0.5)]
        for item in ACI_ABLATION_PRESETS:
            if item.task == task:
                variants.append(
                    (item.variant, 0.6 if item.fusion_mode == "attention" else 0.4)
                )
        return variants


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

    def test_ablation_launcher_writes_all_eight_isolated_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models_root = root / "models"
            for task in ("medical", "thai", "malay"):
                preset = get_task_preset(task)
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

            with (
                patch(
                    "scripts.run_aci_ablations.subprocess.Popen",
                    return_value=SuccessfulProcess(),
                ) as popen,
                patch("scripts.run_aci_ablations.time.sleep"),
            ):
                return_code = run_aci_ablations_main(
                    [
                        "--gpus",
                        "0,1",
                        "--models-root",
                        str(models_root),
                        "--results-root",
                        str(results_root),
                        "--dry-run",
                    ]
                )

            self.assertEqual(return_code, 0)
            self.assertEqual(popen.call_count, 8)
            manifest = json.loads(
                (results_root / "ablation_launch_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(manifest["experiments"]), 8)
            outputs = {item["output"] for item in manifest["experiments"]}
            self.assertEqual(len(outputs), 8)
            for item in manifest["experiments"]:
                command = item["command"]
                self.assertIn("--fusion-mode", command)
                mode_index = command.index("--fusion-mode") + 1
                self.assertEqual(command[mode_index], item["fusion_mode"])


if __name__ == "__main__":
    unittest.main()
