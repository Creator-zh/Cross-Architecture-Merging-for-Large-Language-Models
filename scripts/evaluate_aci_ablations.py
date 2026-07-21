#!/usr/bin/env python
"""Evaluate the fixed ACI module ablations and compare each only with target."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from core.aci.presets import (  # noqa: E402
    ACI_ABLATION_PRESETS,
    TASK_PRESETS,
    aci_run_name,
    get_task_preset,
)
from scripts.evaluate_aci_tasks import main as evaluate_main  # noqa: E402
from scripts.summarize_merge_results import _latest_result, _pick_metric  # noqa: E402


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=_csv, default=["0", "1"])
    parser.add_argument("--models-root", type=Path, default=REPOSITORY_ROOT / "models")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPOSITORY_ROOT / "merge_results" / "aci",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=REPOSITORY_ROOT / "evaluation_results" / "aci_ablations",
    )
    parser.add_argument("--scope", choices=("primary", "extended", "all"), default="primary")
    parser.add_argument("--lm-eval-repo", type=Path, default=None)
    parser.add_argument("--lm-eval-dtype", default="float")
    parser.add_argument("--lm-eval-batch-size", default="8")
    parser.add_argument(
        "--malay-repo",
        type=Path,
        default=REPOSITORY_ROOT / "evaluation" / "malay" / "MalayMMLU",
    )
    parser.add_argument("--malay-token", default=os.environ.get("MALAY_HF_TOKEN", ""))
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Skip evaluation and rebuild CSV/Markdown from existing results",
    )
    return parser


def _expected_benchmarks(task: str, scope: str) -> tuple[str, ...]:
    preset = get_task_preset(task)
    if scope == "primary":
        return preset.primary_eval_tasks
    if scope == "extended":
        return preset.extended_eval_tasks
    return preset.primary_eval_tasks + preset.extended_eval_tasks


def _sample_count(payload: dict, benchmark: str, values: dict) -> int | None:
    for key in ("sample_len", "total"):
        value = values.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    task_counts = payload.get("n-samples", {}).get(benchmark, {})
    for key in ("effective", "original"):
        value = task_counts.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    return None


def _task_variants(
    task: str,
    experiments=ACI_ABLATION_PRESETS,
) -> list[tuple[str, str, float | None]]:
    variants = [("target", "target", None)]
    variants.extend(
        (experiment.variant, experiment.fusion_mode, experiment.beta)
        for experiment in experiments
        if experiment.task == task
    )
    return variants


def summarize(
    eval_root: Path,
    scope: str,
    experiments=ACI_ABLATION_PRESETS,
    *,
    csv_name: str = "aci_ablation_scores.csv",
    markdown_name: str = "aci_ablation_summary.md",
    title: str = "ACI attention/FFN ablation vs target",
) -> tuple[Path, Path]:
    rows = []
    aggregates = []
    for task in TASK_PRESETS:
        expected = _expected_benchmarks(task, scope)
        target_macro = None
        target_micro = None
        task_aggregates = []
        for variant, fusion_mode, beta in _task_variants(task, experiments):
            result_file = _latest_result(eval_root / task / variant / scope)
            payload = json.loads(result_file.read_text(encoding="utf-8"))
            results = payload["results"]
            scores = []
            weighted_score = 0.0
            weighted_count = 0
            for benchmark in expected:
                if benchmark not in results:
                    raise KeyError(f"Missing benchmark {task}:{variant}:{benchmark}")
                metric, score = _pick_metric(benchmark, results[benchmark])
                count = _sample_count(payload, benchmark, results[benchmark])
                scores.append(score)
                if count is not None:
                    weighted_score += score * count
                    weighted_count += count
                rows.append(
                    {
                        "domain": task,
                        "variant": variant,
                        "fusion_mode": fusion_mode,
                        "beta": "" if beta is None else beta,
                        "benchmark": benchmark,
                        "metric": metric,
                        "score": score,
                        "score_percent": 100.0 * score,
                        "sample_count": "" if count is None else count,
                        "result_file": str(result_file),
                    }
                )
            macro = sum(scores) / len(scores)
            micro = weighted_score / weighted_count if weighted_count else None
            if variant == "target":
                target_macro = macro
                target_micro = micro
            task_aggregates.append(
                {
                    "domain": task,
                    "variant": variant,
                    "fusion_mode": fusion_mode,
                    "beta": beta,
                    "score": macro,
                    "micro": micro if task == "medical" else None,
                }
            )
        if target_macro is None:
            raise RuntimeError(f"Target result missing for {task}")
        for aggregate in task_aggregates:
            aggregate["delta"] = aggregate["score"] - target_macro
            aggregate["micro_delta"] = (
                aggregate["micro"] - target_micro
                if aggregate["micro"] is not None and target_micro is not None
                else None
            )
            aggregates.append(aggregate)

    eval_root.mkdir(parents=True, exist_ok=True)
    csv_path = eval_root / csv_name
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# {title}",
        "",
        "All deltas are computed against the target from the same evaluation batch. ",
        "Thai uses XQuAD F1 in its unweighted macro. Medical also reports question-weighted micro accuracy.",
        "",
        "| Domain | Variant | β | Macro / accuracy | Δ vs target | Medical micro | Δ micro |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in aggregates:
        beta = "—" if item["beta"] is None else f"{item['beta']:g}"
        micro = "—" if item["micro"] is None else f"{100.0 * item['micro']:.2f}"
        micro_delta = (
            "—"
            if item["micro_delta"] is None
            else f"{100.0 * item['micro_delta']:+.2f}"
        )
        lines.append(
            f"| {item['domain']} | {item['variant']} | {beta} | "
            f"{100.0 * item['score']:.2f} | {100.0 * item['delta']:+.2f} | "
            f"{micro} | {micro_delta} |"
        )
    lines.append("")
    markdown_path = eval_root / markdown_name
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(markdown_path.read_text(encoding="utf-8"))
    return csv_path, markdown_path


def _evaluation_args(args: argparse.Namespace, task: str, experiment_matrix) -> list[str]:
    task_experiments = [item for item in experiment_matrix if item.task == task]
    variants = ["target", *(item.variant for item in task_experiments)]
    child = [
        "--tasks",
        task,
        "--variants",
        ",".join(variants),
        "--gpus",
        ",".join(args.gpus),
        "--models-root",
        str(args.models_root),
        "--results-root",
        str(args.results_root),
        "--eval-root",
        str(args.eval_root),
        "--scope",
        args.scope,
        "--lm-eval-dtype",
        args.lm_eval_dtype,
        "--lm-eval-batch-size",
        str(args.lm_eval_batch_size),
        "--malay-repo",
        str(args.malay_repo),
    ]
    if args.lm_eval_repo is not None:
        child.extend(("--lm-eval-repo", str(args.lm_eval_repo)))
    if args.malay_token:
        child.extend(("--malay-token", args.malay_token))
    for experiment in task_experiments:
        model = (
            args.results_root
            / aci_run_name(task, experiment.beta, experiment.fusion_mode)
            / "fused_model"
        ).resolve()
        child.extend(("--external-model", f"{task}:{experiment.variant}={model}"))
    return child


def evaluate_experiments(
    args: argparse.Namespace,
    experiments,
    *,
    method: str,
    manifest_name: str,
    csv_name: str,
    markdown_name: str,
    title: str,
) -> int:
    if not args.gpus:
        raise SystemExit("Provide at least one GPU")
    manifest = {
        "method": method,
        "comparison": "target_only",
        "thai_xquad_metric": "f1",
        "experiments": [
            {
                "task": experiment.task,
                "variant": experiment.variant,
                "fusion_mode": experiment.fusion_mode,
                "beta": experiment.beta,
                "model": str(
                    (
                        args.results_root
                        / aci_run_name(
                            experiment.task,
                            experiment.beta,
                            experiment.fusion_mode,
                        )
                        / "fused_model"
                    ).resolve()
                ),
            }
            for experiment in experiments
        ],
    }
    args.eval_root.mkdir(parents=True, exist_ok=True)
    (args.eval_root / manifest_name).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not args.summarize_only:
        for task in TASK_PRESETS:
            return_code = evaluate_main(_evaluation_args(args, task, experiments))
            if return_code:
                return return_code
    try:
        summarize(
            args.eval_root,
            args.scope,
            experiments,
            csv_name=csv_name,
            markdown_name=markdown_name,
            title=title,
        )
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as error:
        print(f"Unable to summarize ablations: {error}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return evaluate_experiments(
        args,
        ACI_ABLATION_PRESETS,
        method="aci_module_ablation_evaluation",
        manifest_name="ablation_manifest.json",
        csv_name="aci_ablation_scores.csv",
        markdown_name="aci_ablation_summary.md",
        title="ACI attention/FFN ablation vs target",
    )


if __name__ == "__main__":
    raise SystemExit(main())
