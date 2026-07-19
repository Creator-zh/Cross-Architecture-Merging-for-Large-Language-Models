#!/usr/bin/env python
"""Evaluate T&M, DFOP, and unfused baselines with one common harness."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from core.dfop.task_presets import (  # noqa: E402
    TASK_PRESETS,
    dfop_fusion_run_name,
    dfop_sft_run_name,
    get_task_preset,
)
from core.dfop.config import ROUTE_GROUPINGS, ROUTE_SOLVERS  # noqa: E402


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class EvaluationJob:
    task: str
    variant: str
    model: Path
    command: list[str]
    cwd: Path
    output: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=_csv, default=["medical", "thai", "malay"])
    parser.add_argument(
        "--variants",
        type=_csv,
        default=["target", "source", "hot", "dfop_attn", "dfop_full"],
        help=(
            "Comma-separated variants. Built-ins: target, source, dfop, "
            "dfop_attn, dfop_full, dfop_sft. The original T&M checkpoint "
            "uses alias 'hot'; known three-task output paths are automatic "
            "and --external-model can override them."
        ),
    )
    parser.add_argument("--gpus", type=_csv, default=["0", "1", "2", "3", "4", "5", "6"])
    parser.add_argument("--models-root", type=Path, default=REPOSITORY_ROOT / "models")
    parser.add_argument(
        "--results-root", type=Path, default=REPOSITORY_ROOT / "transport_results" / "dfop"
    )
    parser.add_argument(
        "--sft-results-root", type=Path, default=REPOSITORY_ROOT / "sft_results" / "dfop"
    )
    parser.add_argument("--eval-root", type=Path, default=REPOSITORY_ROOT / "evaluation_results" / "dfop")
    parser.add_argument("--mode", choices=("full", "attn"), default="full")
    parser.add_argument("--track", choices=("universal", "matched"), default="universal")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--top-source-layers", type=int, default=2)
    parser.add_argument(
        "--route-solver", choices=ROUTE_SOLVERS, default="row_softmax_topk"
    )
    parser.add_argument(
        "--route-grouping", choices=ROUTE_GROUPINGS, default="independent"
    )
    parser.add_argument("--sft-train-mode", choices=("full", "lora"), default="full")
    parser.add_argument("--sft-profile", choices=("declared", "legacy"), default="declared")
    parser.add_argument("--scope", choices=("primary", "extended", "all"), default="primary")
    parser.add_argument("--lm-eval-repo", type=Path, default=None)
    parser.add_argument("--lm-eval-dtype", default="float")
    parser.add_argument("--lm-eval-batch-size", default="8")
    parser.add_argument(
        "--malay-repo",
        type=Path,
        default=REPOSITORY_ROOT / "evaluation" / "malay" / "MalayMMLU",
    )
    parser.add_argument(
        "--hot-workspace-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="Workspace used by the original T&M launcher (Malay output lives here)",
    )
    parser.add_argument("--malay-token", default=os.environ.get("MALAY_HF_TOKEN", ""))
    parser.add_argument(
        "--external-model",
        action="append",
        default=[],
        metavar="TASK:VARIANT=PATH",
        help="Add T&M or another model variant, e.g. medical:hot=/path/to/model",
    )
    return parser


def _external_models(values: list[str]) -> dict[tuple[str, str], Path]:
    parsed = {}
    for value in values:
        try:
            left, path_text = value.split("=", 1)
            task, variant = left.split(":", 1)
        except ValueError as error:
            raise ValueError(f"Invalid --external-model: {value!r}") from error
        if task not in TASK_PRESETS or not variant:
            raise ValueError(f"Invalid --external-model: {value!r}")
        parsed[(task, variant)] = Path(path_text).expanduser().resolve()
    return parsed


def _dfop_model_path(
    args: argparse.Namespace,
    task: str,
    mode: str | None = None,
) -> Path:
    resolved_mode = mode or args.mode
    run_name = dfop_fusion_run_name(
        task,
        resolved_mode,
        args.track,
        args.rank,
        args.top_source_layers,
        route_solver=args.route_solver,
        route_grouping=args.route_grouping,
    )
    return (args.results_root / run_name / "fused_model").resolve()


def _model_for_variant(
    args: argparse.Namespace,
    task: str,
    variant: str,
    external: dict[tuple[str, str], Path],
) -> Path:
    preset = get_task_preset(task)
    if (task, variant) in external:
        return external[(task, variant)]
    if variant == "target":
        return (args.models_root / preset.target_local_dir).resolve()
    if variant == "source":
        return (args.models_root / preset.source_local_dir).resolve()
    if variant == "hot":
        if task == "medical":
            return (args.models_root / "medllama_fused_alpha01_fortrain_1b").resolve()
        if task == "thai":
            return (
                args.models_root / "llamathai_fused_alpha01_fortrain_1b_thai_instruction_sft"
            ).resolve()
        if task == "malay":
            return (
                args.hot_workspace_root / "maly_llama_fused_alpha01_fortrain_1b_select"
            ).resolve()
    if variant == "dfop":
        return _dfop_model_path(args, task)
    if variant == "dfop_attn":
        return _dfop_model_path(args, task, mode="attn")
    if variant == "dfop_full":
        return _dfop_model_path(args, task, mode="full")
    if variant == "dfop_sft":
        return (
            args.sft_results_root
            / dfop_sft_run_name(
                task,
                args.mode,
                args.track,
                args.rank,
                args.top_source_layers,
                route_solver=args.route_solver,
                route_grouping=args.route_grouping,
            )
            / f"{args.sft_train_mode}_{args.sft_profile}"
        ).resolve()
    try:
        return external[(task, variant)]
    except KeyError as error:
        raise ValueError(
            f"No model path for {task}:{variant}; provide --external-model {task}:{variant}=PATH"
        ) from error


def _canonical_class(value: str) -> str:
    """Normalize MalayMMLU labels: numeric index (0->A) or letter -> uppercase letter."""
    value = value.strip()
    if not value:
        return value
    try:
        idx = int(float(value))
        if 0 <= idx < 26:
            return chr(65 + idx)
    except ValueError:
        pass
    return value.upper()[:1]


def _postprocess_malay(job: EvaluationJob) -> None:
    """Turn MalayMMLU prediction CSV into a small lm-eval-like metrics file."""
    candidates = sorted(job.output.glob("MalayMMLU_result_*_True_0shot.csv"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one MalayMMLU prediction CSV in {job.output}, found {len(candidates)}"
        )
    correct = 0
    total = 0
    with candidates[0].open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"golds", "preds"}.issubset(reader.fieldnames):
            raise RuntimeError(f"Missing golds/preds columns in {candidates[0]}")
        for row in reader:
            total += 1
            correct += _canonical_class(row["golds"]) == _canonical_class(row["preds"])
    if total == 0:
        raise RuntimeError(f"Empty MalayMMLU prediction CSV: {candidates[0]}")
    payload = {
        "model": str(job.model),
        "prediction_file": str(candidates[0]),
        "results": {
            "MalayMMLU": {
                "acc,none": correct / total,
                "correct": correct,
                "total": total,
            }
        },
    }
    (job.output / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _tasks_for_scope(args: argparse.Namespace, task: str) -> tuple[str, ...]:
    preset = get_task_preset(task)
    if args.scope == "primary":
        return preset.primary_eval_tasks
    if args.scope == "extended":
        return preset.extended_eval_tasks
    return preset.primary_eval_tasks + preset.extended_eval_tasks


def _build_job(
    args: argparse.Namespace,
    task: str,
    variant: str,
    model: Path,
) -> EvaluationJob:
    preset = get_task_preset(task)
    output = (args.eval_root / task / variant / args.scope).resolve()
    if preset.eval_kind == "lm_eval":
        eval_tasks = _tasks_for_scope(args, task)
        if not eval_tasks:
            raise ValueError(f"No {args.scope} evaluation tasks configured for {task}")
        command = [
            sys.executable,
            "-m",
            "lm_eval",
            "--model",
            "hf",
            "--model_args",
            f"pretrained={model},dtype={args.lm_eval_dtype},trust_remote_code=True",
            "--tasks",
            ",".join(eval_tasks),
            "--device",
            "cuda:0",
            "--batch_size",
            str(args.lm_eval_batch_size),
            "--output_path",
            str(output),
        ]
        cwd = REPOSITORY_ROOT
    else:
        command = [
            sys.executable,
            "src/evaluate.py",
            "--by_letter",
            "--shot",
            "0",
            "--task",
            "MalayMMLU",
            "--base_model",
            str(model),
            "--output_folder",
            str(output),
        ]
        if args.malay_token:
            command.extend(("--token", args.malay_token))
        cwd = args.malay_repo.resolve()
    return EvaluationJob(task, variant, model, command, cwd, output)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        unknown_tasks = set(args.tasks) - set(TASK_PRESETS)
        if unknown_tasks:
            raise ValueError(f"Unknown tasks: {sorted(unknown_tasks)}")
        if not args.gpus:
            raise ValueError("At least one GPU is required")
        external = _external_models(args.external_model)
        jobs = deque()
        for task in args.tasks:
            for variant in args.variants:
                model = _model_for_variant(args, task, variant, external)
                if not model.is_dir():
                    raise FileNotFoundError(f"Model directory does not exist: {model}")
                jobs.append(_build_job(args, task, variant, model))
        if any(job.task == "malay" for job in jobs):
            if not args.malay_repo.is_dir():
                raise FileNotFoundError(f"MalayMMLU repository does not exist: {args.malay_repo}")
            malay_data = args.malay_repo / "data" / "MalayMMLU_0shot.json"
            if not malay_data.is_file():
                raise FileNotFoundError(
                    f"MalayMMLU data is missing: {malay_data}. "
                    "Use a complete MalayMMLU checkout via --malay-repo."
                )
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error

    args.eval_root.mkdir(parents=True, exist_ok=True)
    log_root = args.eval_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    manifest = [
        {
            "task": job.task,
            "variant": job.variant,
            "model": str(job.model),
            "output": str(job.output),
            "command": job.command,
        }
        for job in jobs
    ]
    (args.eval_root / "evaluation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    available_gpus = deque(args.gpus)
    running = []
    failures = []
    try:
        while jobs or running:
            while jobs and available_gpus:
                job = jobs.popleft()
                gpu = available_gpus.popleft()
                job.output.mkdir(parents=True, exist_ok=True)
                log_path = log_root / f"{job.task}_{job.variant}_{args.scope}.log"
                log_file = log_path.open("w", encoding="utf-8")
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                if args.lm_eval_repo is not None:
                    current = environment.get("PYTHONPATH", "")
                    environment["PYTHONPATH"] = str(args.lm_eval_repo.resolve()) + (
                        os.pathsep + current if current else ""
                    )
                print(f"[launch] {job.task}:{job.variant} gpu={gpu} log={log_path}", flush=True)
                process = subprocess.Popen(
                    job.command,
                    cwd=job.cwd,
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
                running.append((job, gpu, process, log_file, log_path))

            next_running = []
            for job, gpu, process, log_file, log_path in running:
                return_code = process.poll()
                if return_code is None:
                    next_running.append((job, gpu, process, log_file, log_path))
                    continue
                log_file.close()
                available_gpus.append(gpu)
                print(f"[complete] {job.task}:{job.variant} exit={return_code}", flush=True)
                if return_code == 0 and get_task_preset(job.task).eval_kind == "malay_mmlu":
                    try:
                        _postprocess_malay(job)
                    except RuntimeError as error:
                        print(f"[postprocess-failed] {job.task}:{job.variant}: {error}", flush=True)
                        failures.append((job.task, job.variant, 2, str(log_path)))
                elif return_code != 0:
                    failures.append((job.task, job.variant, return_code, str(log_path)))
            running = next_running
            if jobs or running:
                time.sleep(5)
    except KeyboardInterrupt:
        for _, _, process, log_file, _ in running:
            process.terminate()
            log_file.close()
        return 130

    if failures:
        print(json.dumps({"failures": failures}, ensure_ascii=False, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
