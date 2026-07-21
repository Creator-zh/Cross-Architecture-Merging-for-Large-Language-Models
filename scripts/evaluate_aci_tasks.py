#!/usr/bin/env python
"""Evaluate unfused targets, ACI merges, and optional external baselines."""

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

from core.aci.presets import (  # noqa: E402
    TASK_PRESETS,
    aci_run_name,
    aci_sft_run_name,
    get_task_preset,
)


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _beta_overrides(values: list[str]) -> dict[str, float]:
    result = {}
    for value in values:
        try:
            task, number = value.split("=", 1)
            beta = float(number)
        except ValueError as error:
            raise ValueError(f"Invalid beta override: {value!r}") from error
        if task not in TASK_PRESETS or not 0 <= beta <= 1:
            raise ValueError(f"Invalid beta override: {value!r}")
        result[task] = beta
    return result


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
    parser.add_argument("--variants", type=_csv, default=["target", "aci"])
    parser.add_argument("--gpus", type=_csv, default=["0", "1"])
    parser.add_argument("--models-root", type=Path, default=REPOSITORY_ROOT / "models")
    parser.add_argument(
        "--results-root", type=Path, default=REPOSITORY_ROOT / "merge_results" / "aci"
    )
    parser.add_argument(
        "--eval-root", type=Path, default=REPOSITORY_ROOT / "evaluation_results" / "aci"
    )
    parser.add_argument(
        "--sft-results-root", type=Path, default=REPOSITORY_ROOT / "sft_results" / "aci"
    )
    parser.add_argument("--sft-train-mode", choices=("full", "lora"), default="full")
    parser.add_argument("--beta", action="append", default=[], metavar="TASK=VALUE")
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
        "--external-model",
        action="append",
        default=[],
        metavar="TASK:VARIANT=PATH",
    )
    return parser


def _external_models(values: list[str]) -> dict[tuple[str, str], Path]:
    parsed = {}
    for value in values:
        try:
            left, path = value.split("=", 1)
            task, variant = left.split(":", 1)
        except ValueError as error:
            raise ValueError(f"Invalid --external-model: {value!r}") from error
        if task not in TASK_PRESETS or not variant:
            raise ValueError(f"Invalid --external-model: {value!r}")
        parsed[(task, variant)] = Path(path).expanduser().resolve()
    return parsed


def _model_for_variant(
    args: argparse.Namespace,
    task: str,
    variant: str,
    betas: dict[str, float],
    external: dict[tuple[str, str], Path],
) -> Path:
    if (task, variant) in external:
        return external[(task, variant)]
    preset = get_task_preset(task)
    if variant == "target":
        return (args.models_root / preset.target_local_dir).resolve()
    if variant == "source":
        return (args.models_root / preset.source_local_dir).resolve()
    if variant == "reference":
        return (args.models_root / preset.reference_local_dir).resolve()
    if variant == "aci":
        beta = betas.get(task, preset.beta)
        return (args.results_root / aci_run_name(task, beta) / "fused_model").resolve()
    if variant in ("aci_attention", "aci_ffn"):
        beta = betas.get(task, preset.beta)
        fusion_mode = variant.removeprefix("aci_")
        return (
            args.results_root
            / aci_run_name(task, beta, fusion_mode)
            / "fused_model"
        ).resolve()
    if variant == "aci_sft":
        beta = betas.get(task, preset.beta)
        return (
            args.sft_results_root
            / aci_sft_run_name(task, beta)
            / args.sft_train_mode
        ).resolve()
    raise ValueError(
        f"No built-in path for {task}:{variant}; use --external-model {task}:{variant}=PATH"
    )


def _canonical_class(value: str) -> str:
    value = value.strip()
    if not value:
        return value
    try:
        index = int(float(value))
        if 0 <= index < 26:
            return chr(65 + index)
    except ValueError:
        pass
    return value.upper()[:1]


def _postprocess_malay(job: EvaluationJob) -> None:
    candidates = sorted(job.output.glob("MalayMMLU_result_*_True_0shot.csv"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one MalayMMLU CSV in {job.output}, found {len(candidates)}"
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
        raise RuntimeError(f"Empty MalayMMLU CSV: {candidates[0]}")
    (job.output / "metrics.json").write_text(
        json.dumps(
            {
                "model": str(job.model),
                "prediction_file": str(candidates[0]),
                "results": {
                    "MalayMMLU": {
                        "acc,none": correct / total,
                        "correct": correct,
                        "total": total,
                    }
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _tasks_for_scope(args: argparse.Namespace, task: str) -> tuple[str, ...]:
    preset = get_task_preset(task)
    if args.scope == "primary":
        return preset.primary_eval_tasks
    if args.scope == "extended":
        return preset.extended_eval_tasks
    return preset.primary_eval_tasks + preset.extended_eval_tasks


def _build_job(args: argparse.Namespace, task: str, variant: str, model: Path) -> EvaluationJob:
    preset = get_task_preset(task)
    output = (args.eval_root / task / variant / args.scope).resolve()
    if preset.eval_kind == "lm_eval":
        tasks = _tasks_for_scope(args, task)
        if not tasks:
            raise ValueError(f"No {args.scope} tasks configured for {task}")
        command = [
            sys.executable,
            "-m",
            "lm_eval",
            "--model",
            "hf",
            "--model_args",
            f"pretrained={model},dtype={args.lm_eval_dtype},trust_remote_code=True",
            "--tasks",
            ",".join(tasks),
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
        unknown = set(args.tasks) - set(TASK_PRESETS)
        if unknown:
            raise ValueError(f"Unknown tasks: {sorted(unknown)}")
        if not args.gpus:
            raise ValueError("At least one GPU is required")
        betas = _beta_overrides(args.beta)
        external = _external_models(args.external_model)
        jobs = deque()
        for task in args.tasks:
            for variant in args.variants:
                model = _model_for_variant(args, task, variant, betas, external)
                if not model.is_dir():
                    raise FileNotFoundError(f"Model directory does not exist: {model}")
                jobs.append(_build_job(args, task, variant, model))
        if any(job.task == "malay" for job in jobs):
            data = args.malay_repo / "data" / "MalayMMLU_0shot.json"
            if not data.is_file():
                raise FileNotFoundError(f"MalayMMLU data is missing: {data}")
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error

    args.eval_root.mkdir(parents=True, exist_ok=True)
    log_root = args.eval_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    (args.eval_root / "evaluation_manifest.json").write_text(
        json.dumps(
            [
                {
                    "task": job.task,
                    "variant": job.variant,
                    "model": str(job.model),
                    "output": str(job.output),
                    "command": job.command,
                }
                for job in jobs
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    available = deque(args.gpus)
    running = []
    failures = []
    try:
        while jobs or running:
            while jobs and available:
                job = jobs.popleft()
                gpu = available.popleft()
                job.output.mkdir(parents=True, exist_ok=True)
                log_path = log_root / f"{job.task}_{job.variant}_{args.scope}.log"
                handle = log_path.open("w", encoding="utf-8")
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                if args.lm_eval_repo is not None:
                    current = environment.get("PYTHONPATH", "")
                    environment["PYTHONPATH"] = str(args.lm_eval_repo.resolve()) + (
                        os.pathsep + current if current else ""
                    )
                process = subprocess.Popen(
                    job.command,
                    cwd=job.cwd,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                running.append((job, gpu, process, handle, log_path))
                print(f"[launch] {job.task}:{job.variant} gpu={gpu} log={log_path}", flush=True)

            remaining = []
            for job, gpu, process, handle, log_path in running:
                return_code = process.poll()
                if return_code is None:
                    remaining.append((job, gpu, process, handle, log_path))
                    continue
                handle.close()
                available.append(gpu)
                print(f"[complete] {job.task}:{job.variant} exit={return_code}", flush=True)
                if return_code == 0 and get_task_preset(job.task).eval_kind == "malay_mmlu":
                    try:
                        _postprocess_malay(job)
                    except RuntimeError as error:
                        print(f"[postprocess-failed] {error}", flush=True)
                        failures.append((job.task, job.variant, 2, str(log_path)))
                elif return_code:
                    failures.append((job.task, job.variant, return_code, str(log_path)))
            running = remaining
            if jobs or running:
                time.sleep(5)
    except KeyboardInterrupt:
        for _, _, process, handle, _ in running:
            process.terminate()
            handle.close()
        return 130
    if failures:
        print(json.dumps({"failures": failures}, ensure_ascii=False, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
