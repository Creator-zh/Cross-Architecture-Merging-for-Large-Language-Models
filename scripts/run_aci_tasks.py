#!/usr/bin/env python
"""Run Medical, Thai, and Malay ACI jobs across a bounded GPU pool."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from core.aci.presets import TASK_PRESETS, aci_run_name, get_task_preset  # noqa: E402


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _beta_overrides(values: list[str]) -> dict[str, float]:
    result = {}
    for value in values:
        try:
            task, beta_text = value.split("=", 1)
            beta = float(beta_text)
        except ValueError as error:
            raise ValueError(f"Invalid --beta value {value!r}; expected TASK=VALUE") from error
        if task not in TASK_PRESETS or not 0.0 <= beta <= 1.0:
            raise ValueError(f"Invalid --beta value {value!r}")
        result[task] = beta
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=_csv, default=["medical", "thai", "malay"])
    parser.add_argument("--gpus", type=_csv, default=["0", "1"])
    parser.add_argument("--models-root", type=Path, default=REPOSITORY_ROOT / "models")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPOSITORY_ROOT / "merge_results" / "aci",
    )
    parser.add_argument("--hf-direct", action="store_true")
    parser.add_argument(
        "--beta",
        action="append",
        default=[],
        metavar="TASK=VALUE",
        help="Override one preset; repeat for multiple tasks",
    )
    parser.add_argument("--model-dtype", default="bfloat16")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Append one raw argument to every child command",
    )
    return parser


def _model_paths(args: argparse.Namespace, task: str) -> tuple[str, str, str]:
    preset = get_task_preset(task)
    if args.hf_direct:
        return preset.target_hf_id, preset.reference_hf_id, preset.source_hf_id
    paths = (
        (args.models_root / preset.target_local_dir).resolve(),
        (args.models_root / preset.reference_local_dir).resolve(),
        (args.models_root / preset.source_local_dir).resolve(),
    )
    missing = [str(path) for path in paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Missing local models for {task}: {missing}. Run scripts/download_models.py."
        )
    return tuple(str(path) for path in paths)  # type: ignore[return-value]


def _command(
    args: argparse.Namespace,
    task: str,
    overrides: dict[str, float],
) -> tuple[list[str], Path, float]:
    preset = get_task_preset(task)
    target, reference, source = _model_paths(args, task)
    beta = overrides.get(task, preset.beta)
    output = (args.results_root / aci_run_name(task, beta)).resolve()
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "run_aci_merge.py"),
        "--target-model",
        target,
        "--reference-model",
        reference,
        "--source-model",
        source,
        "--output-dir",
        str(output),
        "--beta",
        str(beta),
        "--device",
        "cuda:0",
        "--model-dtype",
        args.model_dtype,
    ]
    if not args.hf_direct:
        command.append("--local-files-only")
    if args.dry_run:
        command.append("--dry-run")
    if args.overwrite_output:
        command.append("--overwrite-output")
    command.extend(args.extra_arg)
    return command, output, beta


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        unknown = set(args.tasks) - set(TASK_PRESETS)
        if unknown:
            raise ValueError(f"Unknown tasks: {sorted(unknown)}")
        if not args.gpus:
            raise ValueError("Provide at least one GPU")
        overrides = _beta_overrides(args.beta)
        jobs = [(_command(args, task, overrides), task) for task in args.tasks]
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error

    args.results_root.mkdir(parents=True, exist_ok=True)
    log_root = args.results_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": "aci",
        "tasks": args.tasks,
        "gpus": args.gpus,
        "betas": {task: command_info[2] for command_info, task in jobs},
        "commands": [command_info[0] for command_info, _ in jobs],
    }
    (args.results_root / "launch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    pending = deque(jobs)
    available = deque(args.gpus)
    running = []
    logs = []
    failures = []
    try:
        while pending or running:
            while pending and available:
                (command, output, beta), task = pending.popleft()
                gpu = available.popleft()
                log_path = log_root / f"{task}_beta{beta:g}.log"
                handle = log_path.open("a" if args.overwrite_output else "w", encoding="utf-8")
                logs.append(handle)
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                print(f"[launch] task={task} beta={beta:g} gpu={gpu} log={log_path}", flush=True)
                process = subprocess.Popen(
                    command,
                    cwd=REPOSITORY_ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                running.append((task, gpu, process, log_path, output))

            remaining = []
            for task, gpu, process, log_path, output in running:
                return_code = process.poll()
                if return_code is None:
                    remaining.append((task, gpu, process, log_path, output))
                else:
                    available.append(gpu)
                    print(f"[complete] task={task} exit={return_code} output={output}", flush=True)
                    if return_code:
                        failures.append((task, return_code, str(log_path)))
            running = remaining
            if running and (pending or running):
                time.sleep(5)
    except KeyboardInterrupt:
        for _, _, process, _, _ in running:
            process.terminate()
        return 130
    finally:
        for handle in logs:
            handle.close()
    if failures:
        print(json.dumps({"failures": failures}, ensure_ascii=False, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
