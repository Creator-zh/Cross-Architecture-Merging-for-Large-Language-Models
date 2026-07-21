#!/usr/bin/env python
"""Run the fixed eight-job ACI attention/FFN ablation matrix."""

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

from core.aci.presets import (  # noqa: E402
    ACI_ABLATION_PRESETS,
    ACIAblationPreset,
    aci_run_name,
    get_task_preset,
)


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
    parser.add_argument("--hf-direct", action="store_true")
    parser.add_argument("--model-dtype", default="bfloat16")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Append one raw argument to every run_aci_merge.py command",
    )
    return parser


def _model_paths(
    args: argparse.Namespace,
    experiment: ACIAblationPreset,
) -> tuple[str, str, str]:
    preset = get_task_preset(experiment.task)
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
            f"Missing local models for {experiment.task}: {missing}. "
            "Run scripts/download_models.py."
        )
    return tuple(str(path) for path in paths)  # type: ignore[return-value]


def _command(
    args: argparse.Namespace,
    experiment: ACIAblationPreset,
) -> tuple[list[str], Path]:
    target, reference, source = _model_paths(args, experiment)
    output = (
        args.results_root
        / aci_run_name(experiment.task, experiment.beta, experiment.fusion_mode)
    ).resolve()
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
        str(experiment.beta),
        "--fusion-mode",
        experiment.fusion_mode,
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
    return command, output


def run_experiments(
    args: argparse.Namespace,
    experiments: tuple[ACIAblationPreset, ...],
    *,
    method: str,
    manifest_name: str,
    log_subdirectory: str,
) -> int:
    try:
        if not args.gpus:
            raise ValueError("Provide at least one GPU")
        jobs = [
            (experiment, *_command(args, experiment))
            for experiment in experiments
        ]
        outputs = [str(output) for _, _, output in jobs]
        if len(outputs) != len(set(outputs)):
            raise ValueError("Ablation matrix contains duplicate output directories")
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error

    args.results_root.mkdir(parents=True, exist_ok=True)
    log_root = args.results_root / "logs" / log_subdirectory
    log_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": method,
        "gpus": args.gpus,
        "experiments": [
            {
                "task": experiment.task,
                "fusion_mode": experiment.fusion_mode,
                "beta": experiment.beta,
                "variant": experiment.variant,
                "output": str(output),
                "command": command,
            }
            for experiment, command, output in jobs
        ],
    }
    (args.results_root / manifest_name).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    pending = deque(jobs)
    available = deque(args.gpus)
    running = []
    handles = []
    failures = []
    try:
        while pending or running:
            while pending and available:
                experiment, command, output = pending.popleft()
                gpu = available.popleft()
                run_name = aci_run_name(
                    experiment.task,
                    experiment.beta,
                    experiment.fusion_mode,
                )
                log_path = log_root / f"{run_name}.log"
                handle = log_path.open(
                    "a" if args.overwrite_output else "w",
                    encoding="utf-8",
                )
                handles.append(handle)
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                print(
                    f"[launch] task={experiment.task} mode={experiment.fusion_mode} "
                    f"beta={experiment.beta:g} gpu={gpu} log={log_path}",
                    flush=True,
                )
                process = subprocess.Popen(
                    command,
                    cwd=REPOSITORY_ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                running.append((experiment, gpu, process, log_path, output))

            remaining = []
            for experiment, gpu, process, log_path, output in running:
                return_code = process.poll()
                if return_code is None:
                    remaining.append((experiment, gpu, process, log_path, output))
                    continue
                available.append(gpu)
                print(
                    f"[complete] task={experiment.task} mode={experiment.fusion_mode} "
                    f"beta={experiment.beta:g} exit={return_code} output={output}",
                    flush=True,
                )
                if return_code:
                    failures.append(
                        {
                            "task": experiment.task,
                            "fusion_mode": experiment.fusion_mode,
                            "beta": experiment.beta,
                            "return_code": return_code,
                            "log": str(log_path),
                        }
                    )
            running = remaining
            if pending or running:
                time.sleep(5)
    except KeyboardInterrupt:
        for _, _, process, _, _ in running:
            process.terminate()
        return 130
    finally:
        for handle in handles:
            handle.close()
    if failures:
        print(json.dumps({"failures": failures}, ensure_ascii=False, indent=2))
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_experiments(
        args,
        ACI_ABLATION_PRESETS,
        method="aci_module_ablation",
        manifest_name="ablation_launch_manifest.json",
        log_subdirectory="ablations",
    )


if __name__ == "__main__":
    raise SystemExit(main())
