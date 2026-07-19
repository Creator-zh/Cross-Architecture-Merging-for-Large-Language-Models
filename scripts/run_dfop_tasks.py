#!/usr/bin/env python
"""Launch medical, Thai, and Malay DFOP fusion jobs on separate GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from core.dfop.task_presets import (  # noqa: E402
    TASK_PRESETS,
    dfop_fusion_run_name,
    fusion_beta,
    get_task_preset,
)


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=_csv, default=["medical", "thai", "malay"])
    parser.add_argument("--gpus", type=_csv, default=["0", "1", "2"])
    parser.add_argument("--models-root", type=Path, default=REPOSITORY_ROOT / "models")
    parser.add_argument(
        "--results-root", type=Path, default=REPOSITORY_ROOT / "transport_results" / "dfop"
    )
    parser.add_argument("--hf-direct", action="store_true", help="Load HF ids instead of local model dirs")
    parser.add_argument("--mode", choices=("full", "attn"), default="full")
    parser.add_argument(
        "--track",
        choices=("universal", "matched"),
        default="universal",
        help="Universal beta=.05 or each paper task's original fusion alpha",
    )
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--top-source-layers", type=int, default=2)
    parser.add_argument("--trust-ratio", default="0.10")
    parser.add_argument("--model-dtype", default="bfloat16")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Append one raw argument to every fusion child; repeat as needed",
    )
    return parser


def _validate(args: argparse.Namespace) -> None:
    unknown = set(args.tasks) - set(TASK_PRESETS)
    if unknown:
        raise ValueError(f"Unknown tasks: {sorted(unknown)}")
    if len(args.gpus) < len(args.tasks):
        raise ValueError("Provide at least one GPU id per concurrently launched task")
    if args.rank <= 0:
        raise ValueError("--rank must be positive")
    if args.top_source_layers <= 0:
        raise ValueError("--top-source-layers must be positive")


def _model_paths(args: argparse.Namespace, task_name: str) -> tuple[str, str]:
    preset = get_task_preset(task_name)
    if args.hf_direct:
        return preset.target_hf_id, preset.source_hf_id
    target = (args.models_root / preset.target_local_dir).resolve()
    source = (args.models_root / preset.source_local_dir).resolve()
    if not target.is_dir() or not source.is_dir():
        raise FileNotFoundError(
            f"Missing local model for {task_name}: target={target}, source={source}. "
            "Run scripts/download_models.py or use --hf-direct."
        )
    return str(target), str(source)


def _command(args: argparse.Namespace, task_name: str) -> tuple[list[str], Path]:
    preset = get_task_preset(task_name)
    target, source = _model_paths(args, task_name)
    modules = "q,k,v,o" if args.mode == "attn" else "q,k,v,o,gate,up,down"
    beta = fusion_beta(args.track, task_name)
    run_name = dfop_fusion_run_name(
        task_name, args.mode, args.track, args.rank, args.top_source_layers, beta
    )
    output = (args.results_root / run_name).resolve()
    shared_cache = (
        args.results_root
        / "cache"
        / f"{task_name}_{args.mode}_r{args.rank}_top{args.top_source_layers}"
    ).resolve()
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "run_dfop_fusion.py"),
        "--target-model",
        target,
        "--source-model",
        source,
        "--output-dir",
        str(output),
        "--stage1-cache-dir",
        str(shared_cache / "stage1"),
        "--stage2-cache-dir",
        str(shared_cache / "stage2"),
        "--device",
        "cuda:0",
        "--model-dtype",
        args.model_dtype,
        "--modules",
        modules,
        "--rank",
        str(args.rank),
        "--top-source-layers",
        str(args.top_source_layers),
        "--beta",
        str(beta),
        "--trust-ratio",
        str(args.trust_ratio),
    ]
    if not args.hf_direct:
        command.extend(("--local-files-only", "--copy-target-tokenizer-files"))
    if args.resume:
        command.append("--overwrite-output")
    if args.dry_run:
        command.append("--dry-run")
    command.extend(args.extra_arg)
    return command, output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate(args)
        jobs = []
        for index, task in enumerate(args.tasks):
            command, _ = _command(args, task)
            jobs.append((command, task, args.gpus[index]))
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error

    args.results_root.mkdir(parents=True, exist_ok=True)
    log_root = args.results_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "tasks": args.tasks,
        "gpus": args.gpus[: len(args.tasks)],
        "mode": args.mode,
        "track": args.track,
        "rank": args.rank,
        "top_source_layers": args.top_source_layers,
        "commands": [command for command, _, _ in jobs],
    }
    (args.results_root / "launch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    processes = []
    opened_logs = []
    try:
        for command, task, gpu in jobs:
            log_path = (
                log_root
                / f"{task}_{args.mode}_{args.track}_top{args.top_source_layers}.log"
            )
            log_file = log_path.open("a" if args.resume else "w", encoding="utf-8")
            opened_logs.append(log_file)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print(f"[launch] task={task} physical_gpu={gpu} log={log_path}", flush=True)
            process = subprocess.Popen(
                command,
                cwd=REPOSITORY_ROOT,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            processes.append((task, process, log_path))

        failures = []
        while processes:
            remaining = []
            for task, process, log_path in processes:
                return_code = process.poll()
                if return_code is None:
                    remaining.append((task, process, log_path))
                else:
                    print(f"[complete] task={task} exit={return_code} log={log_path}", flush=True)
                    if return_code != 0:
                        failures.append((task, return_code, str(log_path)))
            processes = remaining
            if processes:
                time.sleep(5)
        if failures:
            print(json.dumps({"failures": failures}, ensure_ascii=False, indent=2))
            return 1
        return 0
    except KeyboardInterrupt:
        for _, process, _ in processes:
            process.terminate()
        return 130
    finally:
        for log_file in opened_logs:
            log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
