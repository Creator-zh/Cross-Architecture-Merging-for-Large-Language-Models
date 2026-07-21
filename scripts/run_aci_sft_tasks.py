#!/usr/bin/env python
"""Run optional post-merge SFT for the three ACI task presets."""

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
    TASK_PRESETS,
    aci_run_name,
    aci_sft_run_name,
    get_task_preset,
)


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _overrides(values: list[str]) -> dict[str, float]:
    result = {}
    for value in values:
        try:
            task, text = value.split("=", 1)
            beta = float(text)
        except ValueError as error:
            raise ValueError(f"Invalid --beta value: {value!r}") from error
        if task not in TASK_PRESETS or not 0 <= beta <= 1:
            raise ValueError(f"Invalid --beta value: {value!r}")
        result[task] = beta
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=_csv, default=["medical", "thai", "malay"])
    parser.add_argument("--gpus", type=_csv, default=["0", "1"])
    parser.add_argument(
        "--fusion-results-root",
        type=Path,
        default=REPOSITORY_ROOT / "merge_results" / "aci",
    )
    parser.add_argument(
        "--sft-results-root",
        type=Path,
        default=REPOSITORY_ROOT / "sft_results" / "aci",
    )
    parser.add_argument("--beta", action="append", default=[], metavar="TASK=VALUE")
    parser.add_argument("--train-mode", choices=("full", "lora"), default="full")
    parser.add_argument("--thai-dataset-path", type=Path, default=None)
    parser.add_argument("--malay-dataset-path", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser


def _command(
    args: argparse.Namespace,
    task: str,
    betas: dict[str, float],
) -> tuple[list[str], Path, float]:
    preset = get_task_preset(task)
    beta = betas.get(task, preset.beta)
    model = (args.fusion_results_root / aci_run_name(task, beta) / "fused_model").resolve()
    if not model.is_dir():
        raise FileNotFoundError(f"Missing ACI model for {task}: {model}")
    output = (
        args.sft_results_root / aci_sft_run_name(task, beta) / args.train_mode
    ).resolve()
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "train_hot_residual_sft.py"),
        "--training_scenario",
        "no_hot",
        "--freeze_strategy",
        "none",
        "--model_type",
        "llama",
        "--dataset_type",
        preset.sft_dataset_type,
        "--model_dir",
        str(model),
        "--output_dir",
        str(output),
        "--per_device_train_batch_size",
        "1",
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--learning_rate",
        str(preset.sft_learning_rate),
        "--num_train_epochs",
        "1",
        "--block_size",
        str(args.block_size),
        "--max_samples_per_subset",
        str(preset.sft_samples),
        "--seed",
        "42",
        "--fp16",
        "--honor_precision_flags",
    ]
    if args.train_mode == "lora":
        command.extend(
            (
                "--use_lora",
                "--lora_target_modules",
                "q_proj,k_proj,v_proj,o_proj",
                "--lora_r",
                str(args.lora_r),
                "--lora_alpha",
                str(args.lora_alpha),
                "--lora_dropout",
                str(args.lora_dropout),
            )
        )
    dataset_path = args.thai_dataset_path if task == "thai" else args.malay_dataset_path
    if dataset_path is not None:
        resolved = dataset_path.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"Dataset path does not exist: {resolved}")
        command.extend(("--local_dataset_path", str(resolved)))
    return command, output, beta


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        unknown = set(args.tasks) - set(TASK_PRESETS)
        if unknown:
            raise ValueError(f"Unknown tasks: {sorted(unknown)}")
        if not args.gpus:
            raise ValueError("At least one GPU is required")
        betas = _overrides(args.beta)
        jobs = deque([(_command(args, task, betas), task) for task in args.tasks])
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error

    args.sft_results_root.mkdir(parents=True, exist_ok=True)
    log_root = args.sft_results_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    (args.sft_results_root / "sft_launch_manifest.json").write_text(
        json.dumps(
            {
                "tasks": args.tasks,
                "gpus": args.gpus,
                "train_mode": args.train_mode,
                "commands": [info[0] for info, _ in jobs],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    available = deque(args.gpus)
    running = []
    failures = []
    while jobs or running:
        while jobs and available:
            (command, output, beta), task = jobs.popleft()
            marker = output / "_ACI_SFT_DONE"
            if args.resume and marker.is_file():
                print(f"[skip] {task} completed={marker}", flush=True)
                continue
            output.mkdir(parents=True, exist_ok=True)
            gpu = available.popleft()
            log_path = log_root / f"{task}_beta{beta:g}_{args.train_mode}.log"
            handle = log_path.open("a" if args.resume else "w", encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(
                command,
                cwd=REPOSITORY_ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running.append((task, gpu, process, handle, log_path, marker))
            print(f"[launch] {task} gpu={gpu} log={log_path}", flush=True)

        remaining = []
        for task, gpu, process, handle, log_path, marker in running:
            return_code = process.poll()
            if return_code is None:
                remaining.append((task, gpu, process, handle, log_path, marker))
                continue
            handle.close()
            available.append(gpu)
            if return_code == 0:
                marker.write_text("complete\n", encoding="utf-8")
            else:
                failures.append((task, return_code, str(log_path)))
            print(f"[complete] {task} exit={return_code}", flush=True)
        running = remaining
        if jobs or running:
            time.sleep(5)
    if failures:
        print(json.dumps({"failures": failures}, ensure_ascii=False, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
