#!/usr/bin/env python
"""Run matched post-merge SFT for DFOP medical, Thai, and Malay models."""

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

from core.dfop.task_presets import TASK_PRESETS, get_task_preset  # noqa: E402


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=_csv, default=["medical", "thai", "malay"])
    parser.add_argument("--gpus", type=_csv, default=["0", "1", "2"])
    parser.add_argument(
        "--fusion-results-root",
        type=Path,
        default=REPOSITORY_ROOT / "transport_results" / "dfop",
    )
    parser.add_argument(
        "--sft-results-root",
        type=Path,
        default=REPOSITORY_ROOT / "sft_results" / "dfop",
    )
    parser.add_argument("--mode", choices=("full", "attn"), default="full")
    parser.add_argument("--track", choices=("universal", "matched"), default="universal")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--train-mode", choices=("full", "lora"), default="full")
    parser.add_argument(
        "--profile",
        choices=("declared", "legacy"),
        default="declared",
        help="Declared uses Thai=8000 and FP16; legacy reproduces Thai=2000 and FP32",
    )
    parser.add_argument("--thai-dataset-path", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser


def _fusion_model(args: argparse.Namespace, task: str) -> Path:
    preset = get_task_preset(task)
    beta = 0.05 if args.track == "universal" else preset.matched_beta
    name = f"{task}_{args.mode}_{args.track}_r{args.rank}_beta{beta:g}"
    return (args.fusion_results_root / name / "fused_model").resolve()


def _command(args: argparse.Namespace, task: str) -> tuple[list[str], Path]:
    preset = get_task_preset(task)
    model = _fusion_model(args, task)
    if not model.is_dir():
        raise FileNotFoundError(f"Missing fused model for {task}: {model}")
    samples = (
        preset.sft_samples_declared
        if args.profile == "declared"
        else preset.sft_samples_legacy
    )
    output = (
        args.sft_results_root
        / f"{task}_{args.mode}_{args.track}_r{args.rank}"
        / f"{args.train_mode}_{args.profile}"
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
        str(samples),
        "--seed",
        "42",
    ]
    if args.profile == "declared":
        command.extend(("--fp16", "--honor_precision_flags"))
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
    if task == "thai" and args.thai_dataset_path is not None:
        dataset_path = args.thai_dataset_path.expanduser().resolve()
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"Thai dataset path does not exist: {dataset_path}")
        command.extend(("--local_dataset_path", str(dataset_path)))
    return command, output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        unknown = set(args.tasks) - set(TASK_PRESETS)
        if unknown:
            raise ValueError(f"Unknown tasks: {sorted(unknown)}")
        if len(args.gpus) < len(args.tasks):
            raise ValueError("Provide at least one GPU per concurrently trained task")
        jobs = [(_command(args, task), task, args.gpus[index]) for index, task in enumerate(args.tasks)]
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error

    args.sft_results_root.mkdir(parents=True, exist_ok=True)
    log_root = args.sft_results_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "tasks": args.tasks,
        "gpus": args.gpus[: len(args.tasks)],
        "train_mode": args.train_mode,
        "profile": args.profile,
        "commands": [command for (command, _), _, _ in jobs],
    }
    (args.sft_results_root / "sft_launch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    running = []
    logs = []
    try:
        for (command, output), task, gpu in jobs:
            marker = output / "_DFOP_SFT_DONE"
            if args.resume and marker.is_file():
                print(f"[skip] task={task} completed={marker}", flush=True)
                continue
            output.mkdir(parents=True, exist_ok=True)
            log_path = log_root / f"{task}_{args.train_mode}_{args.profile}.log"
            log_file = log_path.open("a" if args.resume else "w", encoding="utf-8")
            logs.append(log_file)
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
            running.append((task, output, process, log_path))

        failures = []
        while running:
            remaining = []
            for task, output, process, log_path in running:
                return_code = process.poll()
                if return_code is None:
                    remaining.append((task, output, process, log_path))
                    continue
                print(f"[complete] task={task} exit={return_code}", flush=True)
                if return_code == 0:
                    (output / "_DFOP_SFT_DONE").write_text("complete\n", encoding="utf-8")
                else:
                    failures.append((task, return_code, str(log_path)))
            running = remaining
            if running:
                time.sleep(5)
        if failures:
            print(json.dumps({"failures": failures}, ensure_ascii=False, indent=2))
            return 1
        return 0
    except KeyboardInterrupt:
        for _, _, process, _ in running:
            process.terminate()
        return 130
    finally:
        for log_file in logs:
            log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
