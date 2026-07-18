#!/usr/bin/env python
"""Launch original Transport-and-Merge merge-only jobs for three paper tasks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


HOT_FUSED_OUTPUTS = {
    "medical": lambda workspace, models: models / "medllama_fused_alpha01_fortrain_1b",
    "thai": lambda workspace, models: models
    / "llamathai_fused_alpha01_fortrain_1b_thai_instruction_sft",
    "malay": lambda workspace, models: workspace / "maly_llama_fused_alpha01_fortrain_1b_select",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=_csv, default=["medical", "thai", "malay"])
    parser.add_argument("--gpus", type=_csv, default=["0", "1", "2"])
    parser.add_argument("--models-root", type=Path, default=REPOSITORY_ROOT / "models")
    parser.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--hot-results-root",
        type=Path,
        default=REPOSITORY_ROOT / "transport_results",
    )
    parser.add_argument("--fineweb-thai-cache", type=Path, default=None)
    parser.add_argument("--bash", default="bash")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    unknown = set(args.tasks) - set(HOT_FUSED_OUTPUTS)
    if unknown:
        raise SystemExit(f"Unknown tasks: {sorted(unknown)}")
    if len(args.gpus) < len(args.tasks):
        raise SystemExit("Provide one GPU id per task")

    workspace = args.workspace_root.expanduser().resolve()
    models = args.models_root.expanduser().resolve()
    hot_results = args.hot_results_root.expanduser().resolve()
    if not models.is_dir():
        raise SystemExit(f"Models root does not exist: {models}")
    hot_results.mkdir(parents=True, exist_ok=True)
    log_root = hot_results / "hot_merge_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        task: str(HOT_FUSED_OUTPUTS[task](workspace, models).resolve())
        for task in args.tasks
    }
    manifest = {
        "method": "transport_and_merge",
        "scope": "merge_only_steps_1_and_2",
        "tasks": args.tasks,
        "gpus": args.gpus[: len(args.tasks)],
        "fused_models": outputs,
    }
    (hot_results / "hot_merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    running = []
    opened_logs = []
    try:
        for index, task in enumerate(args.tasks):
            gpu = args.gpus[index]
            log_path = log_root / f"{task}.log"
            log_file = log_path.open("a", encoding="utf-8")
            opened_logs.append(log_file)
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "NUM_GPUS": "1",
                    "NPROC_PER_NODE": "1",
                    "WORKSPACE_ROOT": str(workspace),
                    "MODELS_ROOT": str(models),
                    "HOT_RESULTS_ROOT": str(hot_results),
                    "TASK_NAMES": task,
                    "RUN_STEP1": "true",
                    "RUN_STEP2": "true",
                    "RUN_STEP3": "false",
                    "RUN_STEP4": "false",
                }
            )
            if args.fineweb_thai_cache is not None:
                environment["FINEWEB_THAI_CACHE_DIR"] = str(
                    args.fineweb_thai_cache.expanduser().resolve()
                )
            command = [args.bash, str(REPOSITORY_ROOT / "scripts" / "run_train_final.sh")]
            print(f"[launch] task={task} physical_gpu={gpu} log={log_path}", flush=True)
            process = subprocess.Popen(
                command,
                cwd=REPOSITORY_ROOT,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            running.append((task, process, log_path))

        failures = []
        while running:
            remaining = []
            for task, process, log_path in running:
                return_code = process.poll()
                if return_code is None:
                    remaining.append((task, process, log_path))
                    continue
                print(f"[complete] task={task} exit={return_code}", flush=True)
                if return_code != 0:
                    failures.append((task, return_code, str(log_path)))
            running = remaining
            if running:
                time.sleep(5)
        if failures:
            print(json.dumps({"failures": failures}, ensure_ascii=False, indent=2))
            return 1
        return 0
    except KeyboardInterrupt:
        for _, process, _ in running:
            process.terminate()
        return 130
    finally:
        for log_file in opened_logs:
            log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
