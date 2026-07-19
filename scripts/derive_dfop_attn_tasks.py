#!/usr/bin/env python
"""Derive exact attention-only checkpoints from completed seven-module DFOP runs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from core.dfop.module_registry import collect_module_linears  # noqa: E402
from core.dfop.config import ROUTE_GROUPINGS, ROUTE_SOLVERS  # noqa: E402
from core.dfop.task_presets import (  # noqa: E402
    TASK_PRESETS,
    dfop_fusion_run_name,
    get_task_preset,
)


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=_csv, default=["medical", "thai", "malay"])
    parser.add_argument("--models-root", type=Path, default=REPOSITORY_ROOT / "models")
    parser.add_argument(
        "--results-root", type=Path, default=REPOSITORY_ROOT / "transport_results" / "dfop"
    )
    parser.add_argument("--track", choices=("universal", "matched"), default="universal")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--top-source-layers", type=int, default=2)
    parser.add_argument(
        "--route-solver", choices=ROUTE_SOLVERS, default="row_softmax_topk"
    )
    parser.add_argument(
        "--route-grouping", choices=ROUTE_GROUPINGS, default="independent"
    )
    parser.add_argument("--model-dtype", choices=("auto", "bfloat16", "float16"), default="auto")
    parser.add_argument("--overwrite-output", action="store_true")
    return parser


def _dtype(value: str):
    return {"auto": "auto", "bfloat16": torch.bfloat16, "float16": torch.float16}[value]


def _copy_tokenizer_metadata(source: Path, destination: Path) -> list[str]:
    copied = []
    for name in (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
    ):
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, destination / name)
            copied.append(name)
    return copied


@torch.no_grad()
def _derive_one(args: argparse.Namespace, task: str) -> dict:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise RuntimeError("transformers is required on the experiment server") from error

    preset = get_task_preset(task)
    beta = 0.05 if args.track == "universal" else preset.matched_beta
    full_name = dfop_fusion_run_name(
        task,
        "full",
        args.track,
        args.rank,
        args.top_source_layers,
        beta,
        route_solver=args.route_solver,
        route_grouping=args.route_grouping,
    )
    attn_name = dfop_fusion_run_name(
        task,
        "attn",
        args.track,
        args.rank,
        args.top_source_layers,
        beta,
        route_solver=args.route_solver,
        route_grouping=args.route_grouping,
    )
    full_model_path = (args.results_root / full_name / "fused_model").resolve()
    original_path = (args.models_root / preset.target_local_dir).resolve()
    output_root = (args.results_root / attn_name).resolve()
    output_model_path = output_root / "fused_model"
    if not full_model_path.is_dir() or not original_path.is_dir():
        raise FileNotFoundError(
            f"Missing input for {task}: full={full_model_path}, target={original_path}"
        )
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite_output:
        raise FileExistsError(f"Output is not empty: {output_root}")
    output_model_path.mkdir(parents=True, exist_ok=True)

    load_kwargs = {
        "torch_dtype": _dtype(args.model_dtype),
        "low_cpu_mem_usage": True,
        "local_files_only": True,
    }
    full_model = AutoModelForCausalLM.from_pretrained(full_model_path, **load_kwargs)
    original_model = AutoModelForCausalLM.from_pretrained(original_path, **load_kwargs)
    full_ffn = collect_module_linears(full_model, ("gate", "up", "down"), require_all=True)
    original_ffn = collect_module_linears(
        original_model, ("gate", "up", "down"), require_all=True
    )
    restored = 0
    for module_name in ("gate", "up", "down"):
        if len(full_ffn[module_name]) != len(original_ffn[module_name]):
            raise ValueError(f"Layer-count mismatch for {task}:{module_name}")
        for destination, source in zip(full_ffn[module_name], original_ffn[module_name]):
            if destination.weight.shape != source.weight.shape:
                raise ValueError(f"Weight-shape mismatch for {task}:{module_name}")
            destination.weight.copy_(source.weight.to(destination.weight.dtype))
            if destination.bias is not None or source.bias is not None:
                if destination.bias is None or source.bias is None:
                    raise ValueError(f"Bias mismatch for {task}:{module_name}")
                destination.bias.copy_(source.bias.to(destination.bias.dtype))
            restored += 1

    full_model.save_pretrained(output_model_path, safe_serialization=True, max_shard_size="5GB")
    copied = _copy_tokenizer_metadata(full_model_path, output_model_path)
    report = {
        "task": task,
        "derivation": "restore_target_ffn_from_dfop_full",
        "source_full_model": str(full_model_path),
        "source_original_target": str(original_path),
        "output_model": str(output_model_path),
        "restored_linear_count": restored,
        "restored_modules": ["gate", "up", "down"],
        "retained_modules": ["q", "k", "v", "o"],
        "copied_tokenizer_files": copied,
    }
    (output_root / "derivation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    unknown = set(args.tasks) - set(TASK_PRESETS)
    if unknown:
        raise SystemExit(f"Unknown tasks: {sorted(unknown)}")
    reports = []
    for task in args.tasks:
        print(f"[derive] task={task}", flush=True)
        try:
            reports.append(_derive_one(args, task))
        except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
            print(f"[failed] task={task}: {error}", file=sys.stderr, flush=True)
            return 1
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
