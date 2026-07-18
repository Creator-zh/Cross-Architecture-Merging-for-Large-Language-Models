#!/usr/bin/env python
"""Run data-free DF-OT-Procrustes fusion on two causal language models.

The script intentionally never loads a tokenizer, dataset, or input tensors and
never calls either model's forward method. Model weights stay on CPU; one linear
matrix/pair at a time is streamed to ``--device`` for SVD and OT computation.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Iterable

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from core.dfop.config import (  # noqa: E402
    MODULE_TYPES,
    CoreScaleConfig,
    DFOPConfig,
    FusionConfig,
    OTProcrustesConfig,
    RouteConfig,
    SVDConfig,
    SinkhornConfig,
    SpectralPointConfig,
)
from core.dfop.diagnostics import write_json  # noqa: E402
from core.dfop.pipeline import run_dfop_pipeline  # noqa: E402


def _comma_separated(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(values) - set(MODULE_TYPES)
    if not values or unknown:
        raise argparse.ArgumentTypeError(
            f"modules must be a non-empty subset of {','.join(MODULE_TYPES)}; "
            f"unknown={sorted(unknown)}"
        )
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("modules must not contain duplicates")
    return values


def _optional_positive_int(value: str) -> int | None:
    if value.lower() in {"all", "none"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive or 'all'")
    return parsed


def _optional_positive_float(value: str) -> float | None:
    if value.lower() in {"none", "off", "disabled"}:
        return None
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive or 'none'")
    return parsed


def _rank_overrides(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        try:
            module, rank_text = value.split("=", 1)
            rank = int(rank_text)
        except ValueError as error:
            raise ValueError(f"Invalid --rank-by-module value: {value!r}") from error
        module = module.strip()
        if module not in MODULE_TYPES or rank <= 0:
            raise ValueError(f"Invalid --rank-by-module value: {value!r}")
        result[module] = rank
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fuse a large source LM into a smaller target LM without data or forward passes."
    )
    parser.add_argument("--target-model", required=True, help="Fine-tuned target model path or HF id")
    parser.add_argument("--source-model", required=True, help="General source model path or HF id")
    parser.add_argument("--output-dir", required=True, type=Path, help="New DFOP result directory")
    parser.add_argument("--diagnostics-dir", type=Path, default=None)
    parser.add_argument(
        "--stage1-cache-dir",
        type=Path,
        default=None,
        help="Reusable layer-cost cache; defaults to OUTPUT_DIR/stage1_cache",
    )
    parser.add_argument(
        "--stage2-cache-dir",
        type=Path,
        default=None,
        help="Reusable aggregate-core cache; defaults to OUTPUT_DIR/stage2_cache",
    )
    parser.add_argument("--device", default="auto", help="cpu, cuda, cuda:N, or auto")
    parser.add_argument(
        "--model-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Storage dtype while models remain on CPU",
    )
    parser.add_argument("--modules", type=_comma_separated, default=MODULE_TYPES)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument(
        "--rank-by-module",
        action="append",
        default=[],
        metavar="NAME=RANK",
        help="Repeat for module-specific fixed ranks, e.g. q=128",
    )
    parser.add_argument("--svd-algorithm", choices=("randomized", "exact"), default="randomized")
    parser.add_argument("--svd-oversample", type=int, default=16)
    parser.add_argument("--svd-power-iterations", type=int, default=2)
    parser.add_argument("--sigma-power", type=float, default=1.0)
    parser.add_argument("--center-points", action="store_true")
    parser.add_argument("--disable-width-normalization", action="store_true")
    parser.add_argument("--inner-entropy", type=float, default=0.05)
    parser.add_argument("--sinkhorn-iterations", type=int, default=200)
    parser.add_argument("--sinkhorn-tolerance", type=float, default=1e-6)
    parser.add_argument("--sinkhorn-check-interval", type=int, default=10)
    parser.add_argument("--alternating-iterations", type=int, default=8)
    parser.add_argument("--alternating-tolerance", type=float, default=1e-4)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--route-temperature", type=float, default=0.05)
    parser.add_argument(
        "--top-source-layers",
        type=_optional_positive_int,
        default=2,
        help="Sources retained per target row; use 'all' for a dense route",
    )
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument(
        "--trust-ratio",
        type=_optional_positive_float,
        default=0.10,
        help="Maximum relative Frobenius update; use 'none' to disable",
    )
    parser.add_argument("--disable-core-scale", action="store_true")
    parser.add_argument("--gamma-min", type=float, default=0.25)
    parser.add_argument("--gamma-max", type=float, default=4.0)
    parser.add_argument("--minimum-relative-core-norm", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute everything but do not change/save weights",
    )
    output_mode.add_argument(
        "--updates-only",
        action="store_true",
        help="Save exact rank-k update factors without saving a full model",
    )
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--no-safe-serialization", action="store_true")
    parser.add_argument(
        "--copy-target-tokenizer-files",
        action="store_true",
        help="Copy local target tokenizer metadata without loading a tokenizer",
    )
    return parser


def _torch_dtype(name: str):
    return {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _load_model(identifier: str, args: argparse.Namespace):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise RuntimeError(
            "transformers is required for the CLI. Install requirements.txt on the GPU server."
        ) from error

    kwargs = {
        "torch_dtype": _torch_dtype(args.model_dtype),
        "low_cpu_mem_usage": True,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    if args.revision is not None:
        kwargs["revision"] = args.revision
    model = AutoModelForCausalLM.from_pretrained(identifier, **kwargs)
    model.to("cpu")
    model.eval()
    return model


def _build_config(args: argparse.Namespace) -> DFOPConfig:
    config = DFOPConfig(
        modules=args.modules,
        svd=SVDConfig(
            rank_default=args.rank,
            rank_by_module=_rank_overrides(args.rank_by_module),
            algorithm=args.svd_algorithm,
            oversample=args.svd_oversample,
            power_iterations=args.svd_power_iterations,
            seed=args.seed,
        ),
        spectral_points=SpectralPointConfig(
            sigma_power=args.sigma_power,
            width_normalization=not args.disable_width_normalization,
            center=args.center_points,
        ),
        ot_procrustes=OTProcrustesConfig(
            sinkhorn=SinkhornConfig(
                entropy=args.inner_entropy,
                max_iterations=args.sinkhorn_iterations,
                tolerance=args.sinkhorn_tolerance,
                check_interval=args.sinkhorn_check_interval,
            ),
            max_alternating_iterations=args.alternating_iterations,
            alternating_tolerance=args.alternating_tolerance,
            restarts=args.restarts,
            seed=args.seed,
        ),
        route=RouteConfig(
            temperature=args.route_temperature,
            top_source_layers=args.top_source_layers,
        ),
        core_scale=CoreScaleConfig(
            enabled=not args.disable_core_scale,
            gamma_min=args.gamma_min,
            gamma_max=args.gamma_max,
            minimum_relative_norm=args.minimum_relative_core_norm,
        ),
        fusion=FusionConfig(beta=args.beta, trust_ratio=args.trust_ratio),
    )
    config.validate()
    return config


def _ensure_output_is_safe(output: Path, overwrite: bool) -> None:
    resolved = output.expanduser().resolve()
    if resolved == REPOSITORY_ROOT or resolved.parent == resolved:
        raise ValueError("Refusing to use a repository/filesystem root as --output-dir")
    if resolved.exists() and any(resolved.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {resolved}. Use --overwrite-output to reuse it."
        )
    resolved.mkdir(parents=True, exist_ok=True)


def _copy_local_tokenizer_files(target_identifier: str, destination: Path) -> list[str]:
    source = Path(target_identifier).expanduser()
    if not source.is_dir():
        raise ValueError("--copy-target-tokenizer-files requires a local --target-model directory")
    names = (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
    )
    copied = []
    for name in names:
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, destination / name)
            copied.append(name)
    if not copied:
        raise FileNotFoundError(f"No tokenizer metadata files found in {source}")
    return copied


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _build_config(args)
        _ensure_output_is_safe(args.output_dir, args.overwrite_output)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))

    output_dir = args.output_dir.expanduser().resolve()
    diagnostics_dir = (
        args.diagnostics_dir.expanduser().resolve()
        if args.diagnostics_dir is not None
        else output_dir / "diagnostics"
    )
    stage1_cache_dir = (
        args.stage1_cache_dir.expanduser().resolve()
        if args.stage1_cache_dir is not None
        else output_dir / "stage1_cache"
    )
    stage2_cache_dir = (
        args.stage2_cache_dir.expanduser().resolve()
        if args.stage2_cache_dir is not None
        else output_dir / "stage2_cache"
    )
    invocation = {
        "target_model": args.target_model,
        "source_model": args.source_model,
        "model_dtype": args.model_dtype,
        "device": args.device,
        "dry_run": args.dry_run,
        "updates_only": args.updates_only,
        "argv": sys.argv[1:] if argv is None else argv,
        "data_free_contract": {
            "loads_dataset": False,
            "loads_tokenizer": False,
            "calls_forward": False,
        },
    }
    write_json(output_dir / "invocation.json", invocation)
    write_json(output_dir / "config.json", config.to_dict())

    print(f"[DFOP] Loading target model on CPU: {args.target_model}", flush=True)
    target_model = _load_model(args.target_model, args)
    print(f"[DFOP] Loading source model on CPU: {args.source_model}", flush=True)
    source_model = _load_model(args.source_model, args)
    print(f"[DFOP] Starting data-free computation on {args.device}", flush=True)
    result = run_dfop_pipeline(
        target_model,
        source_model,
        config,
        compute_device=args.device,
        diagnostics_dir=diagnostics_dir,
        stage1_cache_dir=stage1_cache_dir,
        stage2_cache_dir=stage2_cache_dir,
        target_identity=args.target_model,
        source_identity=args.source_model,
        apply_updates=not (args.dry_run or args.updates_only),
        collect_low_rank_updates=args.updates_only,
    )
    write_json(output_dir / "run_report.json", result.report)

    if args.updates_only:
        update_files = {}
        for module_name, updates in result.low_rank_updates.items():
            update_path = output_dir / f"updates_{module_name}.pt"
            torch.save(updates, update_path)
            update_files[module_name] = update_path.name
        write_json(
            output_dir / "updates_manifest.json",
            {
                "format": "dfop_exact_low_rank_v1",
                "target_model": args.target_model,
                "source_model": args.source_model,
                "modules": list(config.modules),
                "rank_by_module": result.report["rank_by_module"],
                "files": update_files,
            },
        )
        print(f"[DFOP] Saved exact low-rank update shards: {output_dir}", flush=True)
    elif not args.dry_run:
        fused_model_dir = output_dir / "fused_model"
        fused_model_dir.mkdir(parents=True, exist_ok=True)
        target_model.save_pretrained(
            fused_model_dir,
            safe_serialization=not args.no_safe_serialization,
            max_shard_size=args.max_shard_size,
        )
        if args.copy_target_tokenizer_files:
            copied = _copy_local_tokenizer_files(args.target_model, fused_model_dir)
            write_json(output_dir / "copied_tokenizer_files.json", {"files": copied})
        print(f"[DFOP] Saved fused target model: {fused_model_dir}", flush=True)
    else:
        print("[DFOP] Dry-run complete; no model weights were changed or saved", flush=True)
    print(json.dumps(result.report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
