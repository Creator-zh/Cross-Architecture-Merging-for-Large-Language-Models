#!/usr/bin/env python
"""Run one strictly data-free Anchor-Compress-Inject merge."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from core.aci import ACIConfig, run_aci_pipeline  # noqa: E402
from core.aci.config import FUSION_MODES  # noqa: E402
from core.aci.io import write_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", required=True)
    parser.add_argument(
        "--reference-model",
        required=True,
        help="Same-architecture generic 1B checkpoint used to preserve the target task vector",
    )
    parser.add_argument("--source-model", required=True, help="Wider/deeper 8B donor")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--beta", required=True, type=float, help="Injection strength and update cap")
    parser.add_argument(
        "--fusion-mode",
        choices=FUSION_MODES,
        default="full",
        help=(
            "Select legacy full/module ablation or safe FFN/circuit-attention "
            "injection"
        ),
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--model-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--anchor-tokens", type=int, default=8192)
    parser.add_argument("--anchor-chunk-size", type=int, default=1024)
    parser.add_argument("--ffn-sketch-dim", type=int, default=32)
    parser.add_argument("--ffn-candidate-k", type=int, default=32)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--no-safe-serialization", action="store_true")
    parser.add_argument(
        "--skip-tokenizer-copy",
        action="store_true",
        help="Do not copy tokenizer metadata when --target-model is a local directory",
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
        raise RuntimeError("transformers is required; install requirements.txt") from error
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


def _prepare_output(path: Path, overwrite: bool) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == REPOSITORY_ROOT or resolved.parent == resolved:
        raise ValueError("Refusing to use a repository or filesystem root as output")
    if resolved.exists() and any(resolved.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output is not empty: {resolved}; use --overwrite-output to resume/replace files"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _copy_tokenizer_metadata(
    target: str,
    destination: Path,
    *,
    revision: str | None,
    local_files_only: bool,
) -> list[str]:
    source = Path(target).expanduser()
    names = (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
    )
    copied = []
    if source.is_dir():
        for name in names:
            candidate = source / name
            if candidate.is_file():
                shutil.copy2(candidate, destination / name)
                copied.append(name)
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError("huggingface_hub is required to copy remote tokenizer metadata") from error
        snapshot_download(
            repo_id=target,
            revision=revision,
            local_dir=str(destination),
            allow_patterns=list(names),
            local_files_only=local_files_only,
        )
        copied = [name for name in names if (destination / name).is_file()]
    if not copied:
        raise FileNotFoundError(
            f"No tokenizer metadata found for {target}; use --skip-tokenizer-copy only "
            "if evaluation supplies a tokenizer separately"
        )
    return copied


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    config = ACIConfig(
        beta=args.beta,
        fusion_mode=args.fusion_mode,
        anchor_tokens=args.anchor_tokens,
        anchor_chunk_size=args.anchor_chunk_size,
        ffn_sketch_dim=args.ffn_sketch_dim,
        ffn_candidate_k=args.ffn_candidate_k,
    )
    try:
        config.validate()
        output = _prepare_output(args.output_dir, args.overwrite_output)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))

    write_json(output / "config.json", config.to_dict())
    write_json(
        output / "invocation.json",
        {
            "target_model": args.target_model,
            "reference_model": args.reference_model,
            "source_model": args.source_model,
            "device": args.device,
            "model_dtype": args.model_dtype,
            "fusion_mode": args.fusion_mode,
            "dry_run": args.dry_run,
            "argv": sys.argv[1:] if argv is None else argv,
            "data_free_contract": {
                "loads_dataset": False,
                "loads_tokenizer": False,
                "calls_forward": False,
            },
        },
    )

    print(f"[ACI] Loading target on CPU: {args.target_model}", flush=True)
    target = _load_model(args.target_model, args)
    print(f"[ACI] Loading 1B reference on CPU: {args.reference_model}", flush=True)
    reference = _load_model(args.reference_model, args)
    print(f"[ACI] Loading 8B source on CPU: {args.source_model}", flush=True)
    source = _load_model(args.source_model, args)
    result = run_aci_pipeline(
        target,
        reference,
        source,
        config,
        compute_device=args.device,
        diagnostics_dir=output / "diagnostics",
        apply_updates=not args.dry_run,
    )
    write_json(output / "run_report.json", result.report)

    if not args.dry_run:
        fused = output / "fused_model"
        fused.mkdir(parents=True, exist_ok=True)
        target.save_pretrained(
            fused,
            safe_serialization=not args.no_safe_serialization,
            max_shard_size=args.max_shard_size,
        )
        copied = [] if args.skip_tokenizer_copy else _copy_tokenizer_metadata(
            args.target_model,
            fused,
            revision=args.revision,
            local_files_only=args.local_files_only,
        )
        write_json(output / "copied_tokenizer_files.json", {"files": copied})
        print(f"[ACI] Saved fused model: {fused}", flush=True)
    else:
        print("[ACI] Dry-run complete; target weights were not changed", flush=True)
    print(json.dumps(result.report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
