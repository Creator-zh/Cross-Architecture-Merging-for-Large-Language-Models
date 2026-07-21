from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .alignment import build_residual_anchor
from .attention import (
    attention_geometry,
    contract_attention,
    contract_attention_circuit,
    validate_attention_pair,
)
from .config import ACIConfig
from .ffn import contract_ffn
from .injection import inject_protected_delta
from .io import write_json, write_jsonl
from .registry import (
    ATTENTION_MODULE_TYPES,
    FFN_MODULE_TYPES,
    MODULE_TYPES,
    collect_blocks,
    input_embedding,
    output_head,
)
from .safe_injection import inject_safe_attention, inject_safe_ffn


LOGGER = logging.getLogger(__name__)


@dataclass
class ACIPipelineResult:
    report: dict
    layer_groups: list[list[int]]
    attention_diagnostics: list[dict]
    ffn_diagnostics: list[dict]
    injection_diagnostics: list[dict]


def resolve_compute_device(value: str | torch.device) -> torch.device:
    if isinstance(value, torch.device):
        device = value
    elif value == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    return device


def monotonic_layer_groups(target_layers: int, source_layers: int) -> list[list[int]]:
    """Partition every source layer into ordered, contiguous target groups."""

    if target_layers <= 0 or source_layers <= 0:
        raise ValueError("layer counts must be positive")
    if source_layers < target_layers:
        raise ValueError("ACI currently contracts a deeper source into a shallower target")
    groups = []
    for target_index in range(target_layers):
        start = target_index * source_layers // target_layers
        end = (target_index + 1) * source_layers // target_layers
        groups.append(list(range(start, max(start + 1, end))))
    flattened = [index for group in groups for index in group]
    if flattened != list(range(source_layers)):
        raise AssertionError("internal error: monotonic groups do not partition source layers")
    return groups


def _validate_reference_shapes(
    target_blocks,
    reference_blocks,
) -> None:
    if len(target_blocks) != len(reference_blocks):
        raise ValueError(
            "target and reference must have the same decoder depth; "
            f"target={len(target_blocks)} reference={len(reference_blocks)}"
        )
    for layer_index, (target, reference) in enumerate(zip(target_blocks, reference_blocks)):
        for module in MODULE_TYPES:
            target_shape = tuple(target.as_dict()[module].weight.shape)
            reference_shape = tuple(reference.as_dict()[module].weight.shape)
            if target_shape != reference_shape:
                raise ValueError(
                    f"target/reference shape mismatch layer={layer_index} module={module}: "
                    f"{target_shape} != {reference_shape}"
                )


def _empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def run_aci_pipeline(
    target_model: nn.Module,
    reference_model: nn.Module,
    source_model: nn.Module,
    config: ACIConfig | None = None,
    *,
    compute_device: str | torch.device = "auto",
    diagnostics_dir: str | Path | None = None,
    apply_updates: bool = True,
) -> ACIPipelineResult:
    """Run ACI without a tokenizer, model forward, or external data."""

    config = config or ACIConfig()
    config.validate()
    active_modules = (
        MODULE_TYPES
        if config.uses_attention and config.uses_ffn
        else ATTENTION_MODULE_TYPES
        if config.uses_attention
        else FFN_MODULE_TYPES
    )
    device = resolve_compute_device(compute_device)
    started = time.perf_counter()

    target_blocks = collect_blocks(target_model)
    reference_blocks = collect_blocks(reference_model)
    source_blocks = collect_blocks(source_model)
    _validate_reference_shapes(target_blocks, reference_blocks)
    target_geometry = attention_geometry(target_model)
    reference_geometry = attention_geometry(reference_model)
    source_geometry = attention_geometry(source_model)
    if target_geometry != reference_geometry:
        raise ValueError(
            "target and reference attention geometry must be identical; "
            f"target={target_geometry} reference={reference_geometry}"
        )
    validate_attention_pair(source_geometry, reference_geometry)
    groups = monotonic_layer_groups(len(target_blocks), len(source_blocks))

    LOGGER.info(
        "Building vocabulary anchor source_hidden=%d reference_hidden=%d",
        source_geometry.hidden_size,
        reference_geometry.hidden_size,
    )
    anchor = build_residual_anchor(
        input_embedding(source_model),
        input_embedding(reference_model),
        output_head(source_model),
        output_head(reference_model),
        config,
        device=device,
    )
    anchor_finished = time.perf_counter()

    attention_diagnostics: list[dict] = []
    ffn_diagnostics: list[dict] = []
    injection_diagnostics: list[dict] = []
    for target_index, source_indices in enumerate(groups):
        LOGGER.info(
            "Contracting target layer %d from source layers %s",
            target_index,
            source_indices,
        )
        reference_block = reference_blocks[target_index]
        target_block = target_blocks[target_index]
        compressed = {
            module: torch.zeros_like(
                reference_block.as_dict()[module].weight,
                device=device,
                dtype=torch.float32,
            )
            for module in active_modules
        }
        ffn_confidence = (
            torch.zeros(
                reference_block.gate.weight.shape[0],
                device=device,
                dtype=torch.float32,
            )
            if config.uses_safe_ffn
            else None
        )
        attention_confidence = (
            torch.zeros(
                reference_geometry.query_heads,
                device=device,
                dtype=torch.float32,
            )
            if config.uses_circuit_attention
            else None
        )
        source_weight = 1.0 / len(source_indices)
        for source_index in source_indices:
            source_block = source_blocks[source_index]
            if config.uses_attention:
                attention_contractor = (
                    contract_attention_circuit
                    if config.uses_circuit_attention
                    else contract_attention
                )
                attention, attention_match = attention_contractor(
                    source_block,
                    reference_block,
                    anchor.source_to_reference,
                    anchor.reference_sketch_basis,
                    source_geometry,
                    reference_geometry,
                )
                for module, value in attention.items():
                    compressed[module].add_(value, alpha=source_weight)
                attention_record = {
                    "target_layer": target_index,
                    "source_layer": source_index,
                    "matching_mode": (
                        "qk_ov_circuit"
                        if config.uses_circuit_attention
                        else "factor_signature"
                    ),
                    "mean_group_cosine": attention_match.mean_group_cosine,
                    "minimum_group_cosine": attention_match.minimum_group_cosine,
                    "mean_query_cosine": attention_match.mean_query_cosine,
                    "minimum_query_cosine": attention_match.minimum_query_cosine,
                    "group_assignment": attention_match.group_assignment,
                    "query_assignment": attention_match.query_assignment,
                }
                if config.uses_circuit_attention:
                    if attention_confidence is None:
                        raise AssertionError(
                            "circuit attention requires accumulated head confidence"
                        )
                    attention_confidence.add_(
                        attention_match.head_confidences.to(device, torch.float32),
                        alpha=source_weight,
                    )
                    attention_record.update(
                        {
                            "selected_group_cosines": (
                                attention_match.selected_group_cosines
                            ),
                            "selected_query_cosines": (
                                attention_match.selected_query_cosines
                            ),
                            "mean_head_confidence": float(
                                attention_match.head_confidences.clamp(0.0, 1.0).mean()
                            ),
                            "qk_gauge_cosine_before": (
                                attention_match.qk_gauge_cosine_before
                            ),
                            "qk_gauge_cosine_after": (
                                attention_match.qk_gauge_cosine_after
                            ),
                            "ov_gauge_cosine_before": (
                                attention_match.ov_gauge_cosine_before
                            ),
                            "ov_gauge_cosine_after": (
                                attention_match.ov_gauge_cosine_after
                            ),
                        }
                    )
                attention_diagnostics.append(attention_record)
                del attention, attention_match
            if config.uses_ffn:
                ffn, match = contract_ffn(
                    source_block,
                    reference_block,
                    anchor.source_to_reference,
                    anchor.reference_sketch_basis,
                    config,
                )
                for module, value in ffn.items():
                    compressed[module].add_(value, alpha=source_weight)
                if ffn_confidence is not None:
                    ffn_confidence.add_(
                        match.selected_cosines.to(device, torch.float32),
                        alpha=source_weight,
                    )
                ffn_diagnostics.append(
                    {
                        "target_layer": target_index,
                        "source_layer": source_index,
                        "mean_match_cosine": match.mean_cosine,
                        "minimum_match_cosine": match.minimum_cosine,
                        "positive_match_fraction": float(
                            (match.selected_cosines > 0).float().mean()
                        ),
                        "reused_sources": match.reused_sources,
                    }
                )
                del ffn, match
            _empty_cache(device)

        if config.uses_circuit_attention:
            if attention_confidence is None:
                raise AssertionError(
                    "circuit attention mode requires accumulated head confidence"
                )
            target_weights = {
                module: target_block.as_dict()[module].weight
                for module in ATTENTION_MODULE_TYPES
            }
            reference_weights = {
                module: reference_block.as_dict()[module].weight
                for module in ATTENTION_MODULE_TYPES
            }
            safe_attention = inject_safe_attention(
                target_weights,
                reference_weights,
                {
                    module: compressed[module]
                    for module in ATTENTION_MODULE_TYPES
                },
                attention_confidence,
                reference_geometry,
                beta=config.beta,
                eps=config.eps,
            )
            for module in ATTENTION_MODULE_TYPES:
                target_linear = target_block.as_dict()[module]
                if apply_updates:
                    target_linear.weight.copy_(
                        safe_attention.weights[module].to(
                            device=target_linear.weight.device,
                            dtype=target_linear.weight.dtype,
                        )
                    )
                pair = "qk" if module in ("q", "k") else "ov"
                injection_diagnostics.append(
                    {
                        "target_layer": target_index,
                        "source_layers": source_indices,
                        "module": module,
                        "injection_mode": "safe_attention_circuit",
                        "circuit_pair": pair,
                        "calibration_scale": (
                            safe_attention.calibration_scales[module]
                        ),
                        "trust_coefficient": (
                            safe_attention.trust_coefficients[pair]
                        ),
                        "relative_update_norm": (
                            safe_attention.module_relative_update_norms[module]
                        ),
                        "joint_relative_update_norm": (
                            safe_attention.joint_relative_update_norm
                        ),
                        "mean_confidence": safe_attention.mean_confidence,
                        "minimum_confidence": safe_attention.minimum_confidence,
                        "maximum_confidence": safe_attention.maximum_confidence,
                        "active_confidence_fraction": (
                            safe_attention.active_confidence_fraction
                        ),
                        "qk_conflict_fraction": (
                            safe_attention.qk_conflict_fraction
                        ),
                        "ov_conflict_fraction": (
                            safe_attention.ov_conflict_fraction
                        ),
                        "qk_source_domain_cosine": (
                            safe_attention.qk_source_domain_cosine
                        ),
                        "ov_source_domain_cosine": (
                            safe_attention.ov_source_domain_cosine
                        ),
                        "removed_conflict_norm_ratio": (
                            safe_attention.removed_conflict_norm_ratio
                        ),
                        "applied": apply_updates,
                    }
                )
                del compressed[module]
            del safe_attention, attention_confidence

        if config.uses_safe_ffn:
            if ffn_confidence is None:
                raise AssertionError("safe FFN mode requires accumulated confidence")
            target_weights = {
                module: target_block.as_dict()[module].weight
                for module in FFN_MODULE_TYPES
            }
            reference_weights = {
                module: reference_block.as_dict()[module].weight
                for module in FFN_MODULE_TYPES
            }
            safe_ffn = inject_safe_ffn(
                target_weights,
                reference_weights,
                {module: compressed[module] for module in FFN_MODULE_TYPES},
                ffn_confidence,
                beta=config.beta,
                eps=config.eps,
            )
            for module in FFN_MODULE_TYPES:
                target_linear = target_block.as_dict()[module]
                if apply_updates:
                    target_linear.weight.copy_(
                        safe_ffn.weights[module].to(
                            device=target_linear.weight.device,
                            dtype=target_linear.weight.dtype,
                        )
                    )
                injection_diagnostics.append(
                    {
                        "target_layer": target_index,
                        "source_layers": source_indices,
                        "module": module,
                        "injection_mode": "safe_ffn",
                        "calibration_scale": safe_ffn.calibration_scales[module],
                        "trust_coefficient": safe_ffn.trust_coefficient,
                        "relative_update_norm": safe_ffn.module_relative_update_norms[module],
                        "joint_relative_update_norm": safe_ffn.joint_relative_update_norm,
                        "mean_confidence": safe_ffn.mean_confidence,
                        "minimum_confidence": safe_ffn.minimum_confidence,
                        "maximum_confidence": safe_ffn.maximum_confidence,
                        "active_confidence_fraction": safe_ffn.active_confidence_fraction,
                        "conflict_fraction": safe_ffn.conflict_fraction,
                        "source_domain_cosine": safe_ffn.source_domain_cosine,
                        "removed_conflict_norm_ratio": safe_ffn.removed_conflict_norm_ratio,
                        "applied": apply_updates,
                    }
                )
                del compressed[module]
            del safe_ffn, ffn_confidence

        standard_modules = tuple(
            module
            for module in active_modules
            if not (
                (config.uses_safe_ffn and module in FFN_MODULE_TYPES)
                or (
                    config.uses_circuit_attention
                    and module in ATTENTION_MODULE_TYPES
                )
            )
        )
        for module in standard_modules:
            target_linear = target_block.as_dict()[module]
            reference_linear = reference_block.as_dict()[module]
            injection = inject_protected_delta(
                target_linear.weight,
                reference_linear.weight,
                compressed[module],
                beta=config.beta,
                eps=config.eps,
            )
            if apply_updates:
                target_linear.weight.copy_(
                    injection.weight.to(
                        device=target_linear.weight.device,
                        dtype=target_linear.weight.dtype,
                    )
                )
            injection_diagnostics.append(
                {
                    "target_layer": target_index,
                    "source_layers": source_indices,
                    "module": module,
                    "calibration_scale": injection.calibration_scale,
                    "trust_coefficient": injection.trust_coefficient,
                    "relative_update_norm": injection.relative_update_norm,
                    "applied": apply_updates,
                }
            )
            del injection, compressed[module]
        _empty_cache(device)

    finished = time.perf_counter()
    report = {
        "method": "anchor_compress_inject",
        "beta": config.beta,
        "fusion_mode": config.fusion_mode,
        "modules": list(active_modules),
        "strategies": {
            "attention": (
                "qk_ov_circuit_gauge_conflict_gate"
                if config.uses_circuit_attention
                else "factor_signature"
                if config.uses_attention
                else "disabled"
            ),
            "ffn": (
                "joint_neuron_conflict_gate"
                if config.uses_safe_ffn
                else "joint_neuron_match"
                if config.uses_ffn
                else "disabled"
            ),
        },
        "target_layers": len(target_blocks),
        "source_layers": len(source_blocks),
        "layer_groups": groups,
        "compute_device": str(device),
        "apply_updates": apply_updates,
        "anchor": {
            "count": anchor.anchor_count,
            "input_cosine": anchor.input_anchor_cosine,
            "output_cosine": anchor.output_anchor_cosine,
            "projection_shape": list(anchor.source_to_reference.shape),
            "orthogonality_error": float(
                (
                    anchor.source_to_reference.transpose(0, 1)
                    @ anchor.source_to_reference
                    - torch.eye(
                        anchor.source_to_reference.shape[1],
                        device=device,
                        dtype=torch.float32,
                    )
                ).abs().max()
            ),
        },
        "data_free_contract": {
            "loads_dataset": False,
            "loads_tokenizer": False,
            "calls_forward": False,
            "uses_only_model_weights_and_config": True,
        },
        "timing_seconds": {
            "anchor": anchor_finished - started,
            "contract_and_inject": finished - anchor_finished,
            "total": finished - started,
        },
    }
    if diagnostics_dir is not None:
        output = Path(diagnostics_dir)
        write_json(output / "config.json", config.to_dict())
        write_json(output / "run_report.json", report)
        write_jsonl(output / "attention_matches.jsonl", attention_diagnostics)
        write_jsonl(output / "ffn_matches.jsonl", ffn_diagnostics)
        write_jsonl(output / "injections.jsonl", injection_diagnostics)
    return ACIPipelineResult(
        report=report,
        layer_groups=groups,
        attention_diagnostics=attention_diagnostics,
        ffn_diagnostics=ffn_diagnostics,
        injection_diagnostics=injection_diagnostics,
    )
