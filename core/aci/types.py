from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AnchorResult:
    source_to_reference: torch.Tensor
    reference_sketch_basis: torch.Tensor
    input_anchor_cosine: float
    output_anchor_cosine: float | None
    anchor_count: int


@dataclass
class FFNMatchResult:
    source_indices: torch.Tensor
    selected_cosines: torch.Tensor
    mean_cosine: float
    minimum_cosine: float
    reused_sources: int


@dataclass
class InjectionResult:
    weight: torch.Tensor
    delta: torch.Tensor
    calibration_scale: float
    trust_coefficient: float
    relative_update_norm: float


@dataclass
class SafeGroupInjectionResult:
    weights: dict[str, torch.Tensor]
    calibration_scales: dict[str, float]
    trust_coefficient: float
    joint_relative_update_norm: float
    module_relative_update_norms: dict[str, float]
    mean_confidence: float
    minimum_confidence: float
    maximum_confidence: float
    active_confidence_fraction: float
    conflict_fraction: float
    source_domain_cosine: float
    removed_conflict_norm_ratio: float


@dataclass
class SafeAttentionInjectionResult:
    weights: dict[str, torch.Tensor]
    calibration_scales: dict[str, float]
    trust_coefficients: dict[str, float]
    joint_relative_update_norm: float
    module_relative_update_norms: dict[str, float]
    mean_confidence: float
    minimum_confidence: float
    maximum_confidence: float
    active_confidence_fraction: float
    qk_conflict_fraction: float
    ov_conflict_fraction: float
    qk_source_domain_cosine: float
    ov_source_domain_cosine: float
    removed_conflict_norm_ratio: float
