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
