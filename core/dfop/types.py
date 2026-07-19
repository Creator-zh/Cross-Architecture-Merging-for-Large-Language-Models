from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch


Tensor = torch.Tensor


@dataclass
class SVDRecord:
    """Truncated SVD and reproducibility metadata for one weight matrix."""

    u: Tensor
    s: Tensor
    v: Tensor
    shape: Tuple[int, int]
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def sigma(self) -> Tensor:
        return torch.diag(self.s)

    def to(self, device: torch.device | str, dtype: torch.dtype = torch.float32) -> "SVDRecord":
        return SVDRecord(
            u=self.u.to(device=device, dtype=dtype),
            s=self.s.to(device=device, dtype=dtype),
            v=self.v.to(device=device, dtype=dtype),
            shape=self.shape,
            rank=self.rank,
            metadata=dict(self.metadata),
        )


@dataclass
class SinkhornResult:
    coupling: Tensor
    marginal_error: float
    iterations: int
    converged: bool


@dataclass
class OTProcrustesResult:
    coupling: Tensor
    rotation: Tensor
    geometric_cost: float
    regularized_objective: float
    marginal_error: float
    sinkhorn_iterations: int
    alternating_iterations: int
    restart: int
    converged: bool


@dataclass
class LayerCostResult:
    cost: Tensor
    pair_results: Dict[Tuple[int, int], OTProcrustesResult] = field(
        default_factory=dict
    )


@dataclass
class LayerRouteResult:
    dense_route: Tensor
    route: Tensor
    selected_mask: Optional[Tensor]
    entropy: Tensor
    effective_source_count: Tensor
    row_marginal_error: float
    column_marginal_error: float
    transport_objective: float
    solver: str


@dataclass
class PairCoreResult:
    core: Tensor
    calibrated_core: Tensor
    scale: float
    output_result: OTProcrustesResult
    input_result: OTProcrustesResult
    valid: bool
    skip_reason: Optional[str] = None


@dataclass
class FusionResult:
    weight: Tensor
    delta: Tensor
    trust_coefficient: float
    relative_update_norm: float
