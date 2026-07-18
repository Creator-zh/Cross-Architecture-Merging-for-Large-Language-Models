from __future__ import annotations

from typing import Sequence

import torch

from .config import FusionConfig
from .types import FusionResult, PairCoreResult, SVDRecord


@torch.no_grad()
def aggregate_cores(
    pair_cores: Sequence[PairCoreResult],
    route_weights: torch.Tensor,
) -> torch.Tensor:
    if len(pair_cores) != int(route_weights.numel()):
        raise ValueError("pair core count must equal route weight count")
    if not pair_cores:
        raise ValueError("at least one pair core is required")
    route_weights = route_weights.to(dtype=torch.float32)
    valid = torch.tensor(
        [result.valid for result in pair_cores],
        device=route_weights.device,
        dtype=torch.bool,
    )
    weights = torch.where(valid, route_weights, torch.zeros_like(route_weights))
    total = weights.sum()
    if float(total) <= 0:
        raise ValueError("no valid pair core has positive route mass")
    weights = weights / total
    reference = next(result.calibrated_core for result in pair_cores if result.valid)
    aggregate = torch.zeros_like(reference, dtype=torch.float32)
    for weight, result in zip(weights, pair_cores):
        if result.valid:
            aggregate = aggregate + weight.to(aggregate.device) * result.calibrated_core.to(
                aggregate.device, torch.float32
            )
    return aggregate.contiguous()


@torch.no_grad()
def fuse_target_weight(
    weight_a: torch.Tensor,
    target_svd: SVDRecord,
    aggregate_core: torch.Tensor,
    config: FusionConfig | None = None,
) -> FusionResult:
    config = config or FusionConfig()
    config.validate()
    if tuple(weight_a.shape) != tuple(target_svd.shape):
        raise ValueError("weight_a shape does not match target SVD record")
    if aggregate_core.shape != (target_svd.rank, target_svd.rank):
        raise ValueError("aggregate_core has the wrong shape")

    device = aggregate_core.device
    u = target_svd.u.to(device=device, dtype=torch.float32)
    v = target_svd.v.to(device=device, dtype=torch.float32)
    sigma = torch.diag(target_svd.s.to(device=device, dtype=torch.float32))
    raw_delta = u @ (aggregate_core.to(torch.float32) - sigma) @ v.transpose(0, 1)
    beta = float(config.beta)
    trust = 1.0
    weight_norm = torch.linalg.matrix_norm(weight_a.detach().to(device, torch.float32))
    delta_norm = torch.linalg.matrix_norm(raw_delta)
    if config.trust_ratio is not None and beta > 0 and float(delta_norm) > 0:
        trust = min(
            1.0,
            float(config.trust_ratio) * float(weight_norm) / (beta * float(delta_norm)),
        )
    folded_delta = beta * trust * raw_delta
    fused = weight_a.detach().to(device, torch.float32) + folded_delta
    relative = float(torch.linalg.matrix_norm(folded_delta) / weight_norm.clamp_min(1.0e-12))
    return FusionResult(
        weight=fused.to(dtype=weight_a.dtype).contiguous(),
        delta=folded_delta.contiguous(),
        trust_coefficient=trust,
        relative_update_norm=relative,
    )
