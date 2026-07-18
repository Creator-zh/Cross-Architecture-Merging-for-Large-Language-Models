from __future__ import annotations

from typing import Callable, Sequence

import torch

from .config import OTProcrustesConfig, SpectralPointConfig
from .ot_procrustes import solve_ot_procrustes
from .spectral_points import build_record_points
from .types import LayerCostResult, SVDRecord


@torch.no_grad()
def compute_module_layer_costs(
    target_records: Sequence[SVDRecord],
    source_records: Sequence[SVDRecord],
    *,
    residual_side: str,
    point_config: SpectralPointConfig | None = None,
    ot_config: OTProcrustesConfig | None = None,
    store_pair_results: bool = False,
    device: torch.device | str | None = None,
    initial_cost: torch.Tensor | None = None,
    checkpoint_callback: Callable[[torch.Tensor], None] | None = None,
) -> LayerCostResult:
    if not target_records or not source_records:
        raise ValueError("target_records and source_records must be non-empty")
    ranks = {record.rank for record in (*target_records, *source_records)}
    if len(ranks) != 1:
        raise ValueError("all records for one module must use the same fixed rank")

    target_points = [build_record_points(record, residual_side, point_config) for record in target_records]
    source_points = [build_record_points(record, residual_side, point_config) for record in source_records]
    if device is not None:
        target_points = [points.to(device=device, dtype=torch.float32) for points in target_points]
        source_points = [points.to(device=device, dtype=torch.float32) for points in source_points]
    device = target_points[0].device
    if any(points.device != device for points in (*target_points, *source_points)):
        raise ValueError("all spectral point clouds must be on the same device")

    expected_shape = (len(target_records), len(source_records))
    if initial_cost is None:
        cost = torch.full(expected_shape, float("nan"), device=device, dtype=torch.float32)
    else:
        if tuple(initial_cost.shape) != expected_shape:
            raise ValueError("initial_cost has the wrong shape")
        if torch.isinf(initial_cost).any():
            raise ValueError("initial_cost must contain only finite values or NaN")
        cost = initial_cost.to(device=device, dtype=torch.float32).clone()
    pair_results = {}
    for target_index, x in enumerate(target_points):
        for source_index, y in enumerate(source_points):
            if torch.isfinite(cost[target_index, source_index]):
                continue
            result = solve_ot_procrustes(x, y, config=ot_config)
            cost[target_index, source_index] = result.geometric_cost
            if store_pair_results:
                pair_results[(target_index, source_index)] = result
        if checkpoint_callback is not None:
            checkpoint_callback(cost.detach().cpu())

    if not torch.isfinite(cost).all():
        raise RuntimeError("Layer-cost computation finished with missing or non-finite entries")

    return LayerCostResult(cost=cost, pair_results=pair_results)
