from __future__ import annotations

import numpy as np
import torch

from .config import RouteConfig
from .types import LayerRouteResult


def _route_statistics(
    route64: torch.Tensor,
    cost64: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float, float, float]:
    target_layers, source_layers = route64.shape
    expected_rows = torch.ones(target_layers, dtype=torch.float64)
    expected_columns = torch.full(
        (source_layers,),
        float(target_layers) / float(source_layers),
        dtype=torch.float64,
    )
    row_error = float((route64.sum(dim=1) - expected_rows).abs().max())
    column_error = float((route64.sum(dim=0) - expected_columns).abs().max())
    objective = float((cost64 * route64).sum() / float(target_layers))
    route = route64.to(dtype=torch.float32)
    safe = route.clamp_min(torch.finfo(route.dtype).tiny)
    entropy = -(route * safe.log()).sum(dim=1)
    return route, entropy, row_error, column_error, objective


def _compute_balanced_exact(
    cost64: torch.Tensor,
    config: RouteConfig,
) -> LayerRouteResult:
    try:
        from scipy.optimize import linprog
        from scipy.sparse import lil_matrix
    except ImportError as error:
        raise RuntimeError(
            "Strict balanced layer OT requires scipy. Install requirements.txt."
        ) from error

    target_layers, source_layers = cost64.shape
    variable_count = target_layers * source_layers
    constraints = lil_matrix(
        (target_layers + source_layers, variable_count), dtype=np.float64
    )
    for target_index in range(target_layers):
        start = target_index * source_layers
        constraints[target_index, start : start + source_layers] = 1.0
    for source_index in range(source_layers):
        constraints[
            target_layers + source_index,
            source_index::source_layers,
        ] = 1.0

    target_mass = np.full(target_layers, 1.0 / target_layers, dtype=np.float64)
    source_mass = np.full(source_layers, 1.0 / source_layers, dtype=np.float64)
    result = linprog(
        cost64.numpy().reshape(-1),
        A_eq=constraints.tocsr(),
        b_eq=np.concatenate((target_mass, source_mass)),
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            "Strict balanced layer OT failed: "
            f"status={result.status} message={result.message}"
        )

    coupling = torch.from_numpy(result.x.reshape(target_layers, source_layers))
    coupling = coupling.clamp_min(0.0)
    # The pipeline consumes a row-stochastic route. Multiplication by L maps
    # the probability coupling back to conditional source weights per target.
    route64 = coupling * float(target_layers)
    route, entropy, row_error, column_error, objective = _route_statistics(
        route64, cost64
    )
    if max(row_error, column_error) > config.marginal_tolerance:
        raise RuntimeError(
            "Strict balanced layer OT violated its marginals: "
            f"row_error={row_error:.3e} column_error={column_error:.3e} "
            f"tolerance={config.marginal_tolerance:.3e}"
        )
    selected_mask = route > float(config.support_tolerance)
    return LayerRouteResult(
        dense_route=route.clone().contiguous(),
        route=route.contiguous(),
        selected_mask=selected_mask.contiguous(),
        entropy=entropy,
        effective_source_count=entropy.exp(),
        row_marginal_error=row_error,
        column_marginal_error=column_error,
        transport_objective=objective,
        solver=config.solver,
    )


def _compute_row_softmax_topk(
    cost64: torch.Tensor,
    config: RouteConfig,
) -> LayerRouteResult:
    cost = cost64.to(dtype=torch.float32)
    dense_route = torch.softmax(-cost / float(config.temperature), dim=1)
    route = dense_route.clone()
    top_s = config.top_source_layers
    if top_s is not None and top_s < cost.shape[1]:
        indices = torch.topk(route, k=top_s, dim=1, largest=True).indices
        selected_mask = torch.zeros_like(route, dtype=torch.bool)
        selected_mask.scatter_(1, indices, True)
        route = torch.where(selected_mask, route, torch.zeros_like(route))
        route = route / route.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(route.dtype).tiny
        )
    else:
        selected_mask = route > float(config.support_tolerance)

    route, entropy, row_error, column_error, objective = _route_statistics(
        route.to(dtype=torch.float64), cost64
    )
    return LayerRouteResult(
        dense_route=dense_route.contiguous(),
        route=route.contiguous(),
        selected_mask=selected_mask.contiguous(),
        entropy=entropy,
        effective_source_count=entropy.exp(),
        row_marginal_error=row_error,
        column_marginal_error=column_error,
        transport_objective=objective,
        solver=config.solver,
    )


@torch.no_grad()
def compute_layer_route(
    cost: torch.Tensor,
    config: RouteConfig | None = None,
) -> LayerRouteResult:
    config = config or RouteConfig()
    config.validate()
    if cost.ndim != 2 or cost.shape[0] == 0 or cost.shape[1] == 0:
        raise ValueError("layer cost must be a non-empty L x M matrix")
    cost64 = cost.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(cost64).all():
        raise ValueError("layer cost contains non-finite values")

    if config.solver == "balanced_exact":
        return _compute_balanced_exact(cost64, config)
    if config.solver == "row_softmax_topk":
        return _compute_row_softmax_topk(cost64, config)
    raise AssertionError(f"unhandled route solver: {config.solver}")
