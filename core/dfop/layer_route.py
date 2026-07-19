from __future__ import annotations

import numpy as np
import torch

from .config import RouteConfig
from .types import LayerRouteResult


@torch.no_grad()
def compute_layer_route(
    cost: torch.Tensor,
    config: RouteConfig | None = None,
) -> LayerRouteResult:
    config = config or RouteConfig()
    config.validate()
    if cost.ndim != 2 or cost.shape[0] == 0 or cost.shape[1] == 0:
        raise ValueError("layer cost must be a non-empty L x M matrix")
    cost = cost.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(cost).all():
        raise ValueError("layer cost contains non-finite values")

    try:
        from scipy.optimize import linprog
        from scipy.sparse import lil_matrix
    except ImportError as error:
        raise RuntimeError(
            "Strict balanced layer OT requires scipy. Install requirements.txt."
        ) from error

    target_layers, source_layers = cost.shape
    variable_count = target_layers * source_layers
    # One equality per target and source layer. The final equality is redundant,
    # but HiGHS removes this harmless transport-polytope redundancy reliably.
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
        cost.numpy().reshape(-1),
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
    # The externally consumed route is row stochastic. Scaling the balanced
    # coupling by L preserves exact row sums and gives every source column L/M.
    route64 = coupling * float(target_layers)
    expected_rows = torch.ones(target_layers, dtype=torch.float64)
    expected_columns = torch.full(
        (source_layers,),
        float(target_layers) / float(source_layers),
        dtype=torch.float64,
    )
    row_error = float((route64.sum(dim=1) - expected_rows).abs().max())
    column_error = float((route64.sum(dim=0) - expected_columns).abs().max())
    if max(row_error, column_error) > config.marginal_tolerance:
        raise RuntimeError(
            "Strict balanced layer OT violated its marginals: "
            f"row_error={row_error:.3e} column_error={column_error:.3e} "
            f"tolerance={config.marginal_tolerance:.3e}"
        )

    route = route64.to(dtype=torch.float32)
    dense_route = route.clone()
    selected_mask = route > float(config.support_tolerance)

    safe = route.clamp_min(torch.finfo(route.dtype).tiny)
    entropy = -(route * safe.log()).sum(dim=1)
    effective_source_count = entropy.exp()
    return LayerRouteResult(
        dense_route=dense_route.contiguous(),
        route=route.contiguous(),
        selected_mask=selected_mask,
        entropy=entropy,
        effective_source_count=effective_source_count,
        row_marginal_error=row_error,
        column_marginal_error=column_error,
        transport_objective=float(result.fun),
        solver=config.solver,
    )
