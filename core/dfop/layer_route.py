from __future__ import annotations

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
    cost = cost.to(dtype=torch.float32)
    if not torch.isfinite(cost).all():
        raise ValueError("layer cost contains non-finite values")

    dense_route = torch.softmax(-cost / float(config.temperature), dim=1)
    route = dense_route.clone()
    selected_mask = None
    top_s = config.top_source_layers
    if top_s is not None and top_s < cost.shape[1]:
        indices = torch.topk(route, k=top_s, dim=1, largest=True).indices
        selected_mask = torch.zeros_like(route, dtype=torch.bool)
        selected_mask.scatter_(1, indices, True)
        route = torch.where(selected_mask, route, torch.zeros_like(route))
        route = route / route.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(route.dtype).tiny
        )

    safe = route.clamp_min(torch.finfo(route.dtype).tiny)
    entropy = -(route * safe.log()).sum(dim=1)
    effective_source_count = entropy.exp()
    return LayerRouteResult(
        dense_route=dense_route.contiguous(),
        route=route.contiguous(),
        selected_mask=selected_mask,
        entropy=entropy,
        effective_source_count=effective_source_count,
    )
