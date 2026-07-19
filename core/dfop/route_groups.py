from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


GROUP_WEIGHTS: Mapping[str, Mapping[str, float]] = {
    "qk": {"q": 0.5, "k": 0.5},
    "vo": {"v": 0.5, "o": 0.5},
    # gate/up are two observations of the residual-input side; down observes
    # the residual-output side. Give the two sides equal total weight.
    "ffn": {"gate": 0.25, "up": 0.25, "down": 0.5},
}


def build_route_groups(
    modules: Sequence[str],
    grouping: str,
) -> dict[str, tuple[str, ...]]:
    modules = tuple(modules)
    if grouping == "independent":
        return {module: (module,) for module in modules}
    if grouping != "qk_vo_ffn":
        raise ValueError(f"unknown route grouping: {grouping}")

    result: dict[str, tuple[str, ...]] = {}
    assigned: set[str] = set()
    for group_name, weights in GROUP_WEIGHTS.items():
        active = tuple(module for module in weights if module in modules)
        if active:
            result[group_name] = active
            assigned.update(active)
    for module in modules:
        if module not in assigned:
            result[module] = (module,)
    return result


def robust_normalize_cost(cost: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    if cost.ndim != 2 or not torch.isfinite(cost).all():
        raise ValueError("cost must be a finite matrix")
    value = cost.detach().to(device="cpu", dtype=torch.float64)
    median = value.median()
    scale = (value - median).abs().median()
    if float(scale) <= eps:
        scale = value.std(unbiased=False)
    if float(scale) <= eps:
        return torch.zeros_like(value, dtype=torch.float32)
    return ((value - median) / scale).to(dtype=torch.float32)


def aggregate_group_cost(
    group_name: str,
    modules: Sequence[str],
    layer_costs: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    modules = tuple(modules)
    if not modules:
        raise ValueError("cannot aggregate an empty route group")
    shapes = {tuple(layer_costs[module].shape) for module in modules}
    if len(shapes) != 1:
        raise ValueError(f"route group {group_name} has inconsistent cost shapes")

    if len(modules) == 1:
        return layer_costs[modules[0]].detach().cpu().to(dtype=torch.float32)

    configured = GROUP_WEIGHTS.get(group_name, {})
    raw_weights = {module: float(configured.get(module, 1.0)) for module in modules}
    total = sum(raw_weights.values())
    if total <= 0:
        raise ValueError(f"route group {group_name} has invalid weights")
    aggregate = torch.zeros(next(iter(shapes)), dtype=torch.float32)
    for module in modules:
        aggregate.add_(
            robust_normalize_cost(layer_costs[module]),
            alpha=raw_weights[module] / total,
        )
    return aggregate
