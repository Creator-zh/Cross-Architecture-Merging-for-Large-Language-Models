from __future__ import annotations

import torch

from .config import SinkhornConfig
from .types import SinkhornResult


def uniform_mass(size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if size <= 0:
        raise ValueError("mass size must be positive")
    return torch.full((size,), 1.0 / size, device=device, dtype=dtype)


def _validate_inputs(
    cost: torch.Tensor,
    mass_x: torch.Tensor,
    mass_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cost.ndim != 2:
        raise ValueError("cost must be a matrix")
    if mass_x.shape != (cost.shape[0],) or mass_y.shape != (cost.shape[1],):
        raise ValueError("mass shapes must match cost rows and columns")
    cost = cost.to(dtype=torch.float32)
    mass_x = mass_x.to(device=cost.device, dtype=cost.dtype)
    mass_y = mass_y.to(device=cost.device, dtype=cost.dtype)
    if not torch.isfinite(cost).all():
        raise ValueError("cost contains non-finite values")
    if (mass_x <= 0).any() or (mass_y <= 0).any():
        raise ValueError("all masses must be strictly positive")
    if not torch.isclose(mass_x.sum(), torch.ones((), device=cost.device), atol=1e-5):
        raise ValueError("mass_x must sum to one")
    if not torch.isclose(mass_y.sum(), torch.ones((), device=cost.device), atol=1e-5):
        raise ValueError("mass_y must sum to one")
    return cost, mass_x, mass_y


def coupling_marginal_error(
    coupling: torch.Tensor,
    mass_x: torch.Tensor,
    mass_y: torch.Tensor,
) -> float:
    row_error = (coupling.sum(dim=1) - mass_x).abs().max()
    col_error = (coupling.sum(dim=0) - mass_y).abs().max()
    return float(torch.maximum(row_error, col_error))


@torch.no_grad()
def log_sinkhorn(
    cost: torch.Tensor,
    mass_x: torch.Tensor,
    mass_y: torch.Tensor,
    config: SinkhornConfig | None = None,
) -> SinkhornResult:
    """Solve balanced entropic OT in the log domain.

    The returned coupling minimizes <cost, pi> - epsilon H(pi) under the
    supplied positive marginals. All arithmetic is FP32 even when model
    weights are stored in lower precision.
    """

    config = config or SinkhornConfig()
    config.validate()
    cost, mass_x, mass_y = _validate_inputs(cost, mass_x, mass_y)

    log_kernel = -cost / float(config.entropy)
    log_a = mass_x.log()
    log_b = mass_y.log()
    log_u = torch.zeros_like(mass_x)
    log_v = torch.zeros_like(mass_y)
    coupling = torch.empty_like(cost)
    marginal_error = float("inf")
    converged = False
    iterations = 0

    for iteration in range(1, config.max_iterations + 1):
        log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
        iterations = iteration

        should_check = iteration % config.check_interval == 0 or iteration == config.max_iterations
        if should_check:
            coupling = torch.exp(log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0))
            marginal_error = coupling_marginal_error(coupling, mass_x, mass_y)
            if marginal_error <= config.tolerance:
                converged = True
                break

    coupling = torch.exp(log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0))
    marginal_error = coupling_marginal_error(coupling, mass_x, mass_y)
    converged = converged or marginal_error <= config.tolerance
    return SinkhornResult(
        coupling=coupling.contiguous(),
        marginal_error=marginal_error,
        iterations=iterations,
        converged=converged,
    )
