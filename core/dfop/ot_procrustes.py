from __future__ import annotations

import math

import torch

from .config import OTProcrustesConfig
from .sinkhorn import log_sinkhorn, uniform_mass
from .types import OTProcrustesResult


def pairwise_rotated_cost(
    x: torch.Tensor,
    y: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """Return 0.5 * ||x_i - R y_j||^2 without materializing differences."""

    cross = x @ rotation @ y.transpose(0, 1)
    cost = 0.5 * (
        x.square().sum(dim=1, keepdim=True)
        + y.square().sum(dim=1).unsqueeze(0)
        - 2.0 * cross
    )
    return cost.clamp_min(0).contiguous()


def _orthogonal_from_cross(cross: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(cross, full_matrices=False)
    return (u @ vh).contiguous()


def _random_orthogonal(
    rank: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    matrix = torch.randn(rank, rank, generator=generator, device=device, dtype=dtype)
    q, r = torch.linalg.qr(matrix)
    signs = torch.where(torch.diag(r) >= 0, 1.0, -1.0).to(dtype=dtype)
    return (q * signs.unsqueeze(0)).contiguous()


def _entropy(coupling: torch.Tensor) -> torch.Tensor:
    safe = coupling.clamp_min(torch.finfo(coupling.dtype).tiny)
    return -(coupling * (safe.log() - 1.0)).sum()


def _validate_points(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be point matrices")
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must share the spectral coordinate dimension")
    if x.shape[0] == 0 or y.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("point matrices must be non-empty")
    if x.device != y.device:
        raise ValueError("x and y must be on the same device")
    x = x.to(dtype=torch.float32)
    y = y.to(dtype=torch.float32)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("point matrices contain non-finite values")
    return x, y


@torch.no_grad()
def solve_ot_procrustes(
    x: torch.Tensor,
    y: torch.Tensor,
    mass_x: torch.Tensor | None = None,
    mass_y: torch.Tensor | None = None,
    config: OTProcrustesConfig | None = None,
) -> OTProcrustesResult:
    """Alternating balanced OT and orthogonal Procrustes.

    Entropy is used only by the Sinkhorn subproblem. The result's
    ``geometric_cost`` excludes entropy and is the layer dissimilarity used by
    DFOP.
    """

    config = config or OTProcrustesConfig()
    config.validate()
    x, y = _validate_points(x, y)
    mass_x = mass_x if mass_x is not None else uniform_mass(
        x.shape[0], device=x.device, dtype=x.dtype
    )
    mass_y = mass_y if mass_y is not None else uniform_mass(
        y.shape[0], device=y.device, dtype=y.dtype
    )
    mass_x = mass_x.to(device=x.device, dtype=x.dtype)
    mass_y = mass_y.to(device=x.device, dtype=x.dtype)

    best: OTProcrustesResult | None = None
    rank = x.shape[1]
    for restart in range(config.restarts):
        if restart == 0:
            rotation = torch.eye(rank, device=x.device, dtype=x.dtype)
        else:
            rotation = _random_orthogonal(
                rank,
                device=x.device,
                dtype=x.dtype,
                seed=config.seed + restart,
            )

        previous = math.inf
        alternating_iterations = 0
        converged = False
        for alternating_iteration in range(1, config.max_alternating_iterations + 1):
            cost = pairwise_rotated_cost(x, y, rotation)
            sinkhorn = log_sinkhorn(cost, mass_x, mass_y, config.sinkhorn)
            cross = x.transpose(0, 1) @ sinkhorn.coupling @ y
            rotation = _orthogonal_from_cross(cross)
            updated_cost = pairwise_rotated_cost(x, y, rotation)
            geometric = float((sinkhorn.coupling * updated_cost).sum())
            alternating_iterations = alternating_iteration
            if math.isfinite(previous):
                relative = abs(previous - geometric) / max(1.0, abs(previous))
                if relative <= config.alternating_tolerance:
                    converged = True
                    break
            previous = geometric

        # One final Procrustes update followed by one final OT update makes the
        # returned coupling exactly optimal for the returned rotation within
        # the configured Sinkhorn tolerance.
        final_cost = pairwise_rotated_cost(x, y, rotation)
        sinkhorn = log_sinkhorn(final_cost, mass_x, mass_y, config.sinkhorn)
        cross = x.transpose(0, 1) @ sinkhorn.coupling @ y
        rotation = _orthogonal_from_cross(cross)
        final_cost = pairwise_rotated_cost(x, y, rotation)
        sinkhorn = log_sinkhorn(final_cost, mass_x, mass_y, config.sinkhorn)
        geometric = float((sinkhorn.coupling * final_cost).sum())
        regularized = float(
            (sinkhorn.coupling * final_cost).sum()
            - float(config.sinkhorn.entropy) * _entropy(sinkhorn.coupling)
        )

        candidate = OTProcrustesResult(
            coupling=sinkhorn.coupling,
            rotation=rotation,
            geometric_cost=geometric,
            regularized_objective=regularized,
            marginal_error=sinkhorn.marginal_error,
            sinkhorn_iterations=sinkhorn.iterations,
            alternating_iterations=alternating_iterations,
            restart=restart,
            converged=converged and sinkhorn.converged,
        )
        if best is None or candidate.geometric_cost < best.geometric_cost:
            best = candidate

    if best is None:
        raise RuntimeError("OT-Procrustes did not produce a result")
    return best
