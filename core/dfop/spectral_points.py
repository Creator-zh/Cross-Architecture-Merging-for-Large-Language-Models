from __future__ import annotations

import math

import torch

from .config import SpectralPointConfig
from .types import SVDRecord


def normalized_spectrum(
    singular_values: torch.Tensor,
    *,
    power: float = 1.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    if singular_values.ndim != 1:
        raise ValueError("singular_values must be a vector")
    powered = singular_values.clamp_min(0).pow(power)
    norm = torch.linalg.vector_norm(powered)
    if not torch.isfinite(norm) or float(norm) <= eps:
        raise ValueError("singular spectrum has zero or non-finite norm")
    return powered / norm


def build_spectral_points(
    basis: torch.Tensor,
    singular_values: torch.Tensor,
    config: SpectralPointConfig | None = None,
) -> torch.Tensor:
    config = config or SpectralPointConfig()
    config.validate()
    if basis.ndim != 2 or singular_values.ndim != 1:
        raise ValueError("basis must be a matrix and singular_values a vector")
    if basis.shape[1] != singular_values.shape[0]:
        raise ValueError("basis columns and singular value count must match")

    spectrum = normalized_spectrum(
        singular_values,
        power=config.sigma_power,
        eps=config.eps,
    )
    points = basis.to(torch.float32) * spectrum.to(torch.float32).unsqueeze(0)
    if config.center:
        points = points - points.mean(dim=0, keepdim=True)
        energy = points.square().sum() / max(1, points.shape[0])
        points = points / torch.sqrt(energy.clamp_min(config.eps))
    elif config.width_normalization:
        points = points * math.sqrt(int(points.shape[0]))
    return points.contiguous()


def residual_basis(record: SVDRecord, residual_side: str) -> torch.Tensor:
    if residual_side == "input":
        return record.v
    if residual_side == "output":
        return record.u
    raise ValueError("residual_side must be 'input' or 'output'")


def build_record_points(
    record: SVDRecord,
    residual_side: str,
    config: SpectralPointConfig | None = None,
) -> torch.Tensor:
    return build_spectral_points(residual_basis(record, residual_side), record.s, config)


def weighted_mean_energy(points: torch.Tensor, masses: torch.Tensor | None = None) -> float:
    if points.ndim != 2:
        raise ValueError("points must be a matrix")
    if masses is None:
        masses = torch.full(
            (points.shape[0],),
            1.0 / points.shape[0],
            dtype=points.dtype,
            device=points.device,
        )
    return float((masses * points.square().sum(dim=1)).sum())

