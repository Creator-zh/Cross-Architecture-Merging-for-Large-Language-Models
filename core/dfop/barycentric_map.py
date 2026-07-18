from __future__ import annotations

import torch


def coupling_to_barycentric_map(
    coupling: torch.Tensor,
    target_mass: torch.Tensor,
    *,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    if coupling.ndim != 2 or target_mass.shape != (coupling.shape[0],):
        raise ValueError("target_mass must match coupling rows")
    target_mass = target_mass.to(device=coupling.device, dtype=coupling.dtype)
    if (target_mass <= 0).any():
        raise ValueError("target masses must be strictly positive")
    row_error = (coupling.sum(dim=1) - target_mass).abs().max()
    if float(row_error) > 1.0e-4:
        raise ValueError(f"coupling rows do not match target masses: {float(row_error):.3e}")
    return (coupling / target_mass.clamp_min(eps).unsqueeze(1)).contiguous()
