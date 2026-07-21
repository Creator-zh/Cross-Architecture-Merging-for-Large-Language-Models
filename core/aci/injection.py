from __future__ import annotations

import torch

from .types import InjectionResult


@torch.no_grad()
def inject_protected_delta(
    target_weight: torch.Tensor,
    reference_weight: torch.Tensor,
    compressed_source_weight: torch.Tensor,
    *,
    beta: float,
    eps: float = 1.0e-8,
) -> InjectionResult:
    """Apply ``target + beta * (calibrated_source - reference)`` safely.

    Calibrating the contracted source to the reference Frobenius norm removes
    the deterministic energy loss caused by width contraction.  ``beta`` also
    caps the final relative change, so there is no independent trust-ratio
    hyperparameter.
    """

    if tuple(target_weight.shape) != tuple(reference_weight.shape):
        raise ValueError("target and reference weights must have identical shapes")
    if tuple(target_weight.shape) != tuple(compressed_source_weight.shape):
        raise ValueError("compressed source does not match target shape")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0, 1]")
    device = compressed_source_weight.device
    target = target_weight.detach().to(device, torch.float32)
    reference = reference_weight.detach().to(device, torch.float32)
    source = compressed_source_weight.to(device, torch.float32)
    reference_norm = torch.linalg.vector_norm(reference)
    source_norm = torch.linalg.vector_norm(source)
    scale = float(reference_norm / source_norm.clamp_min(eps))
    raw_delta = float(beta) * (scale * source - reference)
    target_norm = torch.linalg.vector_norm(target).clamp_min(eps)
    delta_norm = torch.linalg.vector_norm(raw_delta)
    trust = 1.0
    if beta > 0.0 and float(delta_norm) > 0.0:
        trust = min(1.0, float(beta) * float(target_norm) / float(delta_norm))
    delta = trust * raw_delta
    relative = float(torch.linalg.vector_norm(delta) / target_norm)
    return InjectionResult(
        weight=(target + delta).to(dtype=target_weight.dtype).contiguous(),
        delta=delta.contiguous(),
        calibration_scale=scale,
        trust_coefficient=trust,
        relative_update_norm=relative,
    )
