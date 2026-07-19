from __future__ import annotations

import torch

from .config import CoreScaleConfig
from .types import OTProcrustesResult, PairCoreResult, SVDRecord


COUPLING_MARGINAL_TOLERANCE = 1.0e-4


@torch.no_grad()
def compute_pair_core(
    target: SVDRecord,
    source: SVDRecord,
    output_result: OTProcrustesResult,
    input_result: OTProcrustesResult,
    output_target_mass: torch.Tensor,
    input_target_mass: torch.Tensor,
    config: CoreScaleConfig | None = None,
) -> PairCoreResult:
    config = config or CoreScaleConfig()
    config.validate()
    if target.rank != source.rank:
        raise ValueError("target and source records must use the same fixed rank")

    pi_out = output_result.coupling.to(dtype=torch.float32)
    pi_in = input_result.coupling.to(dtype=torch.float32)
    output_target_mass = output_target_mass.to(pi_out.device, torch.float32)
    input_target_mass = input_target_mass.to(pi_in.device, torch.float32)
    if pi_out.shape != (target.shape[0], source.shape[0]):
        raise ValueError("output coupling shape does not match SVD output dimensions")
    if pi_in.shape != (target.shape[1], source.shape[1]):
        raise ValueError("input coupling shape does not match SVD input dimensions")
    if float((pi_out.sum(1) - output_target_mass).abs().max()) > COUPLING_MARGINAL_TOLERANCE:
        raise ValueError("output coupling rows do not match target masses")
    if float((pi_in.sum(1) - input_target_mass).abs().max()) > COUPLING_MARGINAL_TOLERANCE:
        raise ValueError("input coupling rows do not match target masses")

    u_a = target.u.to(device=pi_out.device, dtype=torch.float32)
    u_b = source.u.to(device=pi_out.device, dtype=torch.float32)
    v_a = target.v.to(device=pi_in.device, dtype=torch.float32)
    v_b = source.v.to(device=pi_in.device, dtype=torch.float32)
    sigma_b = torch.diag(source.s.to(device=pi_out.device, dtype=torch.float32))

    # U_A^T diag(a_out)^-1 Pi_out U_B without materializing the dense map T.
    mapped_u_b = (pi_out @ u_b) / output_target_mass.clamp_min(config.eps).unsqueeze(1)
    mapped_v_b = (pi_in @ v_b) / input_target_mass.clamp_min(config.eps).unsqueeze(1)
    align_out = u_a.transpose(0, 1) @ mapped_u_b
    align_in = v_a.transpose(0, 1) @ mapped_v_b
    core = align_out @ sigma_b @ align_in.transpose(0, 1)

    target_norm = torch.linalg.vector_norm(target.s.to(core.device, torch.float32))
    core_norm = torch.linalg.matrix_norm(core)
    relative = float(core_norm / target_norm.clamp_min(config.eps))
    if not torch.isfinite(core).all() or relative < config.minimum_relative_norm:
        return PairCoreResult(
            core=core,
            calibrated_core=torch.zeros_like(core),
            scale=0.0,
            output_result=output_result,
            input_result=input_result,
            valid=False,
            skip_reason="core norm is non-finite or below minimum_relative_norm",
        )

    scale = 1.0
    if config.enabled:
        raw_scale = float(target_norm / core_norm.clamp_min(config.eps))
        scale = min(config.gamma_max, max(config.gamma_min, raw_scale))
    calibrated = core * scale
    return PairCoreResult(
        core=core.contiguous(),
        calibrated_core=calibrated.contiguous(),
        scale=scale,
        output_result=output_result,
        input_result=input_result,
        valid=True,
    )
