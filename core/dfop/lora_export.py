from __future__ import annotations

from dataclasses import dataclass

import torch

from .types import SVDRecord


@dataclass
class ExactLoRAFactors:
    """Factors satisfying ``delta = lora_b @ lora_a`` when scaling is one."""

    lora_a: torch.Tensor
    lora_b: torch.Tensor

    @property
    def rank(self) -> int:
        return int(self.lora_a.shape[0])

    def dense_delta(self) -> torch.Tensor:
        return self.lora_b @ self.lora_a


@torch.no_grad()
def exact_lora_factors(
    target_svd: SVDRecord,
    aggregate_core: torch.Tensor,
    *,
    beta: float,
    trust_coefficient: float,
) -> ExactLoRAFactors:
    """Factor the DFOP top-k update exactly as a rank-k LoRA update.

    PEFT convention multiplies ``B @ A`` by ``lora_alpha / rank``. To retain
    exact equality, set ``lora_alpha=rank`` or fold the inverse scaling into B.
    """

    if aggregate_core.shape != (target_svd.rank, target_svd.rank):
        raise ValueError("aggregate_core has the wrong shape")
    if not 0 <= beta <= 1:
        raise ValueError("beta must lie in [0, 1]")
    if not 0 <= trust_coefficient <= 1:
        raise ValueError("trust_coefficient must lie in [0, 1]")

    device = aggregate_core.device
    u = target_svd.u.to(device=device, dtype=torch.float32)
    v = target_svd.v.to(device=device, dtype=torch.float32)
    sigma = torch.diag(target_svd.s.to(device=device, dtype=torch.float32))
    core_delta = float(beta) * float(trust_coefficient) * (
        aggregate_core.to(dtype=torch.float32) - sigma
    )
    return ExactLoRAFactors(
        lora_a=v.transpose(0, 1).contiguous(),
        lora_b=(u @ core_delta).contiguous(),
    )
