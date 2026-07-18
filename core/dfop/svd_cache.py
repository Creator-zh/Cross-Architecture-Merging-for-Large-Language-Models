from __future__ import annotations

from typing import Optional

import torch

from .config import SVDConfig
from .types import SVDRecord


def _as_compute_tensor(weight: torch.Tensor) -> torch.Tensor:
    if not weight.is_floating_point():
        raise TypeError("SVD input weight must be floating point")
    return weight.detach().to(dtype=torch.float32)


@torch.no_grad()
def exact_svd(weight: torch.Tensor, rank: int) -> SVDRecord:
    matrix = _as_compute_tensor(weight)
    limit = min(matrix.shape)
    if rank <= 0 or rank > limit:
        raise ValueError(f"rank must lie in [1, {limit}], got {rank}")
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    return SVDRecord(
        u=u[:, :rank].contiguous(),
        s=s[:rank].contiguous(),
        v=vh[:rank, :].transpose(0, 1).contiguous(),
        shape=(int(matrix.shape[0]), int(matrix.shape[1])),
        rank=rank,
        metadata={"algorithm": "exact"},
    )


@torch.no_grad()
def randomized_svd(
    weight: torch.Tensor,
    rank: int,
    *,
    oversample: int = 16,
    power_iterations: int = 2,
    seed: int = 42,
) -> SVDRecord:
    matrix = _as_compute_tensor(weight)
    m, n = matrix.shape
    limit = min(m, n)
    if rank <= 0 or rank > limit:
        raise ValueError(f"rank must lie in [1, {limit}], got {rank}")
    sketch_rank = min(limit, rank + max(0, oversample))

    generator_device = matrix.device if matrix.device.type != "meta" else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    omega = torch.randn(
        n,
        sketch_rank,
        generator=generator,
        device=matrix.device,
        dtype=matrix.dtype,
    )
    y = matrix @ omega
    for _ in range(max(0, power_iterations)):
        # Re-orthogonalizing both half steps is more stable than forming
        # (W W^T)^q W Omega directly, especially for low-precision model
        # checkpoints with a steep spectrum.
        q_y, _ = torch.linalg.qr(y, mode="reduced")
        z = matrix.transpose(0, 1) @ q_y
        q_z, _ = torch.linalg.qr(z, mode="reduced")
        y = matrix @ q_z
    q, _ = torch.linalg.qr(y, mode="reduced")
    small = q.transpose(0, 1) @ matrix
    u_hat, s, vh = torch.linalg.svd(small, full_matrices=False)
    u = q @ u_hat
    return SVDRecord(
        u=u[:, :rank].contiguous(),
        s=s[:rank].contiguous(),
        v=vh[:rank, :].transpose(0, 1).contiguous(),
        shape=(int(m), int(n)),
        rank=rank,
        metadata={
            "algorithm": "randomized",
            "oversample": int(oversample),
            "power_iterations": int(power_iterations),
            "seed": int(seed),
        },
    )


@torch.no_grad()
def compute_svd_record(
    weight: torch.Tensor,
    rank: int,
    config: Optional[SVDConfig] = None,
    *,
    seed_offset: int = 0,
) -> SVDRecord:
    config = config or SVDConfig(rank_default=rank)
    if config.algorithm == "exact":
        record = exact_svd(weight, rank)
    else:
        record = randomized_svd(
            weight,
            rank,
            oversample=config.oversample,
            power_iterations=config.power_iterations,
            seed=config.seed + int(seed_offset),
        )
    record.metadata.update({"configured_rank": int(rank)})
    return record
