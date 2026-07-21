from __future__ import annotations

import torch
import torch.nn.functional as functional
import torch.nn as nn

from .config import ACIConfig
from .types import AnchorResult


def deterministic_anchor_indices(vocabulary_size: int, count: int) -> torch.Tensor:
    if vocabulary_size <= 0 or count <= 0:
        raise ValueError("vocabulary_size and count must be positive")
    count = min(vocabulary_size, count)
    if count == vocabulary_size:
        return torch.arange(vocabulary_size, dtype=torch.long)
    # Integer midpoint sampling covers the full vocabulary without relying on
    # tokenizer text, token frequency, or a random calibration set.
    return ((2 * torch.arange(count, dtype=torch.long) + 1) * vocabulary_size // (2 * count))


@torch.no_grad()
def _cross_covariance(
    source_weight: torch.Tensor,
    reference_weight: torch.Tensor,
    indices: torch.Tensor,
    *,
    device: torch.device,
    chunk_size: int,
    eps: float,
) -> torch.Tensor:
    if source_weight.ndim != 2 or reference_weight.ndim != 2:
        raise ValueError("anchor weights must be matrices")
    if int(indices.max()) >= min(source_weight.shape[0], reference_weight.shape[0]):
        raise ValueError("anchor index exceeds the shared vocabulary")
    covariance = torch.zeros(
        (source_weight.shape[1], reference_weight.shape[1]),
        device=device,
        dtype=torch.float32,
    )
    cpu_indices = indices.cpu()
    for start in range(0, indices.numel(), chunk_size):
        selected = cpu_indices[start : start + chunk_size]
        source = source_weight.detach().index_select(0, selected).to(device, torch.float32)
        reference = reference_weight.detach().index_select(0, selected).to(
            device, torch.float32
        )
        source = functional.normalize(source, dim=1, eps=eps)
        reference = functional.normalize(reference, dim=1, eps=eps)
        covariance.add_(source.transpose(0, 1) @ reference)
    return covariance / float(indices.numel())


@torch.no_grad()
def _anchor_cosine(
    source_weight: torch.Tensor,
    reference_weight: torch.Tensor,
    projection: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> float:
    sample = indices[: min(1024, indices.numel())].cpu()
    source = source_weight.detach().index_select(0, sample).to(
        projection.device, torch.float32
    )
    reference = reference_weight.detach().index_select(0, sample).to(
        projection.device, torch.float32
    )
    projected = functional.normalize(source @ projection, dim=1, eps=eps)
    reference = functional.normalize(reference, dim=1, eps=eps)
    return float((projected * reference).sum(dim=1).mean())


@torch.no_grad()
def build_residual_anchor(
    source_embedding: nn.Embedding,
    reference_embedding: nn.Embedding,
    source_head: nn.Linear | None,
    reference_head: nn.Linear | None,
    config: ACIConfig,
    *,
    device: torch.device,
) -> AnchorResult:
    """Learn one global source-to-reference residual map from model weights.

    The rectangular orthogonal Procrustes problem is anchored only by shared
    vocabulary rows already stored in the checkpoints.  No tokenizer text or
    model input is constructed.
    """

    shared_vocabulary = min(
        source_embedding.weight.shape[0], reference_embedding.weight.shape[0]
    )
    indices = deterministic_anchor_indices(shared_vocabulary, config.anchor_tokens)
    covariance = _cross_covariance(
        source_embedding.weight,
        reference_embedding.weight,
        indices,
        device=device,
        chunk_size=config.anchor_chunk_size,
        eps=config.eps,
    )
    used_output = False
    if source_head is not None and reference_head is not None:
        output_vocabulary = min(source_head.weight.shape[0], reference_head.weight.shape[0])
        output_indices = indices[indices < output_vocabulary]
        if output_indices.numel() >= 2:
            covariance.add_(
                _cross_covariance(
                    source_head.weight,
                    reference_head.weight,
                    output_indices,
                    device=device,
                    chunk_size=config.anchor_chunk_size,
                    eps=config.eps,
                )
            )
            covariance.mul_(0.5)
            used_output = True

    if covariance.shape[0] < covariance.shape[1]:
        raise ValueError(
            "ACI currently contracts a wider source into a narrower reference; "
            f"got source hidden={covariance.shape[0]} reference hidden={covariance.shape[1]}"
        )
    u, _, vh = torch.linalg.svd(covariance, full_matrices=False)
    projection = (u @ vh).contiguous()
    sketch_dim = min(config.ffn_sketch_dim, vh.shape[0])
    reference_basis = vh.transpose(0, 1)[:, :sketch_dim].contiguous()
    input_cosine = _anchor_cosine(
        source_embedding.weight,
        reference_embedding.weight,
        projection,
        indices,
        config.eps,
    )
    output_cosine = None
    if used_output and source_head is not None and reference_head is not None:
        output_cosine = _anchor_cosine(
            source_head.weight,
            reference_head.weight,
            projection,
            indices[indices < min(source_head.weight.shape[0], reference_head.weight.shape[0])],
            config.eps,
        )
    return AnchorResult(
        source_to_reference=projection,
        reference_sketch_basis=reference_basis,
        input_anchor_cosine=input_cosine,
        output_anchor_cosine=output_cosine,
        anchor_count=indices.numel(),
    )
