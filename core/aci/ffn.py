from __future__ import annotations

import torch
import torch.nn.functional as functional

from .config import ACIConfig
from .registry import BlockLinears
from .types import FFNMatchResult


def _component_signature(rows: torch.Tensor, basis: torch.Tensor, eps: float) -> torch.Tensor:
    return functional.normalize(rows @ basis, dim=1, eps=eps)


@torch.no_grad()
def ffn_signature(
    block: BlockLinears,
    residual_basis: torch.Tensor,
    *,
    device: torch.device,
    eps: float,
) -> torch.Tensor:
    gate = block.gate.weight.detach().to(device, torch.float32)
    gate_signature = _component_signature(gate, residual_basis, eps)
    del gate
    up = block.up.weight.detach().to(device, torch.float32)
    up_signature = _component_signature(up, residual_basis, eps)
    del up
    down = block.down.weight.detach().to(device, torch.float32).transpose(0, 1)
    down_signature = _component_signature(down, residual_basis, eps)
    del down
    return functional.normalize(
        torch.cat((gate_signature, up_signature, down_signature), dim=1),
        dim=1,
        eps=eps,
    )


@torch.no_grad()
def _unique_greedy_match(
    target_signature: torch.Tensor,
    source_signature: torch.Tensor,
    *,
    candidate_k: int,
) -> FFNMatchResult:
    target_count, source_count = target_signature.shape[0], source_signature.shape[0]
    if target_count > source_count:
        raise ValueError(
            "ACI currently contracts a wider source FFN; "
            f"target neurons={target_count} source neurons={source_count}"
        )
    k = min(candidate_k, source_count)
    candidate_indices = []
    candidate_scores = []
    for start in range(0, target_count, 256):
        similarity = target_signature[start : start + 256] @ source_signature.transpose(0, 1)
        scores, indices = torch.topk(similarity, k=k, dim=1, largest=True, sorted=True)
        candidate_indices.append(indices.cpu())
        candidate_scores.append(scores.cpu())
    candidates = torch.cat(candidate_indices, dim=0)
    scores = torch.cat(candidate_scores, dim=0)

    order = torch.argsort(scores[:, 0], descending=True).tolist()
    assignment = torch.full((target_count,), -1, dtype=torch.long)
    selected_scores = torch.full((target_count,), float("-inf"), dtype=torch.float32)
    used: set[int] = set()
    for target_index in order:
        for rank in range(k):
            source_index = int(candidates[target_index, rank])
            if source_index not in used:
                assignment[target_index] = source_index
                selected_scores[target_index] = scores[target_index, rank]
                used.add(source_index)
                break

    # Candidate collisions are uncommon because the source is wider.  Resolve
    # any remainder exactly instead of silently reusing a source neuron.
    for target_index in torch.nonzero(assignment < 0, as_tuple=False).flatten().tolist():
        similarity = (
            target_signature[target_index : target_index + 1]
            @ source_signature.transpose(0, 1)
        ).flatten()
        if used:
            used_indices = torch.tensor(sorted(used), device=similarity.device)
            similarity[used_indices] = float("-inf")
        source_index = int(torch.argmax(similarity))
        assignment[target_index] = source_index
        selected_scores[target_index] = float(similarity[source_index])
        used.add(source_index)

    return FFNMatchResult(
        source_indices=assignment,
        mean_cosine=float(selected_scores.mean()),
        minimum_cosine=float(selected_scores.min()),
        reused_sources=target_count - len(set(assignment.tolist())),
    )


@torch.no_grad()
def contract_ffn(
    source: BlockLinears,
    reference: BlockLinears,
    projection: torch.Tensor,
    reference_basis: torch.Tensor,
    config: ACIConfig,
) -> tuple[dict[str, torch.Tensor], FFNMatchResult]:
    device = projection.device
    if reference_basis.shape[0] != projection.shape[1]:
        raise ValueError("reference sketch basis does not match target hidden size")
    source_basis = projection @ reference_basis
    target_signature = ffn_signature(
        reference,
        reference_basis,
        device=device,
        eps=config.eps,
    )
    source_signature = ffn_signature(
        source,
        source_basis,
        device=device,
        eps=config.eps,
    )
    match = _unique_greedy_match(
        target_signature,
        source_signature,
        candidate_k=config.ffn_candidate_k,
    )
    indices = match.source_indices.cpu()
    gate = source.gate.weight.detach().index_select(0, indices).to(device, torch.float32)
    up = source.up.weight.detach().index_select(0, indices).to(device, torch.float32)
    down = source.down.weight.detach().index_select(1, indices).to(device, torch.float32)
    contracted = {
        "gate": (gate @ projection).contiguous(),
        "up": (up @ projection).contiguous(),
        "down": (projection.transpose(0, 1) @ down).contiguous(),
    }
    return contracted, match
