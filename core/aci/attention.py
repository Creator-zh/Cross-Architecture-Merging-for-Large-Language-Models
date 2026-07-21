from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from .registry import BlockLinears


@dataclass(frozen=True)
class AttentionGeometry:
    hidden_size: int
    query_heads: int
    kv_heads: int
    head_dim: int


@dataclass(frozen=True)
class AttentionMatchResult:
    group_assignment: torch.Tensor
    query_assignment: torch.Tensor
    mean_group_cosine: float
    minimum_group_cosine: float
    mean_query_cosine: float
    minimum_query_cosine: float


def attention_geometry(model) -> AttentionGeometry:
    config = getattr(model, "config", None)
    if config is None:
        raise AttributeError("model.config is required for structure-aware attention contraction")
    hidden = int(getattr(config, "hidden_size"))
    query_heads = int(getattr(config, "num_attention_heads"))
    kv_heads = int(getattr(config, "num_key_value_heads", query_heads))
    head_dim = int(getattr(config, "head_dim", hidden // query_heads))
    if hidden <= 0 or query_heads <= 0 or kv_heads <= 0 or head_dim <= 0:
        raise ValueError("invalid attention geometry")
    return AttentionGeometry(hidden, query_heads, kv_heads, head_dim)


def rope_frequency_indices(source_head_dim: int, target_head_dim: int) -> torch.Tensor:
    """Select source coordinates whose RoPE base frequencies equal the target's.

    Hugging Face Llama rotates corresponding coordinates in the first and
    second half of each head.  For 128 -> 64 this selects
    ``[0,2,...,62, 64,66,...,126]``.
    """

    if source_head_dim % 2 or target_head_dim % 2:
        raise ValueError("Llama RoPE head dimensions must be even")
    if source_head_dim < target_head_dim or source_head_dim % target_head_dim:
        raise ValueError(
            "source head_dim must be an integer multiple of target head_dim"
        )
    ratio = source_head_dim // target_head_dim
    target_half = target_head_dim // 2
    source_half = source_head_dim // 2
    first = torch.arange(target_half, dtype=torch.long) * ratio
    return torch.cat((first, first + source_half))


def validate_attention_pair(
    source: AttentionGeometry, reference: AttentionGeometry
) -> None:
    if source.query_heads != reference.query_heads:
        raise ValueError(
            "ACI's Llama contraction requires equal query-head counts; "
            f"source={source.query_heads} reference={reference.query_heads}"
        )
    if source.kv_heads != reference.kv_heads:
        raise ValueError(
            "ACI's Llama contraction requires equal KV-head counts; "
            f"source={source.kv_heads} reference={reference.kv_heads}"
        )
    rope_frequency_indices(source.head_dim, reference.head_dim)


def _maximum_assignment(similarity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact target-row to source-column assignment for small head groups."""

    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("head matching requires a square similarity matrix")
    size = similarity.shape[0]
    if size > 16:
        raise ValueError("exact head matching is limited to at most 16 groups")
    values = similarity.detach().cpu()
    states: dict[int, tuple[float, list[int]]] = {0: (0.0, [])}
    for row in range(size):
        next_states: dict[int, tuple[float, list[int]]] = {}
        for mask, (score, assignment) in states.items():
            for column in range(size):
                bit = 1 << column
                if mask & bit:
                    continue
                new_mask = mask | bit
                new_score = score + float(values[row, column])
                previous = next_states.get(new_mask)
                if previous is None or new_score > previous[0]:
                    next_states[new_mask] = (new_score, assignment + [column])
        states = next_states
    assignment = torch.tensor(states[(1 << size) - 1][1], dtype=torch.long)
    selected = values[torch.arange(size), assignment]
    return assignment, selected


def _joint_signature(parts: list[torch.Tensor], eps: float) -> torch.Tensor:
    normalized = [functional.normalize(part.flatten(1), dim=1, eps=eps) for part in parts]
    return functional.normalize(torch.cat(normalized, dim=1), dim=1, eps=eps)


@torch.no_grad()
def _match_gqa_structure(
    contracted: dict[str, torch.Tensor],
    reference: BlockLinears,
    geometry: AttentionGeometry,
    reference_basis: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> tuple[dict[str, torch.Tensor], AttentionMatchResult]:
    device = reference_basis.device
    query_per_kv = geometry.query_heads // geometry.kv_heads
    if query_per_kv * geometry.kv_heads != geometry.query_heads:
        raise ValueError("query heads must divide evenly across KV heads")
    hq, hkv, head_dim, hidden = (
        geometry.query_heads,
        geometry.kv_heads,
        geometry.head_dim,
        geometry.hidden_size,
    )

    def shaped(weights: dict[str, torch.Tensor]):
        return (
            weights["q"].view(hq, head_dim, hidden),
            weights["k"].view(hkv, head_dim, hidden),
            weights["v"].view(hkv, head_dim, hidden),
            weights["o"].view(hidden, hq, head_dim),
        )

    source_q, source_k, source_v, source_o = shaped(contracted)
    reference_weights = {
        name: reference.as_dict()[name].weight.detach().to(device, torch.float32)
        for name in ("q", "k", "v", "o")
    }
    reference_q, reference_k, reference_v, reference_o = shaped(reference_weights)

    def group_signature(q, k, v, o):
        q = q.view(hkv, query_per_kv, head_dim, hidden)
        o = o.view(hidden, hkv, query_per_kv, head_dim)
        return _joint_signature(
            [
                (q @ reference_basis).flatten(1),
                (k @ reference_basis).flatten(1),
                (v @ reference_basis).flatten(1),
                torch.einsum("dk,dgqh->gkqh", reference_basis, o).flatten(1),
            ],
            eps,
        )

    source_group_signature = group_signature(source_q, source_k, source_v, source_o)
    reference_group_signature = group_signature(
        reference_q, reference_k, reference_v, reference_o
    )
    group_assignment, group_scores = _maximum_assignment(
        reference_group_signature @ source_group_signature.transpose(0, 1)
    )

    query_assignment = torch.empty(hq, dtype=torch.long)
    query_scores = []
    for target_group in range(hkv):
        source_group = int(group_assignment[target_group])
        target_start = target_group * query_per_kv
        source_start = source_group * query_per_kv
        target_q = reference_q[target_start : target_start + query_per_kv]
        source_q_group = source_q[source_start : source_start + query_per_kv]
        target_o = reference_o[:, target_start : target_start + query_per_kv]
        source_o_group = source_o[:, source_start : source_start + query_per_kv]

        def query_signature(q, o):
            return _joint_signature(
                [
                    (q @ reference_basis).flatten(1),
                    torch.einsum("dk,dqh->qkh", reference_basis, o).flatten(1),
                ],
                eps,
            )

        target_signature = query_signature(target_q, target_o)
        source_signature = query_signature(source_q_group, source_o_group)
        local_assignment, local_scores = _maximum_assignment(
            target_signature @ source_signature.transpose(0, 1)
        )
        query_assignment[target_start : target_start + query_per_kv] = (
            local_assignment + source_start
        )
        query_scores.append(local_scores)

    query_scores_tensor = torch.cat(query_scores)
    query_indices = query_assignment.to(device)
    group_indices = group_assignment.to(device)
    reordered = {
        "q": source_q.index_select(0, query_indices).reshape(hq * head_dim, hidden),
        "k": source_k.index_select(0, group_indices).reshape(hkv * head_dim, hidden),
        "v": source_v.index_select(0, group_indices).reshape(hkv * head_dim, hidden),
        "o": source_o.index_select(1, query_indices).reshape(hidden, hq * head_dim),
    }
    return reordered, AttentionMatchResult(
        group_assignment=group_assignment,
        query_assignment=query_assignment,
        mean_group_cosine=float(group_scores.mean()),
        minimum_group_cosine=float(group_scores.min()),
        mean_query_cosine=float(query_scores_tensor.mean()),
        minimum_query_cosine=float(query_scores_tensor.min()),
    )


@torch.no_grad()
def contract_attention(
    source: BlockLinears,
    reference: BlockLinears,
    projection: torch.Tensor,
    reference_basis: torch.Tensor,
    source_geometry: AttentionGeometry,
    reference_geometry: AttentionGeometry,
) -> tuple[dict[str, torch.Tensor], AttentionMatchResult]:
    validate_attention_pair(source_geometry, reference_geometry)
    device = projection.device
    indices = rope_frequency_indices(
        source_geometry.head_dim, reference_geometry.head_dim
    ).to(device)
    source_hidden = source_geometry.hidden_size
    reference_hidden = reference_geometry.hidden_size
    if projection.shape != (source_hidden, reference_hidden):
        raise ValueError("residual projection does not match attention hidden sizes")

    def contract_input(weight: torch.Tensor, heads: int) -> torch.Tensor:
        expected = (heads * source_geometry.head_dim, source_hidden)
        if tuple(weight.shape) != expected:
            raise ValueError(f"attention input projection shape {tuple(weight.shape)} != {expected}")
        value = weight.detach().to(device, torch.float32).view(
            heads, source_geometry.head_dim, source_hidden
        )
        value = value.index_select(1, indices).reshape(
            heads * reference_geometry.head_dim, source_hidden
        )
        return (value @ projection).contiguous()

    q = contract_input(source.q.weight, source_geometry.query_heads)
    k = contract_input(source.k.weight, source_geometry.kv_heads)
    v = contract_input(source.v.weight, source_geometry.kv_heads)

    expected_o = (
        source_hidden,
        source_geometry.query_heads * source_geometry.head_dim,
    )
    if tuple(source.o.weight.shape) != expected_o:
        raise ValueError(f"attention output projection shape mismatch: {source.o.weight.shape}")
    o = source.o.weight.detach().to(device, torch.float32).view(
        source_hidden, source_geometry.query_heads, source_geometry.head_dim
    )
    o = o.index_select(2, indices).reshape(
        source_hidden, source_geometry.query_heads * reference_geometry.head_dim
    )
    o = (projection.transpose(0, 1) @ o).contiguous()
    return _match_gqa_structure(
        {"q": q, "k": k, "v": v, "o": o},
        reference,
        reference_geometry,
        reference_basis,
    )
