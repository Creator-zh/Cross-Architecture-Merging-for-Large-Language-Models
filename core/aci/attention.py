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


@dataclass(frozen=True)
class CircuitAttentionMatchResult:
    group_assignment: torch.Tensor
    query_assignment: torch.Tensor
    selected_group_cosines: torch.Tensor
    selected_query_cosines: torch.Tensor
    head_confidences: torch.Tensor
    mean_group_cosine: float
    minimum_group_cosine: float
    mean_query_cosine: float
    minimum_query_cosine: float
    qk_gauge_cosine_before: float
    qk_gauge_cosine_after: float
    ov_gauge_cosine_before: float
    ov_gauge_cosine_after: float


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


def _shaped_attention(
    weights: dict[str, torch.Tensor],
    geometry: AttentionGeometry,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        weights["q"].view(
            geometry.query_heads, geometry.head_dim, geometry.hidden_size
        ),
        weights["k"].view(
            geometry.kv_heads, geometry.head_dim, geometry.hidden_size
        ),
        weights["v"].view(
            geometry.kv_heads, geometry.head_dim, geometry.hidden_size
        ),
        weights["o"].view(
            geometry.hidden_size, geometry.query_heads, geometry.head_dim
        ),
    )


def _reference_attention_weights(
    reference: BlockLinears,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: reference.as_dict()[name].weight.detach().to(device, torch.float32)
        for name in ("q", "k", "v", "o")
    }


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

    source_q, source_k, source_v, source_o = _shaped_attention(contracted, geometry)
    reference_weights = _reference_attention_weights(reference, device=device)
    reference_q, reference_k, reference_v, reference_o = _shaped_attention(
        reference_weights, geometry
    )

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


def _circuit_signatures(
    weights: dict[str, torch.Tensor],
    geometry: AttentionGeometry,
    basis: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    q, k, v, o = _shaped_attention(weights, geometry)
    query_per_kv = geometry.query_heads // geometry.kv_heads
    q = q.view(
        geometry.kv_heads,
        query_per_kv,
        geometry.head_dim,
        geometry.hidden_size,
    )
    o = o.view(
        geometry.hidden_size,
        geometry.kv_heads,
        query_per_kv,
        geometry.head_dim,
    )
    q_basis = torch.einsum("gqad,dk->gqak", q, basis)
    k_basis = torch.einsum("gad,dk->gak", k, basis)
    v_basis = torch.einsum("gad,dk->gak", v, basis)
    o_basis = torch.einsum("dk,dgqa->gqka", basis, o)
    qk = torch.einsum("gqak,gaj->gqkj", q_basis, k_basis)
    ov = torch.einsum("gqka,gaj->gqkj", o_basis, v_basis)
    qk = functional.normalize(qk.flatten(2), dim=2, eps=eps)
    ov = functional.normalize(ov.flatten(2), dim=2, eps=eps)
    return functional.normalize(torch.cat((qk, ov), dim=2), dim=2, eps=eps)


def _proper_rotation(cross_covariance: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(cross_covariance, full_matrices=False)
    correction = torch.eye(
        cross_covariance.shape[0],
        device=cross_covariance.device,
        dtype=cross_covariance.dtype,
    )
    correction[-1, -1] = torch.det(u @ vh)
    return u @ correction @ vh


def _orthogonal_map(cross_covariance: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(cross_covariance, full_matrices=False)
    return u @ vh


def _flat_cosine(left: torch.Tensor, right: torch.Tensor, eps: float) -> float:
    return float(
        functional.cosine_similarity(
            left.flatten(), right.flatten(), dim=0, eps=eps
        )
    )


@torch.no_grad()
def _align_qk_rope_gauge(
    source_q: torch.Tensor,
    source_k: torch.Tensor,
    reference_q: torch.Tensor,
    reference_k: torch.Tensor,
    geometry: AttentionGeometry,
    basis: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    query_per_kv = geometry.query_heads // geometry.kv_heads
    half = geometry.head_dim // 2
    sq = source_q.view(
        geometry.kv_heads,
        query_per_kv,
        geometry.head_dim,
        geometry.hidden_size,
    ).clone()
    sk = source_k.view(
        geometry.kv_heads, geometry.head_dim, geometry.hidden_size
    ).clone()
    rq = reference_q.view_as(sq)
    rk = reference_k.view_as(sk)
    before = []
    after = []
    for group in range(geometry.kv_heads):
        for frequency in range(half):
            indices = torch.tensor(
                (frequency, frequency + half), device=sq.device, dtype=torch.long
            )
            source_q_pair = sq[group].index_select(1, indices)
            reference_q_pair = rq[group].index_select(1, indices)
            source_k_pair = sk[group].index_select(0, indices)
            reference_k_pair = rk[group].index_select(0, indices)
            source_sketch = torch.cat(
                (
                    torch.einsum("qad,dk->qak", source_q_pair, basis)
                    .permute(1, 0, 2)
                    .reshape(2, -1),
                    source_k_pair @ basis,
                ),
                dim=1,
            )
            reference_sketch = torch.cat(
                (
                    torch.einsum("qad,dk->qak", reference_q_pair, basis)
                    .permute(1, 0, 2)
                    .reshape(2, -1),
                    reference_k_pair @ basis,
                ),
                dim=1,
            )
            rotation = _proper_rotation(reference_sketch @ source_sketch.transpose(0, 1))
            before.append(_flat_cosine(source_sketch, reference_sketch, eps))
            after.append(_flat_cosine(rotation @ source_sketch, reference_sketch, eps))
            aligned_q = torch.einsum("ab,qbd->qad", rotation, source_q_pair)
            aligned_k = rotation @ source_k_pair
            sq[group, :, frequency, :] = aligned_q[:, 0, :]
            sq[group, :, frequency + half, :] = aligned_q[:, 1, :]
            sk[group, frequency, :] = aligned_k[0]
            sk[group, frequency + half, :] = aligned_k[1]
    return (
        sq.reshape(geometry.query_heads * geometry.head_dim, geometry.hidden_size),
        sk.reshape(geometry.kv_heads * geometry.head_dim, geometry.hidden_size),
        sum(before) / len(before),
        sum(after) / len(after),
    )


@torch.no_grad()
def _align_ov_gauge(
    source_v: torch.Tensor,
    source_o: torch.Tensor,
    reference_v: torch.Tensor,
    reference_o: torch.Tensor,
    geometry: AttentionGeometry,
    basis: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    query_per_kv = geometry.query_heads // geometry.kv_heads
    sv = source_v.view(
        geometry.kv_heads, geometry.head_dim, geometry.hidden_size
    ).clone()
    so = source_o.view(
        geometry.hidden_size,
        geometry.kv_heads,
        query_per_kv,
        geometry.head_dim,
    ).clone()
    rv = reference_v.view_as(sv)
    ro = reference_o.view_as(so)
    before = []
    after = []
    for group in range(geometry.kv_heads):
        source_o_sketch = torch.einsum("dqa,dk->qak", so[:, group], basis)
        reference_o_sketch = torch.einsum("dqa,dk->qak", ro[:, group], basis)
        source_sketch = torch.cat(
            (
                sv[group] @ basis,
                source_o_sketch.permute(1, 0, 2).reshape(geometry.head_dim, -1),
            ),
            dim=1,
        )
        reference_sketch = torch.cat(
            (
                rv[group] @ basis,
                reference_o_sketch.permute(1, 0, 2).reshape(geometry.head_dim, -1),
            ),
            dim=1,
        )
        transform = _orthogonal_map(
            reference_sketch @ source_sketch.transpose(0, 1)
        )
        before.append(_flat_cosine(source_sketch, reference_sketch, eps))
        after.append(_flat_cosine(transform @ source_sketch, reference_sketch, eps))
        sv[group] = transform @ sv[group]
        so[:, group] = torch.einsum(
            "dqa,ab->dqb", so[:, group], transform.transpose(0, 1)
        )
    return (
        sv.reshape(geometry.kv_heads * geometry.head_dim, geometry.hidden_size),
        so.reshape(geometry.hidden_size, geometry.query_heads * geometry.head_dim),
        sum(before) / len(before),
        sum(after) / len(after),
    )


@torch.no_grad()
def _match_and_align_circuits(
    contracted: dict[str, torch.Tensor],
    reference: BlockLinears,
    geometry: AttentionGeometry,
    reference_basis: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> tuple[dict[str, torch.Tensor], CircuitAttentionMatchResult]:
    device = reference_basis.device
    query_per_kv = geometry.query_heads // geometry.kv_heads
    reference_weights = _reference_attention_weights(reference, device=device)
    source_signatures = _circuit_signatures(
        contracted, geometry, reference_basis, eps=eps
    )
    reference_signatures = _circuit_signatures(
        reference_weights, geometry, reference_basis, eps=eps
    )
    group_similarity = torch.empty(
        geometry.kv_heads,
        geometry.kv_heads,
        device=device,
        dtype=torch.float32,
    )
    local_assignments: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    for target_group in range(geometry.kv_heads):
        for source_group in range(geometry.kv_heads):
            similarity = (
                reference_signatures[target_group]
                @ source_signatures[source_group].transpose(0, 1)
            )
            assignment, scores = _maximum_assignment(similarity)
            local_assignments[(target_group, source_group)] = (assignment, scores)
            group_similarity[target_group, source_group] = scores.mean()
    group_assignment, group_scores = _maximum_assignment(group_similarity)
    query_assignment = torch.empty(geometry.query_heads, dtype=torch.long)
    query_scores = torch.empty(geometry.query_heads, dtype=torch.float32)
    for target_group in range(geometry.kv_heads):
        source_group = int(group_assignment[target_group])
        local_assignment, local_scores = local_assignments[(target_group, source_group)]
        target_start = target_group * query_per_kv
        source_start = source_group * query_per_kv
        query_assignment[target_start : target_start + query_per_kv] = (
            local_assignment + source_start
        )
        query_scores[target_start : target_start + query_per_kv] = local_scores

    source_q, source_k, source_v, source_o = _shaped_attention(contracted, geometry)
    query_indices = query_assignment.to(device)
    group_indices = group_assignment.to(device)
    reordered_q = source_q.index_select(0, query_indices).reshape(
        geometry.query_heads * geometry.head_dim, geometry.hidden_size
    )
    reordered_k = source_k.index_select(0, group_indices).reshape(
        geometry.kv_heads * geometry.head_dim, geometry.hidden_size
    )
    reordered_v = source_v.index_select(0, group_indices).reshape(
        geometry.kv_heads * geometry.head_dim, geometry.hidden_size
    )
    reordered_o = source_o.index_select(1, query_indices).reshape(
        geometry.hidden_size, geometry.query_heads * geometry.head_dim
    )
    reference_q, reference_k, reference_v, reference_o = _shaped_attention(
        reference_weights, geometry
    )
    aligned_q, aligned_k, qk_before, qk_after = _align_qk_rope_gauge(
        reordered_q,
        reordered_k,
        reference_q.reshape_as(reordered_q),
        reference_k.reshape_as(reordered_k),
        geometry,
        reference_basis,
        eps=eps,
    )
    aligned_v, aligned_o, ov_before, ov_after = _align_ov_gauge(
        reordered_v,
        reordered_o,
        reference_v.reshape_as(reordered_v),
        reference_o.reshape_as(reordered_o),
        geometry,
        reference_basis,
        eps=eps,
    )
    group_per_query = group_scores.repeat_interleave(query_per_kv)
    head_confidence = 0.5 * (group_per_query + query_scores)
    return (
        {"q": aligned_q, "k": aligned_k, "v": aligned_v, "o": aligned_o},
        CircuitAttentionMatchResult(
            group_assignment=group_assignment,
            query_assignment=query_assignment,
            selected_group_cosines=group_scores,
            selected_query_cosines=query_scores,
            head_confidences=head_confidence,
            mean_group_cosine=float(group_scores.mean()),
            minimum_group_cosine=float(group_scores.min()),
            mean_query_cosine=float(query_scores.mean()),
            minimum_query_cosine=float(query_scores.min()),
            qk_gauge_cosine_before=qk_before,
            qk_gauge_cosine_after=qk_after,
            ov_gauge_cosine_before=ov_before,
            ov_gauge_cosine_after=ov_after,
        ),
    )


@torch.no_grad()
def _contract_attention_coordinates(
    source: BlockLinears,
    projection: torch.Tensor,
    source_geometry: AttentionGeometry,
    reference_geometry: AttentionGeometry,
) -> dict[str, torch.Tensor]:
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
            raise ValueError(
                f"attention input projection shape {tuple(weight.shape)} != {expected}"
            )
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
    return {"q": q, "k": k, "v": v, "o": o}


@torch.no_grad()
def contract_attention(
    source: BlockLinears,
    reference: BlockLinears,
    projection: torch.Tensor,
    reference_basis: torch.Tensor,
    source_geometry: AttentionGeometry,
    reference_geometry: AttentionGeometry,
) -> tuple[dict[str, torch.Tensor], AttentionMatchResult]:
    contracted = _contract_attention_coordinates(
        source, projection, source_geometry, reference_geometry
    )
    return _match_gqa_structure(
        contracted,
        reference,
        reference_geometry,
        reference_basis,
    )


@torch.no_grad()
def contract_attention_circuit(
    source: BlockLinears,
    reference: BlockLinears,
    projection: torch.Tensor,
    reference_basis: torch.Tensor,
    source_geometry: AttentionGeometry,
    reference_geometry: AttentionGeometry,
) -> tuple[dict[str, torch.Tensor], CircuitAttentionMatchResult]:
    contracted = _contract_attention_coordinates(
        source, projection, source_geometry, reference_geometry
    )
    return _match_and_align_circuits(
        contracted,
        reference,
        reference_geometry,
        reference_basis,
    )
