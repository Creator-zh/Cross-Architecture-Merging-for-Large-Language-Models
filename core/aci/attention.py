from __future__ import annotations

import math
from collections.abc import Mapping
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
    rotary_dim: int
    rope_type: str
    rope_inverse_frequencies: tuple[float, ...]


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


def _rope_parameter_dict(config) -> dict:
    """Read both the legacy and current Transformers RoPE config layouts."""

    for attribute in ("rope_parameters", "rope_scaling"):
        value = getattr(config, attribute, None)
        if isinstance(value, Mapping) and value:
            parameters = dict(value)
            if any(isinstance(item, Mapping) for item in parameters.values()):
                raise ValueError(
                    "layer-specific RoPE parameters are not supported by ACI attention contraction"
                )
            return parameters
    return {}


def _rope_inverse_frequencies(
    config,
    head_dim: int,
) -> tuple[str, int, tuple[float, ...]]:
    """Reproduce the configured Transformers RoPE inverse frequencies on CPU."""

    parameters = _rope_parameter_dict(config)
    rope_type = str(
        parameters.get("rope_type") or parameters.get("type") or "default"
    ).lower()
    theta_value = parameters.get("rope_theta")
    if theta_value is None:
        theta_value = getattr(config, "rope_theta", None) or 10_000.0
    theta = float(theta_value)
    partial_value = parameters.get("partial_rotary_factor")
    if partial_value is None:
        partial_value = getattr(config, "partial_rotary_factor", None) or 1.0
    partial_rotary_factor = float(partial_value)
    rotary_dim = int(head_dim * partial_rotary_factor)
    if theta <= 0.0:
        raise ValueError("rope_theta must be positive")
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError(
            "RoPE rotary dimension must be positive, even, and no larger than head_dim; "
            f"head_dim={head_dim} rotary_dim={rotary_dim}"
        )

    exponents = torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim
    inverse = 1.0 / (theta**exponents)
    if rope_type == "default":
        pass
    elif rope_type == "linear":
        factor = float(parameters["factor"])
        if factor <= 0.0:
            raise ValueError("linear RoPE factor must be positive")
        inverse = inverse / factor
    elif rope_type == "llama3":
        required = (
            "factor",
            "low_freq_factor",
            "high_freq_factor",
            "original_max_position_embeddings",
        )
        missing = [key for key in required if key not in parameters]
        if missing:
            raise ValueError(f"Llama 3 RoPE parameters are missing {missing}")
        factor = float(parameters["factor"])
        low_factor = float(parameters["low_freq_factor"])
        high_factor = float(parameters["high_freq_factor"])
        original_context = float(parameters["original_max_position_embeddings"])
        if factor <= 0.0 or low_factor <= 0.0 or high_factor <= low_factor:
            raise ValueError("invalid Llama 3 RoPE scaling factors")
        if original_context <= 0.0:
            raise ValueError("original_max_position_embeddings must be positive")

        wavelength = (2.0 * math.pi) / inverse
        low_wavelength = original_context / low_factor
        high_wavelength = original_context / high_factor
        scaled = torch.where(wavelength > low_wavelength, inverse / factor, inverse)
        smooth = (
            (original_context / wavelength - low_factor)
            / (high_factor - low_factor)
        )
        interpolated = (1.0 - smooth) * scaled / factor + smooth * scaled
        medium = (wavelength >= high_wavelength) & (wavelength <= low_wavelength)
        inverse = torch.where(medium, interpolated, scaled)
    else:
        raise ValueError(
            f"unsupported context-dependent RoPE type {rope_type!r}; "
            "ACI currently supports default, linear, and llama3"
        )

    return rope_type, rotary_dim, tuple(float(value) for value in inverse)


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
    rope_type, rotary_dim, inverse = _rope_inverse_frequencies(config, head_dim)
    return AttentionGeometry(
        hidden,
        query_heads,
        kv_heads,
        head_dim,
        rotary_dim,
        rope_type,
        inverse,
    )


def _monotonic_frequency_assignment(
    source_inverse: torch.Tensor,
    target_inverse: torch.Tensor,
) -> torch.Tensor:
    """Find the minimum-log-error monotonic one-to-one frequency assignment."""

    source_count = int(source_inverse.numel())
    target_count = int(target_inverse.numel())
    if target_count > source_count:
        raise ValueError("source must provide at least as many RoPE pairs as target")
    if source_count == 0 or target_count == 0:
        raise ValueError("RoPE inverse-frequency lists must be non-empty")
    if bool((source_inverse <= 0).any()) or bool((target_inverse <= 0).any()):
        raise ValueError("RoPE inverse frequencies must be positive")

    cost = (
        source_inverse.to(torch.float64).log().unsqueeze(0)
        - target_inverse.to(torch.float64).log().unsqueeze(1)
    ).abs()
    infinity = float("inf")
    scores = [[infinity] * source_count for _ in range(target_count)]
    parents = [[-1] * source_count for _ in range(target_count)]
    for source_index in range(source_count - target_count + 1):
        scores[0][source_index] = float(cost[0, source_index])

    for target_index in range(1, target_count):
        first_source = target_index
        last_source = source_count - (target_count - target_index)
        for source_index in range(first_source, last_source + 1):
            best_parent = min(
                range(target_index - 1, source_index),
                key=lambda index: scores[target_index - 1][index],
            )
            scores[target_index][source_index] = (
                scores[target_index - 1][best_parent]
                + float(cost[target_index, source_index])
            )
            parents[target_index][source_index] = best_parent

    final_source = min(
        range(target_count - 1, source_count),
        key=lambda index: scores[target_count - 1][index],
    )
    assignment = [final_source]
    for target_index in range(target_count - 1, 0, -1):
        assignment.append(parents[target_index][assignment[-1]])
    assignment.reverse()
    return torch.tensor(assignment, dtype=torch.long)


def _frequency_spec(
    geometry_or_head_dim: AttentionGeometry | int,
) -> tuple[int, int, torch.Tensor]:
    if isinstance(geometry_or_head_dim, int):
        head_dim = geometry_or_head_dim
        if head_dim <= 0 or head_dim % 2:
            raise ValueError("Llama RoPE head dimensions must be positive and even")
        exponents = torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim
        return head_dim, head_dim, 1.0 / (10_000.0**exponents)
    return (
        geometry_or_head_dim.head_dim,
        geometry_or_head_dim.rotary_dim,
        torch.tensor(
            geometry_or_head_dim.rope_inverse_frequencies,
            dtype=torch.float64,
        ),
    )


def rope_frequency_indices(
    source: AttentionGeometry | int,
    target: AttentionGeometry | int,
) -> torch.Tensor:
    """Select source Q/K coordinates by matching the configured RoPE frequencies.

    Hugging Face Llama rotates corresponding coordinates in the first and
    second half of each head.  The pair assignment is monotonic and minimizes
    absolute log-frequency error, so different Llama 3 scaling factors are
    handled rather than silently treated as identical.
    """

    source_head_dim, source_rotary_dim, source_inverse = _frequency_spec(source)
    target_head_dim, target_rotary_dim, target_inverse = _frequency_spec(target)
    if source_rotary_dim != source_head_dim or target_rotary_dim != target_head_dim:
        raise ValueError(
            "ACI attention contraction currently requires full-head rotary embeddings"
        )
    if source_head_dim < target_head_dim:
        raise ValueError("source head_dim must be at least target head_dim")
    pair_indices = _monotonic_frequency_assignment(source_inverse, target_inverse)
    source_half = source_head_dim // 2
    return torch.cat((pair_indices, pair_indices + source_half))


def _uniform_channel_indices(source_dim: int, target_dim: int) -> torch.Tensor:
    """Select non-RoPE V/O channels without imposing Q/K frequency semantics."""

    if source_dim < target_dim or target_dim <= 0:
        raise ValueError("source channel dimension must be at least target dimension")
    return torch.div(
        torch.arange(target_dim, dtype=torch.long) * source_dim,
        target_dim,
        rounding_mode="floor",
    )


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
    rope_frequency_indices(source, reference)
    _uniform_channel_indices(source.head_dim, reference.head_dim)


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
    qk_indices = rope_frequency_indices(source_geometry, reference_geometry).to(device)
    vo_indices = _uniform_channel_indices(
        source_geometry.head_dim, reference_geometry.head_dim
    ).to(device)
    source_hidden = source_geometry.hidden_size
    reference_hidden = reference_geometry.hidden_size
    if projection.shape != (source_hidden, reference_hidden):
        raise ValueError("residual projection does not match attention hidden sizes")

    def contract_input(
        weight: torch.Tensor,
        heads: int,
        indices: torch.Tensor,
    ) -> torch.Tensor:
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

    q = contract_input(source.q.weight, source_geometry.query_heads, qk_indices)
    k = contract_input(source.k.weight, source_geometry.kv_heads, qk_indices)
    v = contract_input(source.v.weight, source_geometry.kv_heads, vo_indices)
    expected_o = (
        source_hidden,
        source_geometry.query_heads * source_geometry.head_dim,
    )
    if tuple(source.o.weight.shape) != expected_o:
        raise ValueError(f"attention output projection shape mismatch: {source.o.weight.shape}")
    o = source.o.weight.detach().to(device, torch.float32).view(
        source_hidden, source_geometry.query_heads, source_geometry.head_dim
    )
    o = o.index_select(2, vo_indices).reshape(
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
