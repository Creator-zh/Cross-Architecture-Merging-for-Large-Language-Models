from __future__ import annotations

import torch

from .attention import AttentionGeometry
from .types import SafeAttentionInjectionResult, SafeGroupInjectionResult


def calibrate_to_reference(
    reference: torch.Tensor,
    compressed_source: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, float]:
    reference_norm = torch.linalg.vector_norm(reference)
    source_norm = torch.linalg.vector_norm(compressed_source)
    scale = float(reference_norm / source_norm.clamp_min(eps))
    return scale * compressed_source, scale


def _joint_norm(values: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.sqrt(sum(torch.sum(value.square()) for value in values.values()))


@torch.no_grad()
def inject_safe_ffn(
    target_weights: dict[str, torch.Tensor],
    reference_weights: dict[str, torch.Tensor],
    compressed_source_weights: dict[str, torch.Tensor],
    neuron_confidence: torch.Tensor,
    *,
    beta: float,
    eps: float,
) -> SafeGroupInjectionResult:
    """Target-aware, confidence-gated injection for coupled SwiGLU neurons."""

    names = ("gate", "up", "down")
    if set(target_weights) != set(names):
        raise ValueError("safe FFN injection requires gate/up/down target weights")
    if set(reference_weights) != set(names) or set(compressed_source_weights) != set(names):
        raise ValueError("safe FFN injection requires gate/up/down for all models")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0, 1]")
    device = compressed_source_weights["gate"].device
    target = {
        name: target_weights[name].detach().to(device, torch.float32) for name in names
    }
    reference = {
        name: reference_weights[name].detach().to(device, torch.float32) for name in names
    }
    source = {
        name: compressed_source_weights[name].to(device, torch.float32) for name in names
    }
    for name in names:
        if target[name].shape != reference[name].shape or target[name].shape != source[name].shape:
            raise ValueError(f"safe FFN shape mismatch for {name}")
    intermediate = target["gate"].shape[0]
    if target["up"].shape[0] != intermediate or target["down"].shape[1] != intermediate:
        raise ValueError("gate/up rows and down columns must share the FFN width")
    confidence = neuron_confidence.detach().to(device, torch.float32).flatten()
    if confidence.numel() != intermediate:
        raise ValueError(
            f"FFN confidence length {confidence.numel()} != intermediate size {intermediate}"
        )
    confidence = confidence.clamp(0.0, 1.0)

    calibrated = {}
    calibration_scales = {}
    for name in names:
        calibrated[name], calibration_scales[name] = calibrate_to_reference(
            reference[name], source[name], eps=eps
        )
    source_delta = {name: calibrated[name] - reference[name] for name in names}
    domain_delta = {name: target[name] - reference[name] for name in names}

    dot = (source_delta["gate"] * domain_delta["gate"]).sum(dim=1)
    dot.add_((source_delta["up"] * domain_delta["up"]).sum(dim=1))
    dot.add_((source_delta["down"] * domain_delta["down"]).sum(dim=0))
    domain_norm_sq = domain_delta["gate"].square().sum(dim=1)
    domain_norm_sq.add_(domain_delta["up"].square().sum(dim=1))
    domain_norm_sq.add_(domain_delta["down"].square().sum(dim=0))
    coefficient = torch.minimum(
        dot / domain_norm_sq.clamp_min(eps),
        torch.zeros_like(dot),
    )
    projected = {
        "gate": source_delta["gate"] - coefficient[:, None] * domain_delta["gate"],
        "up": source_delta["up"] - coefficient[:, None] * domain_delta["up"],
        "down": source_delta["down"] - coefficient[None, :] * domain_delta["down"],
    }
    gated = {
        "gate": confidence[:, None] * projected["gate"],
        "up": confidence[:, None] * projected["up"],
        "down": confidence[None, :] * projected["down"],
    }

    source_norm = _joint_norm(source_delta).clamp_min(eps)
    domain_norm = _joint_norm(domain_delta).clamp_min(eps)
    source_domain_cosine = float(
        sum((source_delta[name] * domain_delta[name]).sum() for name in names)
        / (source_norm * domain_norm)
    )
    removed = {name: source_delta[name] - projected[name] for name in names}
    removed_ratio = float(_joint_norm(removed) / source_norm)

    raw_delta = {name: float(beta) * gated[name] for name in names}
    target_norm = _joint_norm(target).clamp_min(eps)
    raw_norm = _joint_norm(raw_delta)
    trust = 1.0
    if beta > 0.0 and float(raw_norm) > 0.0:
        trust = min(1.0, float(beta) * float(target_norm) / float(raw_norm))
    applied_delta = {name: trust * raw_delta[name] for name in names}
    weights = {
        name: (target[name] + applied_delta[name])
        .to(dtype=target_weights[name].dtype)
        .contiguous()
        for name in names
    }
    module_relative = {
        name: float(
            torch.linalg.vector_norm(applied_delta[name])
            / torch.linalg.vector_norm(target[name]).clamp_min(eps)
        )
        for name in names
    }
    joint_relative = float(_joint_norm(applied_delta) / target_norm)
    return SafeGroupInjectionResult(
        weights=weights,
        calibration_scales=calibration_scales,
        trust_coefficient=trust,
        joint_relative_update_norm=joint_relative,
        module_relative_update_norms=module_relative,
        mean_confidence=float(confidence.mean()),
        minimum_confidence=float(confidence.min()),
        maximum_confidence=float(confidence.max()),
        active_confidence_fraction=float((confidence > 0).float().mean()),
        conflict_fraction=float((coefficient < 0).float().mean()),
        source_domain_cosine=source_domain_cosine,
        removed_conflict_norm_ratio=removed_ratio,
    )


def _pair_trust(
    targets: tuple[torch.Tensor, torch.Tensor],
    deltas: tuple[torch.Tensor, torch.Tensor],
    *,
    beta: float,
    eps: float,
) -> float:
    target_norm = torch.sqrt(sum(value.square().sum() for value in targets)).clamp_min(eps)
    raw_norm = float(beta) * torch.sqrt(sum(value.square().sum() for value in deltas))
    if beta == 0.0 or float(raw_norm) == 0.0:
        return 1.0
    return min(1.0, float(beta) * float(target_norm) / float(raw_norm))


@torch.no_grad()
def inject_safe_attention(
    target_weights: dict[str, torch.Tensor],
    reference_weights: dict[str, torch.Tensor],
    compressed_source_weights: dict[str, torch.Tensor],
    head_confidence: torch.Tensor,
    geometry: AttentionGeometry,
    *,
    beta: float,
    eps: float,
) -> SafeAttentionInjectionResult:
    """Inject QK and OV circuit factors with group-coupled conflict surgery."""

    names = ("q", "k", "v", "o")
    if set(target_weights) != set(names):
        raise ValueError("safe attention injection requires q/k/v/o target weights")
    if set(reference_weights) != set(names) or set(compressed_source_weights) != set(names):
        raise ValueError("safe attention injection requires q/k/v/o for all models")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0, 1]")
    if geometry.query_heads % geometry.kv_heads:
        raise ValueError("query heads must divide evenly across KV heads")
    device = compressed_source_weights["q"].device
    target = {
        name: target_weights[name].detach().to(device, torch.float32) for name in names
    }
    reference = {
        name: reference_weights[name].detach().to(device, torch.float32) for name in names
    }
    source = {
        name: compressed_source_weights[name].to(device, torch.float32) for name in names
    }
    for name in names:
        if target[name].shape != reference[name].shape or target[name].shape != source[name].shape:
            raise ValueError(f"safe attention shape mismatch for {name}")
    confidence = head_confidence.detach().to(device, torch.float32).flatten().clamp(0.0, 1.0)
    if confidence.numel() != geometry.query_heads:
        raise ValueError(
            f"attention confidence length {confidence.numel()} "
            f"!= query heads {geometry.query_heads}"
        )
    query_per_kv = geometry.query_heads // geometry.kv_heads
    group_confidence = confidence.view(geometry.kv_heads, query_per_kv).mean(dim=1)

    calibrated = {}
    calibration_scales = {}
    for name in names:
        calibrated[name], calibration_scales[name] = calibrate_to_reference(
            reference[name], source[name], eps=eps
        )
    source_delta = {name: calibrated[name] - reference[name] for name in names}
    domain_delta = {name: target[name] - reference[name] for name in names}

    def shaped(values: dict[str, torch.Tensor]):
        return (
            values["q"].view(
                geometry.kv_heads,
                query_per_kv,
                geometry.head_dim,
                geometry.hidden_size,
            ),
            values["k"].view(
                geometry.kv_heads, geometry.head_dim, geometry.hidden_size
            ),
            values["v"].view(
                geometry.kv_heads, geometry.head_dim, geometry.hidden_size
            ),
            values["o"].view(
                geometry.hidden_size,
                geometry.kv_heads,
                query_per_kv,
                geometry.head_dim,
            ),
        )

    sq, sk, sv, so = shaped(source_delta)
    dq, dk, dv, do = shaped(domain_delta)
    qk_dot = (sq * dq).sum(dim=(1, 2, 3)) + (sk * dk).sum(dim=(1, 2))
    qk_domain_norm_sq = dq.square().sum(dim=(1, 2, 3)) + dk.square().sum(dim=(1, 2))
    qk_coefficient = torch.minimum(
        qk_dot / qk_domain_norm_sq.clamp_min(eps), torch.zeros_like(qk_dot)
    )
    ov_dot = (sv * dv).sum(dim=(1, 2)) + (so * do).sum(dim=(0, 2, 3))
    ov_domain_norm_sq = dv.square().sum(dim=(1, 2)) + do.square().sum(dim=(0, 2, 3))
    ov_coefficient = torch.minimum(
        ov_dot / ov_domain_norm_sq.clamp_min(eps), torch.zeros_like(ov_dot)
    )
    pq = sq - qk_coefficient[:, None, None, None] * dq
    pk = sk - qk_coefficient[:, None, None] * dk
    pv = sv - ov_coefficient[:, None, None] * dv
    po = so - ov_coefficient[None, :, None, None] * do

    head_confidence_shaped = confidence.view(geometry.kv_heads, query_per_kv)
    gq = head_confidence_shaped[:, :, None, None] * pq
    gk = group_confidence[:, None, None] * pk
    gv = group_confidence[:, None, None] * pv
    go = head_confidence_shaped[None, :, :, None] * po
    gated = {
        "q": gq.reshape_as(source_delta["q"]),
        "k": gk.reshape_as(source_delta["k"]),
        "v": gv.reshape_as(source_delta["v"]),
        "o": go.reshape_as(source_delta["o"]),
    }
    qk_trust = _pair_trust(
        (target["q"], target["k"]),
        (gated["q"], gated["k"]),
        beta=beta,
        eps=eps,
    )
    ov_trust = _pair_trust(
        (target["v"], target["o"]),
        (gated["v"], gated["o"]),
        beta=beta,
        eps=eps,
    )
    applied_delta = {
        "q": qk_trust * float(beta) * gated["q"],
        "k": qk_trust * float(beta) * gated["k"],
        "v": ov_trust * float(beta) * gated["v"],
        "o": ov_trust * float(beta) * gated["o"],
    }
    weights = {
        name: (target[name] + applied_delta[name])
        .to(dtype=target_weights[name].dtype)
        .contiguous()
        for name in names
    }
    target_joint_norm = _joint_norm(target).clamp_min(eps)
    module_relative = {
        name: float(
            torch.linalg.vector_norm(applied_delta[name])
            / torch.linalg.vector_norm(target[name]).clamp_min(eps)
        )
        for name in names
    }
    source_norm = _joint_norm(source_delta).clamp_min(eps)
    projected = {
        "q": pq.reshape_as(source_delta["q"]),
        "k": pk.reshape_as(source_delta["k"]),
        "v": pv.reshape_as(source_delta["v"]),
        "o": po.reshape_as(source_delta["o"]),
    }
    removed = {name: source_delta[name] - projected[name] for name in names}

    def pair_cosine(pair: tuple[str, str]) -> float:
        numerator = sum((source_delta[name] * domain_delta[name]).sum() for name in pair)
        source_pair_norm = torch.sqrt(
            sum(source_delta[name].square().sum() for name in pair)
        ).clamp_min(eps)
        domain_pair_norm = torch.sqrt(
            sum(domain_delta[name].square().sum() for name in pair)
        ).clamp_min(eps)
        return float(numerator / (source_pair_norm * domain_pair_norm))

    return SafeAttentionInjectionResult(
        weights=weights,
        calibration_scales=calibration_scales,
        trust_coefficients={"qk": qk_trust, "ov": ov_trust},
        joint_relative_update_norm=float(_joint_norm(applied_delta) / target_joint_norm),
        module_relative_update_norms=module_relative,
        mean_confidence=float(confidence.mean()),
        minimum_confidence=float(confidence.min()),
        maximum_confidence=float(confidence.max()),
        active_confidence_fraction=float((confidence > 0).float().mean()),
        qk_conflict_fraction=float((qk_coefficient < 0).float().mean()),
        ov_conflict_fraction=float((ov_coefficient < 0).float().mean()),
        qk_source_domain_cosine=pair_cosine(("q", "k")),
        ov_source_domain_cosine=pair_cosine(("v", "o")),
        removed_conflict_norm_ratio=float(_joint_norm(removed) / source_norm),
    )
