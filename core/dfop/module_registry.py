from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch.nn as nn

from .config import MODULE_TYPES


RESIDUAL_SIDE: Mapping[str, str] = {
    "q": "input",
    "k": "input",
    "v": "input",
    "o": "output",
    "gate": "input",
    "up": "input",
    "down": "output",
}


@dataclass(frozen=True)
class ModuleSpec:
    logical_name: str
    residual_side: str


MODULE_SPECS: Mapping[str, ModuleSpec] = {
    name: ModuleSpec(name, RESIDUAL_SIDE[name]) for name in MODULE_TYPES
}


def _pick_linear(obj: Optional[nn.Module], names: Sequence[str]) -> Optional[nn.Linear]:
    if obj is None:
        return None
    for name in names:
        value = getattr(obj, name, None)
        if isinstance(value, nn.Linear):
            return value
    return None


def find_decoder_layers(model: nn.Module) -> List[nn.Module]:
    """Find the ordered decoder block list without depending on Transformers."""

    roots: List[nn.Module] = [model]
    for attr in ("model", "language_model", "transformer", "base_model", "llm"):
        value = getattr(model, attr, None)
        if isinstance(value, nn.Module) and value not in roots:
            roots.append(value)
        nested = getattr(value, "model", None)
        if isinstance(nested, nn.Module) and nested not in roots:
            roots.append(nested)

    for root in roots:
        for attr in ("layers", "h", "blocks"):
            value = getattr(root, attr, None)
            if isinstance(value, (nn.ModuleList, list, tuple)) and len(value) > 0:
                return list(value)

    for module in model.modules():
        if not isinstance(module, nn.ModuleList) or len(module) == 0:
            continue
        first = module[0]
        if any(hasattr(first, name) for name in ("self_attn", "attention", "attn", "mlp")):
            return list(module)

    raise AttributeError("Could not locate an ordered decoder layer list")


def linears_from_layer(layer: nn.Module) -> Dict[str, Optional[nn.Linear]]:
    attn = (
        getattr(layer, "self_attn", None)
        or getattr(layer, "attention", None)
        or getattr(layer, "attn", None)
    )
    mlp = (
        getattr(layer, "mlp", None)
        or getattr(layer, "feed_forward", None)
        or getattr(layer, "ffn", None)
    )

    return {
        "q": _pick_linear(attn, ("q_proj", "query")),
        "k": _pick_linear(attn, ("k_proj", "key")),
        "v": _pick_linear(attn, ("v_proj", "value")),
        "o": _pick_linear(attn, ("o_proj", "out_proj", "dense")),
        # LLaMA/Qwen names are unambiguous. w1/w3 follow the common SwiGLU convention.
        "gate": _pick_linear(mlp, ("gate_proj", "w1")),
        "up": _pick_linear(mlp, ("up_proj", "w3")),
        "down": _pick_linear(mlp, ("down_proj", "w2")),
    }


def collect_module_linears(
    model: nn.Module,
    modules: Iterable[str] = MODULE_TYPES,
    *,
    require_all: bool = True,
) -> Dict[str, List[nn.Linear]]:
    modules = tuple(modules)
    unknown = set(modules) - set(MODULE_TYPES)
    if unknown:
        raise ValueError(f"Unknown module types: {sorted(unknown)}")

    layers = find_decoder_layers(model)
    result: Dict[str, List[nn.Linear]] = {name: [] for name in modules}
    missing: List[str] = []
    for layer_index, layer in enumerate(layers):
        found = linears_from_layer(layer)
        for name in modules:
            linear = found[name]
            if linear is None:
                missing.append(f"layer={layer_index}, module={name}")
            else:
                result[name].append(linear)

    if missing and require_all:
        preview = ", ".join(missing[:12])
        suffix = " ..." if len(missing) > 12 else ""
        raise AttributeError(f"Missing required linear modules: {preview}{suffix}")

    return result


def common_rank_limit(*module_lists: Sequence[nn.Linear]) -> int:
    weights = [linear.weight for modules in module_lists for linear in modules]
    if not weights:
        raise ValueError("Cannot determine a rank limit from an empty module list")
    return min(min(int(weight.shape[0]), int(weight.shape[1])) for weight in weights)

