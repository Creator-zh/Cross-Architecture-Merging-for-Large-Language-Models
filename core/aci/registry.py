from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch.nn as nn


ATTENTION_MODULE_TYPES = ("q", "k", "v", "o")
FFN_MODULE_TYPES = ("gate", "up", "down")
MODULE_TYPES = ATTENTION_MODULE_TYPES + FFN_MODULE_TYPES


@dataclass(frozen=True)
class BlockLinears:
    q: nn.Linear
    k: nn.Linear
    v: nn.Linear
    o: nn.Linear
    gate: nn.Linear
    up: nn.Linear
    down: nn.Linear

    def as_dict(self) -> dict[str, nn.Linear]:
        return {name: getattr(self, name) for name in MODULE_TYPES}


def _pick_linear(obj: Optional[nn.Module], names: Sequence[str]) -> nn.Linear | None:
    if obj is None:
        return None
    for name in names:
        value = getattr(obj, name, None)
        if isinstance(value, nn.Linear):
            return value
    return None


def find_decoder_layers(model: nn.Module) -> list[nn.Module]:
    roots: list[nn.Module] = [model]
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
            if isinstance(value, (nn.ModuleList, list, tuple)) and value:
                return list(value)
    for module in model.modules():
        if isinstance(module, nn.ModuleList) and module:
            first = module[0]
            if any(hasattr(first, name) for name in ("self_attn", "attention", "attn")):
                return list(module)
    raise AttributeError("Could not locate an ordered decoder layer list")


def block_linears(layer: nn.Module, layer_index: int | None = None) -> BlockLinears:
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
    found = {
        "q": _pick_linear(attn, ("q_proj", "query")),
        "k": _pick_linear(attn, ("k_proj", "key")),
        "v": _pick_linear(attn, ("v_proj", "value")),
        "o": _pick_linear(attn, ("o_proj", "out_proj", "dense")),
        "gate": _pick_linear(mlp, ("gate_proj", "w1")),
        "up": _pick_linear(mlp, ("up_proj", "w3")),
        "down": _pick_linear(mlp, ("down_proj", "w2")),
    }
    missing = [name for name, value in found.items() if value is None]
    if missing:
        prefix = "" if layer_index is None else f"layer {layer_index}: "
        raise AttributeError(f"{prefix}missing required linears {missing}")
    return BlockLinears(**found)  # type: ignore[arg-type]


def collect_blocks(model: nn.Module) -> list[BlockLinears]:
    return [block_linears(layer, index) for index, layer in enumerate(find_decoder_layers(model))]


def input_embedding(model: nn.Module) -> nn.Embedding:
    getter = getattr(model, "get_input_embeddings", None)
    value = getter() if callable(getter) else None
    if isinstance(value, nn.Embedding):
        return value
    for path in (("model", "embed_tokens"), ("embed_tokens",)):
        obj: object = model
        for attr in path:
            obj = getattr(obj, attr, None)
        if isinstance(obj, nn.Embedding):
            return obj
    raise AttributeError("Could not locate input embeddings")


def output_head(model: nn.Module) -> nn.Linear | None:
    getter = getattr(model, "get_output_embeddings", None)
    value = getter() if callable(getter) else None
    if isinstance(value, nn.Linear):
        return value
    value = getattr(model, "lm_head", None)
    return value if isinstance(value, nn.Linear) else None
