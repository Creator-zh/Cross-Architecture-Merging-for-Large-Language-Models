from __future__ import annotations

from dataclasses import asdict, dataclass


FUSION_MODES = ("full", "attention", "ffn")


@dataclass(frozen=True)
class ACIConfig:
    """The intentionally small set of ACI controls.

    ``beta`` is the only capability/fusion hyperparameter.  The remaining
    values are numerical batching controls and deterministic sketch sizes.
    """

    beta: float = 0.03
    fusion_mode: str = "full"
    anchor_tokens: int = 8192
    anchor_chunk_size: int = 1024
    ffn_sketch_dim: int = 32
    ffn_candidate_k: int = 32
    eps: float = 1.0e-8

    def validate(self) -> None:
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must lie in [0, 1]")
        if self.fusion_mode not in FUSION_MODES:
            raise ValueError(
                f"fusion_mode must be one of {FUSION_MODES}; got {self.fusion_mode!r}"
            )
        if self.anchor_tokens <= 0 or self.anchor_chunk_size <= 0:
            raise ValueError("anchor sizes must be positive")
        if self.ffn_sketch_dim <= 0 or self.ffn_candidate_k <= 0:
            raise ValueError("FFN sketch settings must be positive")
        if self.eps <= 0:
            raise ValueError("eps must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def uses_attention(self) -> bool:
        return self.fusion_mode in ("full", "attention")

    @property
    def uses_ffn(self) -> bool:
        return self.fusion_mode in ("full", "ffn")
