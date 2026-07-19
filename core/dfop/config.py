from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional


MODULE_TYPES = ("q", "k", "v", "o", "gate", "up", "down")
ROUTE_SOLVERS = ("balanced_exact", "row_softmax_topk")
ROUTE_GROUPINGS = ("independent", "qk_vo_ffn")


@dataclass
class SVDConfig:
    rank_default: int = 128
    rank_by_module: Dict[str, int] = field(default_factory=dict)
    algorithm: str = "randomized"
    oversample: int = 16
    power_iterations: int = 2
    seed: int = 42
    compute_dtype: str = "float32"

    def validate(self) -> None:
        if self.rank_default <= 0:
            raise ValueError("rank_default must be positive")
        if self.algorithm not in {"randomized", "exact"}:
            raise ValueError("svd.algorithm must be 'randomized' or 'exact'")
        if self.oversample < 0 or self.power_iterations < 0:
            raise ValueError("SVD oversample and power_iterations must be non-negative")
        unknown = set(self.rank_by_module) - set(MODULE_TYPES)
        if unknown:
            raise ValueError(f"Unknown rank_by_module keys: {sorted(unknown)}")

    def requested_rank(self, module_type: str) -> int:
        return int(self.rank_by_module.get(module_type, self.rank_default))


@dataclass
class SpectralPointConfig:
    sigma_power: float = 1.0
    width_normalization: bool = True
    center: bool = False
    eps: float = 1.0e-8

    def validate(self) -> None:
        if self.sigma_power < 0:
            raise ValueError("sigma_power must be non-negative")
        if self.eps <= 0:
            raise ValueError("spectral point eps must be positive")


@dataclass
class SinkhornConfig:
    entropy: float = 0.05
    max_iterations: int = 200
    tolerance: float = 1.0e-6
    check_interval: int = 10

    def validate(self) -> None:
        if self.entropy <= 0:
            raise ValueError("Sinkhorn entropy must be positive")
        if self.max_iterations <= 0 or self.check_interval <= 0:
            raise ValueError("Sinkhorn iteration counts must be positive")
        if self.tolerance <= 0:
            raise ValueError("Sinkhorn tolerance must be positive")


@dataclass
class OTProcrustesConfig:
    sinkhorn: SinkhornConfig = field(default_factory=SinkhornConfig)
    max_alternating_iterations: int = 8
    alternating_tolerance: float = 1.0e-4
    restarts: int = 2
    seed: int = 42

    def validate(self) -> None:
        self.sinkhorn.validate()
        if self.max_alternating_iterations <= 0 or self.restarts <= 0:
            raise ValueError("OT-Procrustes iteration counts must be positive")
        if self.alternating_tolerance <= 0:
            raise ValueError("alternating_tolerance must be positive")


@dataclass
class RouteConfig:
    solver: str = "row_softmax_topk"
    grouping: str = "independent"
    temperature: float = 0.05
    top_source_layers: Optional[int] = 2
    marginal_tolerance: float = 1.0e-7
    support_tolerance: float = 1.0e-9

    def validate(self) -> None:
        if self.solver not in ROUTE_SOLVERS:
            raise ValueError(f"route solver must be one of {ROUTE_SOLVERS}")
        if self.grouping not in ROUTE_GROUPINGS:
            raise ValueError(f"route grouping must be one of {ROUTE_GROUPINGS}")
        if self.temperature <= 0:
            raise ValueError("route temperature must be positive")
        if self.top_source_layers is not None and self.top_source_layers <= 0:
            raise ValueError("top_source_layers must be positive or None")
        if self.marginal_tolerance <= 0:
            raise ValueError("route marginal_tolerance must be positive")
        if self.support_tolerance < 0:
            raise ValueError("route support_tolerance must be non-negative")


@dataclass
class CoreScaleConfig:
    enabled: bool = True
    gamma_min: float = 0.25
    gamma_max: float = 4.0
    minimum_relative_norm: float = 1.0e-6
    eps: float = 1.0e-8

    def validate(self) -> None:
        if self.gamma_min <= 0 or self.gamma_max < self.gamma_min:
            raise ValueError("invalid core scale clipping range")
        if self.minimum_relative_norm < 0 or self.eps <= 0:
            raise ValueError("invalid core scale thresholds")


@dataclass
class FusionConfig:
    beta: float = 0.05
    trust_ratio: Optional[float] = 0.10
    preserve_target_tail: bool = True

    def validate(self) -> None:
        if not 0 <= self.beta <= 1:
            raise ValueError("fusion beta must lie in [0, 1]")
        if self.trust_ratio is not None and self.trust_ratio <= 0:
            raise ValueError("trust_ratio must be positive or None")


@dataclass
class DFOPConfig:
    modules: tuple[str, ...] = MODULE_TYPES
    svd: SVDConfig = field(default_factory=SVDConfig)
    spectral_points: SpectralPointConfig = field(default_factory=SpectralPointConfig)
    ot_procrustes: OTProcrustesConfig = field(default_factory=OTProcrustesConfig)
    route: RouteConfig = field(default_factory=RouteConfig)
    core_scale: CoreScaleConfig = field(default_factory=CoreScaleConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    store_all_stage1_couplings: bool = False

    def validate(self) -> None:
        if not self.modules:
            raise ValueError("At least one module type must be enabled")
        unknown = set(self.modules) - set(MODULE_TYPES)
        if unknown:
            raise ValueError(f"Unknown modules: {sorted(unknown)}")
        self.svd.validate()
        self.spectral_points.validate()
        self.ot_procrustes.validate()
        self.route.validate()
        self.core_scale.validate()
        self.fusion.validate()

    def to_dict(self) -> dict:
        return asdict(self)
