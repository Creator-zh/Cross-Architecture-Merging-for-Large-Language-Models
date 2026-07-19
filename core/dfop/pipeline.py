from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch
import torch.nn as nn

from .config import DFOPConfig, MODULE_TYPES
from .diagnostics import write_json, write_jsonl
from .fusion import fuse_target_weight
from .layer_cost import compute_module_layer_costs
from .layer_route import compute_layer_route
from .lora_export import exact_lora_factors
from .module_registry import MODULE_SPECS, collect_module_linears, common_rank_limit
from .ot_procrustes import solve_ot_procrustes
from .pair_core import COUPLING_MARGINAL_TOLERANCE, compute_pair_core
from .sinkhorn import uniform_mass
from .spectral_points import build_spectral_points
from .svd_cache import compute_svd_record
from .types import SVDRecord


LOGGER = logging.getLogger(__name__)


@dataclass
class DFOPPipelineResult:
    report: dict
    layer_costs: Dict[str, torch.Tensor]
    dense_routes: Dict[str, torch.Tensor]
    routes: Dict[str, torch.Tensor]
    low_rank_updates: Dict[str, List[dict]]


def resolve_compute_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return resolved


def _empty_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _stage1_signature(
    config: DFOPConfig,
    rank_by_module: Mapping[str, int],
    target_linears: Mapping[str, Sequence[nn.Linear]],
    source_linears: Mapping[str, Sequence[nn.Linear]],
    *,
    target_identity: str | None,
    source_identity: str | None,
) -> dict:
    full_config = config.to_dict()
    return {
        "schema_version": 1,
        "target_identity": target_identity,
        "source_identity": source_identity,
        "modules": list(config.modules),
        "rank_by_module": dict(rank_by_module),
        "svd": full_config["svd"],
        "spectral_points": full_config["spectral_points"],
        "ot_procrustes": full_config["ot_procrustes"],
        "target_shapes": {
            name: [list(linear.weight.shape) for linear in target_linears[name]]
            for name in config.modules
        },
        "source_shapes": {
            name: [list(linear.weight.shape) for linear in source_linears[name]]
            for name in config.modules
        },
    }


def _prepare_stage1_cache(directory: Path, signature: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "stage1_manifest.json"
    if manifest.is_file():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if existing != signature:
            raise ValueError(
                f"Stage-1 cache signature mismatch at {directory}; use a new cache directory"
            )
    else:
        write_json(manifest, signature)


def _prepare_stage2_cache(directory: Path, signature: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "stage2_manifest.json"
    if manifest.is_file():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if existing != signature:
            raise ValueError(
                f"Stage-2 cache signature mismatch at {directory}; use a new cache directory"
            )
    else:
        write_json(manifest, signature)


def _save_tensor_atomic(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(tensor.detach().cpu(), temporary)
    os.replace(temporary, path)


def _save_object_atomic(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _load_cached_cost(
    path: Path,
    expected_shape: tuple[int, int],
    *,
    allow_incomplete: bool = False,
) -> torch.Tensor:
    cost = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(cost, torch.Tensor) or tuple(cost.shape) != expected_shape:
        raise ValueError(f"Invalid cached layer cost: {path}")
    cost = cost.to(dtype=torch.float32)
    if torch.isinf(cost).any() or (not allow_incomplete and not torch.isfinite(cost).all()):
        raise ValueError(f"Cached layer cost contains invalid values: {path}")
    return cost


@torch.no_grad()
def build_model_svd_cache(
    module_linears: Mapping[str, Sequence[nn.Linear]],
    config: DFOPConfig,
    *,
    compute_device: torch.device,
    model_seed_offset: int,
    rank_by_module: Mapping[str, int],
) -> Dict[str, List[SVDRecord]]:
    records: Dict[str, List[SVDRecord]] = {}
    for module_index, module_name in enumerate(config.modules):
        rank = int(rank_by_module[module_name])
        module_records: List[SVDRecord] = []
        LOGGER.info(
            "SVD module=%s matrices=%d rank=%d device=%s",
            module_name,
            len(module_linears[module_name]),
            rank,
            compute_device,
        )
        for layer_index, linear in enumerate(module_linears[module_name]):
            if linear.weight.device.type == "meta":
                raise RuntimeError(
                    "DFOP does not accept meta/offloaded parameters in the core pipeline; "
                    "load the model on CPU and use compute_device for streamed GPU compute"
                )
            matrix = linear.weight.detach().to(device=compute_device, dtype=torch.float32)
            record = compute_svd_record(
                matrix,
                rank,
                config.svd,
                seed_offset=model_seed_offset + module_index * 10000 + layer_index,
            )
            record.metadata.update(
                {
                    "module": module_name,
                    "layer": layer_index,
                    "original_dtype": str(linear.weight.dtype).replace("torch.", ""),
                }
            )
            module_records.append(record.to("cpu"))
            del matrix, record
            _empty_cuda_cache(compute_device)
        records[module_name] = module_records
    return records


def _resolve_module_ranks(
    target_linears: Mapping[str, Sequence[nn.Linear]],
    source_linears: Mapping[str, Sequence[nn.Linear]],
    config: DFOPConfig,
) -> Dict[str, int]:
    ranks: Dict[str, int] = {}
    for module_name in config.modules:
        requested = config.svd.requested_rank(module_name)
        limit = common_rank_limit(target_linears[module_name], source_linears[module_name])
        if requested > limit:
            raise ValueError(
                f"Requested rank {requested} for {module_name} exceeds the common limit {limit}"
            )
        ranks[module_name] = requested
    return ranks


def _side_points(
    record: SVDRecord,
    side: str,
    config: DFOPConfig,
    device: torch.device,
) -> torch.Tensor:
    if side == "output":
        basis = record.u
    elif side == "input":
        basis = record.v
    else:
        raise ValueError("side must be 'input' or 'output'")
    return build_spectral_points(basis, record.s, config.spectral_points).to(
        device=device, dtype=torch.float32
    )


@torch.no_grad()
def _solve_pair_transport(
    target_points: torch.Tensor,
    source_points: torch.Tensor,
    config: DFOPConfig | None = None,
):
    """Solve one pair OT problem, retrying only numerically infeasible couplings."""
    ot_config = (config or DFOPConfig()).ot_procrustes
    best = solve_ot_procrustes(target_points, source_points, config=ot_config)
    if best.marginal_error <= COUPLING_MARGINAL_TOLERANCE:
        return best

    initial_iterations = ot_config.sinkhorn.max_iterations
    for retry_iterations in (
        max(900, 3 * initial_iterations),
        max(1800, 6 * initial_iterations),
    ):
        LOGGER.warning(
            "Retrying pair OT: marginal_error=%g exceeds tolerance=%g; max_iterations=%d",
            best.marginal_error,
            COUPLING_MARGINAL_TOLERANCE,
            retry_iterations,
        )
        retry_config = replace(
            ot_config,
            sinkhorn=replace(ot_config.sinkhorn, max_iterations=retry_iterations),
        )
        retry = solve_ot_procrustes(target_points, source_points, config=retry_config)
        if retry.marginal_error < best.marginal_error:
            best = retry
        if best.marginal_error <= COUPLING_MARGINAL_TOLERANCE:
            return best

    return best


@torch.no_grad()
def run_dfop_pipeline(
    target_model: nn.Module,
    source_model: nn.Module,
    config: DFOPConfig | None = None,
    *,
    compute_device: str | torch.device = "auto",
    diagnostics_dir: str | Path | None = None,
    stage1_cache_dir: str | Path | None = None,
    stage2_cache_dir: str | Path | None = None,
    target_identity: str | None = None,
    source_identity: str | None = None,
    apply_updates: bool = True,
    collect_low_rank_updates: bool = False,
) -> DFOPPipelineResult:
    """Run the full data-free pipeline without calling either model's forward."""

    config = config or DFOPConfig()
    config.validate()
    device = resolve_compute_device(compute_device)
    started = time.perf_counter()
    target_linears = collect_module_linears(target_model, config.modules, require_all=True)
    source_linears = collect_module_linears(source_model, config.modules, require_all=True)
    rank_by_module = _resolve_module_ranks(target_linears, source_linears, config)
    diagnostics_output = Path(diagnostics_dir) if diagnostics_dir is not None else None
    if diagnostics_output is not None:
        diagnostics_output.mkdir(parents=True, exist_ok=True)
        write_json(diagnostics_output / "config.json", config.to_dict())
    stage1_output = Path(stage1_cache_dir) if stage1_cache_dir is not None else None
    base_cache_signature = _stage1_signature(
        config,
        rank_by_module,
        target_linears,
        source_linears,
        target_identity=target_identity,
        source_identity=source_identity,
    )
    if stage1_output is not None:
        _prepare_stage1_cache(stage1_output, base_cache_signature)
    stage2_output = Path(stage2_cache_dir) if stage2_cache_dir is not None else None
    if stage2_output is not None:
        full_config = config.to_dict()
        _prepare_stage2_cache(
            stage2_output,
            {
                "schema_version": 1,
                "stage1": base_cache_signature,
                "route": full_config["route"],
                "core_scale": full_config["core_scale"],
            },
        )

    target_records = build_model_svd_cache(
        target_linears,
        config,
        compute_device=device,
        model_seed_offset=0,
        rank_by_module=rank_by_module,
    )
    source_records = build_model_svd_cache(
        source_linears,
        config,
        compute_device=device,
        model_seed_offset=1_000_000,
        rank_by_module=rank_by_module,
    )
    svd_finished = time.perf_counter()

    layer_costs: Dict[str, torch.Tensor] = {}
    dense_routes: Dict[str, torch.Tensor] = {}
    routes: Dict[str, torch.Tensor] = {}
    route_diagnostics: List[dict] = []
    for module_name in config.modules:
        expected_shape = (
            len(target_records[module_name]),
            len(source_records[module_name]),
        )
        cache_path = (
            stage1_output / f"layer_cost_{module_name}.pt"
            if stage1_output is not None
            else None
        )
        initial_cost = None
        if cache_path is not None and cache_path.is_file():
            initial_cost = _load_cached_cost(
                cache_path,
                expected_shape,
                allow_incomplete=True,
            )
            completed = int(torch.isfinite(initial_cost).sum())
            total = initial_cost.numel()
            LOGGER.info(
                "Layer-cost cache module=%s completed=%d/%d path=%s",
                module_name,
                completed,
                total,
                cache_path,
            )
        if initial_cost is not None and torch.isfinite(initial_cost).all():
            cost = initial_cost
        else:
            LOGGER.info(
                "Layer costs module=%s target_layers=%d source_layers=%d residual_side=%s",
                module_name,
                expected_shape[0],
                expected_shape[1],
                MODULE_SPECS[module_name].residual_side,
            )
            cost_result = compute_module_layer_costs(
                target_records[module_name],
                source_records[module_name],
                residual_side=MODULE_SPECS[module_name].residual_side,
                point_config=config.spectral_points,
                ot_config=config.ot_procrustes,
                store_pair_results=False,
                device=device,
                initial_cost=initial_cost,
                checkpoint_callback=(
                    (lambda partial, path=cache_path: _save_tensor_atomic(partial, path))
                    if cache_path is not None
                    else None
                ),
            )
            cost = cost_result.cost.detach().cpu()
            if cache_path is not None:
                _save_tensor_atomic(cost, cache_path)
        route_result = compute_layer_route(cost, config.route)
        layer_costs[module_name] = cost
        dense_routes[module_name] = route_result.dense_route.cpu()
        routes[module_name] = route_result.route.cpu()
        LOGGER.info(
            "Layer route complete module=%s mean_entropy=%.6f",
            module_name,
            float(route_result.entropy.mean()),
        )
        for layer_index in range(cost.shape[0]):
            nonzero = torch.nonzero(route_result.route[layer_index] > 0, as_tuple=False).flatten()
            route_diagnostics.append(
                {
                    "module": module_name,
                    "target_layer": layer_index,
                    "route_entropy": float(route_result.entropy[layer_index]),
                    "effective_source_count": float(
                        route_result.effective_source_count[layer_index]
                    ),
                    "selected_source_layers": nonzero.tolist(),
                    "selected_weights": route_result.route[layer_index, nonzero].tolist(),
                }
            )
        if diagnostics_output is not None:
            _save_tensor_atomic(cost, diagnostics_output / f"layer_cost_{module_name}.pt")
            _save_tensor_atomic(
                route_result.dense_route,
                diagnostics_output / f"route_dense_{module_name}.pt",
            )
            _save_tensor_atomic(
                route_result.route,
                diagnostics_output / f"route_{module_name}.pt",
            )
        _empty_cuda_cache(device)
    route_finished = time.perf_counter()

    pair_diagnostics: List[dict] = []
    fusion_diagnostics: List[dict] = []
    low_rank_updates: Dict[str, List[dict]] = {name: [] for name in config.modules}
    for module_name in config.modules:
        route = routes[module_name]
        for target_index, target_record_cpu in enumerate(target_records[module_name]):
            selected = torch.nonzero(route[target_index] > 0, as_tuple=False).flatten().tolist()
            LOGGER.info(
                "Fusion module=%s target_layer=%d selected_sources=%s",
                module_name,
                target_index,
                selected,
            )
            target_record = target_record_cpu.to(device)
            row_cache_path = (
                stage2_output / f"aggregate_core_{module_name}_{target_index:03d}.pt"
                if stage2_output is not None
                else None
            )
            row_pair_diagnostics: List[dict]
            if row_cache_path is not None and row_cache_path.is_file():
                payload = torch.load(row_cache_path, map_location="cpu", weights_only=True)
                if payload.get("selected_source_layers") != selected:
                    raise ValueError(f"Stage-2 selected-layer mismatch: {row_cache_path}")
                aggregate_core = payload.get("aggregate_core")
                row_pair_diagnostics = payload.get("pair_diagnostics", [])
                if (
                    not isinstance(aggregate_core, torch.Tensor)
                    or aggregate_core.shape != (target_record.rank, target_record.rank)
                    or not torch.isfinite(aggregate_core).all()
                ):
                    raise ValueError(f"Invalid stage-2 aggregate core: {row_cache_path}")
                LOGGER.info(
                    "Loaded aggregate core module=%s target_layer=%d path=%s",
                    module_name,
                    target_index,
                    row_cache_path,
                )
            else:
                selected_cores: List[torch.Tensor] = []
                selected_weights: List[float] = []
                row_pair_diagnostics = []
                for source_index in selected:
                    source_record = source_records[module_name][source_index].to(device)
                    x_out = _side_points(target_record, "output", config, device)
                    y_out = _side_points(source_record, "output", config, device)
                    x_in = _side_points(target_record, "input", config, device)
                    y_in = _side_points(source_record, "input", config, device)
                    output_result = _solve_pair_transport(x_out, y_out, config)
                    input_result = _solve_pair_transport(x_in, y_in, config)
                    output_mass = uniform_mass(
                        x_out.shape[0], device=device, dtype=torch.float32
                    )
                    input_mass = uniform_mass(
                        x_in.shape[0], device=device, dtype=torch.float32
                    )
                    pair = compute_pair_core(
                        target_record,
                        source_record,
                        output_result,
                        input_result,
                        output_mass,
                        input_mass,
                        config.core_scale,
                    )
                    weight = float(route[target_index, source_index])
                    if pair.valid:
                        selected_cores.append(pair.calibrated_core.detach().cpu())
                        selected_weights.append(weight)
                    row_pair_diagnostics.append(
                        {
                            "module": module_name,
                            "target_layer": target_index,
                            "source_layer": source_index,
                            "rank": target_record.rank,
                            "target_shape": target_record.shape,
                            "source_shape": source_record.shape,
                            "route_weight": weight,
                            "output_geometric_cost": output_result.geometric_cost,
                            "input_geometric_cost": input_result.geometric_cost,
                            "output_marginal_error": output_result.marginal_error,
                            "input_marginal_error": input_result.marginal_error,
                            "output_converged": output_result.converged,
                            "input_converged": input_result.converged,
                            "output_sinkhorn_iterations": output_result.sinkhorn_iterations,
                            "input_sinkhorn_iterations": input_result.sinkhorn_iterations,
                            "output_alternating_iterations": output_result.alternating_iterations,
                            "input_alternating_iterations": input_result.alternating_iterations,
                            "core_norm": float(torch.linalg.matrix_norm(pair.core)),
                            "core_scale": pair.scale,
                            "valid": pair.valid,
                            "skip_reason": pair.skip_reason,
                        }
                    )
                    del (
                        source_record,
                        x_out,
                        y_out,
                        x_in,
                        y_in,
                        output_result,
                        input_result,
                        pair,
                    )
                    _empty_cuda_cache(device)

                if not selected_cores:
                    raise RuntimeError(
                        f"No valid source core for module={module_name}, target_layer={target_index}"
                    )
                normalized_weights = torch.tensor(selected_weights, dtype=torch.float32)
                normalized_weights = normalized_weights / normalized_weights.sum()
                aggregate_core = torch.zeros_like(selected_cores[0])
                for weight, core in zip(normalized_weights, selected_cores):
                    aggregate_core.add_(core, alpha=float(weight))
                if row_cache_path is not None:
                    _save_object_atomic(
                        {
                            "selected_source_layers": selected,
                            "aggregate_core": aggregate_core,
                            "pair_diagnostics": row_pair_diagnostics,
                        },
                        row_cache_path,
                    )
            pair_diagnostics.extend(row_pair_diagnostics)

            target_linear = target_linears[module_name][target_index]
            fusion = fuse_target_weight(
                target_linear.weight,
                target_record,
                aggregate_core.to(device),
                config.fusion,
            )
            if collect_low_rank_updates:
                factors = exact_lora_factors(
                    target_record,
                    aggregate_core.to(device),
                    beta=config.fusion.beta,
                    trust_coefficient=fusion.trust_coefficient,
                )
                low_rank_updates[module_name].append(
                    {
                        "target_layer": target_index,
                        "target_shape": tuple(target_record.shape),
                        "rank": factors.rank,
                        "lora_a": factors.lora_a.detach().cpu(),
                        "lora_b": factors.lora_b.detach().cpu(),
                    }
                )
            if apply_updates:
                target_linear.weight.copy_(
                    fusion.weight.to(
                        device=target_linear.weight.device,
                        dtype=target_linear.weight.dtype,
                    )
                )
            fusion_diagnostics.append(
                {
                    "module": module_name,
                    "target_layer": target_index,
                    "selected_source_layers": selected,
                    "trust_coefficient": fusion.trust_coefficient,
                    "relative_update_norm": fusion.relative_update_norm,
                    "applied": apply_updates,
                }
            )
            del target_record, aggregate_core, fusion
            _empty_cuda_cache(device)

    fusion_finished = time.perf_counter()
    report = {
        "method": "dfop",
        "modules": list(config.modules),
        "target_layers": {name: len(target_linears[name]) for name in config.modules},
        "source_layers": {name: len(source_linears[name]) for name in config.modules},
        "rank_by_module": rank_by_module,
        "compute_device": str(device),
        "apply_updates": apply_updates,
        "collected_low_rank_updates": collect_low_rank_updates,
        "timing_seconds": {
            "svd": svd_finished - started,
            "layer_cost_and_route": route_finished - svd_finished,
            "pair_transport_and_fusion": fusion_finished - route_finished,
            "total": fusion_finished - started,
        },
    }

    if diagnostics_output is not None:
        output = diagnostics_output
        write_json(output / "run_report.json", report)
        write_jsonl(output / "route_diagnostics.jsonl", route_diagnostics)
        write_jsonl(output / "pair_diagnostics.jsonl", pair_diagnostics)
        write_jsonl(output / "fusion_diagnostics.jsonl", fusion_diagnostics)

    return DFOPPipelineResult(
        report=report,
        layer_costs=layer_costs,
        dense_routes=dense_routes,
        routes=routes,
        low_rank_updates=low_rank_updates,
    )
