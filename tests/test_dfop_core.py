import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from core.dfop.barycentric_map import coupling_to_barycentric_map
from core.dfop.config import (
    CoreScaleConfig,
    FusionConfig,
    OTProcrustesConfig,
    RouteConfig,
    SinkhornConfig,
    SpectralPointConfig,
    SVDConfig,
    DFOPConfig,
)
from core.dfop.fusion import aggregate_cores, fuse_target_weight
from core.dfop.layer_route import compute_layer_route
from core.dfop.layer_cost import compute_module_layer_costs
from core.dfop.lora_export import exact_lora_factors
from core.dfop.ot_procrustes import solve_ot_procrustes
from core.dfop.pipeline import _solve_pair_transport, run_dfop_pipeline
from core.dfop.module_registry import collect_module_linears
from core.dfop.pair_core import compute_pair_core
from core.dfop.sinkhorn import log_sinkhorn, uniform_mass
from core.dfop.spectral_points import build_spectral_points, weighted_mean_energy
from core.dfop.svd_cache import exact_svd
from core.dfop.types import OTProcrustesResult
from core.dfop.task_presets import get_task_preset


def _result(coupling: torch.Tensor) -> OTProcrustesResult:
    rank = min(coupling.shape)
    return OTProcrustesResult(
        coupling=coupling,
        rotation=torch.eye(rank),
        geometric_cost=0.0,
        regularized_objective=0.0,
        marginal_error=0.0,
        sinkhorn_iterations=1,
        alternating_iterations=1,
        restart=0,
        converged=True,
    )


class SpectralPointTests(unittest.TestCase):
    def test_sqrt_width_normalization_sets_uniform_energy_to_one(self):
        q, _ = torch.linalg.qr(torch.randn(13, 4))
        singular_values = torch.tensor([4.0, 2.0, 1.0, 0.5])
        points = build_spectral_points(q, singular_values, SpectralPointConfig())
        self.assertAlmostEqual(weighted_mean_energy(points), 1.0, places=5)

    def test_global_weight_scale_does_not_change_points(self):
        q, _ = torch.linalg.qr(torch.randn(9, 3))
        singular_values = torch.tensor([3.0, 2.0, 1.0])
        first = build_spectral_points(q, singular_values)
        second = build_spectral_points(q, 17.0 * singular_values)
        self.assertTrue(torch.allclose(first, second, atol=1e-6))


class TaskPresetTests(unittest.TestCase):
    def test_three_paper_task_presets_have_expected_models_and_sft_sizes(self):
        medical = get_task_preset("medical")
        thai = get_task_preset("thai")
        malay = get_task_preset("malay")
        self.assertEqual(medical.target_hf_id, "PathFinderKR/Llama-3-1B-Medical-Instruct")
        self.assertEqual(thai.sft_samples_declared, 8000)
        self.assertEqual(thai.sft_samples_legacy, 2000)
        self.assertEqual(malay.sft_dataset_type, "malaysian_sft")
        self.assertEqual(len(thai.primary_eval_tasks), 3)
        self.assertEqual(medical.source_hf_id, malay.source_hf_id)


class SinkhornTests(unittest.TestCase):
    def test_log_sinkhorn_matches_rectangular_uniform_marginals(self):
        cost = torch.rand(5, 8)
        mass_x = uniform_mass(5, device=cost.device, dtype=cost.dtype)
        mass_y = uniform_mass(8, device=cost.device, dtype=cost.dtype)
        result = log_sinkhorn(
            cost,
            mass_x,
            mass_y,
            SinkhornConfig(entropy=0.1, max_iterations=500, tolerance=1e-7),
        )
        self.assertLess(result.marginal_error, 2e-6)
        self.assertTrue(torch.allclose(result.coupling.sum(1), mass_x, atol=2e-6))
        self.assertTrue(torch.allclose(result.coupling.sum(0), mass_y, atol=2e-6))


class OTProcrustesTests(unittest.TestCase):
    def test_pair_transport_retries_when_initial_marginal_is_infeasible(self):
        points = torch.randn(3, 2)
        config = DFOPConfig(
            ot_procrustes=OTProcrustesConfig(
                sinkhorn=SinkhornConfig(max_iterations=200)
            )
        )
        coupling = torch.full((3, 3), 1.0 / 9.0)
        initial = replace(_result(coupling), marginal_error=2.0e-4, converged=False)
        retried = replace(_result(coupling), marginal_error=1.0e-5, converged=False)

        with patch(
            "core.dfop.pipeline.solve_ot_procrustes",
            side_effect=[initial, retried],
        ) as solve:
            result = _solve_pair_transport(points, points, config)

        self.assertIs(result, retried)
        self.assertEqual(solve.call_count, 2)
        self.assertEqual(solve.call_args_list[1].kwargs["config"].sinkhorn.max_iterations, 900)

    def test_recovers_permutation_and_rotation(self):
        torch.manual_seed(3)
        x = torch.randn(7, 3)
        x = x / torch.sqrt(x.square().sum() / x.shape[0])
        q, _ = torch.linalg.qr(torch.randn(3, 3))
        permutation = torch.tensor([4, 1, 6, 0, 3, 5, 2])
        # Row form of x = R y is y = x R.
        y = x[permutation] @ q
        config = OTProcrustesConfig(
            sinkhorn=SinkhornConfig(
                entropy=0.002,
                max_iterations=1000,
                tolerance=1e-7,
                check_interval=10,
            ),
            max_alternating_iterations=30,
            alternating_tolerance=1e-7,
            restarts=4,
            seed=11,
        )
        result = solve_ot_procrustes(x, y, config=config)
        self.assertLess(result.geometric_cost, 2e-3)
        self.assertLess(result.marginal_error, 3e-6)
        identity = result.rotation.transpose(0, 1) @ result.rotation
        self.assertTrue(torch.allclose(identity, torch.eye(3), atol=1e-5))

    def test_row_permutation_does_not_change_cost(self):
        torch.manual_seed(5)
        x = torch.randn(6, 2)
        y = torch.randn(9, 2)
        config = OTProcrustesConfig(
            sinkhorn=SinkhornConfig(entropy=0.03, max_iterations=500),
            max_alternating_iterations=12,
            restarts=2,
        )
        first = solve_ot_procrustes(x, y, config=config).geometric_cost
        second = solve_ot_procrustes(x[torch.randperm(6)], y, config=config).geometric_cost
        self.assertAlmostEqual(first, second, places=5)


class RouteTests(unittest.TestCase):
    def test_route_is_row_normalized_and_topk_sparse(self):
        cost = torch.tensor([[0.1, 0.5, 0.2], [0.8, 0.3, 0.4]])
        result = compute_layer_route(cost, RouteConfig(temperature=0.1, top_source_layers=2))
        self.assertTrue(torch.allclose(result.route.sum(1), torch.ones(2)))
        self.assertTrue(torch.equal((result.route > 0).sum(1), torch.tensor([2, 2])))
        self.assertFalse(torch.allclose(result.route.sum(0), torch.full((3,), 2 / 3)))


class LayerCostCheckpointTests(unittest.TestCase):
    def test_partial_cost_matrix_is_resumed_without_overwriting_completed_pairs(self):
        torch.manual_seed(23)
        targets = [exact_svd(torch.randn(4, 3), rank=2) for _ in range(2)]
        sources = [exact_svd(torch.randn(5, 3), rank=2) for _ in range(2)]
        initial = torch.full((2, 2), float("nan"))
        initial[0, 0] = 0.123
        checkpoints = []
        result = compute_module_layer_costs(
            targets,
            sources,
            residual_side="input",
            ot_config=OTProcrustesConfig(
                sinkhorn=SinkhornConfig(entropy=0.1, max_iterations=50),
                max_alternating_iterations=2,
                restarts=1,
            ),
            initial_cost=initial,
            checkpoint_callback=lambda cost: checkpoints.append(cost.clone()),
        )
        self.assertAlmostEqual(float(result.cost[0, 0]), 0.123, places=6)
        self.assertTrue(torch.isfinite(result.cost).all())
        self.assertEqual(len(checkpoints), 2)


class PairCoreAndFusionTests(unittest.TestCase):
    def test_identity_couplings_recover_source_core(self):
        weight = torch.diag(torch.tensor([4.0, 2.0, 1.0]))
        record = exact_svd(weight, rank=3)
        mass = torch.full((3,), 1 / 3)
        coupling = torch.diag(mass)
        result = _result(coupling)
        pair = compute_pair_core(
            record,
            record,
            result,
            result,
            mass,
            mass,
            CoreScaleConfig(enabled=True),
        )
        self.assertTrue(pair.valid)
        self.assertTrue(torch.allclose(pair.core, torch.diag(record.s), atol=1e-5))
        self.assertTrue(torch.allclose(pair.calibrated_core, pair.core, atol=1e-5))
        self.assertAlmostEqual(pair.scale, 1.0, places=5)

    def test_small_core_matches_explicit_transport_projection(self):
        torch.manual_seed(7)
        target = exact_svd(torch.randn(4, 3), rank=3)
        source = exact_svd(torch.randn(5, 6), rank=3)
        mass_out = torch.full((4,), 1 / 4)
        mass_in = torch.full((3,), 1 / 3)
        pi_out = torch.full((4, 5), 1 / 20)
        pi_in = torch.full((3, 6), 1 / 18)
        out_result = _result(pi_out)
        in_result = _result(pi_in)
        pair = compute_pair_core(
            target,
            source,
            out_result,
            in_result,
            mass_out,
            mass_in,
            CoreScaleConfig(enabled=False, minimum_relative_norm=0.0),
        )
        t_out = coupling_to_barycentric_map(pi_out, mass_out)
        t_in = coupling_to_barycentric_map(pi_in, mass_in)
        explicit_weight = t_out @ (source.u @ torch.diag(source.s) @ source.v.T) @ t_in.T
        explicit_core = target.u.T @ explicit_weight @ target.v
        self.assertTrue(torch.allclose(pair.core, explicit_core, atol=1e-5))

    def test_beta_zero_is_identity_and_trust_bound_is_enforced(self):
        torch.manual_seed(13)
        weight = torch.randn(5, 4)
        record = exact_svd(weight, rank=3)
        aggregate = 20.0 * torch.eye(3)
        identity = fuse_target_weight(
            weight,
            record,
            aggregate,
            FusionConfig(beta=0.0, trust_ratio=0.1),
        )
        self.assertTrue(torch.equal(identity.weight, weight))

        bounded = fuse_target_weight(
            weight,
            record,
            aggregate,
            FusionConfig(beta=1.0, trust_ratio=0.05),
        )
        self.assertLessEqual(bounded.relative_update_norm, 0.050001)

    def test_full_rank_lora_factors_equal_dense_dfop_update(self):
        torch.manual_seed(17)
        weight = torch.randn(6, 5)
        record = exact_svd(weight, rank=3)
        aggregate = torch.randn(3, 3)
        config = FusionConfig(beta=0.2, trust_ratio=0.07)
        fused = fuse_target_weight(weight, record, aggregate, config)
        factors = exact_lora_factors(
            record,
            aggregate,
            beta=config.beta,
            trust_coefficient=fused.trust_coefficient,
        )
        self.assertEqual(factors.rank, 3)
        self.assertTrue(torch.allclose(factors.dense_delta(), fused.delta, atol=1e-6))

    def test_aggregate_renormalizes_after_invalid_pair(self):
        weight = torch.diag(torch.tensor([2.0, 1.0]))
        record = exact_svd(weight, rank=2)
        mass = torch.full((2,), 0.5)
        result = _result(torch.diag(mass))
        valid = compute_pair_core(record, record, result, result, mass, mass)
        invalid = compute_pair_core(
            record,
            exact_svd(torch.zeros_like(weight), rank=2),
            result,
            result,
            mass,
            mass,
            CoreScaleConfig(minimum_relative_norm=1e-3),
        )
        aggregate = aggregate_cores([valid, invalid], torch.tensor([0.2, 0.8]))
        self.assertTrue(torch.allclose(aggregate, valid.calibrated_core))


class _TinyAttention(nn.Module):
    def __init__(self, hidden: int, kv: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, kv, bias=False)
        self.v_proj = nn.Linear(hidden, kv, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)


class _TinyMLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)


class _TinyBlock(nn.Module):
    def __init__(self, hidden: int, kv: int, intermediate: int):
        super().__init__()
        self.self_attn = _TinyAttention(hidden, kv)
        self.mlp = _TinyMLP(hidden, intermediate)


class _NoForwardTinyModel(nn.Module):
    def __init__(self, layers: int, hidden: int, kv: int, intermediate: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [_TinyBlock(hidden, kv, intermediate) for _ in range(layers)]
        )

    def forward(self, *args, **kwargs):
        raise AssertionError("DFOP must never call model.forward")


class PipelineTests(unittest.TestCase):
    def test_end_to_end_is_data_free_and_supports_heterogeneous_shapes(self):
        torch.manual_seed(19)
        target = _NoForwardTinyModel(layers=2, hidden=4, kv=3, intermediate=6)
        source = _NoForwardTinyModel(layers=3, hidden=5, kv=4, intermediate=7)
        target_state = {name: value.clone() for name, value in target.state_dict().items()}
        source_state = {name: value.clone() for name, value in source.state_dict().items()}
        before = target.layers[0].self_attn.q_proj.weight.detach().clone()
        config = DFOPConfig(
            svd=SVDConfig(rank_default=2, algorithm="exact"),
            ot_procrustes=OTProcrustesConfig(
                sinkhorn=SinkhornConfig(
                    entropy=0.1,
                    max_iterations=100,
                    tolerance=1e-5,
                    check_interval=5,
                ),
                max_alternating_iterations=3,
                restarts=1,
            ),
            route=RouteConfig(temperature=0.2, top_source_layers=1),
            fusion=FusionConfig(beta=0.1, trust_ratio=0.2),
        )
        with tempfile.TemporaryDirectory() as temporary:
            stage1_cache = Path(temporary) / "stage1_cache"
            stage2_cache = Path(temporary) / "stage2_cache"
            result = run_dfop_pipeline(
                target,
                source,
                config,
                compute_device="cpu",
                diagnostics_dir=temporary,
                stage1_cache_dir=stage1_cache,
                stage2_cache_dir=stage2_cache,
                target_identity="tiny-target",
                source_identity="tiny-source",
            )
            for module_name in config.modules:
                self.assertEqual(tuple(result.layer_costs[module_name].shape), (2, 3))
                self.assertTrue(
                    torch.allclose(result.routes[module_name].sum(1), torch.ones(2))
                )
                self.assertTrue(
                    torch.equal(
                        (result.routes[module_name] > 0).sum(1), torch.ones(2, dtype=torch.long)
                    )
                )
            self.assertFalse(
                torch.equal(before, target.layers[0].self_attn.q_proj.weight)
            )
            diagnostics = Path(temporary)
            self.assertTrue((diagnostics / "run_report.json").is_file())
            self.assertTrue((diagnostics / "route_dense_q.pt").is_file())
            self.assertTrue((diagnostics / "pair_diagnostics.jsonl").is_file())
            self.assertTrue((stage1_cache / "stage1_manifest.json").is_file())
            self.assertTrue((stage1_cache / "layer_cost_down.pt").is_file())
            self.assertTrue((stage2_cache / "stage2_manifest.json").is_file())
            self.assertTrue((stage2_cache / "aggregate_core_q_000.pt").is_file())

            resumed_target = _NoForwardTinyModel(
                layers=2, hidden=4, kv=3, intermediate=6
            )
            resumed_source = _NoForwardTinyModel(
                layers=3, hidden=5, kv=4, intermediate=7
            )
            resumed_target.load_state_dict(target_state)
            resumed_source.load_state_dict(source_state)
            with patch(
                "core.dfop.layer_cost.solve_ot_procrustes",
                side_effect=AssertionError("stage-1 OT should be cached"),
            ), patch(
                "core.dfop.pipeline.solve_ot_procrustes",
                side_effect=AssertionError("stage-2 OT should be cached"),
            ):
                resumed = run_dfop_pipeline(
                    resumed_target,
                    resumed_source,
                    config,
                    compute_device="cpu",
                    stage1_cache_dir=stage1_cache,
                    stage2_cache_dir=stage2_cache,
                    target_identity="tiny-target",
                    source_identity="tiny-source",
                )
            self.assertEqual(resumed.report["rank_by_module"]["q"], 2)

    def test_attention_checkpoint_can_be_derived_exactly_from_full_checkpoint(self):
        torch.manual_seed(29)
        full_target = _NoForwardTinyModel(layers=1, hidden=4, kv=3, intermediate=5)
        direct_target = _NoForwardTinyModel(layers=1, hidden=4, kv=3, intermediate=5)
        direct_target.load_state_dict(full_target.state_dict())
        original_target = _NoForwardTinyModel(layers=1, hidden=4, kv=3, intermediate=5)
        original_target.load_state_dict(full_target.state_dict())
        full_source = _NoForwardTinyModel(layers=2, hidden=5, kv=4, intermediate=6)
        direct_source = _NoForwardTinyModel(layers=2, hidden=5, kv=4, intermediate=6)
        direct_source.load_state_dict(full_source.state_dict())
        config = DFOPConfig(
            svd=SVDConfig(rank_default=2, algorithm="exact"),
            ot_procrustes=OTProcrustesConfig(
                sinkhorn=SinkhornConfig(
                    entropy=0.1,
                    max_iterations=300,
                    tolerance=1e-6,
                ),
                max_alternating_iterations=2,
                restarts=1,
            ),
            route=RouteConfig(temperature=0.2, top_source_layers=1),
            fusion=FusionConfig(beta=0.1, trust_ratio=0.2),
        )
        run_dfop_pipeline(full_target, full_source, config, compute_device="cpu")
        attn_config = replace(config, modules=("q", "k", "v", "o"))
        run_dfop_pipeline(direct_target, direct_source, attn_config, compute_device="cpu")

        derived_ffn = collect_module_linears(full_target, ("gate", "up", "down"))
        original_ffn = collect_module_linears(original_target, ("gate", "up", "down"))
        with torch.no_grad():
            for module_name in ("gate", "up", "down"):
                for destination, source in zip(
                    derived_ffn[module_name], original_ffn[module_name]
                ):
                    destination.weight.copy_(source.weight)

        for name, direct_weight in direct_target.state_dict().items():
            self.assertTrue(
                torch.allclose(full_target.state_dict()[name], direct_weight, atol=1e-6),
                msg=name,
            )


if __name__ == "__main__":
    unittest.main()
