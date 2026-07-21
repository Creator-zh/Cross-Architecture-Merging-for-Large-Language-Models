import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from core.aci.alignment import build_residual_anchor, deterministic_anchor_indices
from core.aci.attention import (
    _align_ov_gauge,
    _align_qk_rope_gauge,
    attention_geometry,
    contract_attention,
    contract_attention_circuit,
    rope_frequency_indices,
)
from core.aci.config import ACIConfig
from core.aci.ffn import contract_ffn
from core.aci.injection import inject_protected_delta
from core.aci.pipeline import monotonic_layer_groups, run_aci_pipeline
from core.aci.registry import block_linears
from core.aci.safe_injection import inject_safe_attention, inject_safe_ffn


class _TinyAttention(nn.Module):
    def __init__(self, hidden: int, query_heads: int, kv_heads: int, head_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden, query_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(query_heads * head_dim, hidden, bias=False)


class _TinyMLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)


class _TinyBlock(nn.Module):
    def __init__(
        self,
        hidden: int,
        intermediate: int,
        query_heads: int,
        kv_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.self_attn = _TinyAttention(hidden, query_heads, kv_heads, head_dim)
        self.mlp = _TinyMLP(hidden, intermediate)


class _TinyModel(nn.Module):
    def __init__(
        self,
        *,
        layers: int,
        hidden: int,
        intermediate: int,
        query_heads: int,
        kv_heads: int,
        head_dim: int,
        vocabulary: int = 32,
        rope_theta: float = 10_000.0,
        rope_scaling: dict | None = None,
    ):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden,
            num_attention_heads=query_heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.embed_tokens = nn.Embedding(vocabulary, hidden)
        self.layers = nn.ModuleList(
            [
                _TinyBlock(hidden, intermediate, query_heads, kv_heads, head_dim)
                for _ in range(layers)
            ]
        )
        self.lm_head = nn.Linear(hidden, vocabulary, bias=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, *args, **kwargs):
        raise AssertionError("ACI must not call model.forward")


def _reference_model() -> _TinyModel:
    return _TinyModel(
        layers=2,
        hidden=4,
        intermediate=6,
        query_heads=2,
        kv_heads=1,
        head_dim=2,
    )


def _source_model() -> _TinyModel:
    return _TinyModel(
        layers=4,
        hidden=8,
        intermediate=10,
        query_heads=2,
        kv_heads=1,
        head_dim=4,
    )


class AnchorTests(unittest.TestCase):
    def test_anchor_indices_are_unique_and_cover_vocab(self):
        indices = deterministic_anchor_indices(101, 16)
        self.assertEqual(indices.numel(), 16)
        self.assertEqual(torch.unique(indices).numel(), 16)
        self.assertGreaterEqual(int(indices.min()), 0)
        self.assertLess(int(indices.max()), 101)

    def test_rectangular_anchor_has_orthonormal_columns(self):
        torch.manual_seed(3)
        source = _source_model()
        reference = _reference_model()
        result = build_residual_anchor(
            source.embed_tokens,
            reference.embed_tokens,
            source.lm_head,
            reference.lm_head,
            ACIConfig(anchor_tokens=16, anchor_chunk_size=5, ffn_sketch_dim=2),
            device=torch.device("cpu"),
        )
        self.assertEqual(tuple(result.source_to_reference.shape), (8, 4))
        identity = result.source_to_reference.T @ result.source_to_reference
        self.assertTrue(torch.allclose(identity, torch.eye(4), atol=1e-5))
        self.assertEqual(tuple(result.reference_sketch_basis.shape), (4, 2))


class StructureTests(unittest.TestCase):
    def test_rope_selection_preserves_frequency_coordinates(self):
        self.assertTrue(
            torch.equal(
                rope_frequency_indices(8, 4),
                torch.tensor([0, 2, 4, 6]),
            )
        )

    def test_llama3_rope_selection_uses_real_scaling_frequencies(self):
        source_model = _TinyModel(
            layers=1,
            hidden=128,
            intermediate=8,
            query_heads=1,
            kv_heads=1,
            head_dim=128,
            rope_theta=500_000.0,
            rope_scaling={
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
        )
        target_model = _TinyModel(
            layers=1,
            hidden=64,
            intermediate=8,
            query_heads=1,
            kv_heads=1,
            head_dim=64,
            rope_theta=500_000.0,
            rope_scaling={
                "factor": 32.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
        )
        source_geometry = attention_geometry(source_model)
        target_geometry = attention_geometry(target_model)
        indices = rope_frequency_indices(source_geometry, target_geometry)
        pair_indices = indices[: target_geometry.head_dim // 2]
        legacy_pairs = torch.arange(target_geometry.head_dim // 2) * 2
        source_inverse = torch.tensor(source_geometry.rope_inverse_frequencies)
        target_inverse = torch.tensor(target_geometry.rope_inverse_frequencies)
        matched_error = (
            source_inverse[pair_indices].log() - target_inverse.log()
        ).abs().sum()
        legacy_error = (
            source_inverse[legacy_pairs].log() - target_inverse.log()
        ).abs().sum()
        self.assertTrue(torch.all(pair_indices[1:] > pair_indices[:-1]))
        self.assertFalse(torch.equal(pair_indices, legacy_pairs))
        self.assertLess(float(matched_error), float(legacy_error))

    def test_attention_and_ffn_contract_to_reference_shapes(self):
        torch.manual_seed(5)
        source_model = _source_model()
        reference_model = _reference_model()
        source = block_linears(source_model.layers[0])
        reference = block_linears(reference_model.layers[0])
        projection, _ = torch.linalg.qr(torch.randn(8, 4))
        basis, _ = torch.linalg.qr(torch.randn(4, 2))
        attention, match_attention = contract_attention(
            source,
            reference,
            projection,
            basis,
            attention_geometry(source_model),
            attention_geometry(reference_model),
        )
        self.assertEqual(tuple(attention["q"].shape), (4, 4))
        self.assertEqual(tuple(attention["k"].shape), (2, 4))
        self.assertEqual(tuple(attention["o"].shape), (4, 4))
        self.assertEqual(torch.unique(match_attention.query_assignment).numel(), 2)
        self.assertEqual(torch.unique(match_attention.group_assignment).numel(), 1)
        ffn, match = contract_ffn(
            source,
            reference,
            projection,
            basis,
            ACIConfig(ffn_sketch_dim=2, ffn_candidate_k=4),
        )
        self.assertEqual(tuple(ffn["gate"].shape), (6, 4))
        self.assertEqual(tuple(ffn["up"].shape), (6, 4))
        self.assertEqual(tuple(ffn["down"].shape), (4, 6))
        self.assertEqual(torch.unique(match.source_indices).numel(), 6)
        self.assertEqual(match.reused_sources, 0)

    def test_gqa_group_and_query_assignments_are_bijections(self):
        torch.manual_seed(7)
        source_model = _TinyModel(
            layers=1,
            hidden=16,
            intermediate=20,
            query_heads=4,
            kv_heads=2,
            head_dim=4,
        )
        reference_model = _TinyModel(
            layers=1,
            hidden=8,
            intermediate=12,
            query_heads=4,
            kv_heads=2,
            head_dim=2,
        )
        projection, _ = torch.linalg.qr(torch.randn(16, 8))
        basis, _ = torch.linalg.qr(torch.randn(8, 3))
        _, match = contract_attention(
            block_linears(source_model.layers[0]),
            block_linears(reference_model.layers[0]),
            projection,
            basis,
            attention_geometry(source_model),
            attention_geometry(reference_model),
        )
        self.assertEqual(torch.unique(match.group_assignment).numel(), 2)
        self.assertEqual(torch.unique(match.query_assignment).numel(), 4)

    def test_circuit_matching_aligns_gauges_and_returns_head_confidence(self):
        torch.manual_seed(9)
        source_model = _source_model()
        reference_model = _reference_model()
        projection, _ = torch.linalg.qr(torch.randn(8, 4))
        basis, _ = torch.linalg.qr(torch.randn(4, 2))
        attention, match = contract_attention_circuit(
            block_linears(source_model.layers[0]),
            block_linears(reference_model.layers[0]),
            projection,
            basis,
            attention_geometry(source_model),
            attention_geometry(reference_model),
        )
        self.assertEqual(tuple(attention["q"].shape), (4, 4))
        self.assertEqual(tuple(attention["k"].shape), (2, 4))
        self.assertEqual(tuple(match.head_confidences.shape), (2,))
        self.assertGreaterEqual(
            match.qk_gauge_cosine_after + 1.0e-6,
            match.qk_gauge_cosine_before,
        )
        self.assertGreaterEqual(
            match.ov_gauge_cosine_after + 1.0e-6,
            match.ov_gauge_cosine_before,
        )

    def test_gauge_alignment_preserves_qk_and_ov_circuits(self):
        torch.manual_seed(10)
        geometry = attention_geometry(_reference_model())
        basis = torch.eye(geometry.hidden_size)
        q = torch.randn(
            geometry.query_heads * geometry.head_dim,
            geometry.hidden_size,
        )
        k = torch.randn(
            geometry.kv_heads * geometry.head_dim,
            geometry.hidden_size,
        )
        v = torch.randn_like(k)
        o = torch.randn(
            geometry.hidden_size,
            geometry.query_heads * geometry.head_dim,
        )
        aligned_q, aligned_k, _, _ = _align_qk_rope_gauge(
            q,
            k,
            torch.randn_like(q),
            torch.randn_like(k),
            geometry,
            basis,
            eps=1.0e-8,
        )
        aligned_v, aligned_o, _, _ = _align_ov_gauge(
            v,
            o,
            torch.randn_like(v),
            torch.randn_like(o),
            geometry,
            basis,
            eps=1.0e-8,
        )
        left = torch.randn(geometry.hidden_size)
        right = torch.randn(geometry.hidden_size)
        angle_q, angle_k = 0.37, -0.22
        rope_q = torch.tensor(
            [
                [torch.cos(torch.tensor(angle_q)), -torch.sin(torch.tensor(angle_q))],
                [torch.sin(torch.tensor(angle_q)), torch.cos(torch.tensor(angle_q))],
            ]
        )
        rope_k = torch.tensor(
            [
                [torch.cos(torch.tensor(angle_k)), -torch.sin(torch.tensor(angle_k))],
                [torch.sin(torch.tensor(angle_k)), torch.cos(torch.tensor(angle_k))],
            ]
        )
        for head in range(geometry.query_heads):
            q_slice = slice(head * geometry.head_dim, (head + 1) * geometry.head_dim)
            before_qk = (rope_q @ (q[q_slice] @ left)) @ (
                rope_k @ (k @ right)
            )
            after_qk = (rope_q @ (aligned_q[q_slice] @ left)) @ (
                rope_k @ (aligned_k @ right)
            )
            self.assertTrue(torch.allclose(before_qk, after_qk, atol=1.0e-5))
            o_slice = slice(head * geometry.head_dim, (head + 1) * geometry.head_dim)
            before_ov = o[:, o_slice] @ (v @ right)
            after_ov = aligned_o[:, o_slice] @ (aligned_v @ right)
            self.assertTrue(torch.allclose(before_ov, after_ov, atol=1.0e-5))

    def test_layer_groups_are_monotonic_and_exhaustive(self):
        self.assertEqual(
            monotonic_layer_groups(4, 8),
            [[0, 1], [2, 3], [4, 5], [6, 7]],
        )


class InjectionTests(unittest.TestCase):
    def test_reference_delta_is_added_without_scaling_target_task_vector(self):
        reference = torch.eye(3)
        domain_delta = torch.full((3, 3), 0.1)
        target = reference + domain_delta
        source = 2.0 * reference
        result = inject_protected_delta(
            target,
            reference,
            source,
            beta=0.05,
        )
        reconstructed_domain = result.weight - (
            reference + result.delta
        )
        self.assertTrue(torch.allclose(reconstructed_domain, domain_delta, atol=1e-6))
        self.assertLessEqual(result.relative_update_norm, 0.050001)

    def test_beta_zero_is_exact_identity(self):
        target = torch.randn(4, 3)
        result = inject_protected_delta(
            target,
            torch.randn(4, 3),
            torch.randn(4, 3),
            beta=0.0,
        )
        self.assertTrue(torch.equal(result.weight, target))

    def test_safe_ffn_projects_conflicts_and_respects_neuron_gate(self):
        reference = {
            "gate": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "up": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "down": torch.eye(2),
        }
        target = {name: 2.0 * value for name, value in reference.items()}
        source = {name: torch.zeros_like(value) for name, value in reference.items()}
        result = inject_safe_ffn(
            target,
            reference,
            source,
            torch.tensor([1.0, 0.0]),
            beta=0.1,
            eps=1.0e-8,
        )
        self.assertEqual(result.conflict_fraction, 1.0)
        self.assertAlmostEqual(result.removed_conflict_norm_ratio, 1.0, places=5)
        for name in ("gate", "up", "down"):
            self.assertTrue(torch.allclose(result.weights[name], target[name]))

        swapped = {
            "gate": torch.flip(reference["gate"], dims=(0,)),
            "up": torch.flip(reference["up"], dims=(0,)),
            "down": torch.flip(reference["down"], dims=(1,)),
        }
        gated = inject_safe_ffn(
            reference,
            reference,
            swapped,
            torch.tensor([1.0, 0.0]),
            beta=0.1,
            eps=1.0e-8,
        )
        self.assertFalse(torch.equal(gated.weights["gate"][0], reference["gate"][0]))
        self.assertFalse(torch.equal(gated.weights["up"][0], reference["up"][0]))
        self.assertFalse(torch.equal(gated.weights["down"][:, 0], reference["down"][:, 0]))
        self.assertTrue(torch.equal(gated.weights["gate"][1], reference["gate"][1]))
        self.assertTrue(torch.equal(gated.weights["up"][1], reference["up"][1]))
        self.assertTrue(torch.equal(gated.weights["down"][:, 1], reference["down"][:, 1]))

    def test_safe_attention_zero_confidence_is_identity(self):
        model = _reference_model()
        geometry = attention_geometry(model)
        block = block_linears(model.layers[0]).as_dict()
        target = {
            name: block[name].weight.detach().clone()
            for name in ("q", "k", "v", "o")
        }
        reference = {name: value - 0.01 for name, value in target.items()}
        source = {name: torch.randn_like(value) for name, value in target.items()}
        result = inject_safe_attention(
            target,
            reference,
            source,
            torch.zeros(geometry.query_heads),
            geometry,
            beta=0.1,
            eps=1.0e-8,
        )
        for name in ("q", "k", "v", "o"):
            self.assertTrue(torch.equal(result.weights[name], target[name]))
        self.assertEqual(result.joint_relative_update_norm, 0.0)


class PipelineTests(unittest.TestCase):
    def test_end_to_end_is_data_free_and_updates_only_seven_linears(self):
        torch.manual_seed(11)
        reference = _reference_model()
        target = _reference_model()
        target.load_state_dict(reference.state_dict())
        with torch.no_grad():
            for block in target.layers:
                block.self_attn.q_proj.weight.add_(0.01)
        source = _source_model()
        embedding_before = target.embed_tokens.weight.detach().clone()
        q_before = target.layers[0].self_attn.q_proj.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            result = run_aci_pipeline(
                target,
                reference,
                source,
                ACIConfig(
                    beta=0.05,
                    anchor_tokens=16,
                    anchor_chunk_size=5,
                    ffn_sketch_dim=2,
                    ffn_candidate_k=4,
                ),
                compute_device="cpu",
                diagnostics_dir=directory,
            )
            self.assertEqual(result.layer_groups, [[0, 1], [2, 3]])
            self.assertFalse(torch.equal(q_before, target.layers[0].self_attn.q_proj.weight))
            self.assertTrue(torch.equal(embedding_before, target.embed_tokens.weight))
            self.assertEqual(len(result.injection_diagnostics), 14)
            self.assertEqual(len(result.attention_diagnostics), 4)
            self.assertTrue(
                all(row["relative_update_norm"] <= 0.050001 for row in result.injection_diagnostics)
            )
            output = Path(directory)
            self.assertTrue((output / "run_report.json").is_file())
            self.assertTrue((output / "attention_matches.jsonl").is_file())
            self.assertTrue((output / "ffn_matches.jsonl").is_file())
            self.assertTrue((output / "injections.jsonl").is_file())

    def test_attention_only_never_computes_or_updates_ffn(self):
        torch.manual_seed(13)
        reference = _reference_model()
        target = _reference_model()
        target.load_state_dict(reference.state_dict())
        source = _source_model()
        q_before = target.layers[0].self_attn.q_proj.weight.detach().clone()
        ffn_before = {
            name: parameter.detach().clone()
            for name, parameter in target.layers[0].mlp.named_parameters()
        }
        with patch(
            "core.aci.pipeline.contract_ffn",
            side_effect=AssertionError("attention-only must not contract FFN"),
        ):
            result = run_aci_pipeline(
                target,
                reference,
                source,
                ACIConfig(
                    beta=0.05,
                    fusion_mode="attention",
                    anchor_tokens=16,
                    anchor_chunk_size=5,
                    ffn_sketch_dim=2,
                    ffn_candidate_k=4,
                ),
                compute_device="cpu",
            )
        self.assertFalse(torch.equal(q_before, target.layers[0].self_attn.q_proj.weight))
        for name, parameter in target.layers[0].mlp.named_parameters():
            self.assertTrue(torch.equal(ffn_before[name], parameter))
        self.assertEqual(result.ffn_diagnostics, [])
        self.assertEqual(len(result.attention_diagnostics), 4)
        self.assertEqual(
            {row["module"] for row in result.injection_diagnostics},
            {"q", "k", "v", "o"},
        )
        self.assertEqual(result.report["fusion_mode"], "attention")
        self.assertIsNotNone(result.report["rope_frequency_mapping"])
        self.assertEqual(
            len(result.report["rope_frequency_mapping"]["source_pair_indices"]),
            attention_geometry(reference).head_dim // 2,
        )

    def test_ffn_only_never_computes_or_updates_attention(self):
        torch.manual_seed(17)
        reference = _reference_model()
        target = _reference_model()
        target.load_state_dict(reference.state_dict())
        source = _source_model()
        attention_before = {
            name: parameter.detach().clone()
            for name, parameter in target.layers[0].self_attn.named_parameters()
        }
        gate_before = target.layers[0].mlp.gate_proj.weight.detach().clone()
        with patch(
            "core.aci.pipeline.contract_attention",
            side_effect=AssertionError("ffn-only must not contract attention"),
        ):
            result = run_aci_pipeline(
                target,
                reference,
                source,
                ACIConfig(
                    beta=0.05,
                    fusion_mode="ffn",
                    anchor_tokens=16,
                    anchor_chunk_size=5,
                    ffn_sketch_dim=2,
                    ffn_candidate_k=4,
                ),
                compute_device="cpu",
            )
        for name, parameter in target.layers[0].self_attn.named_parameters():
            self.assertTrue(torch.equal(attention_before[name], parameter))
        self.assertFalse(torch.equal(gate_before, target.layers[0].mlp.gate_proj.weight))
        self.assertEqual(result.attention_diagnostics, [])
        self.assertEqual(len(result.ffn_diagnostics), 4)
        self.assertEqual(
            {row["module"] for row in result.injection_diagnostics},
            {"gate", "up", "down"},
        )
        self.assertEqual(result.report["fusion_mode"], "ffn")

    def test_ffn_safe_alias_uses_original_ffn_only_injection(self):
        torch.manual_seed(19)
        reference = _reference_model()
        target_ffn = _reference_model()
        target_ffn.load_state_dict(reference.state_dict())
        target_alias = copy.deepcopy(target_ffn)
        source = _source_model()
        baseline = run_aci_pipeline(
            target_ffn,
            reference,
            source,
            ACIConfig(
                beta=0.05,
                fusion_mode="ffn",
                anchor_tokens=16,
                anchor_chunk_size=5,
                ffn_sketch_dim=2,
                ffn_candidate_k=4,
            ),
            compute_device="cpu",
        )
        alias = run_aci_pipeline(
            target_alias,
            reference,
            source,
            ACIConfig(
                beta=0.05,
                fusion_mode="ffn_safe",
                anchor_tokens=16,
                anchor_chunk_size=5,
                ffn_sketch_dim=2,
                ffn_candidate_k=4,
            ),
            compute_device="cpu",
        )
        for baseline_parameter, alias_parameter in zip(
            target_ffn.parameters(), target_alias.parameters()
        ):
            self.assertTrue(torch.equal(baseline_parameter, alias_parameter))
        self.assertEqual(len(alias.injection_diagnostics), 6)
        self.assertEqual(
            {row["injection_mode"] for row in alias.injection_diagnostics},
            {"legacy_tensor"},
        )
        self.assertEqual(
            baseline.report["strategies"]["ffn"],
            alias.report["strategies"]["ffn"],
        )

    def test_attention_circuit_does_not_touch_ffn(self):
        torch.manual_seed(23)
        reference = _reference_model()
        target = _reference_model()
        target.load_state_dict(reference.state_dict())
        source = _source_model()
        ffn_before = {
            name: parameter.detach().clone()
            for name, parameter in target.layers[0].mlp.named_parameters()
        }
        result = run_aci_pipeline(
            target,
            reference,
            source,
            ACIConfig(
                beta=0.05,
                fusion_mode="attention_circuit",
                anchor_tokens=16,
                anchor_chunk_size=5,
                ffn_sketch_dim=2,
                ffn_candidate_k=4,
            ),
            compute_device="cpu",
        )
        for name, parameter in target.layers[0].mlp.named_parameters():
            self.assertTrue(torch.equal(ffn_before[name], parameter))
        self.assertEqual(len(result.injection_diagnostics), 8)
        self.assertEqual(
            {row["injection_mode"] for row in result.injection_diagnostics},
            {"safe_attention_circuit"},
        )
        self.assertEqual(
            {row["matching_mode"] for row in result.attention_diagnostics},
            {"qk_ov_circuit"},
        )
        self.assertTrue(
            all(
                row["joint_relative_update_norm"] <= 0.050001
                for row in result.injection_diagnostics
            )
        )

    def test_safe_combined_uses_circuit_attention_and_original_ffn(self):
        torch.manual_seed(29)
        reference = _reference_model()
        target = _reference_model()
        target.load_state_dict(reference.state_dict())
        ffn_target = copy.deepcopy(target)
        source = _source_model()
        result = run_aci_pipeline(
            target,
            reference,
            source,
            ACIConfig(
                beta=0.05,
                fusion_mode="safe_combined",
                anchor_tokens=16,
                anchor_chunk_size=5,
                ffn_sketch_dim=2,
                ffn_candidate_k=4,
            ),
            compute_device="cpu",
        )
        run_aci_pipeline(
            ffn_target,
            reference,
            source,
            ACIConfig(
                beta=0.05,
                fusion_mode="ffn",
                anchor_tokens=16,
                anchor_chunk_size=5,
                ffn_sketch_dim=2,
                ffn_candidate_k=4,
            ),
            compute_device="cpu",
        )
        self.assertEqual(len(result.injection_diagnostics), 14)
        self.assertEqual(
            {row["injection_mode"] for row in result.injection_diagnostics},
            {"safe_attention_circuit", "legacy_tensor"},
        )
        for combined_block, ffn_block in zip(target.layers, ffn_target.layers):
            for combined_parameter, ffn_parameter in zip(
                combined_block.mlp.parameters(), ffn_block.mlp.parameters()
            ):
                self.assertTrue(torch.equal(combined_parameter, ffn_parameter))


if __name__ == "__main__":
    unittest.main()
