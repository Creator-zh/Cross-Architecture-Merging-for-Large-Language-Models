import tempfile
import unittest
from pathlib import Path

import torch

from generate_hot_residual import (
    _expand_hot_p_from_file,
    _svd_spectral_contribution,
    fuse_attention_only_from_hot_dir_svd,
)


class SvdSpectralFusionTests(unittest.TestCase):
    def test_svd_contribution_matches_a_shape_and_scale(self):
        weight_a = torch.diag(torch.tensor([4.0, 2.0, 1.0]))
        weight_b = torch.diag(torch.tensor([30.0, 10.0, 5.0, 1.0]))

        contribution, info = _svd_spectral_contribution(
            weight_a,
            weight_b,
            scale_mode="match_l2",
        )

        self.assertEqual(tuple(contribution.shape), tuple(weight_a.shape))
        self.assertEqual(info["rank_a"], 3)
        self.assertEqual(info["rank_b_used"], 3)
        self.assertAlmostEqual(
            torch.linalg.svdvals(contribution).norm().item(),
            torch.linalg.svdvals(weight_a).norm().item(),
            places=5,
        )

    def test_expand_hot_p_from_file_uses_kept_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            hot_file = Path(tmp) / "hot_Q.pt"
            torch.save(
                {
                    "P": torch.tensor([[0.25, 0.75]], dtype=torch.float32),
                    "kept_layer_idx": [1],
                    "kept_layer_idx_B": [0, 2],
                    "L": 1,
                    "M": 2,
                },
                hot_file,
            )

            expanded, kept_a, kept_b, _ = _expand_hot_p_from_file(
                str(hot_file),
                full_l=3,
                full_m=4,
            )

        expected = torch.zeros(3, 4)
        expected[1, 0] = 0.25
        expected[1, 2] = 0.75
        self.assertTrue(torch.equal(expanded, expected))
        self.assertEqual(kept_a, [1])
        self.assertEqual(kept_b, [0, 2])

    def test_svd_fusion_uses_hot_p_without_q_list(self):
        class TinyAttention(torch.nn.Module):
            def __init__(self, width):
                super().__init__()
                self.q_proj = torch.nn.Linear(width, width, bias=False)
                self.k_proj = torch.nn.Linear(width, width, bias=False)
                self.v_proj = torch.nn.Linear(width, width, bias=False)
                self.o_proj = torch.nn.Linear(width, width, bias=False)

        class TinyLayer(torch.nn.Module):
            def __init__(self, width):
                super().__init__()
                self.self_attn = TinyAttention(width)

        class TinyModel(torch.nn.Module):
            def __init__(self, width):
                super().__init__()
                self.layers = torch.nn.ModuleList([TinyLayer(width)])

        model_a = TinyModel(width=3)
        model_b = TinyModel(width=3)
        before = model_a.layers[0].self_attn.q_proj.weight.detach().clone()

        with tempfile.TemporaryDirectory() as tmp:
            for name in ["Q", "K", "V", "O"]:
                for suffix in ["", "_pre"]:
                    torch.save(
                        {"P": torch.ones(1, 1), "L": 1, "M": 1},
                        Path(tmp) / f"hot_{name}{suffix}.pt",
                    )

            report = fuse_attention_only_from_hot_dir_svd(
                modelA=model_a,
                modelB=model_b,
                hot_dir=tmp,
                alpha=0.25,
                attn_device="cpu",
                svd_scale_mode="match_l2",
            )

        after = model_a.layers[0].self_attn.q_proj.weight.detach()
        self.assertFalse(torch.allclose(before, after))
        self.assertEqual(report["fusion_method"], "svd")
        self.assertEqual(report["components"]["Q"]["pairs_used"], 1)


if __name__ == "__main__":
    unittest.main()
