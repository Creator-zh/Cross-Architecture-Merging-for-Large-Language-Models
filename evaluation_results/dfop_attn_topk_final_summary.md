# DFOP results summary (full + attn top-1/top-2)

Primary metrics. Medical/Thai = unweighted mean of primary tasks (Thai xquad uses EM). Malay = MalayMMLU accuracy.

| Method | Medical | Thai | Malay |
|--------|---------|------|-------|
| target | 0.4052 | 0.3809 | 0.4095 |
| full merge | 0.3614 (−4.38pp) | 0.3766 (−0.43pp) | 0.3967 (−1.28pp) |
| full + SFT | 0.3757 (−2.95pp) | 0.3823 (+0.14pp) | 0.3834 (−2.61pp) |
| attn top-1 merge | 0.3587 (−4.65pp) | 0.3865 (+0.56pp) | 0.4011 (−0.84pp) |
| attn top-1 + SFT | 0.3670 (−3.82pp) | 0.3879 (+0.70pp) | 0.3853 (−2.42pp) |
| attn top-2 merge | 0.3707 (−3.44pp) | 0.3869 (+0.60pp) | 0.3633 (−4.63pp) |
| attn top-2 + SFT | 0.3771 (−2.81pp) | 0.3915 (+1.06pp) | 0.4103 (+0.08pp) |

Artifacts:
- Full: `evaluation_results/dfop_pre_sft`, `evaluation_results/dfop_post_sft`
- Attn: `evaluation_results/dfop_attn_pipeline/top{1,2}/`
- Light fusion diagnostics under `transport_results/dfop/*_attn_*/` (no weights)
