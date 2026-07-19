# DFOP full-universal r128 (β=0.05, top-2) — primary scores

Source: Llama-3.1-8B-Instruct → task 1B targets. Modules: Q/K/V/O/Gate/Up/Down.

| Task | Metric | target | DFOP-full | DFOP-SFT |
|------|--------|--------|-----------|----------|
| Medical | mean(8 acc) | 0.4052 | 0.3614 | 0.3757 |
| Thai | mean(xcopa/xnli/xquad-EM) | 0.3809 | 0.3766 | 0.3823 |
| Malay | MalayMMLU acc | 0.4095 | 0.3967 | 0.3834 |

Notes:
- Malay `metrics.json` was rewritten to map index golds (0→A) to letter preds.
- Raw lm-eval / MalayMMLU artifacts live under `evaluation_results/dfop_pre_sft` and `evaluation_results/dfop_post_sft`.
- Model weights and datasets are not in git.
