# ACI attention/FFN ablation vs target

All deltas are computed against the target from the same evaluation batch. 
Thai uses XQuAD F1 in its unweighted macro. Medical also reports question-weighted micro accuracy.

| Domain | Variant | β | Macro / accuracy | Δ vs target | Medical micro | Δ micro |
|---|---|---:|---:|---:|---:|---:|
| medical | target | — | 40.52 | +0.00 | 37.39 | +0.00 |
| medical | aci_attention_beta0.03 | 0.03 | 40.36 | -0.16 | 37.04 | -0.35 |
| medical | aci_ffn_beta0.03 | 0.03 | 41.41 | +0.90 | 37.52 | +0.14 |
| thai | target | — | 42.70 | +0.00 | — | — |
| thai | aci_attention_beta0.01 | 0.01 | 42.74 | +0.04 | — | — |
| thai | aci_attention_beta0.1 | 0.1 | 40.08 | -2.63 | — | — |
| thai | aci_ffn_beta0.01 | 0.01 | 42.57 | -0.13 | — | — |
| thai | aci_ffn_beta0.1 | 0.1 | 40.26 | -2.45 | — | — |
| malay | target | — | 41.03 | +0.00 | — | — |
| malay | aci_attention_beta0.1 | 0.1 | 35.03 | -6.00 | — | — |
| malay | aci_ffn_beta0.1 | 0.1 | 45.60 | +4.57 | — | — |
