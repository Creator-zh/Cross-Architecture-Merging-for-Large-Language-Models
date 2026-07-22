# ACI legacy FFN and circuit-attention vs target

All deltas are computed against the target from the same evaluation batch. 
Thai uses XQuAD F1 in its unweighted macro. Medical also reports question-weighted micro accuracy.

| Domain | Variant | β | Macro / accuracy | Δ vs target | Medical micro | Δ micro |
|---|---|---:|---:|---:|---:|---:|
| medical | target | — | 40.52 | +0.00 | 37.39 | +0.00 |
| medical | aci_ffn_safe_beta0.03 | 0.03 | 41.41 | +0.90 | 37.52 | +0.14 |
| medical | aci_attention_circuit_beta0.03 | 0.03 | 41.10 | +0.58 | 37.56 | +0.17 |
| medical | aci_safe_combined_beta0.03 | 0.03 | 41.27 | +0.76 | 37.37 | -0.02 |
| thai | target | — | 42.70 | +0.00 | — | — |
| thai | aci_ffn_safe_beta0.01 | 0.01 | 42.57 | -0.13 | — | — |
| thai | aci_attention_circuit_beta0.01 | 0.01 | 42.70 | +0.00 | — | — |
| thai | aci_safe_combined_beta0.01 | 0.01 | 42.59 | -0.11 | — | — |
| malay | target | — | 41.03 | +0.00 | — | — |
| malay | aci_ffn_safe_beta0.1 | 0.1 | 45.60 | +4.57 | — | — |
| malay | aci_attention_circuit_beta0.1 | 0.1 | 40.36 | -0.67 | — | — |
| malay | aci_safe_combined_beta0.1 | 0.1 | 45.36 | +4.33 | — | — |
