# ACI safe/circuit experiments (partial)

Pipeline stopped before Thai `safe_combined` and all Malay evaluations finished.
Scores are percentages. Incomplete variants are marked —.

## medical

| Benchmark | target | aci_ffn_safe_beta0.03 | aci_attention_circuit_beta0.03 | aci_safe_combined_beta0.03 |
|---|---:|---:|---:|---:|
| medqa_4options | 37.71 | 37.63 | 37.71 | 37.55 |
| medmcqa | 36.39 | 36.22 | 36.55 | 36.39 |
| mmlu_anatomy | 49.63 | 48.15 | 51.11 | 49.63 |
| mmlu_clinical_knowledge | 43.40 | 41.89 | 44.15 | 43.02 |
| mmlu_college_biology | 38.19 | 39.58 | 37.50 | 40.28 |
| mmlu_college_medicine | 29.48 | 30.06 | 30.06 | 30.06 |
| mmlu_medical_genetics | 50.00 | 50.00 | 49.00 | 50.00 |
| mmlu_professional_medicine | 39.34 | 41.18 | 40.07 | 41.18 |
| **macro** | **40.52** | **40.59** | **40.77** | **41.01** |

## thai

| Benchmark | target | aci_ffn_safe_beta0.01 | aci_attention_circuit_beta0.01 | aci_safe_combined_beta0.01 |
|---|---:|---:|---:|---:|
| xcopa_th | 58.00 | 58.20 | 58.00 | 58.20 |
| xnli_th | 45.18 | 45.22 | 45.18 | 45.22 |
| xquad_th | 24.93 | 24.60 | 24.73 | 24.23 |
| **macro** | **42.70** | **42.67** | **42.64** | **42.55** |

## malay

| Benchmark | target | aci_ffn_safe_beta0.1 | aci_attention_circuit_beta0.1 | aci_safe_combined_beta0.1 |
|---|---:|---:|---:|---:|
| MalayMMLU | — | — | — | — |
| **macro** | — | — | — | — |

