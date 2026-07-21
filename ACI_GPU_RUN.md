# ACI 双 A100 运行指南

以下命令从仓库根目录运行。ACI 合并阶段不读取数据、不加载 tokenizer、不执行 forward。

## 1. 下载模型

三个目标模型、共享 8B 源模型和两个 1B 参考模型：

```bash
python scripts/download_models.py \
  --tasks medical thai malay base_8b reference_1b_base reference_1b_instruct
```

预期额外参考目录：

```text
models/Llama-3.2-1B
models/Llama-3.2-1B-Instruct
```

## 2. 两张 GPU 运行三任务合并

启动器最多同时运行两个任务；第三个任务会在任一 GPU 空闲后自动开始。

```bash
python scripts/run_aci_tasks.py \
  --tasks medical,thai,malay \
  --gpus 0,1 \
  --models-root ./models \
  --results-root ./merge_results/aci
```

默认 β：Medical 0.03、Thai 0.01、Malay 0.10。输出为：

```text
merge_results/aci/
├── medical_aci_beta0.03/fused_model/
├── thai_aci_beta0.01/fused_model/
└── malay_aci_beta0.1/fused_model/
```

如需显式覆盖：

```bash
python scripts/run_aci_tasks.py \
  --tasks medical,thai,malay \
  --gpus 0,1 \
  --beta medical=0.02 \
  --beta thai=0.02 \
  --beta malay=0.05
```

覆盖实验应作为 β 消融完整报告，不能只保留测试集最优点。

单任务示例：

```bash
python scripts/run_aci_merge.py \
  --target-model ./models/llama3-1b-med \
  --reference-model ./models/Llama-3.2-1B \
  --source-model ./models/Llama-3.1-8B-Instruct \
  --output-dir ./merge_results/aci/medical_aci_beta0.03 \
  --beta 0.03 \
  --device cuda:0 \
  --local-files-only
```

## 3. Merge-only 评测

```bash
python scripts/evaluate_aci_tasks.py \
  --tasks medical,thai,malay \
  --variants target,aci \
  --gpus 0,1 \
  --models-root ./models \
  --results-root ./merge_results/aci \
  --eval-root ./evaluation_results/aci \
  --lm-eval-repo /path/to/lm-evaluation-harness \
  --malay-repo ./evaluation/malay/MalayMMLU
```

汇总：

```bash
python scripts/summarize_merge_results.py \
  --eval-root ./evaluation_results/aci \
  --tasks medical,thai,malay \
  --variants target,aci \
  --output-prefix aci_merge_only
```

验收标准是 Medical、Thai、Malay 各自的宏平均分别超过目标模型，不把三个领域再合成一个总平均。

## 4. Attention/FFN 严格消融

以下入口固定运行已预注册的 8 个 checkpoint：Medical 两个、Thai 四个、Malay 两个。最多同时占用两张 GPU，已有 full ACI 输出不会被覆盖。

```bash
python scripts/run_aci_ablations.py \
  --gpus 0,1 \
  --models-root ./models \
  --results-root ./merge_results/aci
```

输出目录：

```text
merge_results/aci/
├── medical_aci_attention_beta0.03/
├── medical_aci_ffn_beta0.03/
├── thai_aci_attention_beta0.01/
├── thai_aci_attention_beta0.1/
├── thai_aci_ffn_beta0.01/
├── thai_aci_ffn_beta0.1/
├── malay_aci_attention_beta0.1/
└── malay_aci_ffn_beta0.1/
```

严格定义：

- `attention`：只计算和更新 Q/K/V/O；Gate/Up/Down 与 target 逐位一致；
- `ffn`：只计算和更新 Gate/Up/Down；Q/K/V/O 与 target 逐位一致；
- 禁用模块不会执行压缩，诊断中也没有对应记录。

评测 8 个消融模型，并为每个领域重新评测同批次 target：

```bash
python scripts/evaluate_aci_ablations.py \
  --gpus 0,1 \
  --models-root ./models \
  --results-root ./merge_results/aci \
  --eval-root ./evaluation_results/aci_ablations \
  --lm-eval-repo /path/to/lm-evaluation-harness \
  --malay-repo ./evaluation/malay/MalayMMLU
```

自动生成：

```text
evaluation_results/aci_ablations/
├── ablation_manifest.json
├── aci_ablation_scores.csv
└── aci_ablation_summary.md
```

汇总只和 target 比较，不读取或比较 DFOP。Thai 宏平均使用 XQuAD F1；Medical 同时报告无权 macro 和按题数加权的 micro accuracy。若评测已经完成，只需重新汇总：

```bash
python scripts/evaluate_aci_ablations.py \
  --eval-root ./evaluation_results/aci_ablations \
  --summarize-only
```

单任务通用入口也支持 `--fusion-mode full|attention|ffn`；省略时仍为原来的 `full`。

## 5. 可选 SFT 对比

SFT 不属于无数据 ACI 主方法，只作为独立对比：

```bash
python scripts/run_aci_sft_tasks.py \
  --tasks medical,thai,malay \
  --gpus 0,1 \
  --fusion-results-root ./merge_results/aci \
  --sft-results-root ./sft_results/aci
```

SFT 完成后可运行：

```bash
python scripts/evaluate_aci_tasks.py \
  --tasks medical,thai,malay \
  --variants target,aci,aci_sft \
  --gpus 0,1 \
  --models-root ./models \
  --results-root ./merge_results/aci \
  --sft-results-root ./sft_results/aci \
  --eval-root ./evaluation_results/aci_with_sft
```

## 6. 诊断检查

每个运行目录优先检查：

1. `run_report.json` 中 `data_free_contract` 四项是否满足；
2. `anchor.orthogonality_error` 是否接近浮点误差；
3. `diagnostics/attention_matches.jsonl` 是否为完整的一一 GQA/head 置换；
4. `diagnostics/ffn_matches.jsonl` 的 `reused_sources` 是否恒为 0；
5. `diagnostics/injections.jsonl` 的 `relative_update_norm` 是否均不大于 β；
6. tokenizer 文件是否从目标模型目录复制到 `fused_model/`。

旧 `evaluation_results/dfop_*` 与 `transport_results/dfop/` 只用于追溯旧实验，不能与新输出混用。
