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

## 4. 可选 SFT 对比

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

## 5. 诊断检查

每个运行目录优先检查：

1. `run_report.json` 中 `data_free_contract` 四项是否满足；
2. `anchor.orthogonality_error` 是否接近浮点误差；
3. `diagnostics/attention_matches.jsonl` 是否为完整的一一 GQA/head 置换；
4. `diagnostics/ffn_matches.jsonl` 的 `reused_sources` 是否恒为 0；
5. `diagnostics/injections.jsonl` 的 `relative_update_norm` 是否均不大于 β；
6. tokenizer 文件是否从目标模型目录复制到 `fused_model/`。

旧 `evaluation_results/dfop_*` 与 `transport_results/dfop/` 只用于追溯旧实验，不能与新输出混用。
