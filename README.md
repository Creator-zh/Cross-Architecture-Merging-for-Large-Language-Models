# Cross-Architecture Merging for Large Language Models

本仓库保留两条清晰分离的路线：

1. **Anchor–Compress–Inject（ACI，当前新增主方法）**：严格无数据、无 tokenizer、无 forward，将 Llama-3.1 8B Instruct 一次性压缩并注入领域 Llama-3.2 1B，同时显式保护目标模型的领域任务向量。
2. **Transport and Merge（T&M/HOT，原仓库方法）**：使用领域数据与 forward activation 计算 OT，再进行融合与可选训练。原复现入口保持不变。

旧 DF-OT-Procrustes 已从当前实现和默认入口移除。它的提交历史、`evaluation_results/dfop_*` 和 `transport_results/dfop/` 仍保留用于追溯，但这些结果不是 ACI 结果。

## ACI 一句话方法

ACI 使用同架构通用 1B checkpoint \(W_0\) 作为保护锚点，把 8B 源模型结构化压缩为 \(\mathcal C(W_S)\)，然后对七个 Transformer 线性模块执行：

\[
W_{out}=W_T+\beta\bigl(\mathcal C(W_S)-W_0\bigr).
\]

因此目标领域任务向量 \(W_T-W_0\) 的系数保持为 1。模型只修改 Q/K/V/O/Gate/Up/Down；embedding、LM head、norm、tokenizer、参数形状和推理架构保持目标模型原值。

压缩包含三个固定结构规则：

- **全局坐标**：用共享词表的 embedding/lm-head 权重求一个矩形正交残差映射；
- **Attention**：固定相对深度，按相同 RoPE 基频压缩 128→64 head dimension，并让 Q/K/V/O 共享 GQA 组与 query-head 一一匹配；
- **FFN**：Gate 行、Up 行和 Down 列组成不可拆分的 SwiGLU 神经元，三者共享同一个无重复匹配。

没有 OT、Sinkhorn、自由层路由、rank、top-k、temperature 或独立 trust-ratio。唯一效果超参数是每个任务的注入强度 β。

完整定义见 [ACI_METHOD.md](ACI_METHOD.md)，双 A100 命令见 [ACI_GPU_RUN.md](ACI_GPU_RUN.md)。

## 快速开始

安装：

```bash
pip install -r requirements.txt
```

下载 Medical、Thai、Malay 的目标/参考/源模型：

```bash
python scripts/download_models.py \
  --tasks medical thai malay base_8b reference_1b_base reference_1b_instruct
```

在两张 GPU 上运行三个任务；启动器自动排队：

```bash
python scripts/run_aci_tasks.py \
  --tasks medical,thai,malay \
  --gpus 0,1 \
  --models-root ./models \
  --results-root ./merge_results/aci
```

默认 β 为 Medical 0.03、Thai 0.01、Malay 0.10。它们沿用原 T&M 的 merge-only 任务强度，不是根据 ACI 测试结果反向挑选。

统一评测：

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

python scripts/summarize_merge_results.py \
  --eval-root ./evaluation_results/aci \
  --tasks medical,thai,malay \
  --variants target,aci \
  --output-prefix aci_merge_only
```

Attention/FFN 严格消融一次生成 8 个独立 checkpoint：

```bash
python scripts/run_aci_ablations.py \
  --gpus 0,1 \
  --models-root ./models \
  --results-root ./merge_results/aci

python scripts/evaluate_aci_ablations.py \
  --gpus 0,1 \
  --models-root ./models \
  --results-root ./merge_results/aci \
  --eval-root ./evaluation_results/aci_ablations \
  --lm-eval-repo /path/to/lm-evaluation-harness \
  --malay-repo ./evaluation/malay/MalayMMLU
```

消融只与同批次 target 比较；Thai 使用 XQuAD F1，Medical 同时报告 macro 和按题数加权的 micro。`attention` 模式完全不计算或修改 FFN，`ffn` 模式完全不计算或修改 Attention。

主验收标准是 Medical、Thai、Malay 三个领域宏平均分别超过原目标模型；不把三个领域合并成一个总平均。仓库当前尚未包含 ACI 的服务器评测结果，因此不预先声称提升。

## 可选 SFT

SFT 不是 ACI 主方法，仅作为独立对比：

```bash
python scripts/run_aci_sft_tasks.py \
  --tasks medical,thai,malay \
  --gpus 0,1 \
  --fusion-results-root ./merge_results/aci \
  --sft-results-root ./sft_results/aci
```

## 代码结构

```text
core/aci/
├── alignment.py       # embedding/lm-head 残差空间锚定
├── attention.py       # head/RoPE 保持的 attention 压缩
├── ffn.py             # Gate/Up/Down 联合神经元匹配
├── injection.py       # 领域任务向量保护注入
├── pipeline.py        # 三步主流程与诊断
├── presets.py         # Medical/Thai/Malay 模型、β 与评测预设
└── config.py          # 一个效果参数 + 数值批处理参数

scripts/
├── run_aci_merge.py       # 单任务合并
├── run_aci_tasks.py       # 双 GPU 三任务调度
├── run_aci_ablations.py   # 固定 8-run Attention/FFN 消融
├── evaluate_aci_tasks.py  # merge-only 统一评测
├── evaluate_aci_ablations.py # 消融评测与 target 对比汇总
├── run_aci_sft_tasks.py   # 可选 SFT 对比
└── summarize_merge_results.py
```

## 原 T&M/HOT 复现

原方法仍使用：

```text
run_activs_and_hot.py
generate_hot_residual.py
train_hot_residual_sft.py
scripts/run_train_final.sh
```

完整步骤见 [REPRODUCE.md](REPRODUCE.md)，模型与数据配置见 [MODELS.md](MODELS.md)。

## 测试

```bash
python -m unittest discover -s tests -v
```
