# DF-OT-Procrustes GPU 运行指南

本文档对应 `core/dfop/` 与 `scripts/run_dfop_fusion.py` 的当前实现。主流程只读取两个模型的配置和权重，不加载 tokenizer、数据集或输入张量，也不调用模型 `forward`。外层路由支持两种求解器：行 softmax 后 top-k，以及使用 SciPy HiGHS 的精确严格平衡 OT；也支持七算子独立路由，或 QK、VO、FFN 三组共享路由。

当前三组共享实验仍分别计算 Q/K/V/O/Gate/Up/Down 的原始跨模型层代价。为消除不同算子的代价尺度差异，对算子 $c$ 使用全矩阵中位数与 MAD 标准化：

$$
\widehat C^c
=
\frac{C^c-\operatorname{median}(C^c)}
{\operatorname{median}(|C^c-\operatorname{median}(C^c)|)+\epsilon}.
$$

若 MAD 退化为零，实现会回退到总体标准差。三组代价为

$$
C^{QK}=\frac{1}{2}\widehat C^Q+\frac{1}{2}\widehat C^K,
$$

$$
C^{VO}=\frac{1}{2}\widehat C^V+\frac{1}{2}\widehat C^O,
$$

$$
C^{FFN}=\frac{1}{4}\widehat C^{Gate}
+\frac{1}{4}\widehat C^{Up}
+\frac{1}{2}\widehat C^{Down}.
$$

因此 $P^Q=P^K=P^{QK}$、$P^V=P^O=P^{VO}$，且 $P^{Gate}=P^{Up}=P^{Down}=P^{FFN}$。FFN 权重使两个 residual-input 投影合计占一半，residual-output 投影占一半；应在实验中与三者等权比较。

## 1. 本地 CPU 验证

本地环境不需要下载大模型即可验证数学核与异构维度支持：

```powershell
python -m unittest tests.test_dfop_core -v
python scripts/run_dfop_fusion.py --help
```

端到端测试使用目标 2 层、源 3 层的假模型，并令两边的 hidden、KV 和 FFN 维度均不相同。假模型的 `forward` 会主动抛出异常，因此测试通过也验证了流程没有前向传播。

## 2. 远端环境

建议使用独立环境，并安装与服务器 CUDA 匹配的 PyTorch：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
PY
```

两套模型权重驻留 CPU；当前处理的权重矩阵或 OT 层对才被送到指定 GPU。因此服务器除 GPU 显存外，还需要足够的 CPU 内存容纳两个模型和 FP32 截断 SVD 缓存。加载参数 `--model-dtype bfloat16` 可明确限制权重存储精度。

## 3. 三级试运行

先运行一个低成本、只计算不修改权重的 Q 路由检查：

```bash
python scripts/run_dfop_fusion.py \
  --target-model /path/to/domain-small-target \
  --source-model /path/to/general-large-source \
  --output-dir transport_results/dfop_q_smoke \
  --device cuda:0 \
  --model-dtype bfloat16 \
  --modules q \
  --rank 32 \
  --alternating-iterations 2 \
  --restarts 1 \
  --dry-run
```

确认代价、边缘误差与运行内存正常后，做 attention-only 对比：

```bash
python scripts/run_dfop_fusion.py \
  --target-model /path/to/domain-small-target \
  --source-model /path/to/general-large-source \
  --output-dir transport_results/dfop_attn \
  --device cuda:0 \
  --model-dtype bfloat16 \
  --modules q,k,v,o \
  --rank 128 \
  --beta 0.05 \
  --trust-ratio 0.10
```

最后运行七模块主方法：

```bash
python scripts/run_dfop_fusion.py \
  --target-model /path/to/domain-small-target \
  --source-model /path/to/general-large-source \
  --output-dir transport_results/dfop_full \
  --device cuda:0 \
  --model-dtype bfloat16 \
  --modules q,k,v,o,gate,up,down \
  --rank 128 \
  --svd-algorithm randomized \
  --svd-oversample 16 \
  --svd-power-iterations 2 \
  --inner-entropy 0.05 \
  --route-solver row_softmax_topk \
  --route-grouping qk_vo_ffn \
  --top-source-layers 2 \
  --beta 0.05 \
  --trust-ratio 0.10
```

默认拒绝复用非空输出目录，以免覆盖原 HOT 或 DFOP 结果。明确需要续写同一目录时才使用 `--overwrite-output`。

## 4. 输出

每次运行生成：

```text
output_dir/
├── invocation.json
├── config.json
├── run_report.json
├── diagnostics/
│   ├── layer_cost_{q,k,v,o,gate,up,down}.pt
│   ├── route_group_cost_{qk,vo,ffn}.pt
│   ├── route_dense_{q,k,v,o,gate,up,down}.pt
│   ├── route_{q,k,v,o,gate,up,down}.pt
│   ├── route_diagnostics.jsonl
│   ├── pair_diagnostics.jsonl
│   └── fusion_diagnostics.jsonl
└── fused_model/
    ├── config.json
    └── model-*.safetensors
```

使用 `row_softmax_topk` 时，`route_dense_*.pt` 是截断前的行 softmax，`route_*.pt` 是 top-k 后重新逐行归一化的路由。使用 `balanced_exact` 时两者相同，且不再执行会破坏源层边缘约束的逐行 top-k。共享组内各算子的 `route_*.pt` 内容相同，同时保存一次组代价 `route_group_cost_*.pt`。`run_report.json` 记录求解器、分组、行列边缘误差和各组统计量。对于严格平衡 OT 的目标 16 层、源 32 层，应满足每行总质量为 1、每列总质量为 0.5。

严格无数据约束下，脚本不会把 tokenizer 保存进 `fused_model/`。评测时应显式使用目标模型原有 tokenizer；融合模型的输入/输出语义仍由目标架构定义。

### 4.1 严格平衡 OT 的实际操作

对任一算子或共享组的层代价 $C\in\mathbb R^{L\times M}$，令每个目标层携带质量 $a_\ell=1/L$，每个源层携带质量 $b_m=1/M$。求解

$$
\Pi^*
=
\underset{\Pi\ge 0}{\arg\min}
\sum_{\ell=1}^{L}\sum_{m=1}^{M}C_{\ell m}\Pi_{\ell m},
$$

满足

$$
\sum_{m=1}^{M}\Pi_{\ell m}=\frac{1}{L},
\qquad \ell=1,\ldots,L,
$$

$$
\sum_{\ell=1}^{L}\Pi_{\ell m}=\frac{1}{M},
\qquad m=1,\ldots,M.
$$

实现把 $LM$ 个耦合元素展平成线性规划变量，用 SciPy `linprog(method="highs")` 精确求解；这里没有熵正则，也没有 Sinkhorn 近似。融合阶段需要每个目标层的源层条件权重，因此再转换为

$$
P_{\ell m}=\frac{\Pi^*_{\ell m}}{a_\ell}=L\Pi^*_{\ell m}.
$$

于是

$$
\sum_{m=1}^{M}P_{\ell m}=1,
$$

$$
\sum_{\ell=1}^{L}P_{\ell m}=\frac{L}{M}.
$$

例如 $L=16,M=32$ 时，每个目标层仍有一单位聚合预算，但每个源层在所有目标层上的总贡献被固定为 $0.5$。这正是它避免所有目标层退化到同一两个源层的机制。精确线性 OT 的最优基本可行解通常很稀疏，非退化情形至多有 $L+M-1$ 个正元素；求解后不能再做逐行 top-k，否则会破坏列边缘约束。启用方式为 `--route-solver balanced_exact`，此时 `--top-source-layers` 不参与求解。

## 5. 显存与故障定位

单个 OT 耦合的形状是 `目标侧神经元数 × 源侧神经元数`。FP32 单张量占用约为

$$
4n_A n_B\ \text{bytes}.
$$

log-domain Sinkhorn 同时需要代价、log-kernel、耦合和若干临时张量，实际峰值是单张量的数倍。阶段二的 `gate/up` 输出侧和 `down` 输入侧使用 FFN 宽度，通常是峰值显存来源。当前版本按层对串行处理，不会同时保留多个大耦合；若单个 FFN 耦合仍超出显存，需要根据实际模型尺寸启用后续的分块 Sinkhorn 实现，不能简单降低模型精度，因为 OT 固定使用 FP32。

诊断时优先检查：

- `pair_diagnostics.jsonl` 中两个边缘误差是否接近 `--sinkhorn-tolerance`；
- `output_converged` 与 `input_converged`；
- `run_report.json` 中严格平衡路由的行、列边缘误差是否低于 `1e-7`；
- 精确路由支持集是否异常集中在少数目标层；
- `core_scale` 是否大量命中 `gamma_min`/`gamma_max`；
- `relative_update_norm` 是否大量被 trust ratio 截断。

只有在 Q-only smoke 成功后再运行完整七模块，以免在 FFN 大耦合处才发现环境配置问题。

## 6. Medical、Thai、Malay 三任务实验

下载三套 1B 目标模型和共享的 8B 源模型：

```bash
python scripts/download_models.py \
  --tasks medical thai malay base_8b \
  --models-root ./models
```

当前 QK/VO/FFN 共享实验先使用行 softmax top-1 和 top-2，不启用严格平衡 OT。先运行 top-1：

```bash
python scripts/run_dfop_tasks.py \
  --tasks medical,thai,malay \
  --gpus 0,1,2 \
  --models-root ./models \
  --results-root ./transport_results/dfop \
  --mode full \
  --track universal \
  --rank 128 \
  --route-solver row_softmax_topk \
  --route-grouping qk_vo_ffn \
  --top-source-layers 1 \
  --trust-ratio 0.10
```

top-1 完成后，再运行 top-2：

```bash
python scripts/run_dfop_tasks.py \
  --tasks medical,thai,malay \
  --gpus 0,1,2 \
  --models-root ./models \
  --results-root ./transport_results/dfop \
  --mode full \
  --track universal \
  --rank 128 \
  --route-solver row_softmax_topk \
  --route-grouping qk_vo_ffn \
  --top-source-layers 2 \
  --trust-ratio 0.10
```

两次实验应顺序运行。它们自动复用 `cache/stage1/<task>_full_r128` 中与路由无关的七个原始代价矩阵，而将路由和层对传输分别保存到以完整运行名隔离的 `cache/stage2/` 目录。不要让两个独立启动器同时写同一个阶段一缓存。输出目录分别为 `<task>_full_universal_r128_top1_qk_vo_ffn_beta0.05` 与 `<task>_full_universal_r128_top2_qk_vo_ffn_beta0.05`。

`universal` 对三个任务固定使用 `beta=0.05`。`matched` 则使用原 Transport and Merge 融合阶段的任务参数：medical 0.03、Thai 0.01、Malay 0.10。主结果应先报告 universal track，matched track 作为对照：

```bash
python scripts/run_dfop_tasks.py \
  --tasks medical,thai,malay \
  --gpus 0,1,2 \
  --models-root ./models \
  --mode full \
  --track matched \
  --route-solver row_softmax_topk \
  --route-grouping qk_vo_ffn \
  --top-source-layers 2
```

任务中断后使用完全相同的模型与数学配置，并添加 `--resume`：

```bash
python scripts/run_dfop_tasks.py \
  --tasks medical,thai,malay \
  --gpus 0,1,2 \
  --models-root ./models \
  --mode full \
  --track universal \
  --route-solver row_softmax_topk \
  --route-grouping qk_vo_ffn \
  --top-source-layers 2 \
  --resume
```

阶段一每完成一个 `q/k/v/o/gate/up/down` 模块便原子保存 `layer_cost_*.pt`。恢复时，manifest 中的模型标识、矩阵形状、rank、SVD、谱点和 OT 参数必须完全一致；不一致时程序拒绝误用缓存。

由于七模块仍分别计算和写入各自的权重更新，attention-only checkpoint 可以从 full checkpoint 精确派生：保留 Q/K/V/O，恢复原目标模型的 gate/up/down。该操作不需要重新计算任何 OT。下面派生 top-1：

```bash
python scripts/derive_dfop_attn_tasks.py \
  --tasks medical,thai,malay \
  --models-root ./models \
  --results-root ./transport_results/dfop \
  --track universal \
  --rank 128 \
  --route-solver row_softmax_topk \
  --route-grouping qk_vo_ffn \
  --top-source-layers 1
```

对 top-2 再执行一次相同命令，并把 `--top-source-layers` 改为 `2`。派生 checkpoint 保留同一次 full 运行的 Q/K/V/O 更新并恢复原目标模型 FFN，因此是精确的 FFN 开关消融。

单元测试同时直接运行了 attention-only 流程，并验证派生 checkpoint 与直接结果逐权重一致。`run_dfop_tasks.py --mode attn` 仍保留为独立复核路径。

## 7. 在同一服务器生成原 T&M 与 DFOP

### 7.1 原 T&M merge-only

原方法的 Step 1 会读取领域数据并执行 forward，Step 2 根据 activation OT 生成融合模型。下面的启动器只执行这两步，不执行 SFT 或原脚本内置评测：

```bash
python scripts/run_hot_merge_tasks.py \
  --tasks medical,thai,malay \
  --gpus 0,1,2 \
  --models-root ./models \
  --workspace-root "$PWD" \
  --hot-results-root ./transport_results \
  --fineweb-thai-cache ./hf_cache_eval/fineweb_thai_cli
```

三个 T&M checkpoint 的固定位置是：

```text
models/medllama_fused_alpha01_fortrain_1b
models/llamathai_fused_alpha01_fortrain_1b_thai_instruction_sft
maly_llama_fused_alpha01_fortrain_1b_select
```

`scripts/run_hot_merge_tasks.py` 会为每个任务分配一张物理 GPU，并写出 `transport_results/hot_merge_manifest.json`。它通过环境变量令 `scripts/run_train_final.sh` 只执行 Step 1/2；原脚本已有的结果目录非空时会跳过对应步骤，因此可安全续跑。Thai 数据若不存在，原脚本会自动调用 `huggingface-cli` 下载，也可提前执行：

```bash
huggingface-cli download \
  --repo-type dataset ChavyvAkvar/fineweb-2-1M-Sample-Thai \
  --local-dir ./hf_cache_eval/fineweb_thai_cli
```

### 7.2 DFOP-Full 与 DFOP-Attn

使用另外三张卡运行无数据主方法：

```bash
python scripts/run_dfop_tasks.py \
  --tasks medical,thai,malay \
  --gpus 3,4,5 \
  --models-root ./models \
  --results-root ./transport_results/dfop \
  --mode full \
  --track universal \
  --rank 128 \
  --route-solver row_softmax_topk \
  --route-grouping qk_vo_ffn \
  --top-source-layers 2 \
  --trust-ratio 0.10
```

Full 完成后从其 checkpoint 精确派生 attention-only 版本，不重新计算 OT：

```bash
python scripts/derive_dfop_attn_tasks.py \
  --tasks medical,thai,malay \
  --models-root ./models \
  --results-root ./transport_results/dfop \
  --track universal \
  --rank 128 \
  --route-solver row_softmax_topk \
  --route-grouping qk_vo_ffn \
  --top-source-layers 2
```

7 张 A100 上可让 T&M 使用 0–2、DFOP 使用 3–5 同时运行，保留第 6 张卡做监控或短测。若共享文件系统吞吐成为瓶颈，则先跑 T&M，再跑 DFOP；这不会改变结果。

## 8. 公平的五组 merge-only 评测

主表固定为以下五列：

| 评测别名 | 含义 | 是否使用融合数据/forward |
|---|---|---|
| `target` | 原始领域 1B 目标模型 | 否 |
| `source` | 原始通用 8B 源模型 | 否 |
| `hot` | 原 T&M merge-only 1B | 是 |
| `dfop_attn` | 仅融合 Q/K/V/O 的 DFOP 1B | 否 |
| `dfop_full` | 融合 Q/K/V/O/Gate/Up/Down 的 DFOP 1B | 否 |

用同一个启动器、相同 dtype、相同 batch size 和相同任务集合评测全部 checkpoint：

```bash
python scripts/evaluate_dfop_tasks.py \
  --tasks medical,thai,malay \
  --variants target,source,hot,dfop_attn,dfop_full \
  --gpus 0,1,2,3,4,5,6 \
  --models-root ./models \
  --results-root ./transport_results/dfop \
  --eval-root ./evaluation_results/qk_vo_ffn_top2 \
  --track universal \
  --rank 128 \
  --route-solver row_softmax_topk \
  --route-grouping qk_vo_ffn \
  --top-source-layers 2 \
  --scope primary \
  --lm-eval-repo /path/to/lm-evaluation-harness \
  --malay-repo /path/to/complete/MalayMMLU \
  --external-model medical:hot="$PWD/models/medllama_fused_alpha01_fortrain_1b" \
  --external-model thai:hot="$PWD/models/llamathai_fused_alpha01_fortrain_1b_thai_instruction_sft" \
  --external-model malay:hot="$PWD/maly_llama_fused_alpha01_fortrain_1b_select"
```

一共是 3 个任务乘 5 个模型的 15 个 job。启动器最多同时使用 7 张卡，某个 job 完成后自动调度下一个。命令、checkpoint 路径与输出目录会保存到 `evaluation_results/merge_only/evaluation_manifest.json`，日志位于其 `logs/` 子目录。

仓库内 `evaluation/malay/MalayMMLU/` 只保留了原仓库随附的评测代码，不包含上游仓库的 `data/MalayMMLU_0shot.json`。因此 `--malay-repo` 必须指向一个带完整 `data/` 的 MalayMMLU checkout；启动器会在占用 GPU 前检查该文件。Malay 预测结束后，启动器会比较 `golds` 与 `preds` 并生成统一的 `metrics.json`。

Medical 使用 8 项医疗任务；Thai 的 `primary` 严格使用 `xcopa_th,xquad_th,xnli_th`，`all` 才会额外加入四项扩展任务；Malay 使用零样本、按选项字母计分的 MalayMMLU。主表先报告 `primary`，Thai 扩展集另做补充表：

```bash
python scripts/evaluate_dfop_tasks.py \
  --tasks thai \
  --variants target,source,hot,dfop_attn,dfop_full \
  --gpus 0,1,2,3,4 \
  --models-root ./models \
  --results-root ./transport_results/dfop \
  --eval-root ./evaluation_results/merge_only_thai_extended \
  --track universal \
  --rank 128 \
  --route-solver row_softmax_topk \
  --route-grouping qk_vo_ffn \
  --top-source-layers 2 \
  --scope extended \
  --lm-eval-repo /path/to/lm-evaluation-harness \
  --external-model thai:hot="$PWD/models/llamathai_fused_alpha01_fortrain_1b_thai_instruction_sft"
```

公平性要求：五组 checkpoint 都不再 SFT；所有模型使用同一版评测仓库、同一任务定义、相同 `--lm-eval-dtype` 和 batch size；分别报告每个子任务分数与任务内宏平均，不把 Medical、Thai、Malay 混成一个总平均。T&M 使用论文/原仓库各任务的 `FUSE_ALPHA`；DFOP 主结果对所有任务固定 `beta=0.05`，另把 `matched` track 作为控制实验，不能用它替代 universal 主结果。

全部 job 成功后生成逐任务 CSV 和三领域对比 Markdown：

```bash
python scripts/summarize_merge_results.py \
  --eval-root ./evaluation_results/merge_only \
  --tasks medical,thai,malay \
  --variants target,source,hot,dfop_attn,dfop_full \
  --scope primary
```

输出是 `merge_only_scores.csv` 与 `merge_only_summary.md`。汇总器对 XQuAD-Thai 使用 F1，其余主任务优先使用 accuracy；任务缺失或指标名不兼容时返回非零状态，不会静默跳过后仍生成“完整”结果。

## 9. 融合后 SFT 对照

SFT 不是无数据主方法，仅用于比较不同 merge-only 初始化经过相同领域训练后的表现。三任务单卡并行、全参数 SFT：

```bash
python scripts/run_dfop_sft_tasks.py \
  --tasks medical,thai,malay \
  --gpus 0,1,2 \
  --fusion-results-root ./transport_results/dfop \
  --sft-results-root ./sft_results/dfop \
  --mode full \
  --track universal \
  --rank 128 \
  --train-mode full \
  --profile declared \
  --thai-dataset-path ./hf_cache_eval/fineweb_thai_cli
```

`declared` profile 使用文档声明的设置：Medical 2000 样本、Thai 8000、Malay 2000，学习率分别为 `3e-7`、`1e-7`、`1e-6`，batch size 1、梯度累积 8、1 epoch、block size 2048、FP16。

仓库旧执行路径存在两个行为差异：Thai 实际回落到 2000 样本，且 `--fp16` 没有传入 `TrainingArguments`。需要复现该旧行为时使用：

```bash
python scripts/run_dfop_sft_tasks.py \
  --tasks medical,thai,malay \
  --gpus 0,1,2 \
  --train-mode full \
  --profile legacy
```

训练代码新增了 `--honor_precision_flags`；只有显式传入该开关时才真正启用 `--fp16`，因此原有 `run_train_final.sh` 的行为没有被静默改变。

Q/K/V/O LoRA 工程消融：

```bash
python scripts/run_dfop_sft_tasks.py \
  --tasks medical,thai,malay \
  --gpus 3,4,5 \
  --train-mode lora \
  --profile declared
```

默认 LoRA 配置与仓库声明一致：rank 64、alpha 16、dropout 0.05，只作用于 `q_proj,k_proj,v_proj,o_proj`。评测 SFT 输出时，在评测启动器的 `--variants` 中加入 `dfop_sft`：

```bash
python scripts/evaluate_dfop_tasks.py \
  --tasks medical,thai,malay \
  --variants dfop_sft \
  --gpus 0,1,2 \
  --sft-train-mode full \
  --sft-profile declared \
  --lm-eval-repo /path/to/lm-evaluation-harness
```
