# Anchor–Compress–Inject：无数据异构模型合并

## 1. 目标与边界

给定三个 checkpoint：

- 领域目标模型 \(W_T\)：Llama-3.2 1B 的 Medical、Thai 或 Malay 版本；
- 同架构通用参考模型 \(W_0\)：对应的通用 Llama-3.2 1B；
- 异构能力源模型 \(W_S\)：Llama-3.1 8B Instruct。

ACI 将 8B 源模型压缩到 1B 参数形状，再把通用能力增量注入领域目标。输出仍是原 1B 架构，且只改动 Q/K/V/O/Gate/Up/Down。embedding、LM head、norm、bias 和 tokenizer 全部保持目标模型原值。

ACI 的合并阶段：

- 不加载数据集；
- 不加载 tokenizer；
- 不构造 token ID 或合成文本；
- 不调用任一模型的 `forward`；
- 只读取模型权重与结构配置；
- 一次计算直接输出合并模型，不训练。

当前实现有意限定于仓库的 Llama-3.x 1B←8B 实验。它要求源与参考具有相同的 query-head 数和 KV-head 数，源 head dimension 是参考的整数倍。

## 2. 为什么替换 DF-OT-Procrustes

旧 DFOP 的已有诊断显示：

1. 行 softmax 路由大量集中到源模型第 0、30、31 层；
2. 已保存层对的 OT-Procrustes 解全部未达到收敛判据；
3. 跨宽度核心的缩放系数几乎全部触及上限；
4. Q/K/V/O 被当作普通矩阵独立映射，未保持 attention head、GQA 和 RoPE 坐标结构；
5. Gate/Up/Down 独立对齐，不能保证同一个 SwiGLU 神经元仍被共同处理；
6. 直接修改目标 top-rank 子空间，没有显式保护目标的领域微调增量。

这些问题不是继续增加 OT 求解器、路由分组或超参数就能简洁解决的，因此当前主线完全移除 OT、Sinkhorn、逐层路由和谱核心融合。

## 3. 主方法

ACI 只有三个概念步骤：Anchor、Compress、Inject。

### 3.1 Anchor：共享词表锚定残差空间

记源模型和参考模型的 embedding 权重为

\[
E_S\in\mathbb R^{V\times d_S},\qquad
E_0\in\mathbb R^{V\times d_0},\qquad d_S>d_0.
\]

从共享词表行中确定性地等距选取固定数量的锚点。每行做 L2 归一化，并在可用时同样加入 LM-head 行。构造交叉协方差

\[
C=\hat E_S^\top\hat E_0.
\]

若

\[
C=U\Sigma V^\top,
\]

则全局残差映射为

\[
P=UV^\top\in\mathbb R^{d_S\times d_0},
\qquad P^\top P=I.
\]

同一个 \(P\) 用于所有层和所有模块。Transformer 的 residual stream 跨 block 共享坐标，因此不再为七种矩阵分别猜测互不一致的神经元坐标。

### 3.2 Compress：按模型结构压缩 8B

#### 深度

把全部源层按相对深度划分为连续、有序且互不重叠的组。当前 32→16 层时：

\[
\mathcal G_\ell=\{2\ell,2\ell+1\}.
\]

每个源层只使用一次，不再从全深度范围自由选择边界层。组内先分别压缩到参考坐标，再求均值。

#### Attention

Llama-3.1 8B 与 Llama-3.2 1B 都有 32 个 query heads 和 8 个 KV heads，但 head dimension 分别为 128 和 64。

ACI 在每个 head 内选择具有相同 RoPE 基频的坐标。128→64 时选择：

```text
[0, 2, ..., 62, 64, 66, ..., 126]
```

频率压缩后，ACI 以 Q/K/V/O 的联合权重签名对 8 个 GQA 组做一一匹配，再在每组内对 4 个 query heads 做一一匹配。Q/K/V/O 共用这套置换，既消除 head 编号的任意性，又不拆散 GQA 依赖。记包含频率选择与联合置换的算子为 \(H_Q,H_{KV}\)，则：

\[
\widetilde W_Q=H_QW_Q^SP,
\]

\[
\widetilde W_K=H_{KV}W_K^SP,
\qquad
\widetilde W_V=H_{KV}W_V^SP,
\]

\[
\widetilde W_O=P^\top W_O^SH_Q^\top.
\]

Q/K 共享同一套 RoPE 频率选择；Q/O 共享 query-head 置换；K/V 共享 KV-group 置换。

#### FFN

SwiGLU 的第 \(j\) 个中间神经元由以下三部分共同定义：

- Gate 的第 \(j\) 行；
- Up 的第 \(j\) 行；
- Down 的第 \(j\) 列。

ACI 绝不拆开这三部分。利用 Anchor SVD 得到的参考侧低维基 \(R\)，构造联合签名：

\[
z_j=
\operatorname{norm}\left[
\operatorname{norm}(g_jR),
\operatorname{norm}(u_jR),
\operatorname{norm}(d_j^\top R)
\right].
\]

源侧先通过 \(P\) 投到参考 residual space，再计算同样签名。对参考 FFN 神经元与更宽的源 FFN 神经元做确定性、无重复的余弦贪心匹配，得到选择矩阵 \(S\)。随后：

\[
\widetilde W_{gate}=SW_{gate}^SP,
\]

\[
\widetilde W_{up}=SW_{up}^SP,
\]

\[
\widetilde W_{down}=P^\top W_{down}^SS^\top.
\]

同一个 \(S\) 同时作用于 Gate/Up/Down，保持 SwiGLU 神经元闭包。

### 3.3 Inject：保护领域任务向量

压缩会确定性地损失部分范数。对每层每模块，只做一次透明的 Frobenius 范数校准：

\[
\mathcal C(W_S)
=
\frac{\lVert W_0\rVert_F}
{\lVert\widetilde W_S\rVert_F+\epsilon}
\widetilde W_S.
\]

最终合并公式为：

\[
\boxed{
W_{out}
=
W_T+\beta\bigl(\mathcal C(W_S)-W_0\bigr)
}
\]

令领域任务向量为

\[
\Delta_{domain}=W_T-W_0,
\]

则

\[
W_{out}
=
W_0+\Delta_{domain}
+\beta\bigl(\mathcal C(W_S)-W_0\bigr).
\]

因此 \(\Delta_{domain}\) 的系数严格保持为 1，而不是被 \((1-\beta)\) 缩小。为避免单个张量异常，最终更新满足固定保护条件：

\[
\frac{\lVert W_{out}-W_T\rVert_F}
{\lVert W_T\rVert_F}
\le \beta.
\]

这里没有第二个 trust-ratio 超参数；\(\beta\) 同时表示注入强度和最大相对更新。

## 4. 唯一效果超参数

主方法只有 \(\beta\)。当前预注册值沿用原 T&M 在三个任务上的 merge-only 强度，而不是根据 ACI 测试结果倒推：

| Task | β |
|---|---:|
| Medical | 0.03 |
| Thai | 0.01 |
| Malay | 0.10 |

锚点数量、分块大小、FFN sketch 维度和候选数是计算参数，不改变方法定义。若进行 β 网格实验，必须完整报告网格，不能只报告测试集最优点并称为预设主结果。

## 5. 诊断与失败条件

每次运行写出：

- `run_report.json`：锚点余弦、投影正交误差、层分组和耗时；
- `attention_matches.jsonl`：GQA 组与 query-head 的联合匹配质量和置换；
- `ffn_matches.jsonl`：联合 FFN 匹配的均值/最小余弦及重复数；
- `injections.jsonl`：范数校准、保护系数和实际相对更新。

实现会直接拒绝以下情况：

- 目标与参考的层数或七模块形状不一致；
- 源模型比参考模型更窄或更浅；
- query-head 或 KV-head 数不同；
- head dimension 不是可整除的收缩关系；
- FFN 源宽度小于参考宽度。

## 6. 实验声明

仓库现有 `evaluation_results/dfop_*` 和 `transport_results/dfop/` 是旧方法的历史产物，不是 ACI 结果。ACI 尚需在服务器生成 checkpoint 并按 Medical、Thai、Malay 三个宏平均分别与原目标模型比较；在结果产生前，文档不声称已经提升。
