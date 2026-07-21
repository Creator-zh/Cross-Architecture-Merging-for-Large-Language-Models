# ACI：完整的无数据异构模型融合思路

> Anchor–Compress–Inject（ACI，锚定—压缩—注入）
>
> 本文中的行间公式统一使用标准 Markdown/LaTeX 数学块 `$$...$$`，可在支持数学公式的 Markdown 渲染器中正常显示。

## 1. 研究目标

给定三个模型：

- 领域目标模型 $M_T$：已经具备 Medical、Thai 或 Malay 子领域能力的 Llama-3.2 1B 模型；
- 同架构参考模型 $M_0$：与目标模型架构完全相同的通用 Llama-3.2 1B 模型；
- 异构源模型 $M_S$：能力更强但更宽、更深的 Llama-3.1 8B Instruct 模型。

目标是在不改变目标模型架构的前提下，把 8B 源模型中的通用能力迁移到 1B 领域模型，同时尽量完整保留目标模型已有的领域能力。输出模型仍然是可直接加载的 Llama-3.2 1B checkpoint。

ACI 的融合过程满足以下限制：

- 不使用训练集、验证集或校准数据；
- 不加载 tokenizer，不构造 token ID，也不生成合成文本；
- 不执行模型 `forward`；
- 不进行梯度更新或训练；
- 只读取模型权重和结构配置；
- 一次计算直接得到融合权重。

融合仅修改每个 Transformer block 中的七个线性矩阵：

$$
\mathcal{M}
=
\{Q,K,V,O,\mathrm{Gate},\mathrm{Up},\mathrm{Down}\}.
$$

目标模型的 embedding、LM head、normalization、bias、tokenizer 和网络结构全部保持不变。

## 2. 三模型分工

三个模型并不是简单做参数平均，而是分别承担不同角色：

| 模型 | 作用 |
|---|---|
| 领域目标模型 $M_T$ | 提供最终架构以及需要保护的领域能力 |
| 同架构参考模型 $M_0$ | 定义目标参数空间，并分离目标模型的领域任务向量 |
| 异构源模型 $M_S$ | 提供希望注入的较强通用能力 |

记三者对应的权重为 $W_T$、$W_0$ 和 $W_S$。同架构参考模型使目标模型的领域任务向量可以明确写成：

$$
\Delta_{\mathrm{domain}}
=
W_T-W_0.
$$

ACI 的核心原则是：融合后 $\Delta_{\mathrm{domain}}$ 的系数保持为 $1$，只在此基础上增加异构源模型的能力增量。

当前三个任务的模型组合为：

| 任务 | 目标模型 $M_T$ | 参考模型 $M_0$ | 源模型 $M_S$ |
|---|---|---|---|
| Medical | `PathFinderKR/Llama-3-1B-Medical-Instruct` | `unsloth/Llama-3.2-1B` | `unsloth/Llama-3.1-8B-Instruct` |
| Thai | `typhoon-ai/llama3.2-typhoon2-1b-instruct` | `unsloth/Llama-3.2-1B-Instruct` | `unsloth/Llama-3.1-8B-Instruct` |
| Malay | `mesolitica/Malaysian-Llama-3.2-1B-Instruct` | `unsloth/Llama-3.2-1B-Instruct` | `unsloth/Llama-3.1-8B-Instruct` |

Medical 使用 base 参考模型；Thai 和 Malay 使用 instruct 参考模型。这样可以尽量让 $W_T-W_0$ 表示领域或语言适配，而不是把通用指令微调差异错误地当作领域差异。

## 3. 为什么不能直接平均

异构模型不能直接执行如下线性平均：

$$
W_{\mathrm{out}}
=
(1-\beta)W_T+\beta W_S,
$$

因为 8B 与 1B 模型在以下方面不一致：

- residual hidden size 不同；
- Transformer 层数不同；
- attention head dimension 不同；
- FFN 中间维度不同；
- 神经元、attention head 和层编号不存在天然的一一对应关系。

即使先把两个张量裁剪到相同形状，直接平均仍然会破坏以下结构：

- Q/K 的 RoPE 频率对应关系；
- GQA 中 query head 与 KV head 的从属关系；
- Q 与 O 的成对关系；
- K 与 V 的成对关系；
- SwiGLU 中 Gate 行、Up 行与 Down 列所表示的同一个 FFN 神经元。

因此，异构融合必须先建立共享 residual 坐标，再按 Transformer 的结构压缩源模型，最后才进行能力增量注入。

## 4. 符号与矩阵形状

记目标/参考模型的 residual hidden size 为 $d_0$，源模型的 hidden size 为 $d_S$，且 $d_S>d_0$。记 query head 数为 $h_Q$，KV head 数为 $h_{KV}$，head dimension 分别为 $a_0$ 和 $a_S$。FFN 中间维度分别为 $m_0$ 和 $m_S$。

Hugging Face Llama 线性层的权重形状如下：

| 模块 | 源模型形状 | 目标/参考模型形状 |
|---|---:|---:|
| $W_Q$ | $(h_Qa_S)\times d_S$ | $(h_Qa_0)\times d_0$ |
| $W_K,W_V$ | $(h_{KV}a_S)\times d_S$ | $(h_{KV}a_0)\times d_0$ |
| $W_O$ | $d_S\times(h_Qa_S)$ | $d_0\times(h_Qa_0)$ |
| $W_{\mathrm{gate}},W_{\mathrm{up}}$ | $m_S\times d_S$ | $m_0\times d_0$ |
| $W_{\mathrm{down}}$ | $d_S\times m_S$ | $d_0\times m_0$ |

当前实验对应：

$$
L_S=32,\qquad L_0=16,
$$

$$
d_S=4096,\qquad d_0=2048,
$$

$$
h_Q=32,\qquad h_{KV}=8,
$$

$$
a_S=128,\qquad a_0=64,
$$

$$
m_S=14336,\qquad m_0=8192.
$$

## 5. 方法总览

ACI 包含三个连续步骤：

1. **Anchor**：利用共享词表中的 embedding/LM-head 权重，求一个全局 residual 映射 $P$；
2. **Compress**：用 $P$、深度分组、attention 结构匹配和 FFN 神经元匹配，把 8B 权重压缩成 1B 形状；
3. **Inject**：把“压缩源模型相对于通用 1B 参考模型的增量”注入领域目标模型。

概念上的主公式是：

$$
W_{\mathrm{out}}
=
W_T
+
\beta
\left(
\mathcal{C}(W_S)-W_0
\right),
$$

其中 $\mathcal{C}$ 是结构保持的异构压缩算子，$\beta$ 是唯一直接控制融合效果的超参数。这是旧模式与新安全模式共享的概念主式。

代码还会对上式的更新量施加由权重自动确定的安全操作。旧 `full/attention/ffn` 使用第 9.2 节的逐张量范数上限；当前优先验证的 `ffn_safe/attention_circuit/safe_combined` 还使用第 9.4–9.5 节的目标冲突投影与匹配置信门控。

## 6. Anchor：建立全局 residual 坐标映射

### 6.1 共享词表锚点

记源模型和参考模型的输入 embedding 为：

$$
E_S\in\mathbb{R}^{V\times d_S},
\qquad
E_0\in\mathbb{R}^{V\times d_0}.
$$

从二者共享的词表行中，确定性地等距选取 $n$ 个锚点。第 $r$ 个锚点的索引为：

$$
i_r
=
\left\lfloor
\frac{(2r+1)V}{2n}
\right\rfloor,
\qquad
r=0,1,\ldots,n-1.
$$

这一步只按词表行号读取权重，不需要知道 token 对应的文本，也不需要 tokenizer。

将选中的每个 embedding 行做 L2 归一化，得到 $\widehat E_S$ 和 $\widehat E_0$。如果两个模型都存在 LM head，则同样处理 LM-head 权重，并把输入与输出两部分交叉协方差取平均。

仅使用输入 embedding 时，交叉协方差为：

$$
C
=
\frac{1}{n}
\widehat E_S^{\mathsf T}
\widehat E_0
\in
\mathbb{R}^{d_S\times d_0}.
$$

### 6.2 矩形正交 Procrustes

ACI 求解如下矩形正交对齐问题：

$$
P^{\star}
=
\underset{P^{\mathsf T}P=I_{d_0}}{\arg\min}
\left\|
\widehat E_SP-\widehat E_0
\right\|_F^2.
$$

对交叉协方差做薄 SVD：

$$
C
=
U\Sigma V^{\mathsf T}.
$$

则闭式解为：

$$
P
=
UV^{\mathsf T}
\in
\mathbb{R}^{d_S\times d_0},
$$

并满足：

$$
P^{\mathsf T}P
=
I_{d_0}.
$$

同一个 $P$ 用于所有层以及 Q/K/V/O/Gate/Up/Down 七类矩阵。这样可以避免不同模块各自学习一套互相矛盾的 residual 坐标。

SVD 同时给出参考侧的右奇异向量。取前 $k$ 个方向组成：

$$
R
\in
\mathbb{R}^{d_0\times k}.
$$

$R$ 只用于后续 attention/FFN 匹配签名的低维计算，不改变最终模型维度。

## 7. Compress：结构保持的异构压缩

### 7.1 深度压缩

设源模型有 $L_S$ 层，目标模型有 $L_0$ 层。目标第 $\ell$ 层对应的源层集合定义为：

$$
\mathcal{G}_{\ell}
=
\left\{
\left\lfloor\frac{\ell L_S}{L_0}\right\rfloor,
\ldots,
\left\lfloor\frac{(\ell+1)L_S}{L_0}\right\rfloor-1
\right\}.
$$

在当前 $32\rightarrow16$ 的设置下：

$$
\mathcal{G}_{\ell}
=
\{2\ell,2\ell+1\}.
$$

这些组连续、单调、互不重叠，并且完整覆盖源模型的 32 层。每个源层只使用一次，避免自由路由集中到少数边界层。

设第 $s$ 个源层中模块 $m$ 的结构压缩结果为 $\widetilde W_{S,s}^{(m)}$，则目标第 $\ell$ 层接收的源权重为：

$$
\overline W_{S,\ell}^{(m)}
=
\frac{1}{|\mathcal{G}_{\ell}|}
\sum_{s\in\mathcal{G}_{\ell}}
\widetilde W_{S,s}^{(m)}.
$$

### 7.2 Attention 的 RoPE 坐标压缩

当前源模型和目标模型的 query-head 数与 KV-head 数相同，但源 head dimension 是目标的两倍。ACI 只选择具有对应 RoPE 基频的源坐标。

对 $128\rightarrow64$，每个 head 选择：

```text
[0, 2, 4, ..., 62, 64, 66, 68, ..., 126]
```

记单个 head 的频率选择矩阵为：

$$
D
\in
\{0,1\}^{a_0\times a_S}.
$$

选择矩阵满足每行恰有一个 $1$，表示从源 head 中抽取一个对应频率坐标。

### 7.3 GQA 与 query head 联合匹配

head 编号本身没有语义保证，因此不能假设源模型第 $g$ 个 GQA 组就对应参考模型第 $g$ 个组。ACI 先把源权重压缩到参考 residual 空间，再使用 Q/K/V/O 的联合低维签名进行一一匹配。

对每个 GQA 组构造联合签名 $z_g$，并求解：

$$
\pi_{G}^{\star}
=
\underset{\pi_G\in\mathfrak{S}_{h_{KV}}}{\arg\max}
\sum_{g=1}^{h_{KV}}
\operatorname{cos}
\left(
z_{0,g},
z_{S,\pi_G(g)}
\right).
$$

其中 $\mathfrak{S}_{h_{KV}}$ 表示所有 KV-group 双射。代码使用精确最大分配求解八个 GQA 组的匹配。

在每个已匹配的 GQA 组内部，再根据 Q 与 O 的联合签名，对该组中的 query heads 求双射：

$$
\pi_{Q\mid g}^{\star}
=
\underset{\pi\in\mathfrak{S}_{h_Q/h_{KV}}}{\arg\max}
\sum_q
\operatorname{cos}
\left(
z_{0,g,q},
z_{S,\pi_G(g),\pi(q)}
\right).
$$

由频率选择和匹配置换组成两个整体算子：

$$
H_Q
\in
\{0,1\}^{(h_Qa_0)\times(h_Qa_S)},
$$

$$
H_{KV}
\in
\{0,1\}^{(h_{KV}a_0)\times(h_{KV}a_S)}.
$$

最终 attention 压缩公式为：

$$
\widetilde W_Q
=
H_QW_Q^SP,
$$

$$
\widetilde W_K
=
H_{KV}W_K^SP,
$$

$$
\widetilde W_V
=
H_{KV}W_V^SP,
$$

$$
\widetilde W_O
=
P^{\mathsf T}W_O^SH_Q^{\mathsf T}.
$$

这套共享结构保证：

- Q 与 K 使用相同的 RoPE 频率压缩规则；
- K 与 V 使用相同的 KV-group 置换；
- Q 与 O 使用相同的 query-head 置换；
- 每个 query head 仍然属于正确的 GQA 组。

### 7.4 FFN 的 SwiGLU 神经元闭包

SwiGLU 中第 $j$ 个 FFN 神经元由三部分共同定义：

- $W_{\mathrm{gate}}$ 的第 $j$ 行；
- $W_{\mathrm{up}}$ 的第 $j$ 行；
- $W_{\mathrm{down}}$ 的第 $j$ 列。

这三部分必须一起选择，不能独立排序或独立匹配。

对参考模型第 $j$ 个 FFN 神经元，定义联合签名：

$$
z_{0,j}
=
\operatorname{norm}
\left[
\operatorname{norm}(g_{0,j}R),
\operatorname{norm}(u_{0,j}R),
\operatorname{norm}(d_{0,j}^{\mathsf T}R)
\right].
$$

其中 $g_{0,j}$ 和 $u_{0,j}$ 分别是 Gate/Up 的第 $j$ 行，$d_{0,j}$ 是 Down 的第 $j$ 列。

源模型先通过 $P$ 映射到参考 residual 空间，其签名为：

$$
z_{S,i}
=
\operatorname{norm}
\left[
\operatorname{norm}(g_{S,i}PR),
\operatorname{norm}(u_{S,i}PR),
\operatorname{norm}(d_{S,i}^{\mathsf T}PR)
\right].
$$

实现使用确定性的唯一贪心匹配：每个目标 FFN 神经元选择一个余弦相似度高的源神经元，同时禁止两个目标神经元重复使用同一个源神经元。候选冲突无法在候选集合中解决时，再从所有未使用源神经元中做确定性回退选择。

记最终选择矩阵为：

$$
S
\in
\{0,1\}^{m_0\times m_S},
$$

其中每一行恰有一个 $1$，每一列至多一个 $1$。FFN 压缩为：

$$
\widetilde W_{\mathrm{gate}}
=
SW_{\mathrm{gate}}^SP,
$$

$$
\widetilde W_{\mathrm{up}}
=
SW_{\mathrm{up}}^SP,
$$

$$
\widetilde W_{\mathrm{down}}
=
P^{\mathsf T}W_{\mathrm{down}}^SS^{\mathsf T}.
$$

同一个 $S$ 同时用于 Gate、Up 和 Down，从而保持完整的 SwiGLU 神经元结构。

## 8. 压缩后的范数校准

宽度选择和深度平均会确定性地改变权重能量。为避免源增量仅仅由尺寸压缩造成的范数变化主导，ACI 对每个目标层、每个模块分别进行一次 Frobenius 范数校准。

对目标第 $\ell$ 层的模块 $m$，定义：

$$
\alpha_{\ell}^{(m)}
=
\frac{
\left\|W_{0,\ell}^{(m)}\right\|_F
}{
\max\left(
\left\|\overline W_{S,\ell}^{(m)}\right\|_F,
\varepsilon
\right)
}.
$$

校准后的压缩源权重为：

$$
\mathcal{C}_{\ell}^{(m)}(W_S)
=
\alpha_{\ell}^{(m)}
\overline W_{S,\ell}^{(m)}.
$$

这里不匹配逐行范数，也不引入可学习缩放；只做每个张量一次可解释的整体能量校准。

## 9. Inject：保护领域任务向量并限制更新

### 9.1 能力增量

压缩源模型相对通用参考模型的能力增量定义为：

$$
\Delta_{S,\ell}^{(m)}
=
\mathcal{C}_{\ell}^{(m)}(W_S)
-
W_{0,\ell}^{(m)}.
$$

未经保护的原始更新为：

$$
U_{\ell}^{(m)}
=
\beta
\Delta_{S,\ell}^{(m)}.
$$

### 9.2 自动安全系数

为防止单个张量出现异常大更新，代码根据权重范数自动计算安全系数：

$$
\tau_{\ell}^{(m)}
=
\min
\left(
1,
\frac{
\beta
\left\|W_{T,\ell}^{(m)}\right\|_F
}{
\max\left(
\left\|U_{\ell}^{(m)}\right\|_F,
\varepsilon
\right)
}
\right).
$$

当 $\beta=0$ 或原始更新为零时，定义 $\tau=1$。

与代码完全一致的最终融合公式是：

$$
\boxed{
W_{\mathrm{out},\ell}^{(m)}
=
W_{T,\ell}^{(m)}
+
\tau_{\ell}^{(m)}\beta
\left(
\mathcal{C}_{\ell}^{(m)}(W_S)
-
W_{0,\ell}^{(m)}
\right)
}
$$

因此每个张量都满足：

$$
\frac{
\left\|
W_{\mathrm{out},\ell}^{(m)}
-
W_{T,\ell}^{(m)}
\right\|_F
}{
\max\left(
\left\|W_{T,\ell}^{(m)}\right\|_F,
\varepsilon
\right)
}
\leq
\beta.
$$

$\tau$ 不是需要调节的第二个超参数，而是由当前张量自动计算的保护系数。实际有效注入强度为：

$$
\lambda_{\ell}^{(m)}
=
\tau_{\ell}^{(m)}\beta,
\qquad
0\leq\lambda_{\ell}^{(m)}\leq\beta.
$$

### 9.3 为什么领域能力不会被线性稀释

把目标权重写成：

$$
W_T
=
W_0+\Delta_{\mathrm{domain}}.
$$

代入最终公式可得：

$$
W_{\mathrm{out}}
=
W_0
+
\Delta_{\mathrm{domain}}
+
\lambda
\left(
\mathcal{C}(W_S)-W_0
\right).
$$

可以看到，领域任务向量 $\Delta_{\mathrm{domain}}$ 的系数严格为 $1$。这与普通线性插值不同：普通插值会把目标模型整体乘以 $1-\beta$，从而直接缩小领域任务向量。

需要注意：系数保持为 $1$ 只表示“不在线性公式中主动削弱领域增量”，并不构成对所有评测任务必然提升的数学保证。新增源增量仍可能与领域能力发生干扰，最终效果必须通过服务器实验验证。

### 9.4 FFN：目标冲突投影与神经元置信门控

`ffn_safe` 不再把 Gate、Up、Down 当作三个独立更新。对第 $j$ 个完整 SwiGLU 神经元，记源能力增量和目标领域增量分别为：

$$
s_j=
\left(
\Delta_{S,j}^{\mathrm{gate}},
\Delta_{S,j}^{\mathrm{up}},
\Delta_{S,j}^{\mathrm{down}}
\right),
\qquad
d_j=
\left(
\Delta_{D,j}^{\mathrm{gate}},
\Delta_{D,j}^{\mathrm{up}},
\Delta_{D,j}^{\mathrm{down}}
\right),
$$

其中

$$
\Delta_D=W_T-W_0.
$$

三部分的内积与范数联合计算。当源增量和领域增量冲突时，只去掉源增量在领域反方向上的分量：

$$
\widehat s_j
=
s_j
-
\min\left(
\frac{\langle s_j,d_j\rangle}
{\max(\lVert d_j\rVert_2^2,\varepsilon)},
0
\right)d_j.
$$

因此有

$$
\langle \widehat s_j,d_j\rangle\ge 0.
$$

神经元匹配阶段保存被选源神经元的余弦相似度。对同一目标层所接收的源层取平均，并定义无需调节的置信门：

$$
q_j
=
\operatorname{clip}
\left(
\frac{1}{|\mathcal G_\ell|}
\sum_{s\in\mathcal G_\ell}
\operatorname{cos}(z_{0,j},z_{S,s,\pi_s(j)}),
0,1
\right).
$$

最终 FFN 原始更新为：

$$
U_j^{\mathrm{FFN}}=\beta q_j\widehat s_j.
$$

Gate 行、Up 行和 Down 列共享同一个 $q_j$，并对整层三个张量联合计算一次范数上限；不会出现三个因子各自选择不同神经元或不同安全缩放。

### 9.5 Attention：QK/OV 电路匹配、gauge 对齐与安全注入

旧 Attention 的裸 Q/K/V/O 签名余弦很低，且不同因子分解之间存在不影响功能的 gauge 自由度。`attention_circuit` 因此改为在参考低维基 $R$ 上比较实际功能电路。对 GQA 组 $g$ 中的 query head $q$，定义：

$$
A_{gq}^{QK}
=
(Q_{gq}R)^{\mathsf T}(K_gR),
$$

$$
A_{gq}^{OV}
=
(R^{\mathsf T}O_{gq})(V_gR),
$$

$$
z_{gq}^{\mathrm{circuit}}
=
\operatorname{norm}
\left[
\operatorname{vec}(A_{gq}^{QK}),
\operatorname{vec}(A_{gq}^{OV})
\right].
$$

组间分数是组内 query-head 最优双射的平均余弦；先求 GQA-group 双射，再采用对应的组内 query-head 双射。匹配后还要消除功能等价但参数坐标不同的问题：

1. 对每个 RoPE 频率对求 $G_{g,r}^{QK}\in SO(2)$，并同时变换

   $$
   Q'_{gq,r}=G_{g,r}^{QK}Q_{gq,r},
   \qquad
   K'_{g,r}=G_{g,r}^{QK}K_{g,r}.
   $$

   因为二维旋转彼此可交换且 $G^{\mathsf T}G=I$，任意位置上的 RoPE QK 双线性形式保持不变。

2. 对每个 GQA 组求 $G_g^{OV}\in O(a_0)$，并同时变换

   $$
   V'_g=G_g^{OV}V_g,
   \qquad
   O'_{gq}=O_{gq}(G_g^{OV})^{\mathsf T}.
   $$

   从而严格保持 $O'_{gq}V'_g=O_{gq}V_g$。

head 置信度由最终选中的组分数和 query 分数组成：

$$
q_{gq}
=
\operatorname{clip}
\left(
\frac{c_g+c_{gq}}{2},0,1
\right).
$$

Q 和 O 使用逐 head 的 $q_{gq}$；共享的 K 和 V 使用组内置信度均值。目标冲突投影分别在每个 GQA 组的联合 $(Q,K)$ 因子和联合 $(V,O)$ 因子上执行，公式与第 9.4 节相同。最后 QK 与 OV 各共享一个自动范数上限，避免破坏成对因子关系。

`safe_combined` 同时启用第 9.4 与第 9.5 节；没有新增需要搜索的效果超参数。

## 10. 完整算法

输入：目标模型 $M_T$、同架构参考模型 $M_0$、异构源模型 $M_S$、融合强度 $\beta$。

1. 检查 $M_T$ 与 $M_0$ 的层数、七类线性层形状和 attention 几何完全一致。
2. 检查 $M_S$ 比 $M_0$ 更深且更宽，并满足当前 attention 收缩条件。
3. 从共享 embedding/LM-head 行构造交叉协方差，SVD 求全局映射 $P$ 和签名基 $R$。
4. 按相对深度把全部源层划分为连续组 $\mathcal{G}_{\ell}$。
5. 对每个目标层 $\ell$：
   1. 对每个 $s\in\mathcal{G}_{\ell}$，执行 RoPE 频率选择；
   2. 用 QK/OV 功能电路联合匹配 GQA group 和组内 query head，并做保持功能的 gauge 对齐；
   3. 联合匹配 SwiGLU 神经元，压缩 Gate/Up/Down，并保存逐神经元匹配置信度；
   4. 对组内各源层的压缩结果求均值；
   5. 对七个模块分别做 Frobenius 范数校准；
   6. 对 FFN 神经元及 QK/OV 组投影目标冲突、应用匹配置信门，再计算联合安全系数和最终融合权重。
6. 保持目标模型的其余参数不变。
7. 保存 1B 融合模型、目标 tokenizer 元数据和全部诊断文件。

可将整体过程概括为：

$$
M_S
\xrightarrow[\text{无数据}]{\text{Anchor}}
P
\xrightarrow{\text{结构化 Compress}}
\mathcal{C}(W_S)
\xrightarrow[W_0]{\text{Inject into }W_T}
W_{\mathrm{out}}.
$$

## 11. 超参数

### 11.1 唯一的效果超参数

唯一直接控制能力注入大小的超参数是 $\beta$：

| 任务 | 当前预注册值 |
|---|---:|
| Medical | $0.03$ |
| Thai | $0.01$ |
| Malay | $0.10$ |

这些值沿用原 T&M merge-only 实验的任务强度，并不是根据 ACI 测试结果反向挑选的最优值，因此只能作为首轮服务器实验的预设起点。

### 11.2 数值与计算参数

以下参数用于控制显存、计算量或确定性近似，不应包装成额外的任务调优参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `anchor_tokens` | 8192 | 用于求 $P$ 的确定性共享词表行数 |
| `anchor_chunk_size` | 1024 | 分块累计交叉协方差，控制显存 |
| `ffn_sketch_dim` | 32 | FFN 匹配签名维度 $k$ |
| `ffn_candidate_k` | 32 | 唯一贪心匹配的候选数 |
| `eps` | $10^{-8}$ | 数值稳定项 |

## 12. 与旧 DF-OT-Procrustes 的关键区别

| 方面 | 旧 DFOP | ACI |
|---|---|---|
| 层映射 | 学习式软路由，可能塌缩到少数层 | 连续单调分组，全部源层各使用一次 |
| 坐标对齐 | 多个局部 OT/Procrustes 问题 | 共享词表锚定的单个全局映射 $P$ |
| Attention | 把 Q/K/V/O 当普通矩阵独立处理 | 匹配 QK/OV 电路，保持 RoPE/GQA 并消除等价 gauge |
| FFN | Gate/Up/Down 可能独立对齐 | 三者共享 $S$、目标冲突投影和逐神经元置信门 |
| 领域保护 | 直接改写目标子空间 | 显式保留 $W_T-W_0$ 的单位系数 |
| 效果超参数 | 路由、OT、rank、scale、trust 等多个参数 | 仅 $\beta$ |
| 求解稳定性 | 依赖迭代收敛 | SVD 闭式映射 + 确定性离散匹配 |

ACI 仍在 Anchor 阶段使用一次闭式正交 Procrustes，但不再使用 OT、Sinkhorn、逐层软路由或谱核心融合。

## 13. 诊断与正确性检查

每次融合会生成以下诊断文件：

- `run_report.json`：模型层数、层分组、锚点余弦、投影形状、正交误差和耗时；
- `attention_matches.jsonl`：GQA/query-head 置换、QK/OV 电路余弦、head 置信度和 gauge 对齐前后余弦；
- `ffn_matches.jsonl`：FFN 匹配的平均/最小余弦、正匹配比例和源神经元重复数；
- `injections.jsonl`：范数校准、置信度、冲突比例、移除冲突范数、自动安全系数和实际相对更新量。

结构正确性应满足：

$$
\left\|P^{\mathsf T}P-I\right\|_{\infty}
\approx 0,
$$

$$
\texttt{reused\_sources}=0,
$$

$$
0
\leq
\texttt{joint\_relative\_update\_norm}
\leq
\beta
\qquad
\text{（安全模式）}.
$$

旧模式对应的逐模块 `relative_update_norm` 仍满足同一上界。

以下情况会被实现直接拒绝：

- 目标模型与参考模型的层数或七类模块形状不同；
- 目标模型与参考模型的 attention geometry 不同；
- 源模型比参考模型更窄或更浅；
- 源/参考 query-head 数或 KV-head 数不同；
- source head dimension 不能被 target head dimension 整除；
- 源 FFN 宽度小于目标 FFN 宽度。

## 14. 实验设计与验收标准

### 14.1 必须比较的模型

每个领域至少评测：

1. 原领域目标模型 `target`；
2. 同架构通用参考模型 `reference`；
3. 8B 异构源模型 `source`；
4. 无数据融合模型 `ffn_safe`、`attention_circuit` 和 `safe_combined`。

可选的 `aci_sft` 只作为独立对比，不属于无数据 ACI 主方法。

### 14.2 主验收标准

对领域 $D$ 的 $N_D$ 个任务，宏平均为：

$$
\operatorname{MacroAvg}(D,M)
=
\frac{1}{N_D}
\sum_{i=1}^{N_D}
\operatorname{Score}(D_i,M).
$$

主验收条件是三个领域分别满足：

$$
\operatorname{MacroAvg}(D,M_{\mathrm{ACI}})
>
\operatorname{MacroAvg}(D,M_T),
$$

其中：

$$
D\in\{\mathrm{Medical},\mathrm{Thai},\mathrm{Malay}\}.
$$

不把三个领域继续合成为一个总平均，以免一个领域的大幅提升掩盖另一个领域的退化。

理想结果是每个子任务都超过原目标模型；现实的最低标准是每个领域自身的宏平均超过目标模型。

### 14.3 合法选择 $\beta$

如需搜索每个领域独立的 $\beta$，应预先声明完整网格，例如：

$$
\beta
\in
\{0.005,0.01,0.02,0.03,0.05,0.10\}.
$$

选择规则必须基于独立验证集或开发集。若任务没有可用验证集，应完整报告整个网格，并保留预注册默认值作为主结果；不能只保留测试集最高分再声称它是预设结果。

## 15. 方法假设与局限

ACI 的能力提升建立在以下假设上：

1. 共享词表的 embedding/LM-head 行可以为 8B 与 1B residual space 提供稳定锚点；
2. 较强 8B 模型相对通用 1B 参考模型的差异包含可迁移的通用能力；
3. RoPE/GQA/SwiGLU 结构保持能减少异构压缩引入的功能破坏；
4. 小幅、受限的源增量可以在不明显覆盖领域任务向量的情况下改善能力。

当前局限包括：

- 静态权重对齐无法完全消除神经网络内部的置换与缩放等价性；
- 单个全局线性映射 $P$ 未必能描述所有层的非线性表征差异；
- FFN 使用确定性贪心匹配而非全局最优的大规模双射；
- 当前实现只面向本仓库的 Llama-3.x 1B←8B 几何关系；
- 理论上的领域向量保护不能保证评测分数必然提升；
- 当前仓库尚无 ACI 服务器评测结果，因此不能提前宣称已经超过目标模型。

## 16. 一句话总结

ACI 先用共享词表权重建立唯一的 8B→1B residual 映射，再按深度、RoPE、GQA 和 SwiGLU 结构压缩 8B，最后把压缩源模型相对于通用 1B 参考模型的受限增量加到领域 1B 模型上：

$$
\boxed{
W_{\mathrm{out}}
=
W_T
+
\lambda
\left(
\mathcal{C}(W_S)-W_0
\right),
\qquad
0\leq\lambda\leq\beta
}
$$

其核心不是把三个模型直接平均，而是“先解决异构坐标与结构对应，再只注入源模型相对于通用参考模型的能力增量”。
