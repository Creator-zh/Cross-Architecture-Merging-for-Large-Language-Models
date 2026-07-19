# DF-OT-Procrustes：无数据异构模型融合实现规范

## 0. 文档目的

本文档定义一个可直接转化为代码的、完全无数据的异构 Transformer 权重融合方法。方法暂称 **DF-OT-Procrustes**（Data-Free Optimal-Transport Procrustes）。

主方法满足以下约束：

- 对齐与融合阶段不读取任何训练集、校准集或合成文本；
- 不执行模型 forward pass，不提取 activation；
- 只读取源模型和目标模型的权重及结构配置；
- 支持源模型与目标模型的层数、隐藏宽度、FFN 宽度和注意力投影维度不同；
- 只处理功能同源的线性模块：`q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`；
- embedding、LM head、RMSNorm/LayerNorm 和 bias 默认保持目标模型不变；
- 主输出为折叠后的稠密目标模型；同一更新可以等价保存为 LoRA；
- 融合后 SFT 是独立的对比实验，不属于无数据主方法。

方法的核心目标不是证明两个模型具有相同的语义神经元，而是利用权重诱导的谱几何，在不同大小的神经元集合之间建立可计算的软对应。

---

## 1. 适用范围与方法假设

### 1.1 模型角色

- 目标模型 \(\mathcal M_A\)：已经在特定领域或低资源语言上微调过的小模型，共 \(L\) 个 Transformer block；
- 源模型 \(\mathcal M_B\)：参数规模更大的通用模型，共 \(M\) 个 Transformer block；
- 输出模型保持 \(\mathcal M_A\) 的架构和全部参数形状不变。

### 1.2 结构同源假设

两个模型应当是 decoder-only Transformer，并且只在同名、同功能模块之间传输：

$$
\mathcal C=
\{Q,K,V,O,\mathrm{gate},\mathrm{up},\mathrm{down}\}.
$$

方法允许 GQA/MQA/MHA、隐藏维度和 FFN 维度不同，但不允许把功能不同的模块直接匹配，例如不能把 `q_proj` 与 `down_proj` 融合。

### 1.3 可识别性边界

在没有数据和外部锚点时，无法从数学上保证权重几何对应真实语义。本文方法依赖以下工作假设：

> 同源 Transformer 模块的权重所诱导的神经元谱几何，在不同模型之间保留了足够的关系结构，可以用于构造启发式但置换不敏感的跨模型对应。

因此，论文表述应使用“weight-induced spectral geometry alignment”，不应宣称恢复了唯一的语义坐标变换。

---

## 2. 记号与张量形状

对于模块 \(c\in\mathcal C\)，目标层 \(\ell\) 和源层 \(m\) 的权重分别记为：

$$
W^c_{A,\ell}
\in
\mathbb R^{a^c_{o,\ell}\times a^c_{i,\ell}},
$$

$$
W^c_{B,m}
\in
\mathbb R^{b^c_{o,m}\times b^c_{i,m}}.
$$

一般情况下：

$$
a^c_{o,\ell}\neq b^c_{o,m},
\qquad
a^c_{i,\ell}\neq b^c_{i,m}.
$$

主要张量形状如下。

| 符号 | 形状 | 含义 |
|---|---:|---|
| \(U_A\) | \(a_o\times k\) | 目标左奇异子空间 |
| \(U_B\) | \(b_o\times k\) | 源左奇异子空间 |
| \(V_A\) | \(a_i\times k\) | 目标右奇异子空间 |
| \(V_B\) | \(b_i\times k\) | 源右奇异子空间 |
| \(X_U,Y_U\) | \(a_o\times k,b_o\times k\) | 输出神经元谱点云 |
| \(X_V,Y_V\) | \(a_i\times k,b_i\times k\) | 输入神经元谱点云 |
| \(\Pi_U\) | \(a_o\times b_o\) | 输出神经元 OT coupling |
| \(\Pi_V\) | \(a_i\times b_i\) | 输入神经元 OT coupling |
| \(T_U\) | \(a_o\times b_o\) | 源输出坐标到目标输出坐标的重心映射 |
| \(T_V\) | \(a_i\times b_i\) | 源输入坐标到目标输入坐标的重心映射 |
| \(P^c\) | \(L\times M\) | 模块 \(c\) 的外层 OT coupling |
| \(\bar P^c\) | \(L\times M\) | 按目标层条件化后的层聚合权重 |
| \(K^c_{\ell m}\) | \(k\times k\) | 源层传输到目标公共子空间后的核心矩阵 |

除特别说明外，推导中省略 \(c,\ell,m\) 上标和下标。

---

## 3. 总体流程

从论文叙事上，方法保持两个概念阶段：

1. **层间匹配**：由权重谱几何计算每个层对的兼容性，再求外层 OT 得到 \(P^c\)；
2. **层内传输与融合**：复用层间匹配过程中得到的神经元 coupling，将候选源权重传输到目标形状并融合。

为方便代码实现，这两个概念阶段细分为四个计算阶段：

1. **谱分解与点云构造**：对各层同名模块做截断 SVD，将不同数量的输入/输出神经元表示为相同 \(k\) 维的谱点云；
2. **内层 OT–Procrustes**：在不同大小点云之间联合求解神经元 coupling 和谱坐标旋转，构造跨维重心映射，并计算层对代价；
3. **外层层 OT**：由所有目标层—源层代价得到模块特定的全局层对应 \(P^c\)；
4. **公共子空间融合**：复用内层 coupling，将源权重传输到目标形状，在目标 top-\(k\) 子空间内融合稠密核心，保留目标谱尾部。

概念流程为：

```text
权重矩阵
  -> 截断 SVD
  -> 输出/输入谱点云
  -> 内层 OT-Procrustes
  -> 神经元 coupling + 跨维映射 + 层对代价
  -> 外层 OT
  -> 层路由 P
  -> 源权重跨维传输
  -> 目标公共子空间核心融合
  -> 信任域裁剪
  -> 稠密目标模型 / 等价 LoRA
```

---

## 4. 阶段一：截断 SVD 与缓存

### 4.1 截断 SVD

对每个权重矩阵：

$$
W\approx U_r\Sigma_rV_r^\top,
$$

其中：

$$
\Sigma_r=\operatorname{diag}(\sigma_1,\ldots,\sigma_r),
\qquad
\sigma_1\geq\cdots\geq\sigma_r\geq0.
$$

每个矩阵的缓存秩定义为：

$$
r(W)=
\min\left(
r_{\max},
\left\lfloor
\rho\min(d_{\mathrm{out}},d_{\mathrm{in}})
\right\rfloor
\right).
$$

实现时使用 randomized SVD，并至少缓存：

```python
SVDRecord(
    U: Tensor[d_out, r],
    S: Tensor[r],
    Vh: Tensor[r, d_in],
    fro_norm: float,
    shape: tuple[int, int],
    explained_energy: float,
)
```

### 4.2 层对公共秩

对于层对 \((\ell,m)\)，使用：

$$
k_{\ell m}^c
=
\min\left(
r(W^c_{A,\ell}),
r(W^c_{B,m}),
k_{\max}
\right).
$$

从缓存中截取前 \(k_{\ell m}^c\) 个奇异三元组。不能通过截断矩阵的神经元行来解决宽度差异；只允许截断奇异模式列。

### 4.3 数值要求

- SVD 和所有 OT 代价至少使用 FP32；
- 原始权重可以保存在 FP16/BF16，计算时分块转换；
- 对奇异值使用 `clamp_min(svd_eps)`；
- 对未达到最小有效秩的矩阵直接标记该层对无效；
- 缓存必须包含模型、层、模块、权重形状和 SVD 配置哈希，防止误复用。

---

## 5. 阶段二：构造不同大小的谱点云

### 5.1 输出侧点云

定义：

$$
X_U=U_A\Sigma_A^\gamma
\in\mathbb R^{a_o\times k},
$$

$$
Y_U=U_B\Sigma_B^\gamma
\in\mathbb R^{b_o\times k}.
$$

主设置取 \(\gamma=1\)。此时：

$$
X_UX_U^\top
=U_A\Sigma_A^2U_A^\top
\approx W_AW_A^\top,
$$

点云行之间的几何对应权重行空间的几何。

### 5.2 输入侧点云

$$
X_V=V_A\Sigma_A^\gamma
\in\mathbb R^{a_i\times k},
$$

$$
Y_V=V_B\Sigma_B^\gamma
\in\mathbb R^{b_i\times k}.
$$

当 \(\gamma=1\) 时：

$$
X_VX_V^\top
=V_A\Sigma_A^2V_A^\top
\approx W_A^\top W_A.
$$

### 5.3 中心化与尺度归一化

对任意点云：

$$
X=[x_1^\top,\ldots,x_n^\top]^\top,
\qquad
a_i=\frac1n,
$$

定义：

$$
\mu_X=\sum_i a_ix_i,
$$

$$
s_X^2=\sum_i a_i\|x_i-\mu_X\|_2^2,
$$

$$
\overline X
=
\frac{X-\mathbf1\mu_X^\top}{s_X+\epsilon}.
$$

默认使用均匀质量，以避免大型模型中高范数坐标完全主导 coupling。可选的 leverage mass 为：

$$
a_i
=
\frac{\|x_i\|_2^2+\epsilon}
{\sum_j\|x_j\|_2^2+n\epsilon},
$$

但只能作为消融项。

### 5.4 置换与符号性质

- 神经元置换只会置换点云的行，后续 OT 会同步置换 coupling，层对代价应保持不变；
- SVD 单列符号翻转属于谱空间的正交变换，由 Procrustes 旋转吸收；
- 重复奇异值对应的子空间旋转同样由 Procrustes 处理。

---

## 6. 阶段三：内层 OT–Procrustes

### 6.1 输出侧目标

令目标输出神经元质量为：

$$
a^U=\frac1{a_o}\mathbf1_{a_o},
$$

源输出神经元质量为：

$$
b^U=\frac1{b_o}\mathbf1_{b_o}.
$$

运输多面体：

$$
\mathcal U(a^U,b^U)
=
\left\{
\Pi\in\mathbb R_+^{a_o\times b_o}:
\Pi\mathbf1_{b_o}=a^U,
\Pi^\top\mathbf1_{a_o}=b^U
\right\}.
$$

将点视为列向量，联合求解：

$$
\boxed{
(\Pi_U^*,R_U^*)
=
\arg\min_{
\substack{
\Pi_U\in\mathcal U(a^U,b^U)\\
R_U\in\mathcal O(k)
}}
\sum_{p,q}
\Pi^U_{pq}
\|\bar x^U_p-R_U\bar y^U_q\|_2^2
-\varepsilon_UH(\Pi_U)
}
$$

其中：

$$
H(\Pi)
=
-\sum_{p,q}\Pi_{pq}
(\log(\Pi_{pq}+\epsilon)-1).
$$

该联合问题关于 \((\Pi,R)\) 不是联合凸问题。实现只能保证交替下降到局部稳定点，因此必须记录初始化和最终目标值，并支持少量重启。

### 6.2 固定旋转时更新 coupling

使用行存储点云时，旋转后的源点云为：

$$
\overline Y_U R_U^\top.
$$

代价矩阵：

$$
C^U_{pq}(R_U)
=
\|\bar x^U_p\|_2^2
+\|\bar y^U_q\|_2^2
-2\bar x_p^{U\top}R_U\bar y_q^U.
$$

矩阵形式的交叉项为：

$$
\overline X_U R_U\overline Y_U^\top.
$$

求解：

$$
\Pi_U
=
\arg\min_{
\Pi\in\mathcal U(a^U,b^U)}
\langle C^U(R_U),\Pi\rangle
-\varepsilon_UH(\Pi).
$$

使用 log-domain Sinkhorn，禁止直接构造 `exp(-C / eps)` 后进行普通除法。

### 6.3 固定 coupling 时更新旋转

定义加权交叉矩阵：

$$
S_U
=
\overline X_U^\top
\Pi_U
\overline Y_U
\in\mathbb R^{k\times k}.
$$

求解：

$$
R_U^*
=
\arg\max_{R\in\mathcal O(k)}
\operatorname{tr}(R^\top S_U).
$$

对 \(S_U\) 做 SVD：

$$
S_U=L_U\Gamma_UH_U^\top,
$$

则：

$$
\boxed{R_U^*=L_UH_U^\top.}
$$

### 6.4 输入侧

将 \(a_o,b_o,X_U,Y_U\) 替换为 \(a_i,b_i,X_V,Y_V\)，得到：

$$
(\Pi_V^*,R_V^*).
$$

### 6.5 交替算法

```text
Input: normalized point clouds X[n_A,k], Y[n_B,k]
       marginals a[n_A], b[n_B]
       inner entropy eps, max_alt, sinkhorn settings

for each initialization R_0:
    R <- R_0
    previous_objective <- +inf

    for t in 1..max_alt:
        C[p,q] <- ||x_p||^2 + ||y_q||^2 - 2 x_p^T R y_q
        Pi <- LogDomainSinkhorn(C, a, b, eps)

        S <- X^T Pi Y
        L, _, Ht <- svd(S)
        R <- L Ht

        C_new[p,q] <- ||x_p||^2 + ||y_q||^2 - 2 x_p^T R y_q
        objective <- sum(Pi * C_new) - eps * H(Pi)
        if relative_change(objective) < alt_tol:
            break

return the restart with minimum final objective
```

推荐初始化：

1. \(R_0=I_k\)；
2. 仅用奇异值排序和列符号规范化得到的旋转；
3. 预算允许时加入 1–2 个随机正交初始化。

### 6.6 层对可比代价

用于比较不同层对时，不使用包含熵项的优化目标，而使用最终几何残差：

$$
D_{\mathrm{OP},U}
=
\langle C^U(R_U^*),\Pi_U^*\rangle,
$$

$$
D_{\mathrm{OP},V}
=
\langle C^V(R_V^*),\Pi_V^*\rangle.
$$

原因是不同点数、不同熵正则强度下，熵项不适合作为层兼容性的直接比较量。

---

## 7. 从 coupling 构造跨维映射

### 7.1 重心映射

输出侧：

$$
\boxed{
T_U
=
\operatorname{diag}(a^U)^{-1}\Pi_U^*
\in\mathbb R^{a_o\times b_o}
}
$$

输入侧：

$$
\boxed{
T_V
=
\operatorname{diag}(a^V)^{-1}\Pi_V^*
\in\mathbb R^{a_i\times b_i}.
}
$$

它们满足：

$$
T_U\mathbf1_{b_o}=\mathbf1_{a_o},
\qquad
T_V\mathbf1_{b_i}=\mathbf1_{a_i}.
$$

主实现使用该重心映射进行权重传输，并在公共核心空间做显式尺度校准。

### 7.2 可选部分等距映射

软重心映射会收缩范数。若计算预算允许，可以使用最近部分等距映射。

对 \(T\) 做薄 SVD：

$$
T=L_T\Sigma_TH_T^\top,
$$

定义：

$$
Q_{\mathrm{polar}}=L_TH_T^\top.
$$

当目标维度不大于源维度且满行秩时：

$$
Q_{\mathrm{polar}}Q_{\mathrm{polar}}^\top=I.
$$

也可以使用正则白化：

$$
Q_{\mathrm{white}}
=(TT^\top+\delta I)^{-1/2}T.
$$

但完整 SVD 或逆平方根对大宽度矩阵代价高，因此：

- 默认 `map_mode="barycentric"`；
- `polar` 和 `whiten` 作为小规模实验或消融；
- 不允许用神经元行截断、补零代替 \(T_U,T_V\)。

---

## 8. 映射后的主角度与 Grassmann 距离

### 8.1 输出侧

将源左奇异子空间映射到目标输出环境空间：

$$
Z^U_{B\rightarrow A}=T_UU_B
\in\mathbb R^{a_o\times k}.
$$

对其做经济 QR 或 SVD 正交化：

$$
\widetilde U_B=\operatorname{qf}(Z^U_{B\rightarrow A}).
$$

此时：

$$
U_A,\widetilde U_B\in\mathbb R^{a_o\times k'}.
$$

计算：

$$
\operatorname{svdvals}(U_A^\top\widetilde U_B)
=(\cos\theta_1^U,\ldots,\cos\theta_{k'}^U).
$$

所有余弦必须裁剪到 \([0,1]\)：

$$
\theta_i^U
=
\arccos\left(
\operatorname{clip}(\cos\theta_i^U,0,1)
\right).
$$

Grassmann geodesic 距离：

$$
d_{\mathrm{Gr},U}^2
=
\sum_{i=1}^{k'}(\theta_i^U)^2.
$$

更稳定的 chordal 距离：

$$
d_{\mathrm{ch},U}^2
=
k'-\|U_A^\top\widetilde U_B\|_F^2.
$$

主实现推荐 chordal 距离，geodesic 距离用于报告和消融。

### 8.2 输入侧

$$
Z^V_{B\rightarrow A}=T_VV_B,
$$

$$
\widetilde V_B=\operatorname{qf}(Z^V_{B\rightarrow A}),
$$

得到：

$$
d_{\mathrm{ch},V}^2,
\qquad
d_{\mathrm{Gr},V}^2.
$$

### 8.3 秩亏惩罚

若 \(T_UU_B\) 和 \(T_VV_B\) 的数值秩分别为 \(k'_U,k'_V\)，加入：

$$
D_{\mathrm{rank}}
=
\frac{k-k'_U}{k}
+
\frac{k-k'_V}{k}.
$$

禁止通过补随机正交列掩盖映射秩亏。

---

## 9. 层对代价

### 9.1 宽度无关谱形状距离

对奇异值定义：

$$
p_i(W)=\frac{\sigma_i^2}{\sum_j\sigma_j^2},
\qquad
x_i=\frac{i-\frac12}{r}.
$$

谱能量测度：

$$
\mu_W=\sum_i p_i(W)\delta_{x_i}.
$$

使用一维 Wasserstein 距离：

$$
D_\Sigma(W_A,W_B)
=
W_1(\mu_{W_A},\mu_{W_B}).
$$

该项只比较相对秩上的谱能量形状，不比较原始奇异值尺度。

### 9.2 完整层对代价

对于模块 \(c\) 的层对 \((\ell,m)\)：

$$
\boxed{
\begin{aligned}
C^c_{\ell m}
={}&
\lambda_{\mathrm{OP}}
\left(
D^c_{\mathrm{OP},U}[\ell,m]
+D^c_{\mathrm{OP},V}[\ell,m]
\right)\\
&+
\lambda_{\mathrm{ch}}
\left(
d^c_{\mathrm{ch},U}[\ell,m]^2
+d^c_{\mathrm{ch},V}[\ell,m]^2
\right)\\
&+
\lambda_\Sigma
D_\Sigma
(W^c_{A,\ell},W^c_{B,m})\\
&+
\lambda_{\mathrm{rank}}
D^c_{\mathrm{rank}}[\ell,m].
\end{aligned}
}
$$

建议将每个组成项除以其在当前模块全部有效层对上的中位数：

$$
\mathcal N(D_{\ell m})
=
\frac{D_{\ell m}}
{\operatorname{median}_{i,j}D_{ij}+\epsilon}.
$$

归一化后再使用 \(\lambda\) 加权，避免某个项因量纲过大支配代价。

### 9.3 可选相对深度项

相对深度只能作为消融或弱先验：

$$
D_{\mathrm{depth}}[\ell,m]
=
\left|
\frac{\ell+\frac12}{L}
-
\frac{m+\frac12}{M}
\right|^2.
$$

主方法默认 \(\lambda_{\mathrm{depth}}=0\)，并单独报告加入该项后的结果。

---

## 10. 外层层 OT

对每个模块 \(c\) 独立构造：

$$
C^c\in\mathbb R^{L\times M}.
$$

均匀层质量：

$$
a^L=\frac1L\mathbf1_L,
\qquad
b^M=\frac1M\mathbf1_M.
$$

求解：

$$
\boxed{
P^c
=
\arg\min_{P\in\mathcal U(a^L,b^M)}
\langle C^c,P\rangle
}
$$

主实现使用 HiGHS 线性规划求精确基本可行解，不加入外层熵正则。

得到：

$$
P^c\mathbf1_M=\frac1L\mathbf1_L,
\qquad
(P^c)^\top\mathbf1_L=\frac1M\mathbf1_M.
$$

用于融合前必须按目标层条件化：

$$
\boxed{
\bar P^c_{\ell m}
=
\frac{P^c_{\ell m}}
{\sum_jP^c_{\ell j}}
=LP^c_{\ell m}.
}
$$

因此：

$$
\sum_m\bar P^c_{\ell m}=1.
$$

### 10.1 稀疏支持集

精确线性 OT 的基本可行解天然稀疏，非零元素数量至多为：

$$
L+M-1.
$$

主实现直接使用该非零支持集进行阶段三计算，不再逐行 top-\(s\)，避免破坏严格列边缘。

---

## 11. 阶段四：跨维权重传输

对于保留的层对 \((\ell,m,c)\)，复用内层 OT 得到的重心映射：

$$
T^c_{U,\ell m}
\in\mathbb R^{a_o\times b_o},
\qquad
T^c_{V,\ell m}
\in\mathbb R^{a_i\times b_i}.
$$

将源权重传输到目标形状：

$$
\boxed{
\widetilde W^c_{B\rightarrow A,\ell m}
=
T^c_{U,\ell m}
W^c_{B,m}
(T^c_{V,\ell m})^\top.
}
$$

形状检查：

$$
(a_o\times b_o)
(b_o\times b_i)
(b_i\times a_i)
=
(a_o\times a_i).
$$

该乘法必须分块执行，禁止同时在 GPU 上保留所有层对的完整 \(\Pi,T,\widetilde W\)。

---

## 12. 公共子空间核心融合

### 12.1 投影到目标公共子空间

目标矩阵写为：

$$
W^c_{A,\ell}
=
R^c_{A,\ell}
+U^c_{A,\ell}
\Sigma^c_{A,\ell}
(V^c_{A,\ell})^\top,
$$

其中：

$$
R^c_{A,\ell}
=
W^c_{A,\ell}
-U^c_{A,\ell}\Sigma^c_{A,\ell}(V^c_{A,\ell})^\top
$$

是未参与融合的目标谱尾部。

将传输后的源矩阵投影到目标 top-\(k\) 左右子空间：

$$
\boxed{
K^c_{\ell m}
=
(U^c_{A,\ell})^\top
\widetilde W^c_{B\rightarrow A,\ell m}
V^c_{A,\ell}
\in\mathbb R^{k\times k}.
}
$$

\(K^c_{\ell m}\) 一般是稠密矩阵。不得强行只保留其对角线，否则会再次退化为仅传输奇异值。

### 12.2 核心尺度校准

由于重心映射和模型规模会改变范数，定义：

$$
\gamma^c_{\ell m}
=
\frac{
\|\Sigma^c_{A,\ell}\|_F
}{
\|K^c_{\ell m}\|_F+\epsilon
}.
$$

对缩放因子裁剪：

$$
\widehat\gamma^c_{\ell m}
=
\operatorname{clip}
(\gamma^c_{\ell m},\gamma_{\min},\gamma_{\max}).
$$

得到：

$$
\widehat K^c_{\ell m}
=
\widehat\gamma^c_{\ell m}K^c_{\ell m}.
$$

若 \(\|K_{\ell m}\|_F\) 小于阈值，不应使用巨大缩放强行放大，而应将该层对标记为无效，并在剩余源层上重新归一化 \(\bar P\)。

### 12.3 聚合多个源层

$$
\boxed{
\bar K^c_\ell
=
\sum_{m=1}^{M}
\bar P^c_{\ell m}
\widehat K^c_{\ell m}.
}
$$

### 12.4 最终融合

令 \(\beta_c\in[0,1]\) 为源融合强度：

$$
K^{c,\mathrm{fused}}_\ell
=
(1-\beta_c)\Sigma^c_{A,\ell}
+\beta_c\bar K^c_\ell.
$$

保留目标谱尾部：

$$
\boxed{
W^{c,\mathrm{fused}}_{A,\ell}
=
R^c_{A,\ell}
+U^c_{A,\ell}
K^{c,\mathrm{fused}}_\ell
(V^c_{A,\ell})^\top.
}
$$

实现时不需要显式构造 \(R_A\)，使用等价残差形式：

$$
\boxed{
W^{c,\mathrm{fused}}_{A,\ell}
=
W^c_{A,\ell}
+\beta_c
U^c_{A,\ell}
(\bar K^c_\ell-\Sigma^c_{A,\ell})
(V^c_{A,\ell})^\top.
}
$$

这保证未进入 top-\(k\) 的目标谱尾部完全不变。

---

## 13. 无数据安全约束

不进行 adaptation 时，必须限制更新规模。

定义：

$$
\Delta W^c_\ell
=
U^c_{A,\ell}
(\bar K^c_\ell-\Sigma^c_{A,\ell})
(V^c_{A,\ell})^\top.
$$

全局信任域系数：

$$
\rho^c_\ell
=
\min\left(
1,
\frac{
\delta_{\max}\|W^c_{A,\ell}\|_F
}{
\beta_c\|\Delta W^c_\ell\|_F+\epsilon
}
\right).
$$

最终写回：

$$
\boxed{
W^{c,\mathrm{fused}}_{A,\ell}
=
W^c_{A,\ell}
+\beta_c\rho^c_\ell\Delta W^c_\ell.
}
$$

必须在报告中记录：

- \(\|\Delta W\|_F/\|W_A\|_F\)；
- \(\rho_\ell^c\)；
- 每个层对的 \(\gamma_{\ell m}^c\)；
- 每层使用的源层及 \(\bar P_{\ell m}^c\)；
- 被判定为无效或秩亏的层对。

---

## 14. LoRA 等价实现

主方法输出稠密权重，但更新秩不超过 \(k\)。定义公共核心增量：

$$
D^c_\ell
=
\beta_c\rho^c_\ell
(\bar K^c_\ell-\Sigma^c_{A,\ell}).
$$

对 \(D^c_\ell\) 做 SVD：

$$
D^c_\ell
=
L_D\Sigma_DR_D^\top.
$$

则折叠进目标权重的实际更新为：

$$
\Delta W^{c,\mathrm{fold}}_\ell
=
(U_A L_D)
\Sigma_D
(V_A R_D)^\top.
$$

定义 LoRA 因子：

$$
B_{\mathrm{LoRA}}
=
U_A L_D\Sigma_D^{1/2},
$$

$$
A_{\mathrm{LoRA}}
=
\Sigma_D^{1/2}(V_A R_D)^\top.
$$

因此：

$$
B_{\mathrm{LoRA}}A_{\mathrm{LoRA}}
=
\Delta W^{c,\mathrm{fold}}_\ell.
$$

若只允许 LoRA rank \(r_{\mathrm{LoRA}}<k\)，截断 \(D\) 的 SVD，得到 Frobenius 意义下的最佳 rank-\(r_{\mathrm{LoRA}}\) 近似。

---

## 15. 模块结构协调

主方法为七个模块分别计算：

$$
P^Q,P^K,P^V,P^O,
P^{\mathrm{gate}},P^{\mathrm{up}},P^{\mathrm{down}}.
$$

作为可选消融，可以在外层 OT 后加入弱一致性正则或后处理：

$$
\begin{aligned}
\Omega_{\mathrm{struct}}
={}&
\lambda_{QK}\|\bar P^Q-\bar P^K\|_F^2
+\lambda_{VO}\|\bar P^V-\bar P^O\|_F^2\\
&+\lambda_{GU}\|\bar P^{\mathrm{gate}}-\bar P^{\mathrm{up}}\|_F^2\\
&+\lambda_{\mathrm{MLP}}
\left\|
\bar P^{\mathrm{down}}
-\frac{\bar P^{\mathrm{gate}}+\bar P^{\mathrm{up}}}{2}
\right\|_F^2.
\end{aligned}
$$

第一版实现应先保持各模块完全独立，结构协调作为后续消融，避免同时引入过多变量。

---

## 16. 计算复杂度与可扩展实现

### 16.1 主要瓶颈

单个输出侧 coupling 需要：

$$
O(a_ob_ok)
$$

时间构造低秩欧氏代价，并需要：

$$
O(a_ob_o)
$$

空间存储 coupling。输入侧同理。

若对所有：

$$
L\times M\times|\mathcal C|
$$

层对运行完整 OT–Procrustes，1B/8B 模型上的内存和时间通常不可接受。

### 16.2 正确性优先版本

第一版应先在小模型或少数模块上实现全层对计算，用于验证：

- 数学公式；
- coupling 边缘；
- 置换不变性；
- 跨维形状；
- 融合信任域。

### 16.3 可扩展粗到细版本

大模型使用以下流程：

1. 对全部 \(L\times M\) 层对只计算廉价谱形状代价 \(D_\Sigma\)；
2. 每个目标层保留谱代价最小的 top-\(s\) 源层；
3. 同时保证每个源层至少被一个候选边覆盖；
4. 只对候选层对运行完整 OT–Procrustes；
5. 对候选图求 masked OT 或改用逐行归一化层路由；
6. 只保存最终被 \(\bar P\) 使用的 coupling。

其他扩展方案：

- 对谱点云做 k-means/landmark 压缩后求 OT；
- 按 attention head 分块匹配，避免无约束跨头 coupling；
- 使用低秩或在线 Sinkhorn；
- coupling 以 FP16/BF16 存储，但对数缩放变量和目标值使用 FP32；
- 传输三矩阵乘法使用源输出维和目标列维的双重分块。

### 16.4 不建议的近似

- 直接截断或补零 \(U_B,V_B\) 的神经元行；
- 只比较 \(|U|\) 或 \(|V|\) 后进行最近邻；
- 将不同宽度的投影矩阵直接插值到相同尺寸；
- 忽略输入侧或输出侧其中之一；
- Procrustes 后仍只融合对角 \(\Sigma_B\)。

---

## 17. 推荐代码结构

建议新增独立包，避免继续扩展现有单文件脚本：

```text
core/df_ot_procrustes/
├── __init__.py
├── config.py              # 全部配置 dataclass
├── module_registry.py     # Llama/Qwen 模块发现与形状检查
├── svd_cache.py           # randomized SVD 与磁盘缓存
├── point_cloud.py         # UΣ/VΣ 点云、中心化、质量
├── sinkhorn.py            # log-domain Sinkhorn
├── ot_procrustes.py       # 内层交替优化
├── barycentric_map.py     # coupling -> T/Q
├── grassmann.py           # QR、主角度、chordal/geodesic 距离
├── spectral_cost.py       # 宽度无关谱距离
├── layer_cost.py          # C^c[L,M] 构造与归一化
├── outer_ot.py            # 外层 OT、条件化、稀疏化
├── transport.py           # T_U W_B T_V^T 分块计算
├── fusion.py              # 核心投影、聚合、信任域、写回
├── lora_export.py         # 核心增量转 LoRA
├── reports.py             # JSON/CSV 诊断报告
└── pipeline.py            # 总流程编排
```

### 17.1 核心接口

```python
def truncated_svd(weight, cfg) -> SVDRecord: ...

def build_spectral_clouds(
    svd_a: SVDRecord,
    svd_b: SVDRecord,
    k: int,
    gamma: float,
) -> PairPointClouds: ...

def solve_ot_procrustes(
    x: Tensor,          # [n_target, k]
    y: Tensor,          # [n_source, k]
    mass_x: Tensor,     # [n_target]
    mass_y: Tensor,     # [n_source]
    cfg,
) -> OTProcrustesResult: ...

def coupling_to_map(
    coupling: Tensor,   # [n_target, n_source]
    target_mass: Tensor,
    mode: str,
) -> Tensor: ...

def mapped_subspace_cost(
    basis_target: Tensor,
    basis_source: Tensor,
    mapping: Tensor,
    cfg,
) -> GrassmannResult: ...

def solve_layer_transport(
    cost: Tensor,       # [L, M]
    cfg,
) -> LayerTransportResult: ...

def transport_weight(
    map_out: Tensor,
    weight_source: Tensor,
    map_in: Tensor,
    cfg,
) -> Tensor: ...

def fuse_pair_cores(
    weight_target: Tensor,
    svd_target: SVDRecord,
    transported_sources: list[Tensor],
    layer_weights: Tensor,
    beta: float,
    cfg,
) -> FusionResult: ...
```

---

## 18. 配置建议

以下是实现起点，不是经过任务验证的最优超参数。

```yaml
svd:
  rank_max: 128
  rank_ratio: 0.0625
  oversample: 16
  power_iters: 2
  eps: 1.0e-8

point_cloud:
  sigma_power: 1.0
  mass: uniform
  center: true
  normalize_variance: true

inner_ot:
  entropy: 0.05
  sinkhorn_iters: 200
  sinkhorn_tol: 1.0e-6
  alternating_iters: 8
  alternating_tol: 1.0e-4
  restarts: 2
  log_domain: true

layer_cost:
  lambda_op: 1.0
  lambda_chordal: 0.25
  lambda_spectrum: 0.5
  lambda_rank: 0.5
  lambda_depth: 0.0
  normalize_each_term: true

outer_ot:
  solver: balanced_exact
  marginal_tolerance: 1.0e-7
  support_tolerance: 1.0e-9

mapping:
  mode: barycentric
  polar_eps: 1.0e-5

fusion:
  beta: 0.05
  core_scale: match_fro
  gamma_min: 0.25
  gamma_max: 4.0
  delta_max: 0.05
  preserve_target_tail: true

runtime:
  compute_dtype: float32
  storage_dtype: bfloat16
  device: cuda
  save_couplings: selected_only
```

严格无数据主实验应尽量固定同一套超参数跨任务使用，避免利用测试集选择 \(\beta\)、rank 或 OT 正则。

---

## 19. 输出文件与可复现信息

建议输出：

```text
df_ot_results/
├── config.yaml
├── model_meta.json
├── svd_meta.json
├── layer_cost_Q.pt
├── layer_cost_K.pt
├── ...
├── layer_P_Q.pt
├── layer_Pbar_Q.pt
├── ...
├── selected_pairs.json
├── pair_metrics.jsonl
├── fusion_report.json
├── lora/                  # 可选
└── fused_model/           # 稠密主输出
```

每个层对记录至少包含：

```json
{
  "component": "Q",
  "target_layer": 0,
  "source_layer": 1,
  "target_shape": [0, 0],
  "source_shape": [0, 0],
  "rank": 0,
  "op_cost_out": 0.0,
  "op_cost_in": 0.0,
  "chordal_out": 0.0,
  "chordal_in": 0.0,
  "spectral_cost": 0.0,
  "coupling_entropy_out": 0.0,
  "coupling_entropy_in": 0.0,
  "map_rank_out": 0,
  "map_rank_in": 0,
  "core_scale": 0.0,
  "valid": true,
  "skip_reason": null
}
```

---

## 20. 测试规范

### 20.1 单元测试

1. **SVD 重构**：截断重构误差与缓存 energy 一致；
2. **点云归一化**：加权均值接近 0，总方差接近 1；
3. **Sinkhorn 边缘**：\(\Pi\mathbf1=a\)、\(\Pi^\top\mathbf1=b\)；
4. **Procrustes 恢复**：对已知正交旋转生成的点云恢复低残差；
5. **置换不变性**：独立置换源神经元后，最优层对代价基本不变；
6. **符号不变性**：SVD 列符号翻转后代价基本不变；
7. **跨维形状**：\(T_UW_BT_V^\top\) 的形状严格等于目标形状；
8. **主角度范围**：所有余弦在数值裁剪后位于 \([0,1]\)；
9. **外层 OT 边缘**：\(P\) 满足均匀行列边缘；
10. **条件化**：\(\bar P\) 每行和为 1；
11. **\(\beta=0\)**：融合权重与目标权重逐元素相同；
12. **相同权重**：源目标相同时更新接近 0；
13. **谱尾保持**：目标 top-\(k\) 正交补上的投影更新接近 0；
14. **信任域**：实际更新范数不超过设定上限；
15. **LoRA 等价**：完整 rank 下 LoRA 更新与稠密更新一致。

### 20.2 合成恢复测试

构造低秩潜在算子 \(K\)，使用不同的矩形嵌入和神经元置换生成：

$$
W_A=A_UKB_V^\top,
\qquad
W_B=B_UKB_I^\top.
$$

目标和源宽度不同，但共享已知潜在核心。测试：

- OT–Procrustes 是否得到低于随机 coupling 的几何残差；
- 传输后 \(T_UW_BT_V^\top\) 是否接近 \(W_A\)；
- 公共核心 \(K_{\ell m}\) 是否恢复已知潜在结构。

### 20.3 小模型集成测试

使用随机初始化的两组小型 decoder block，确保：

- 七个模块均能被发现；
- GQA 导致的 Q/K/V 形状差异不会导致形状错误；
- 完整流水线不调用 tokenizer、dataset 或 forward；
- 保存并重新加载融合模型后参数一致。

---

## 21. 消融与对照实验

至少报告：

1. 目标模型，不融合；
2. 纯相对深度层映射；
3. 只用奇异值谱代价的外层 OT；
4. OT–Procrustes 残差，不使用 Grassmann 项；
5. OT–Procrustes + mapped Grassmann；
6. 只用输出侧 / 只用输入侧 / 左右两侧；
7. barycentric / polar / whiten 映射；
8. 对角谱融合 / 稠密公共核心融合；
9. 不同 top-\(k\)；
10. 不同 coupling entropy；
11. 不同 \(\beta\) 与信任域；
12. 稠密融合与 LoRA 近似；
13. 正确源模型、随机源模型、源层打乱；
14. 无数据主方法与融合后 residual-frozen SFT。

“随机源”和“源层打乱”是必要的识别性对照。如果正确源模型并不优于这些对照，则不能声称方法捕获了源模型特定结构。

---

## 22. 可选融合后 SFT

该阶段不是主方法。若进行对比，参数化为：

$$
W^c_\ell
=
W^{c,\mathrm{base}}_{A,\ell}
+\Delta W^{c,\mathrm{DFOT}}_\ell,
$$

固定传输残差：

$$
\frac{\partial\mathcal L}
{\partial\Delta W^{c,\mathrm{DFOT}}_\ell}=0,
$$

只更新目标基础参数：

$$
\frac{\partial\mathcal L}
{\partial W^{c,\mathrm{base}}_{A,\ell}}
\neq0.
$$

训练完成后将残差折叠回基础权重。实验名称应明确区分：

- `DF-OT-Procrustes`：完全无数据、无训练；
- `DF-OT-Procrustes + SFT`：无数据融合后使用目标领域数据适配。

---

## 23. 实现验收条件

第一版代码只有同时满足以下条件才视为完成：

1. 1B/8B 异宽矩阵不经过神经元行截断或补零；
2. 内层 coupling 和外层 \(P\) 的边缘误差均通过测试；
3. 每个源权重经 \(T_UW_BT_V^\top\) 后严格得到目标形状；
4. mapped Grassmann 距离只在共同目标环境空间内计算；
5. 公共子空间融合使用稠密 \(K_{\ell m}\)，而非只融合奇异值对角线；
6. 未参与融合的目标谱尾部保持不变；
7. 更新满足全局信任域；
8. 整个主流程不加载数据、不调用 tokenizer、不运行 forward；
9. 稠密输出和完整 rank LoRA 输出数值等价；
10. 所有随机过程、SVD、重启和 OT 配置可复现。

---

## 24. 方法边界总结

DF-OT-Procrustes 通过以下链条解决异构维度问题：

$$
\boxed{
\begin{aligned}
&U_A\in\mathbb R^{a\times k},
\quad
U_B\in\mathbb R^{b\times k},
\quad a\neq b\\
&\Downarrow\\
&\text{将行视为共同 }\mathbb R^k\text{ 中的谱点云}\\
&\Downarrow\\
&\text{OT–Procrustes 求 }\Pi,R\\
&\Downarrow\\
&T=\operatorname{diag}(\mathbf a)^{-1}\Pi\\
&\Downarrow\\
&TU_B\in\mathbb R^{a\times k}\\
&\Downarrow\\
&U_A^\top\operatorname{qf}(TU_B)\text{ 可定义}\\
&\Downarrow\\
&\text{mapped Grassmann 层代价与跨维权重传输}.
\end{aligned}
}
$$

该方法在数学上避免了跨宽度 SVD 子空间直接相乘的问题，也避免了补零和神经元截断的任意坐标假设。但它仍然是基于权重几何的无数据估计，是否对应可迁移能力必须通过严格的源模型特异性对照和下游实验验证。

