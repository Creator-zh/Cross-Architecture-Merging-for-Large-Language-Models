# DF-OT-Procrustes：无数据跨架构融合实验与代码实现方案

## 0. 文档状态

本文档冻结当前讨论形成的主方法，用于后续代码实现、单元测试以及在 **Transport and Merge: Cross-Architecture Merging for Large Language Models** 实验上的受控对比。

主方法暂称 **DF-OT-Procrustes**，也可在实验表中缩写为 **DFOP**。

本文档中的“无数据”严格指：

- 不加载训练集、校准集、提示文本或合成文本；
- 不调用 tokenizer；
- 不执行目标模型或源模型的 forward pass；
- 不读取 activation、gradient、loss 或数据统计；
- 对齐与融合阶段只读取模型配置和权重；
- 融合完成后的 SFT 仅作为独立对比，不属于无数据主方法。

论文基线参考：

- [Transport and Merge 论文](https://arxiv.org/abs/2602.05495)
- [仓库复现说明](REPRODUCE.md)
- [仓库模型与任务配置](MODELS.md)

---

## 1. 研究目标与核心假设

### 1.1 问题定义

给定目标模型 \(\mathcal M_A\) 和源模型 \(\mathcal M_B\)：

$$
\mathcal M_A=\{A_1,\ldots,A_L\},
\qquad
\mathcal M_B=\{B_1,\ldots,B_M\},
$$

其中目标模型通常是较小的领域或低资源语言模型，源模型通常是较大的通用模型。允许：

$$
L\neq M,
$$

$$
d_A\neq d_B.
$$

目标是在不使用任何数据的条件下，将源模型参数中的可迁移谱结构注入目标模型，并保持输出模型的架构、参数形状、tokenizer 和推理成本与目标模型一致。

### 1.2 七种独立线性层类型

定义模块集合：

$$
\mathcal C
=
\{Q,K,V,O,\mathrm{gate},\mathrm{up},\mathrm{down}\}.
$$

对每个 \(c\in\mathcal C\) 单独计算层代价和层路由。只允许同名比较：

$$
W^c_{A,\ell}
\longleftrightarrow
W^c_{B,m}.
$$

禁止 Q–K、V–O、gate–up 等异类型比较。最终输出七个独立路由矩阵，而不是 Attention 或 FFN 共享路由。

### 1.3 核心假设

主方法依赖以下可检验假设：

1. 同族或结构相近 Transformer 的同名线性层具有可比较的低秩谱几何；
2. 奇异向量的行可以视为神经元在 top-\(k\) 谱模式中的载荷点；
3. 不同宽度模型的谱载荷点云可通过矩形 OT coupling 建立软对应；
4. SVD 符号与谱坐标旋转可通过正交 Procrustes 消除；
5. 较小的最佳对齐残差意味着该源层更适合向目标层传输；
6. 在目标 top-\(k\) 子空间中限制更新并保留目标谱尾，能降低无数据融合的破坏性。

该方法不声称仅凭权重即可证明两个层具有相同语义。OT–Procrustes 值是权重谱几何的不相似度，而不是严格的功能语义度量。

---

## 2. 与 Transport and Merge 的受控关系

Transport and Merge 使用少量输入提取 pre/post activation，以相关距离构造 feature cost，先求 feature-level OT，再求 layer-level balanced OT，并利用 transport plan 进行权重融合；论文还包含激活 top-neuron 选择和可选 residual-frozen adaptation。

DFOP 保留其总体结构：

$$
\text{层对代价}
\rightarrow
P_{\mathrm{eff}}
\rightarrow
\text{神经元传输}
\rightarrow
\text{权重融合},
$$

但替换以下部分：

| 环节 | Transport and Merge | DFOP |
|---|---|---|
| 对齐输入 | 约 2000 条刺激数据的 activation | 权重 SVD |
| forward pass | 需要 | 不允许 |
| 神经元特征 | pre/post activation channel | 归一化谱载荷点云 |
| 层代价 | activation feature OT objective | 直接 OT–Procrustes 几何残差 |
| 层路由 | 均匀行列边际的 balanced OT | 只固定目标行边际 |
| 模块粒度 | pre/post 与投影模块 | 七种线性层独立路由 |
| 神经元选择 | activation top-neuron | 主方法不使用数据掩码 |
| 权重输出 | 传输残差 | 目标 top-\(k\) 稠密核心更新 |
| 后训练 | 可选 residual-frozen adaptation | 仅作为独立对比 |

为保证公平，论文复现实验中的模型对、评测集、随机种子、生成设置和可选 SFT 预算应尽量保持不变。主结果必须单独报告“无 adaptation”的结果。

---

## 3. 记号与形状

对模块 \(c\)、目标层 \(\ell\) 和源层 \(m\)：

$$
W^c_{A,\ell}
\in
\mathbb R^{d^c_{A,o}\times d^c_{A,i}},
$$

$$
W^c_{B,m}
\in
\mathbb R^{d^c_{B,o}\times d^c_{B,i}}.
$$

主要张量如下。

| 张量 | 形状 | 含义 |
|---|---:|---|
| \(U_A\) | \(d_{A,o}\times k_c\) | 目标左奇异向量 |
| \(V_A\) | \(d_{A,i}\times k_c\) | 目标右奇异向量 |
| \(U_B\) | \(d_{B,o}\times k_c\) | 源左奇异向量 |
| \(V_B\) | \(d_{B,i}\times k_c\) | 源右奇异向量 |
| \(X,Y\) | \(n_A\times k_c,n_B\times k_c\) | 阶段一谱点云 |
| \(\Pi\) | \(n_A\times n_B\) | 神经元 OT coupling |
| \(R\) | \(k_c\times k_c\) | 谱坐标正交旋转 |
| \(C^c\) | \(L\times M\) | 模块 \(c\) 的层代价 |
| \(P^c_{\mathrm{eff}}\) | \(L\times M\) | 模块 \(c\) 的行归一化路由 |
| \(T_U\) | \(d_{A,o}\times d_{B,o}\) | 输出侧重心映射 |
| \(T_V\) | \(d_{A,i}\times d_{B,i}\) | 输入侧重心映射 |
| \(K_{\ell m}\) | \(k_c\times k_c\) | 传输后的源核心 |

---

## 4. 阶段零：模块发现与 SVD 缓存

### 4.1 模块注册

LLaMA/Qwen 类模型的默认映射为：

| 逻辑名 | 常见参数路径 |
|---|---|
| Q | `self_attn.q_proj.weight` |
| K | `self_attn.k_proj.weight` |
| V | `self_attn.v_proj.weight` |
| O | `self_attn.o_proj.weight` 或 `out_proj.weight` |
| gate | `mlp.gate_proj.weight` 或 `w1.weight` |
| up | `mlp.up_proj.weight` 或 `w3.weight` |
| down | `mlp.down_proj.weight` 或 `w2.weight` |

必须先根据模型类和实际张量形状验证别名，不能只依赖字符串猜测。embedding、LM head、norm 和 bias 默认保持目标模型不变。

### 4.2 固定模块秩

对每类模块指定固定秩 \(k_c\)：

$$
k_c
\leq
\min_{
X\in\{A,B\},r
}
\min
\left(
d^c_{X,o,r},
d^c_{X,i,r}
\right).
$$

同一模块的 \(k_c\) 不能随层对 \((\ell,m)\) 改变，否则不同层对的代价定义不一致。

### 4.3 截断 SVD

计算：

$$
W^c_{A,\ell}
\approx
U^c_{A,\ell}
\Sigma^c_{A,\ell}
(V^c_{A,\ell})^\top,
$$

$$
W^c_{B,m}
\approx
U^c_{B,m}
\Sigma^c_{B,m}
(V^c_{B,m})^\top.
$$

随机 SVD 必须记录随机种子、过采样量和幂迭代次数。SVD 与后续 OT 计算至少使用 FP32。

### 4.4 缓存键

缓存键至少包含：

```text
model_id_or_weight_hash
model_revision
module_type
layer_index
weight_shape
rank
svd_algorithm
oversample
power_iterations
seed
```

---

## 5. 阶段一：无数据直接层对代价

### 5.1 residual-stream 一侧

定义阶段一使用的奇异子空间基：

$$
H^c_{X,r}
=
\begin{cases}
V^c_{X,r},
&c\in\{Q,K,V,\mathrm{gate},\mathrm{up}\},
\\
U^c_{X,r},
&c\in\{O,\mathrm{down}\}.
\end{cases}
$$

选择规则来自线性层与 residual stream 的连接方向：Q/K/V/gate/up 的输入属于 residual stream，O/down 的输出属于 residual stream。

阶段一不使用每层私有的注意力头或 FFN 中间神经元一侧，以避免层代价依赖任意局部神经元编号。

### 5.2 谱强度归一化

主方法取：

$$
S^c_{X,r}
=
\frac{
\Sigma^c_{X,r}
}{
\|\Sigma^c_{X,r}\|_F+\epsilon
}.
$$

可选消融使用：

$$
S^c_{X,r}(\gamma)
=
\frac{
(\Sigma^c_{X,r})^\gamma
}{
\|(\Sigma^c_{X,r})^\gamma\|_F+\epsilon
},
$$

其中 \(\gamma=0\) 表示只使用子空间基，\(\gamma=1\) 为主设置。

### 5.3 宽度归一化谱点云

设 \(H_A\) 和 \(H_B\) 的行数分别为 \(n_A^c,n_B^c\)。构造：

$$
X^c_\ell
=
\sqrt{n_A^c}
H^c_{A,\ell}
S^c_{A,\ell},
$$

$$
Y^c_m
=
\sqrt{n_B^c}
H^c_{B,m}
S^c_{B,m}.
$$

由于 \(H^\top H=I\) 且 \(\|S\|_F=1\)，均匀质量下有：

$$
\frac1{n_A^c}
\sum_i\|x^c_{\ell,i}\|_2^2
=1,
$$

$$
\frac1{n_B^c}
\sum_j\|y^c_{m,j}\|_2^2
=1.
$$

因此 \(\sqrt n\) 用于消除正交奇异向量行范数随宽度产生的 \(1/\sqrt n\) 缩放。

主方法不再额外中心化点云。中心化会改变由 \(WW^\top\) 或 \(W^\top W\) 诱导的原始投影几何，只作为消融。

### 5.4 运输多面体

定义均匀质量：

$$
a_i^c=\frac1{n_A^c},
\qquad
b_j^c=\frac1{n_B^c}.
$$

$$
\mathcal U(a^c,b^c)
=
\left\{
\Pi\in\mathbb R_+^{n_A^c\times n_B^c}:
\Pi\mathbf1_{n_B^c}=a^c,
\quad
\Pi^\top\mathbf1_{n_A^c}=b^c
\right\}.
$$

### 5.5 唯一代价原语

对目标层 \(\ell\) 与源层 \(m\) 的同名模块 \(c\)，定义：

$$
\boxed{
C^c_{\ell m}
=
\frac12
\min_{
\Pi\in\mathcal U(a^c,b^c),
\;R\in\mathcal O(k_c)
}
\sum_{i,j}
\Pi_{ij}
\|x^c_{\ell,i}-Ry^c_{m,j}\|_2^2
}
$$

其中：

$$
\mathcal O(k_c)
=
\{R\in\mathbb R^{k_c\times k_c}:R^\top R=I\}.
$$

该代价表示在最佳神经元软匹配和最佳谱坐标正交旋转后仍无法解释的归一化谱几何残差。

定义：

$$
M(\Pi)=X^\top\Pi Y.
$$

则等价地：

$$
\boxed{
C^c_{\ell m}
=
1-
\max_{\Pi\in\mathcal U(a^c,b^c)}
\|X^\top\Pi Y\|_*
}
$$

理论上：

$$
0\leq C^c_{\ell m}\leq1.
$$

主层代价不包含额外的谱距离、mapped Grassmann 距离、秩惩罚或相对深度惩罚。

### 5.6 交替求解器

固定旋转时：

$$
G_{ij}(R)
=
\frac12\|x_i-Ry_j\|_2^2.
$$

用 log-domain Sinkhorn 求：

$$
\Pi
=
\arg\min_{\Pi\in\mathcal U(a,b)}
\langle G(R),\Pi\rangle
-\varepsilon H(\Pi).
$$

固定 coupling 时：

$$
M=X^\top\Pi Y.
$$

若：

$$
M=L_M\Gamma_MH_M^\top,
$$

则：

$$
R=L_MH_M^\top.
$$

最终层代价使用：

$$
C^c_{\ell m}
=
\langle G(R^*),\Pi^*\rangle,
$$

不把熵项写入 \(C^c_{\ell m}\)。

建议支持单位阵、符号规范化和随机正交初始化，并保存最终几何残差最小的重启。该问题非联合凸，报告中必须记录重启间方差。

---

## 6. 阶段二：七个独立层路由

对每个 \(c\in\mathcal C\) 得到：

$$
C^c\in\mathbb R^{L\times M}.
$$

主方法只固定目标层行边际：

$$
P^{c,*}
=
\arg\min_{
P\geq0,
\;P\mathbf1_M=\frac1L\mathbf1_L
}
\langle C^c,P\rangle
-\tau H(P).
$$

不强制源层列边际均匀。按目标层条件化：

$$
P^c_{\mathrm{eff},\ell m}
=
L P^{c,*}_{\ell m}.
$$

等价闭式为：

$$
\boxed{
P^c_{\mathrm{eff},\ell m}
=
\frac{
\exp(-C^c_{\ell m}/\tau)
}{
\sum_{r=1}^{M}\exp(-C^c_{\ell r}/\tau)
}
}
$$

得到七个矩阵：

$$
P^Q_{\mathrm{eff}},
P^K_{\mathrm{eff}},
P^V_{\mathrm{eff}},
P^O_{\mathrm{eff}},
P^{\mathrm{gate}}_{\mathrm{eff}},
P^{\mathrm{up}}_{\mathrm{eff}},
P^{\mathrm{down}}_{\mathrm{eff}}.
$$

理论主方法允许所有源层参与。为控制第二阶段成本，可以保留每行 top-\(s\) 并重新归一化：

$$
\widehat P^c_{\ell m}
=
\frac{
P^c_{\mathrm{eff},\ell m}
\mathbf1[m\in\mathcal N_c(\ell)]
}{
\sum_{r\in\mathcal N_c(\ell)}
P^c_{\mathrm{eff},\ell r}
}.
$$

top-\(s\) 是路由稀疏化，不改变最终权重更新是稠密矩阵这一事实。

---

## 7. 阶段三：候选层对的双侧神经元传输

阶段一只对 residual-stream 一侧求 coupling。完整权重传输需要输出侧和输入侧两个 coupling。

### 7.1 双侧点云

输出侧：

$$
X_U
=
\sqrt{d_{A,o}}U_A S_A,
\qquad
Y_U
=
\sqrt{d_{B,o}}U_B S_B.
$$

输入侧：

$$
X_V
=
\sqrt{d_{A,i}}V_A S_A,
\qquad
Y_V
=
\sqrt{d_{B,i}}V_B S_B.
$$

分别求：

$$
(\Pi_U,R_U),
\qquad
(\Pi_V,R_V).
$$

复用规则：

- Q/K/V/gate/up 复用阶段一的 \(\Pi_V\)，补算 \(\Pi_U\)；
- O/down 复用阶段一的 \(\Pi_U\)，补算 \(\Pi_V\)。

### 7.2 重心映射

$$
T_U
=
\operatorname{diag}(a_U)^{-1}\Pi_U
\in
\mathbb R^{d_{A,o}\times d_{B,o}},
$$

$$
T_V
=
\operatorname{diag}(a_V)^{-1}\Pi_V
\in
\mathbb R^{d_{A,i}\times d_{B,i}}.
$$

满足：

$$
T_U\mathbf1=\mathbf1,
\qquad
T_V\mathbf1=\mathbf1.
$$

源权重的 top-\(k_c\) 部分为：

$$
W^{c,(k)}_{B,m}
=
U^c_{B,m}
\Sigma^c_{B,m}
(V^c_{B,m})^\top.
$$

传输到目标形状：

$$
\widetilde W^c_{B\rightarrow A,\ell m}
=
T^c_{U,\ell m}
W^{c,(k)}_{B,m}
(T^c_{V,\ell m})^\top.
$$

### 7.3 只计算小核心

定义：

$$
A^c_{U,\ell m}
=
(U^c_{A,\ell})^\top
T^c_{U,\ell m}
U^c_{B,m},
$$

$$
A^c_{V,\ell m}
=
(V^c_{A,\ell})^\top
T^c_{V,\ell m}
V^c_{B,m}.
$$

传输后的目标子空间核心为：

$$
\boxed{
K^c_{\ell m}
=
A^c_{U,\ell m}
\Sigma^c_{B,m}
(A^c_{V,\ell m})^\top
}
$$

其形状为：

$$
K^c_{\ell m}\in\mathbb R^{k_c\times k_c}.
$$

该公式与先构造 \(\widetilde W\) 再投影到目标 \(U_A,V_A\) 完全等价，但避免构造大矩阵。\(K_{\ell m}\) 一般为稠密矩阵，主方法不得强制只保留对角线。

---

## 8. 阶段四：尺度校准、聚合与融合

### 8.1 核心尺度校准

重心映射可能收缩范数，定义：

$$
\gamma^c_{\ell m}
=
\frac{
\|\Sigma^c_{A,\ell}\|_F
}{
\|K^c_{\ell m}\|_F+\epsilon
}.
$$

$$
\widehat\gamma^c_{\ell m}
=
\operatorname{clip}
(\gamma^c_{\ell m},\gamma_{\min},\gamma_{\max}).
$$

$$
\widehat K^c_{\ell m}
=
\widehat\gamma^c_{\ell m}K^c_{\ell m}.
$$

若 \(\|K^c_{\ell m}\|_F\) 小于相对阈值，则层对无效，并在剩余源层上重新归一化路由；禁止通过巨大 \(\gamma\) 放大数值噪声。

### 8.2 源层聚合

$$
\boxed{
\overline K^c_\ell
=
\sum_{m=1}^{M}
P^c_{\mathrm{eff},\ell m}
\widehat K^c_{\ell m}
}
$$

若启用 top-\(s\)，使用重新归一化后的 \(\widehat P^c\)。

### 8.3 目标谱尾保持

目标权重写为：

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

是保持不变的目标谱尾。

### 8.4 稠密核心融合

定义融合强度 \(\beta\in[0,1]\)：

$$
K^{c,\mathrm{fused}}_\ell
=
(1-\beta)\Sigma^c_{A,\ell}
+\beta\overline K^c_\ell.
$$

残差形式为：

$$
\Delta W^c_\ell
=
U^c_{A,\ell}
(\overline K^c_\ell-\Sigma^c_{A,\ell})
(V^c_{A,\ell})^\top.
$$

### 8.5 信任域

$$
\rho^c_\ell
=
\min
\left(
1,
\frac{
\delta_{\max}\|W^c_{A,\ell}\|_F
}{
\beta\|\Delta W^c_\ell\|_F+\epsilon
}
\right).
$$

最终写回：

$$
\boxed{
W^{c,\mathrm{fused}}_{A,\ell}
=
W^c_{A,\ell}
+\beta\rho^c_\ell\Delta W^c_\ell
}
$$

保证：

$$
\|W^{c,\mathrm{fused}}_{A,\ell}-W^c_{A,\ell}\|_F
\leq
\delta_{\max}\|W^c_{A,\ell}\|_F.
$$

### 8.6 LoRA 等价形式

由于：

$$
\operatorname{rank}(\Delta W^c_\ell)\leq k_c,
$$

完整 rank LoRA 可以精确表示同一更新。主实验输出稠密折叠模型；LoRA 仅作为等价工程变体。

---

## 9. 总算法伪代码

```text
Input:
    target model A with L blocks
    source model B with M blocks
    module set C = {Q,K,V,O,gate,up,down}
    fixed ranks k_c
    inner OT settings
    route temperature tau
    fusion beta and trust radius delta_max

Phase 0: SVD cache
for model X in {A,B}:
    for module c and layer r:
        locate W_X[r,c]
        compute top-k_c SVD
        cache U, Sigma, V and metadata

Phase 1: direct layer costs
for module c:
    for target layer l:
        for source layer m:
            choose residual-side basis H
            build X = sqrt(n_A) H_A Sigma_A / ||Sigma_A||_F
            build Y = sqrt(n_B) H_B Sigma_B / ||Sigma_B||_F
            solve OT-Procrustes(X,Y)
            C_c[l,m] = final geometric residual
            cache residual-side coupling

Phase 2: seven routes
for module c:
    P_eff_c = row_softmax(-C_c / tau)
    optionally retain and renormalize top-s source layers

Phase 3: pairwise transport cores
for module c and selected pair (l,m):
    reuse residual-side coupling
    solve missing-side OT-Procrustes
    construct T_U and T_V
    A_U = U_A^T T_U U_B
    A_V = V_A^T T_V V_B
    K[l,m,c] = A_U Sigma_B A_V^T
    calibrate K and record diagnostics

Phase 4: fuse
for module c and target layer l:
    K_bar = sum_m P_eff_c[l,m] K_hat[l,m,c]
    DeltaW = U_A (K_bar - Sigma_A) V_A^T
    rho = trust_region(DeltaW, W_A)
    W_fused = W_A + beta rho DeltaW
    write W_fused into target model

Output:
    dense fused target model
    optional exact LoRA factors
    diagnostics and reproducibility metadata
```

---

## 10. 建议代码结构

新增独立实现，避免直接把无数据逻辑混入 activation HOT 文件：

```text
core/dfop/
├── __init__.py
├── config.py
├── module_registry.py
├── svd_cache.py
├── spectral_points.py
├── sinkhorn.py
├── ot_procrustes.py
├── layer_cost.py
├── layer_route.py
├── barycentric_map.py
├── pair_core.py
├── fusion.py
├── lora_export.py
├── diagnostics.py
└── pipeline.py

scripts/
├── run_dfop_transport.py
├── run_dfop_fusion.py
├── run_dfop_all_tasks.py
└── compare_dfop_vs_hot.py
```

不应修改或覆盖原始 HOT 输出目录。建议：

```text
transport_results/
├── hot_results_.../
└── dfop_results_.../
```

### 10.1 核心接口

```python
@dataclass
class ModuleSpec:
    logical_name: str
    target_path: str
    source_path: str
    residual_side: Literal["input", "output"]
    rank: int

@dataclass
class SVDRecord:
    u: Tensor
    s: Tensor
    v: Tensor
    shape: tuple[int, int]
    rank: int
    metadata: dict

@dataclass
class OTProcrustesResult:
    coupling: Tensor
    rotation: Tensor
    geometric_cost: float
    regularized_objective: float
    marginal_error: float
    iterations: int
    restart: int

@dataclass
class LayerRouteResult:
    cost: Tensor
    route: Tensor
    selected_mask: Tensor | None
    diagnostics: dict

def build_spectral_points(
    basis: Tensor,
    singular_values: Tensor,
    sigma_power: float = 1.0,
) -> Tensor: ...

def solve_ot_procrustes(
    x: Tensor,
    y: Tensor,
    mass_x: Tensor,
    mass_y: Tensor,
    cfg,
) -> OTProcrustesResult: ...

def compute_module_layer_costs(
    svd_a: list[SVDRecord],
    svd_b: list[SVDRecord],
    spec: ModuleSpec,
    cfg,
) -> LayerRouteResult: ...

def compute_pair_core(
    target: SVDRecord,
    source: SVDRecord,
    pi_out: Tensor,
    pi_in: Tensor,
    cfg,
) -> Tensor: ...

def fuse_target_weight(
    weight_a: Tensor,
    svd_a: SVDRecord,
    aggregate_core: Tensor,
    beta: float,
    trust_ratio: float,
) -> Tensor: ...
```

---

## 11. 起始配置与搜索范围

以下为实现起点，不应被描述为已经验证的最优值。

```yaml
method: dfop
seed: 42

modules:
  enabled: [q, k, v, o, gate, up, down]
  rank_default: 128

svd:
  algorithm: randomized
  oversample: 16
  power_iterations: 2
  compute_dtype: float32
  cache_dtype: float32

spectral_points:
  sigma_power: 1.0
  width_normalization: sqrt_n
  center: false
  masses: uniform

inner_ot_procrustes:
  entropy: 0.05
  sinkhorn_iterations: 200
  sinkhorn_tolerance: 1.0e-6
  alternating_iterations: 8
  alternating_tolerance: 1.0e-4
  restarts: 2
  log_domain: true
  compute_dtype: float32

layer_route:
  mode: row_softmax
  temperature: 0.05
  top_source_layers: 2
  save_dense_route: true

core_scale:
  enabled: true
  gamma_min: 0.25
  gamma_max: 4.0
  minimum_relative_norm: 1.0e-6

fusion:
  beta: 0.05
  trust_ratio: 0.10
  dense_output: true
  preserve_target_tail: true

runtime:
  device: cuda
  pair_batch_size: 1
  save_selected_couplings_only: true
```

最低消融网格：

| 参数 | 候选值 |
|---|---|
| \(k_c\) | 32, 64, 128, 256 |
| inner entropy | 0.01, 0.03, 0.05, 0.10 |
| route temperature | 0.01, 0.03, 0.05, 0.10, 0.20 |
| top source layers | 1, 2, 4, all |
| \(\beta\) | 0.005, 0.01, 0.03, 0.05, 0.10, 0.20 |
| trust ratio | 0.02, 0.05, 0.10, disabled |
| \(\gamma\) clip | disabled, 0.25–4, 0.5–2 |

严格无数据结果不得使用目标测试集选择超参数。建议设置两个报告轨道：

1. **Universal track**：在预注册配置中固定一套跨任务参数；
2. **Matched track**：复用 Transport and Merge 仓库每个任务已公开的 fusion \(\alpha\)，不额外搜索。

---

## 12. 输出文件与诊断信息

建议输出：

```text
dfop_results/<task>/
├── config.yaml
├── run_meta.json
├── model_meta_A.json
├── model_meta_B.json
├── svd_meta.json
├── layer_cost_q.pt
├── layer_cost_k.pt
├── layer_cost_v.pt
├── layer_cost_o.pt
├── layer_cost_gate.pt
├── layer_cost_up.pt
├── layer_cost_down.pt
├── route_q.pt
├── route_k.pt
├── route_v.pt
├── route_o.pt
├── route_gate.pt
├── route_up.pt
├── route_down.pt
├── pair_diagnostics.jsonl
├── layer_diagnostics.jsonl
├── runtime.json
└── fused_model/
```

每个层对至少记录：

```json
{
  "module": "q",
  "target_layer": 0,
  "source_layer": 0,
  "rank": 128,
  "target_shape": [0, 0],
  "source_shape": [0, 0],
  "layer_cost": 0.0,
  "regularized_objective": 0.0,
  "marginal_error": 0.0,
  "alternating_iterations": 0,
  "restart": 0,
  "coupling_entropy": 0.0,
  "core_norm": 0.0,
  "core_scale": 0.0,
  "route_weight": 0.0,
  "valid": true,
  "skip_reason": null
}
```

每个目标层至少记录：

- 路由熵；
- 有效源层数 \(\exp(H(P_{\ell,:}))\)；
- top-1/top-2 源层及质量；
- \(\|\Delta W\|_F/\|W_A\|_F\)；
- 信任域系数 \(\rho\)；
- 被丢弃的层对和原因。

---

## 13. 单元测试与数学验收

### 13.1 SVD 与点云

1. 截断 SVD 形状正确；
2. \(U^\top U\approx I\)、\(V^\top V\approx I\)；
3. 点云满足均匀质量平均能量为 1；
4. 权重整体缩放后，归一化层代价基本不变；
5. SVD 列符号翻转后层代价基本不变。

### 13.2 OT–Procrustes

1. \(\Pi\mathbf1=a\)、\(\Pi^\top\mathbf1=b\)；
2. \(R^\top R\approx I\)；
3. 固定 \(R\) 的 Sinkhorn 更新和固定 \(\Pi\) 的 Procrustes 更新不增加对应子问题目标；
4. 相同点云的代价接近 0；
5. 已知神经元置换和正交旋转的合成点云可恢复低残差；
6. 独立置换目标或源点云行后代价不变；
7. 不同宽度但由已知复制/聚合机制生成的点云保持低残差；
8. 所有层代价有限并位于理论范围容差内。

### 13.3 路由

1. 七个路由文件全部存在且互不复用；
2. 每行和为 1；
3. top-\(s\) 后每行重新归一化；
4. \(\tau\rightarrow0\) 时趋向硬路由；
5. 大 \(\tau\) 时趋向均匀路由；
6. 不要求列和均匀。

### 13.4 传输与融合

1. \(T_UW_BT_V^\top\) 形状等于目标权重；
2. 小核心公式与显式大矩阵公式数值一致；
3. \(K_{\ell m}\) 保留非对角项；
4. \(\beta=0\) 时输出逐元素等于目标模型；
5. 目标 top-\(k\) 正交补上的更新为 0；
6. 实际更新满足信任域；
7. 完整 rank LoRA 与稠密更新一致；
8. 任意模块失败时不得写回半完成模型。

### 13.5 无数据审计

主流程测试应通过 monkeypatch 使以下调用直接抛错：

- tokenizer 加载；
- dataset 加载；
- model forward；
- generate；
- backward。

主流程仍应能够完成 SVD、路由、融合和模型保存。

---

## 14. Transport and Merge 对比实验

### 14.1 六个仓库任务

优先复用仓库当前六任务脚本和模型配置：

| 任务 | 目标模型 | 源模型 | 主评测 |
|---|---|---|---|
| medical | 1B medical target | LLaMA 8B source | medical benchmarks |
| thai | Typhoon2 1B target | LLaMA 8B source | Thai benchmarks |
| finance | 1B target | 7B/8B general source | finance benchmarks |
| cantonese | LLaMA 1B target | Chinese LLaMA 8B source | CMMLU-Yue |
| indonesian | Indonesian 1B target | LLaMA 8B source | Indonesian benchmarks |
| malay | Malaysian 1B target | LLaMA 8B source | MalayMMLU |

具体 checkpoint 以 [MODELS.md](MODELS.md) 和实际下载脚本为准。论文文本与仓库 checkpoint 若不完全一致，结果中应明确标注“paper setting”或“repository reproduction setting”。

### 14.2 主对比组

第一组只比较无 adaptation：

| 编号 | 方法 | 对齐数据 | forward | 后训练 |
|---|---|---:|---:|---:|
| A0 | Target Base | 0 | 否 | 否 |
| A1 | Transport and Merge, fused w/o adaptation | 约 2000 | 是 | 否 |
| A2 | DFOP-Attn，仅 Q/K/V/O | 0 | 否 | 否 |
| A3 | DFOP-Full，七模块 | 0 | 否 | 否 |
| A4 | 相对深度硬映射 + 同一融合器 | 0 | 否 | 否 |
| A5 | 奇异值分布距离 + 同一融合器 | 0 | 否 | 否 |
| A6 | 随机行路由 + 同一融合器 | 0 | 否 | 否 |

DFOP-Attn 用于和仓库默认 `LM_ONLY=true` 的 attention-only 融合严格对齐；DFOP-Full 用于评价 gate/up/down 的额外贡献。

第二组比较相同 SFT 预算：

| 编号 | 方法 | 融合 | SFT |
|---|---|---|---|
| B0 | Target SFT / no-HOT | 无传输 | 原论文预算 |
| B1 | Transport and Merge + adaptation | activation HOT | 原论文预算 |
| B2 | DFOP-Attn + adaptation | 无数据谱传输 | 与 B1 相同 |
| B3 | DFOP-Full + adaptation | 无数据谱传输 | 与 B1 相同 |

B2/B3 不属于无数据主方法，只回答“无数据融合能否提供更好的 SFT 初始化”。

### 14.3 公平性约束

所有方法必须固定：

- 完全相同的目标和源 checkpoint；
- 相同 tokenizer 和 chat template；
- 相同评测代码、few-shot 设置、生成参数和最大长度；
- 相同 SFT 数据、样本数、epoch、batch size、学习率预算；
- 相同 fusion \(\beta\) 搜索预算；
- 相同随机种子集合；
- 相同目标模块集合。

若原 HOT 只融合 Q/K/V/O，则必须同时报告 DFOP-Attn；不能只用七模块 DFOP 对比 attention-only HOT。

### 14.4 指标

任务指标沿用仓库现有 evaluation 脚本。除此之外统一报告：

1. 目标任务平均分；
2. 相对 Target Base 的绝对增益；
3. 通用能力平均分；
4. 与 Target Base 的通用能力差值；
5. 融合耗时；
6. 峰值 GPU/CPU 内存；
7. 对齐所需样本数和 token 数；
8. forward 次数；
9. 输出模型参数量和推理吞吐。

核心主张应优先使用：

$$
\Delta_{\mathrm{task}}
=
\mathrm{Score}_{\mathrm{fused}}
-\mathrm{Score}_{\mathrm{base}},
$$

$$
\Delta_{\mathrm{general}}
=
\mathrm{General}_{\mathrm{fused}}
-\mathrm{General}_{\mathrm{base}}.
$$

### 14.5 随机性与统计

- randomized SVD、OT–Procrustes 初始化和随机基线至少使用 3 个种子；
- 报告均值、标准差，并对评测样本做 bootstrap 置信区间；
- 路由稳定性使用不同种子路由矩阵的 Spearman 相关、top-1 一致率和 Jensen–Shannon 散度；
- 不得只报告最优种子。

---

## 15. 必需消融实验

### 15.1 层代价

1. OT–Procrustes 主代价；
2. 只用奇异值分布；
3. 只用相对深度；
4. 去掉 Procrustes，固定 \(R=I\)；
5. full orthogonal \(R\) 与 signed-permutation \(R\)；
6. 去掉 \(\sqrt n\)；
7. 点云中心化与不中心化；
8. \(\gamma=0,0.5,1\)。

full orthogonal \(R\) 比 SVD 严格符号不确定性更灵活，可能把有意义的谱方向差异也消除，因此 signed-permutation 消融是必要的。

### 15.2 层路由

1. 七个独立路由；
2. Q/K/V/O 共享路由；
3. gate/up/down 共享路由；
4. balanced outer OT；
5. 逐行 softmax；
6. hard argmin；
7. top-1/top-2/top-4/全路由。

### 15.3 神经元传输

1. 双侧 OT 重心映射；
2. 只传 residual-stream 一侧；
3. barycentric 与 polar/whiten 映射；
4. 复用阶段一 coupling 与重新求解；
5. 显式 \(T_UW_BT_V^\top\) 与小核心等价实现。

### 15.4 融合器

1. 稠密核心与仅对角奇异值融合；
2. 有/无核心尺度校准；
3. 有/无信任域；
4. 保留/替换目标谱尾；
5. 稠密折叠与完整 rank LoRA；
6. Q/K/V/O 与七模块融合。

---

## 16. 反事实与特异性对照

无数据权重方法容易只学习模型规模或深度先验，因此必须加入：

1. 正确源模型；
2. 源层顺序随机打乱；
3. 模块内神经元随机置换但函数等价的源模型；
4. 随机初始化且形状相同的源模型；
5. 错误领域或错误语言源模型；
6. 同规模源与更大规模源；
7. 源模型权重整体缩放；
8. 仅目标模型自身作为源的自传输。

预期：

- 函数等价神经元置换不应显著改变代价或结果；
- 随机源和错误源不应获得与正确源相同的收益；
- 自传输应产生接近恒等的更新；
- 源层打乱应改变层路由并降低任务收益，否则方法可能没有识别层特异性。

---

## 17. 结果表模板

### 17.1 无 adaptation 主表

| Task | Target Base | T&M w/o Adapt | DFOP-Attn | DFOP-Full | Spectral-only | Depth-only |
|---|---:|---:|---:|---:|---:|---:|
| Medical |  |  |  |  |  |  |
| Thai |  |  |  |  |  |  |
| Finance |  |  |  |  |  |  |
| Cantonese |  |  |  |  |  |  |
| Indonesian |  |  |  |  |  |  |
| Malay |  |  |  |  |  |  |

### 17.2 相同 SFT 预算

| Task | Target SFT | T&M + SFT | DFOP-Attn + SFT | DFOP-Full + SFT |
|---|---:|---:|---:|---:|
| Medical |  |  |  |  |
| Thai |  |  |  |  |
| Finance |  |  |  |  |
| Cantonese |  |  |  |  |
| Indonesian |  |  |  |  |
| Malay |  |  |  |  |

### 17.3 计算成本

| Method | Alignment samples | Forward passes | Alignment time | Fusion time | Peak memory |
|---|---:|---:|---:|---:|---:|
| T&M | 2000 | >0 |  |  |  |
| DFOP-Attn | 0 | 0 |  |  |  |
| DFOP-Full | 0 | 0 |  |  |  |

---

## 18. 实施里程碑

### M0：数学与小矩阵测试

- 随机矩形矩阵；
- 已知置换、旋转和宽度变化；
- 验证代价、coupling、核心和 LoRA 等价性。

### M1：单模块端到端

- 只实现 Q；
- 只处理少量层；
- 保存 \(C^Q\)、\(P^Q_{\mathrm{eff}}\) 和融合模型；
- 验证模型能加载并完成独立推理测试。

### M2：Attention-only Malay 试点

- Q/K/V/O 四路；
- 对齐仓库 `LM_ONLY=true`；
- 比较 Target、HOT w/o adaptation、DFOP-Attn；
- 记录完整时间和显存。

### M3：七模块完整实现

- 加入 gate/up/down；
- 验证七路独立；
- 运行 DFOP-Full 与模块消融。

### M4：六任务无 adaptation

- 冻结 universal 配置；
- 完成所有任务与通用能力评测；
- 不根据测试结果修改方法。

### M5：同预算 adaptation

- 复用原仓库 SFT 配置；
- 比较 HOT + SFT、DFOP + SFT 和 no-HOT SFT；
- 单独标记为非无数据实验。

---

## 19. 实现验收条件

只有满足以下条件才视为主方法实现完成：

1. 主流程不加载 tokenizer、dataset，不执行 forward；
2. 七种模块分别生成 \(C^c\) 和 \(P^c_{\mathrm{eff}}\)；
3. 同名模块之间才允许比较；
4. 每个路由矩阵形状为 \(L\times M\)，每行和为 1；
5. 层代价只使用直接 OT–Procrustes 几何残差；
6. 不把熵、rank、depth 或 mapped Grassmann 加入主层代价；
7. 阶段一点云满足宽度归一化能量检查；
8. 阶段二同时具有输出和输入 coupling；
9. 源核心通过稠密 \(k_c\times k_c\) 矩阵传输；
10. 目标谱尾保持不变；
11. 更新满足信任域；
12. 稠密输出和完整 rank LoRA 数值等价；
13. 结果目录包含配置、权重哈希、种子、路由和诊断；
14. 至少完成 attention-only 的严格 HOT 对照；
15. 无 adaptation 与 adaptation 结果分开报告。

---

## 20. 已知风险

1. **非凸性**：OT–Procrustes 交替优化只能得到局部稳定点，需要重启和稳定性报告；
2. **过度对齐**：完整正交 \(R\) 可能消除有意义的谱方向差异，必须做 signed-permutation 消融；
3. **计算量**：七模块全 \(L\times M\) coupling 成本高，需要缓存、分块和 top-\(s\)；
4. **语义不可识别**：权重几何相似不保证数据分布上的功能相似；
5. **独立路由协调性**：Q/K/V/O/gate/up/down 独立选择源层可能破坏 block 内协同，需与共享路由消融比较；
6. **重心收缩**：barycentric map 会压缩范数，核心尺度校准和信任域必须同时报告；
7. **实验泄漏**：不得用下游测试集选择 \(k,\tau,\varepsilon,\beta\)；
8. **基线不对齐**：仓库默认 attention-only 时，必须报告 DFOP-Attn，不能只比较 DFOP-Full。

---

## 21. 最终冻结的主方法

$$
\boxed{
\begin{aligned}
&\text{七种同名线性层独立处理}
\\
&\Downarrow
\\
&X=\sqrt n\,H\Sigma/\|\Sigma\|_F
\\
&\Downarrow
\\
&C^c_{\ell m}
=
\frac12
\min_{\Pi,R}
\sum_{i,j}\Pi_{ij}\|x_i-Ry_j\|_2^2
\\
&\Downarrow
\\
&P^c_{\mathrm{eff}}
=
\operatorname{rowsoftmax}(-C^c/\tau)
\\
&\Downarrow
\\
&K^c_{\ell m}
=
(U_A^\top T_UU_B)
\Sigma_B
(V_A^\top T_VV_B)^\top
\\
&\Downarrow
\\
&\overline K^c_\ell
=
\sum_mP^c_{\mathrm{eff},\ell m}\widehat K^c_{\ell m}
\\
&\Downarrow
\\
&W^{c,\mathrm{fused}}_{A,\ell}
=
W^c_{A,\ell}
+\beta\rho^c_\ell
U_A(\overline K^c_\ell-\Sigma_A)V_A^\top.
\end{aligned}
}
$$

该流程是后续实现、实验和论文表述的当前基准版本。任何新增层代价项、共享路由、数据选择掩码或 adaptation 都必须作为单独变体或消融，不得静默加入主方法。
