---
created: 2026-11-26
updated: 2026-11-26
source: 综合 FiLM、SPADE、AdaIN、Conditional BN、StyleGAN2、DiT/adaLN-Zero 等原文与综述整理
type: 概念
tags: [概念, Norm, Attention, 扩散模型]
---

# 调制 Modulation

> 「调制」在深度学习里是一族操作，不是单个算子。它的统一形式只有一行：用一路信号生成 $\gamma$ 与 $\beta$，作用到另一路特征上。真正决定性质的不是公式，而是 $\gamma,\beta$ **从哪里来、有没有空间维、能不能装下序列**。这篇按这个顺序梳理 FiLM → AdaIN → SPADE → adaLN 谱系，并把批评文献（AdaIN 的全局性缺陷、StyleGAN2 的水滴伪影、adaLN 装不下长 prompt）讲清楚。

## 是什么

### 1. 一句话与统一形式

**调制（modulation）= 用条件信号生成 scale 与 shift，去重新分配特征的增益与阈值。**

$$\mathrm{Mod}(F)=\gamma\odot F+\beta,\qquad (\gamma,\beta)=g(\text{condition})$$

关键全在 $g$ 是什么、$\gamma,\beta$ 的形状是什么。$F$ 只是被作用的对象。

### 2. 先分清两个家族

「调制」在 DL 里至少有两个互不相同的用法：

1. **特征调制（feature modulation）**：条件决定 $\gamma,\beta$，改动特征的分布与增益。这是当前的主流含义（FiLM / AdaIN / SPADE / adaLN）。[1]
2. **信号相乘式调制**：把一路信号乘到另一路上，如频域/子带调制（你笔记里 Light-MambaIR 用低频系数去调制三方向高频子带即属此类）。

两者共享「用 A 去缩放 B」的直觉，但机制完全不同。**本文主要讲第一个**。

## 为什么需要

### 3. 它比拼接强在哪

最朴素的条件注入是**拼接（concatenate）**，缺陷是：把高维条件拼到隐层，隐层实际上会对它做一次线性降维，条件维度一高信息就被压掉。[7]

调制则不改输入维度，只在通道上重新分配增益，参数量只有 $2C$，且与空间分辨率无关。[1]

FiLM 给出了量化证据（CLEVR 消融）：[1]

- 测试时把 $\boldsymbol\beta$ 换成训练集均值（抹掉跨样本条件信息）：准确率仅掉 **1.0%**
- 同样操作 $\boldsymbol\beta$ 换成 $\boldsymbol\gamma$：准确率暴跌 **65.4%**

→ **条件信息几乎全靠 scale（$\gamma$）承载**，shift 是辅助。[1]

### 4. scale/shift 到底能做什么

| 取值 | 效果 |
|---|---|
| $\gamma>1$ | 放大某个特征检测器 |
| $\gamma=0$ | 关掉整个通道 |
| $\gamma<0$ | 激活取反，改变 ReLU 后哪些信号通过 |
| $\beta$ | 平移激活阈值，让非线性拐点随条件移动 |

FiLM 实测 $\gamma$ 取值跨 $[-15,19]$，其中 $36\%$ 为负——模型确实在用这些自由度，而不是学出接近 1 的平凡解。[1]

> 所以调制的本质是**由条件决定的特征门控**，$C$ 个旋钮各自可放大 / 置零 / 取反。[1]

## 怎么做 / 关键细节

### 5. 谱系：从 FiLM 到 adaLN-Zero

| 方法 | $\gamma,\beta$ 从哪来 | 空间维 | 代表模型 |
|---|---|---|---|
| Conditional BN / CIN | 由类别或风格嵌入预测 | 每通道一标量 | MODERN、BigGAN [4] |
| **FiLM** | 任意生成器（GRU / MLP） | 每通道一标量 | CLEVR 视觉推理 [1] |
| **AdaIN** | 直接取风格图的均值与方差 | 每通道一标量 | 任意风格迁移 [3] |
| **SPADE** | 由分割图经卷积得到 | **逐像素张量 $C\times H\times W$** | GauGAN [2] |
| **adaLN** | 由 $t+c$ 嵌入回归 | 每通道一标量 | DiT [6] |
| **adaLN-Zero** | 同上 + 残差门控 $\alpha$ | 每通道一标量 | DiT / SD3 / FLUX [6] |

三处关键差异：

- **AdaIN** 没有可学习参数，$\gamma,\beta$ 直接来自风格图统计量；它等价于把内容特征的均值方差换成风格的。[3]
- **SPADE** 把标量升级成**空间张量**，因为分割图在不同位置有不同语义，全局标量无法区分天空与草地。[2]
- **adaLN-Zero** 多回归一个 $\alpha$ 并**零初始化**，使每个 block 在训练起点等价于恒等映射，深层训练更稳。[6]

adaLN-Zero 的实现只有六个数（每个子层一组 scale/shift/gate）：

```python
shift, scale, gate = adaLN_modulation(c).chunk(3, dim=1)
x = x + gate * attn(norm(x) * (1 + scale) + shift)
```

### 6. 调制在现代架构里的地位

DiT 系统对比了四种条件注入方式，结果很有说服力：[6][14]

| 方式 | Gflops | 参数量 | FID-50K @400K |
|---|---|---|---|
| in-context（追加 token） | 119.37 | 449M | 35.24 |
| cross-attention | 137.62 | 598M | 26.14 |
| adaLN | 118.56 | 600M | 25.21 |
| **adaLN-Zero** | 118.64 | 675M | **19.47** |

两个反直觉点：**调制比 cross-attention 更便宜**（不增加序列长度上的算力），却**效果更好**；零初始化本身（adaLN → adaLN-Zero）就带来显著提升。[6][14]

这也是 SD3、FLUX 把时间步与池化文本向量交给 adaLN 的原因。[14]

### 7. 关键区分：调制 ≠ 注意力

| | 调制（Modulation） | 注意力（Attention） |
|---|---|---|
| 粒度 | **全局标量**，一个 $\gamma_c$ 作用于整通道所有位置 | **逐 token 加权** |
| 能力 | 整层倾向（哪些特征重要） | 局部绑定（哪个位置对应哪个词） |
| 成本 | 几乎为零 | $O(T^2)$ |
| 短板 | 无法做空间选择性 | 更贵 |

**这正是 adaLN 装不下长 prompt 的原因**：一个池化向量装不下 77 个 token 的语序与绑定关系。[14]

于是 MMDiT（SD3）的做法是**分工**：模态级（时间步、池化文本）用 adaLN，逐 token 语义用联合注意力。[14]

> 一句话：**调制管「整层倾向」，注意力管「局部绑定」。**[14]

## 常见误区 / 注意事项

### 8. 代价与批评

**① AdaIN 是全局操作，丢掉局部结构。**
AdaIN 只迁移全局统计量（均值/方差），风格图里的局部几何结构无法传递。AdaConv 因此改为从风格图回归整个卷积核。[10]

**② AdaIN 忽略通道间相关性。**
它只对齐每个通道各自的边缘统计，不对齐协方差。AdaWCT 改用白化-着色（whitening & coloring）处理通道相关结构。[11]

**③ StyleGAN2 发现 AdaIN 制造「水滴」伪影。**
机制很精彩：生成器会**故意制造局部尖峰**去主导统计量，从而把信号强度信息偷渡过归一化。解决办法是把调制搬进卷积权重——先 $w'_{ijk}=s_i w_{ijk}$，再按 L2 范数解调制 $w''=w'/\sqrt{\sum_{i,k}w'^2+\epsilon}$，彻底不做数据相关的归一化。[5]

**④ 全局调制的容量有硬上限。**
一个向量装不下序列级信息，这是结构性的，不是调参能解决的。[14]

**⑤ 拼接并不总是输。**
对低维条件，拼接简单、参数少，调制优势不明显；条件维度与是否需空间定位才决定选型。[12]

### 9. 判断口诀

- 条件是**低维全局向量**（类别、时间步、池化文本）→ **用调制**，便宜且有效。[6]
- 条件是**长序列**（prompt、图像、掩码）→ **用注意力**或**空间张量调制（SPADE）**。[2][14]
- 条件需要**逐位置不同** → 必须让 $\gamma,\beta$ 带空间维，否则信息被抹平。[2]
- 只想**极小代价微调** → scale/shift 也能当 PEFT 用，SSF 仅约 0.3M 参数即超越全量微调。[8]

## 各变体的具体公式

记号统一如下：$x$ 为输入特征，$\mu,\sigma$ 为归一化统计量，$\gamma,\beta$ 为调制参数，$c$ 为条件信号，$\odot$ 为逐元素乘。。[1]

### 0. 母公式：所有变体都是它的特例

$$\boxed{\;y=\gamma(c)\odot\hat{x}+\beta(c)\;}$$

差别只有三处：$\hat{x}$ 怎么归一化、$\gamma,\beta$ 的形状（标量 / 向量 / 张量）、$\gamma,\beta$ 从哪里来。。[1][2]

### 1. FiLM（Perez et al. 2018）

生成器由条件 $x_i$ 回归两组参数，再逐特征做仿射。[1]

$$\gamma_{i,c}=f_c(x_i),\qquad \beta_{i,c}=h_c(x_i)$$

$$\mathrm{FiLM}(F_{i,c}\mid\gamma_{i,c},\beta_{i,c})=\gamma_{i,c}F_{i,c}+\beta_{i,c}$$

实现上用残差形式输出尺度，避免零初始化抹掉梯度。[1]

$$\gamma_{i,c}=1+\Delta\gamma_{i,c}$$

**特点**：$\gamma,\beta$ 是每通道一个标量，**没有归一化步骤**，直接对原始激活做仿射。参数量为 $2C$，与空间分辨率无关。[1]

### 2. Conditional Instance Normalization（Dumoulin et al. 2017）

把 IN 的 $\gamma,\beta$ 扩展成 $N\times C$ 矩阵，按风格索引 $s$ 取行。[15]

$$\mathrm{CIN}(x;s)=\gamma_s\left(\frac{x-\mu(x)}{\sigma(x)}\right)+\beta_s$$

$\mu,\sigma$ 沿空间轴计算，$\gamma_s,\beta_s$ 即矩阵的第 $s$ 行。。[15]

**特点**：参数量 $O(N\times L)$，$N$ 为风格数、$L$ 为特征图总数；32 种风格的参数只占网络约 0.2%。[15]

**副产品**：$\gamma,\beta$ 可做凸组合插值来混合风格：

$$\gamma=\alpha\gamma_1+(1-\alpha)\gamma_2$$

### 3. Conditional Batch Normalization（de Vries et al. 2017）

BN 的 $\gamma,\beta$ 由语言嵌入 $e_q$ 经 MLP 预测，且**预测的是增量**，保证训练初期等价于预训练 ResNet。[4]

$$\hat{\beta}_c=\beta_c+\Delta\beta_c,\qquad \hat{\gamma}_c=\gamma_c+\Delta\gamma_c$$

$$\mathrm{BN}(F_{i,c,h,w}\mid\hat{\gamma}_c,\hat{\beta}_c)$$

其中 $\Delta\beta=\mathrm{MLP}(e_q)$、$\Delta\gamma=\mathrm{MLP}(e_q)$，输出维度为 $C$。

**关键细节**：ResNet 全部参数（含 $\gamma,\beta$）冻结，只训练 MLP；BN 参数不到总参数 1%。[4]

### 4. AdaIN（Huang & Belongie 2017）

**没有可学习参数**，$\gamma,\beta$ 直接取自风格图的统计量。[3]

$$\mathrm{AdaIN}(x,y)=\sigma(y)\left(\frac{x-\mu(x)}{\sigma(x)}\right)+\mu(y)$$

逐通道展开写就是：

$$\mathrm{AdaIN}(x,y)_c=\sigma_c(y)\cdot\frac{x_c-\mu_c(x)}{\sigma_c(x)}+\mu_c(y)$$

**特点**：把内容特征的一阶、二阶统计量直接替换成风格的，等价于对齐通道均值与方差；比 Gram 矩阵快几个数量级。[3]

### 5. SPADE（Park et al. 2019）

$\gamma,\beta$ 从**标量升级为空间张量**，由分割图 $m$ 经两层卷积得到。[2]

$$\gamma^i_{c,y,x}(m)\,\frac{h^i_{n,c,y,x}-\mu^i_c}{\sigma^i_c}+\beta^i_{c,y,x}(m)$$

其中通道统计量为：

$$\mu^i_c=\frac{1}{NH^iW^i}\sum_{n,y,x}h^i_{n,c,y,x}$$

$$\sigma^i_c=\sqrt{\frac{1}{NH^iW^i}\sum_{n,y,x}\left((h^i_{n,c,y,x})^2-(\mu^i_c)^2\right)}$$

**退化关系**（论文明确给出）：令 $\gamma,\beta$ 与位置无关即得 Conditional BN；再令 $N=1$ 并用真实图像替换掩码即得 AdaIN。[2]

### 6. AdaGN（Dhariwal & Nichol 2021）

时间步与类别嵌入线性投影出 $y=[y_s,y_b]$，调制 GroupNorm 的输出。[16]

$$\mathrm{AdaGN}(h,y)=y_s\,\mathrm{GroupNorm}(h)+y_b$$

**归属提醒**：AdaGN 出自《Diffusion Models Beat GANs》，不是 DDPM；DDPM 只是把时间步嵌入加性注入。消融显示 AdaGN 的 FID 为 13.06，而「加性 + GroupNorm」为 15.08。[16]

### 7. adaLN / adaLN-Zero（Peebles & Xie 2023）

条件 $c=t_{\text{emb}}+y_{\text{emb}}$ 经一个 MLP 回归出**六个**向量，分别对应两个子层的 scale / shift / gate。[6]

$$(\gamma_1,\beta_1,\alpha_1,\gamma_2,\beta_2,\alpha_2)=\mathrm{MLP}(c)\in\mathbb{R}^{6D}$$

$$h_l=x_l+\alpha_1(c)\odot\mathrm{Attn}\Big((1+\gamma_1(c))\odot\mathrm{LN}(x_l)+\beta_1(c)\Big)$$

$$x_{l+1}=h_l+\alpha_2(c)\odot\mathrm{FFN}\Big((1+\gamma_2(c))\odot\mathrm{LN}(h_l)+\beta_2(c)\Big)$$

其中 $\mathrm{LN}$ 是**无可学习参数**的 LayerNorm（`elementwise_affine=False`）。[6]

**adaLN-Zero 的关键**：把调制 MLP 的最后一层权重与偏置**零初始化**，于是训练第 0 步 $\gamma_k=\beta_k=\alpha_k=\mathbf{0}$，每个 block 严格等价于恒等映射：

$$W_{\text{cond}}=\mathbf{0},\quad b_{\text{cond}}=\mathbf{0}\;\Longrightarrow\;x_{l+1}=x_l$$

对应 PyTorch 实现：

```python
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

shift_msa, scale_msa, gate_msa, \
shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
```

**为什么用 $1+\gamma$ 而不是 $\gamma$**：零初始化时 $\gamma=0$，若直接乘 $\gamma$ 会把特征整体清零；写成 $1+\gamma$ 则恒等退化为恒等映射，这也是 FiLM 用 $\Delta\gamma$ 的同一动机。[6][1]

消融结果：adaLN-Zero 的 FID 19.47，优于 adaLN 的 25.21 与 cross-attention 的 26.14。[6]

### 8. StyleGAN2 权重调制与解调制（Karras et al. 2020）

调制**搬进卷积权重**，不做数据相关归一化。[5]

调制（按输入通道 $i$ 缩放）：

$$w'_{ijk}=s_i\cdot w_{ijk}$$

解调制（按输出通道的 L2 范数归一）：

$$w''_{ijk}=\frac{w'_{ijk}}{\sqrt{\sum_{i,k}(w'_{ijk})^2+\epsilon}}$$

**推导依据**：设输入激活 i.i.d. 且单位方差，则调制加卷积后输出标准差为

$$\sigma_j=\sqrt{\sum_{i,k}(w'_{ijk})^2}$$

故除以 $\sigma_j$ 即可把输出拉回单位方差——**不用看实际特征内容**。这正是它能消除水滴伪影的原因。[5]

### 9. SSF：把 scale/shift 当参数高效微调（Lian et al. 2022）

冻结预训练权重，只在每个算子后插入可学习的 scale/shift。[8]

$$y=\gamma\odot x+\beta$$

与前面变体的区别：$\gamma,\beta$ **与输入无关**（不是由条件生成，而是直接作为自由参数学习），且推理时可重参数化合并回原权重。[8]

### 10. 一张公式对照表

| 变体 | 归一化 | $\gamma,\beta$ 形状 | 来源 | 是否有参数 |
|---|---|---|---|---|
| FiLM | 无 | $\mathbb{R}^{C}$ | MLP(条件) | 是 [1] |
| CIN | IN | $\mathbb{R}^{N\times C}$ | 按风格索引查表 | 是 [15] |
| CBN | BN | $\mathbb{R}^{C}$ | MLP(条件) 预测增量 | 是 [4] |
| AdaIN | IN | $\mathbb{R}^{C}$ | 风格图统计量 | **否** [3] |
| SPADE | BN/IN | $\mathbb{R}^{C\times H\times W}$ | 掩码经卷积 | 是 [2] |
| AdaGN | GroupNorm | $\mathbb{R}^{C}$ | 时间步/类别投影 | 是 [16] |
| adaLN | LayerNorm（无参） | $\mathbb{R}^{6D}$ | MLP($t+y$) | 是 [6] |
| StyleGAN2 | 无（权重解调制） | 权重张量 | 风格向量逐通道 | 是 [5] |
| SSF | 无 | $\mathbb{R}^{C}$ | 自由参数 | 是 [8] |

### 11. 三个易混淆点

**① AdaIN 无参数，AdaGN 有参数**。前者统计量来自风格图，后者来自条件投影，常被混为一谈。[3][16]

**② FiLM 不归一化，adaLN 归一化**。FiLM 直接对激活仿射；adaLN 先做无参 LayerNorm 再仿射，两者的 $\gamma$ 语义不同。[1][6]

**③ $\gamma$ 用 $1+\gamma$ 还是 $\gamma$，取决于初始化**。零初始化配 $1+\gamma$（恒等起点）；自由学习配 $\gamma$。用错会让训练初期整层失效。[6][1]

## 相关

- [[各种Norm层的摆放]] — adaLN / AdaGN 在扩散模型中的位置
- [[通道注意力]] — 同样做通道增益，但权重由输入自身统计得到，而非外部条件
- [[小波变换]] — Light-MambaIR 用低频系数调制高频子带（第二个家族）

### 待建笔记

以下为本文涉及但库中尚无独立笔记的概念，暂以纯文本记录：MMDiT（SD3 的双流联合注意力）、SPADE 的空间张量调制、StyleGAN2 的权重解调制。

## 参考来源

[1] Perez et al. (2018). *FiLM: Visual Reasoning with a General Conditioning Layer*. https://arxiv.org/html/1709.07871
[2] Park et al. (2019). *Semantic Image Synthesis with Spatially-Adaptive Normalization (SPADE)*. https://arxiv.org/abs/1903.07291
[3] Huang & Belongie (2017). *Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization*. https://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.pdf
[4] de Vries et al. (2017). *Modulating early visual processing by language (Conditional BatchNorm / MODERN)*. https://proceedings.neurips.cc/paper_files/paper/2017/file/6fab6e3aa34248ec1e34a4aeedecddc8-Paper.pdf
[5] Karras et al. (2020). *Analyzing and Improving the Image Quality of StyleGAN (weight demodulation)*. https://openaccess.thecvf.com/content_CVPR_2020/papers/Karras_Analyzing_and_Improving_the_Image_Quality_of_StyleGAN_CVPR_2020_paper.pdf
[6] Peebles & Xie (2023). *Scalable Diffusion Models with Transformers (DiT / adaLN-Zero)*. https://arxiv.org/html/2212.09748
[7] Sitzmann et al. (2022). *Attention Beats Concatenation for Conditioning Neural Fields*. https://arxiv.org/abs/2209.10684
[8] Lian et al. (2022). *Scaling & Shifting Your Features (SSF)*. https://arxiv.org/abs/2210.08823
[9] Chandran et al. (2021). *Adaptive Convolutions for Structure-Aware Style Transfer*（批评 AdaIN 全局性）. https://openaccess.thecvf.com/content/CVPR2021/papers/Chandran_Adaptive_Convolutions_for_Structure-Aware_Style_Transfer_CVPR_2021_paper.pdf
[10] AdaWCT (2022). *Adaptive Whitening and Coloring Style Injection*（批评 AdaIN 忽略通道相关性）. https://arxiv.org/abs/2208.00921
[11] (2024). *Hyper Recurrent Neural Network*（拼接 vs FiLM 条件容量对比）. https://arxiv.org/abs/2408.04829
[12] Wang et al. (2020). *Normalization Techniques in Training DNNs: Methodology, Analysis and Application*. https://arxiv.org/abs/2009.12836
[13] Zero to Frontier. *U-Nets, DiT, and MMDiT*. https://zerotofrontier.app/image-video-generation/architectures
[14] Neel Mishra. *Conditional Normalization: AdaIN, FiLM, SPADE, and DiT AdaLN-Zero*. https://neelmishra.github.io/blog/dl/normalization/conditional-normalization-adain-film-adaln.html

### 公式章节补充来源

[15] Dumoulin, Shlens, Kudlur (2017). *A Learned Representation for Artistic Style*（Conditional Instance Normalization，Eq.5）. https://ar5iv.labs.arxiv.org/html/1610.07629v5
[16] Dhariwal & Nichol (2021). *Diffusion Models Beat GANs on Image Synthesis*（AdaGN 出处，非 DDPM）. https://arxiv.org/html/2105.05233v4