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