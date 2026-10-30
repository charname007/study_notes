---
created: 2026-10-30
source: 默认专题/Zamir 等 - 2022 - Restormer Efficient Transformer for High-Resolution Image Restoration.pdf
type: 论文
tags:
  - 论文
  - Transformer
  - 图像修复
  - Attention
---

# Restormer：面向高分辨率图像修复的高效 Transformer

## 一句话总结
把自注意力从**空间维**搬到**通道维**（MDTA），配合门控+深度卷积的前馈网络（GDFN）、4 级无参重采样金字塔与渐进式 patch 训练，使 Transformer 能直接处理高分辨率图像，并在 16 个 benchmark 上取得 SOTA。

## 核心问题
自注意力对图像修复是必要的（CNN 感受野有限、权重静态，第 2 页），但标准 SA 的开销随空间分辨率**二次增长**：`O(W²H²)`（第 3 页）。当时方案（SwinIR/Uformer 的 8×8 窗、IPT 的 48×48 patch）都在"收缩注意力空间范围"，论文原话是这样做 *"defying the main motivation of using self-attention over convolutions"*（第 3 页）。

## 架构图（论文 Fig. 2，第 3 页）

![Restormer 架构图 Fig2](Restormer%20架构图%20Fig2.png)

<small>裁自论文第 3 页 Figure 2，仅取图本体与图例，已去除该页正文与图注文字。</small>

**怎么读这张图：**

- **骨架（左→右）**：`Degraded I` → 3×3 卷积得浅层特征 `F₀`（H×W×C） → 4 级编码器 `Block×L₁₋₄`，每级经 **Downsample**（红色▼，pixel-unshuffle）分辨率减半、通道翻倍 → 最深 `H/8×W/8×8C`（`F_l`）→ 解码器逐级 **Upsample**（绿色▲，pixel-shuffle）回升，每级与编码器特征 **Concatenation**（⊕外的 ○ 符号）后接 1×1 把通道砍半 → `F_d`（H×W×2C）→ **Refinement**（`Block×L_r`，高分辨率精修）→ `F_r` → 3×3 卷积出残差图 `R`（H×W×3）→ 与输入 **Element-wise Addition** 得 `Restored (Î)`。
- **解码器最上层**那条横向长箭头是 `H×W×C` 的跳连，直接跨过整个网络接到 level-1，用于保住细粒度纹理。
- **图右下 (a) MDTA**：输入 `X` 经 Norm 得 `Y`，分成 Q/K/V 三支，每支都是 `1×1`（跨通道）→ `3×3 Dconv`（跨空间）→ Reshape。Q、K 转置后做 **矩阵乘法 ⊗**（图上标 `C×HW`，即通道维变长），Softmax 得 **`Ĉ×Ĉ` 的转置注意力图 A**（那个灰色小方块就是它，注意尺寸与输入无关）→ 与 V 相乘 → Reshape → 1×1 → 与输入 **⊕**。
- **图右上 (b) GDFN**：输入 `X` 经 Norm 后分两支，各走 `1×1` → `3×3 Dconv`，通道升到 `Ĥ×Ŵ×γĈ`（γ 为扩展比，论文取 2.66）；**上支过 GELU（⊙里的 φ）**，两支做 **逐元素乘 ⊙**（门控）→ 1×1 降回 `C` → 与输入 **⊕**。
- **对照代码**：图中 (a)(b) 的连线与代码笔记 [[Restormer 的 MDTA 与 GDFN 实现]] 里的 `Attention`/`FeedForward` 一一对应；红色▼/绿色▲ 对应 [[像素重排：PixelUnshuffle 与 PixelShuffle]] 里的 `Downsample`/`Upsample`。

## 关键方法（四条创新点）

### 1. MDTA：注意力搬到通道维（第 3–4 页）
常规 SA 得到 `HW×HW` 注意力图；MDTA 让 `K·Qᵀ` 得到 **`C×C`** 的转置注意力图（通道间协方差），复杂度线性化。公式 (1)：

$$\hat{X}=W_p\,\text{Attn}(\hat{Q},\hat{K},\hat{V})+X,\qquad \text{Attn}=\hat{V}\cdot\text{Softmax}\!\left(\hat{K}\hat{Q}/\alpha\right)$$

第二要点：**先混局部再算协方差** —— Q/K/V 由 1×1 卷积（跨通道）+ 3×3 深度卷积（跨空间）生成，论文称这样做兼顾"卷积的局部归纳偏置"与"像素间全局关系的隐式建模"（第 2 页）。

| | 注意力图 | 时间 | 显存 |
|---|---|---|---|
| 标准 SA | `N×N` | `O(N²C)` | `O(N²)` |
| MDTA | `C×C` | `O(NC²)` | `O(C²)` |

（N = H·W 为 token 数）

### 2. GDFN：门控 + 深度卷积的前馈网络（第 4 页）
标准 FFN 逐像素独立同构、无空间交互。GDFN 加门控：两条分支一条过 GELU、一条不过，逐元素相乘。公式 (2)：

$$\hat{X}=W_p^0\,\text{Gating}(X)+X,\qquad \text{Gating}(X)=\varphi(W_d^1W_p^1(\mathrm{LN}(X)))\ \odot\ (W_d^2W_p^2(\mathrm{LN}(X)))$$

动机是"抑制信息量低的特征，只让有用信息继续往下传"（第 2、4 页）。因运算更多，扩展比 γ 从常规 4 降到 **2.66** 以保持参数量相当。

### 3. 4 级金字塔 + 无参重采样（第 3 页）
编码器 `C → 2C → 4C → 8C`，分辨率 `1 → 1/2 → 1/4 → 1/8`，block 数由浅入深 4/6/6/8。降/升采样用 pixel-unshuffle / pixel-shuffle（引 [69] ESPCN）：零参数、零信息损失，每级 通道²·token 近似恒定。细节：拼接后 1×1 卷积砍半通道（顶层除外）；level-1 也放 Transformer block 以保住精细纹理。

### 4. 渐进式学习（第 4 页 §3.3，第 5 页给调度）
patch 由小到大 `128²→160²→192²→256²→320²→384²`，batch 同步缩放 `64→40→32→16→8→8`，切换迭代 `92K/156K/204K/240K/276K`，共 300K iter，学习率 3e-4 → 1e-6 余弦退火。

## 消融与结果（第 5、6、8 页）
- MDTA 比标准 SA 基线 **+0.32 dB**；门控单独 **+0.12 dB**；GDFN 完整比标准 FFN **+0.26 dB**；**整个 block 合计 +0.51 dB**（Table 7）
- 渐进式：固定 128² 得 29.71 dB / 22.5h；渐进式 29.78 dB / 23.1h（Table 9）
- 去雨：平均比 SPAIR **+1.05 dB**，单数据集最大 **+2.06 dB**（Rain100L，第 5 页）
- 离焦去模糊：比 Uformer **+1.01 dB**（第 6 页）
- 覆盖 16 个 benchmark、4 类任务（第 2 页）

## 结论与局限
主要贡献在 MDTA（换了注意力的作用维度），而非继续在空间范围上妥协。

局限（我的判断）：
- 4 级 ×2 下采样要求输入 H/W 为 **8 的倍数**，否则重排算子报错
- 通道注意力丢失显式像素-像素对应，依赖前置 depthwise 补局部
- 300K iter 训练量对复现不友好
- 渐进式学习基本沿用既有课程学习思路，原创性弱于 MDTA/GDFN

## 我的疑问
- [ ] MDTA 的 L2 归一化为何未写进公式？
- [ ] 通道注意力在多大程度上等价于空间注意力？（需要可视化注意力图）
- [ ] `heads=[1,2,4,8]` 中 level-1 用单头的设计依据

## 关键摘录
> "the time and memory complexity of the key-query dot-product interaction grows quadratically with the spatial resolution of input, i.e., O(W 2H2) for images of W ×H pixels."（第 3 页）

> "To reduce the computational loads, these methods either apply SA on small spatial windows of size 8×8 around each pixel [44, 80], or divide the input image into non-overlapping patches of size 48×48 and compute SA on each patch independently [13]."（第 1 页）

> "For feature downsampling and upsampling, we apply pixel-unshuffle and pixel-shuffle operations [69], respectively."（第 3 页）

> "this design choice restricts the context aggregation within local neighbourhood, defying the main motivation of using self-attention over convolutions, thus not ideally suited for image-restoration tasks."（第 3 页）

## 相关笔记
- [[像素重排：PixelUnshuffle 与 PixelShuffle]] — 第 3 页两个重采样算子的原理与代码
- [[Restormer 的 MDTA 与 GDFN 实现]] — 公式对应的官方代码
- [[transformer的2道残差连接]] — MDTA/GDFN 外层的两道残差为何必要
- [[PointWise和DepthWise]] — MDTA/GDFN 里 1×1 与 3×3 depthwise 的排布

#论文 #Transformer #图像修复 #Attention
