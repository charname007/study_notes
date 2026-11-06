---
created: 2026-11-06
source: E:\\.dsh-paper-reader\\data\\默认专题\\Long 等 - 2025 - Progressive Focused Transformer for Single Image Super-Resolution.pdf（arXiv:2503.20337v1，2025-03-26）
type: 论文
tags: [论文, Transformer, Attention, 超分辨率]
---

# PFT：跨层继承注意力图的渐进聚焦 Transformer（单图超分）

## 一句话总结

把上一层已经算出的注意力图**乘进**本层（Hadamard 乘积 + 行归一化），连乘天然让弱相关 token 的权重指数衰减；再把这个衰减当成"先验"，在算相似度**之前**就生成 0/1 稀疏索引跳过计算。于是用更少的算力（−27.7% FLOPs）换来更高的 PSNR（Urban100 ×4 +0.36dB），并撑得起 32×32 的大窗口。

## 核心问题

Transformer 做 SISR 时，窗口内每个 query 都要和**全部** token 算相似度，其中大量 token 与它无关（p.1 摘要）。这不只是浪费算力，还会污染特征聚合、降低重建质量。两类已有方案各有短板（p.2）：

- **扩大信息源**：增大窗口（SwinIR/SRFormer）、跨窗口（CAT）、外部 token 字典（ATD）—— 但相似度计算量随 token 数平方涨，只能妥协；
- **改造权重**：稀疏注意力 / top-k（NLSA、DRSformer）—— 但 top-k 是"**算完再扔**"，稠密相似度一步没省（p.3），而且无法区分"高度相关"与"中等相关"。

论文的切入点是：**相邻层的注意力图不是独立的**，本层的相关位置在上一层就已经部分暴露。利用这个跨层冗余，既能改权重，又能提前跳过计算。

## 关键方法

### 1. Progressive Attention：跨层乘法继承（p.3–p.4）

$$A^l = \mathrm{Norm}\!\left(A^{l-1} \odot A^l_{\mathrm{cal}}\right) \tag{3}$$

`Norm(·)` 是按行归一化（保证每行和为 1）。因为归一化后元素都 <1，跨层连乘会让"只在某一层偶然高分"的 token 迅速趋零，而"每层都高分"的 token 被相对放大 —— 相当于把**跨层一致性**当成证据。全文的实验证据是：仅这一项（Table 3 的 Progressive Attention 行）就把 Urban100 从 26.26 提到 26.41，FLOPs 与 vanilla 完全相同（70.4G），即**只提精度、不省算力**。

### 2. Progressive Focused Attention：把继承结果当成计算掩码（p.4–p.5）

$$\begin{cases}
A^l_{sc} = \mathrm{Softmax}\left(\Psi\left(Q^l, (K^l)^\top, I^{l-1}\right)\right) \\
A^l = S_{K^l}\left(\mathrm{Norm}\left(A^l_{sc} \odot A^{l-1}\right)\right) \\
I^l = \mathrm{Sign}\left(A^l\right)
\end{cases} \tag{5}$$

- `Ψ` 是 SMM（Sparse Matrix Multiplication）：只有 `I^{l−1}(i,j)=1` 时才计算 `A^l_{sc}(i,j)`，其余直接置 0；
- `S_{K^l}` 是**逐行**取 top-K（"selects the top K^l values in each row"，p.4；图 2 也写死为 "Select the k largest attention **by row**"）；
- `I^l` 是二值索引矩阵，供下一层跳过计算。

**关键量**：`K₁ = N`（首层不稀疏，退化为标准自注意力，p.5），之后 `K_l = α·K_{l−1}`，α 是 focus ratio。经典设置下每 block 的保留数 `[1024, 256, 128, 64, 32, 16]`，轻量版 `[1024, 256, 128, 64, 32]`（p.5）。窗口 32×32 共 1024 个 token，所以 1024 是"每行保留全部 key"的上界 —— **K 是每行（每个 query）的保留数，不是每层的总数**。

### 3. 三个容易漏的实现细节

1. **首层特判**：没有上一层注意力图，`A₀ = I₀ = 全 1 矩阵`，第 1 层按 Eq.(1) 算标准自注意力（p.5）。
2. **移位窗口 → 两条继承链**：因为用了 Swin 的 shifted window，"attention map ... is passed from the first layer to the last **through two pathways**, divided by the shift operation into even and odd layers"（p.5）。移位改了窗口几何，偶数层与奇数层的坐标对不上，不能互相乘。
3. **LePE 位置编码**：`O = A·V + DWConv(V)` 式的值端位置编码（出自 CSWin Transformer），分辨率/窗口大小无关，适配大窗口 + 移位窗口（p.5，全文仅一句提及）。

### 4. 计算复杂度：省的是第二项（p.5）

$$\Omega(\mathrm{SA}) = 4hwLC^2 + 2W^2hwLC \tag{7}$$
$$\Omega(\mathrm{PFA}) = \sum_{l=1}^{L}\left(4hwLC^2 + 2\alpha^{(l-1)}W^2hwLC\right) \tag{8}$$

第一项 `4hwC²`（Q/K/V + 输出投影）不动；第二项（相似度 + 聚合）被 α^(l−1) 指数压低，α=0.5 时四次衰减后只剩 **6.25%**。这也是为什么实测只降 27.69% 而不是 90%：**投影项的常数开销变成了新瓶颈**。

## 结论与局限

- **主结果**（Table 1，p.6）：PFT ×2 19.6M / 5.03T，Set5 38.68、Urban100 34.90、Manga109 40.49；同 FLOPs 档位全面压过 ATD（38.61/34.70/40.37，6.07T），论文口径是"比 ATD 少 **17.1%** 计算量且多数数据集 PSNR 最高"。×4 为 33.15 / 29.29 / 28.02 / 28.20 / 32.63。
- **轻量版**（Table 2，p.7）：PFT-light ×4 仅 792K 参数、69.6G FLOPs，Urban100 上比 ATD-light 高 0.23dB、比 IPG-Tiny 高 0.42dB；×2 Urban100 分别高 0.40dB / 0.63dB，算力少 20.1%。
- **消融很有信息量**（Table 3，p.7）：vanilla 26.26 / top-k 26.29（70.4G）→ Progressive Attention 26.41（70.4G）→ PFA 26.62（50.9G）。**继承买精度、聚焦买速度**，两者叠加才到 −27.69% FLOPs / +0.36dB。
- **focus ratio 有甜点**（Table 4，p.8）：α = 0.1/0.3/0.5/0.7/0.9 → 26.48/26.58/**26.62**/26.61/26.56。太小会"premature choices"（早层误杀救不回来），太大则留着无关 token 降不了噪。
- **窗口越大越好，且 PFA 付得起**（Table 5，p.8）：8×8 → 16×16 → 32×32 时 Urban100 26.40 → 26.62 → 26.81。

**局限（作者未明说，我的观察）**：
1. **剪枝不可逆**：p.5 明说被判为不相关的 token"后续层完全不考虑"，早期误判永久丢失，只靠 `K₁=N` 和 α≈0.5 缓解，没有回退机制。
2. **Eq.(8) 是按层连续衰减的理想模型**，实际配置是**按 block** 阶梯下降（block 内多层共用同一 K），所以理论复杂度是个乐观上界。
3. **没做"全局 top-K vs 逐行 top-K"的对照消融**，Table 3 的 top-k 基线本身也是逐行的（p.3），"全局过滤不可行"在本文只是推断、未经实验证伪。
4. **LePE 只被引用、没有消融**，无法判断它对 PFT 的实际贡献。
5. **K 对所有 query 一视同仁**，没有 per-row 自适应预算；平坦区与边缘区的 query 需求显然不同，这里是可做的改进空间。

## 我的疑问

- [ ] `A^{l-1}` 跨 block 传递时，如果 block 之间特征分布漂移较大，旧注意力图是否还是有效先验？论文只说"passed from the first layer to the last"，没讨论跨 block 的有效性衰减。
- [ ] SMM 的 CUDA kernel 是本工作的工程核心（p.5、p.11 提到推理速度可比），但论文没有给出 kernel 层面的加速比，无法判断"50.9G FLOPs"折算成墙钟时间后的真实收益。
- [ ] 若把逐行 top-K 改成"逐行动态阈值"，能否在同等 FLOPs 下再涨点？论文完全没试。
- [ ] progressive 这个词在不同论文里指代不同（PAGRN 是 SA+CA 串联、PAN 是在 CNN 中间层反复施加注意力），PFT 指的是**跨层注意力图继承**，三者只是同名不同物，对比时要小心。

## 关键摘录

> "Instead of combining input features merely according to the calculated attention map A_cal^l, we inherit attention information from the previous block in a productive manner A^l = Norm(A^{l−1} ⊙ A_cal^l), where Norm(·) is the row-wise normalization operation."（p.2）

> "the weights for less relevant tokens will be wiped out during the successive multiplication process, while important tokens with high similarity values across layers will be assigned with larger weights after normalization."（p.4）

> "Since tokens which have been determined as irrelevant tokens to a specific query token in previous layers will not be considered in the following layers at all, to avoid filtering-out important tokens at early stages, we set K1 = N and gradually reduce the nonzero values with a focus ratio: Kl = αKl−1."（p.5）

> "Self-Attention and sparse attention have the same computational cost, as both measure similarity with neighboring positions by interacting with tokens within non-overlapping windows."（p.5）

## 关键图

![[PFT 整体架构与 PFA 模块 Fig2.png]]

**怎么读这张图**：上半部分是网络主干 —— 输入过 Conv 进 PFA Block，每个 Block 内含 M 个 PFAL（Pre-LN 的 PFA + 残差，再 Pre-LN 的 MLP + 残差），最后 Conv + Upsample。下半部分把 PFA 内部拆开：左侧 `A^{l−1}` 沿红线进入两个 SMM —— 第一个 SMM 用 `I^{l−1}` 只算被选中的 `Q·Kᵀ` 位置（跳过计算），得到 calculated attention map；与上一层的 map 做 Hadamard 乘积后经 Sparse（逐行取 top-K）得到本层的 focused map；第二个 SMM 再与本层 `I^l` 一起把 `A^l` 乘到 `V` 上得到输出 `O`。**红线的走向就是"注意力图跨层传递"这条主线**。

![[PFT 注意力分布对比 Fig3.png]]

**怎么读这张图**：第 18 层、两张典型图（建筑/栏杆）上的注意力分布。Query 列是被查询的 patch；SA 与 top-k 的分布**弥散且偏离**真实相关区域；PFA 的分布**稀疏、成簇、且落在同构纹理上**（栏杆的斜纹、建筑的窗户格）。这就是"先选后算"的定性证据 —— 不是权重更平滑，而是**只有该亮的位置亮**。

## 相关笔记

- [[渐进式注意力]] —— 同名不同物的三个工作（PAGRN / PAN / PFT）挂在这里
- [[PAN 渐进式注意力网络与视觉属性预测]] —— "渐进"指在 CNN 多层反复施加注意力，与本文的"跨层继承"不是一回事
- [[PAGRN 渐进式注意力引导循环网络]] —— 渐进式注意力（SA+CA 串联）+ 多路循环
- [[Restormer 高效高分辨率图像修复 Transformer]] —— 另一条 SR 效率路线：在通道维做注意力，与本文"窗口内稀疏化"互补
- [[FlashAttention 精确注意力与IO感知]] —— 同样想省 attention 的开销，但走 IO 感知的精确计算，不改变注意力语义
- [[各种Norm层的摆放]] —— 本文 PFA 前的 Pre-LN 结构
