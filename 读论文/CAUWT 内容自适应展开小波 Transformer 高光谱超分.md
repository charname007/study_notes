---
created: 2026-11-07
source: E:\.dsh-paper-reader\data\小波变换\Fang 等 - 2026 - Content-Adaptive Unfolding Wavelet Transformer for Hyperspectral Image Super-Resolution.pdf（IEEE TIP, vol. 35, 2026, DOI 10.1109/TIP.2025.3636789；论文未提及开源代码）
type: 论文
tags: [论文, 超分辨率, 小波变换, Transformer, 频域]
---

# CAUWT：内容自适应的展开小波 Transformer 做高光谱超分

> Yuan Fang, Yipeng Liu, Zhen Long, Chong-Yung Chi, Ce Zhu，*IEEE Transactions on Image Processing* vol. 35 (2026)。下文页码均指本机 PDF 页码。

## 一句话总结

在**深度展开（deep unfolding）**框架里补两处短板：让数据模块的步长/惩罚/噪声水平 **μ、η、γ 每一阶段按当前重建结果自适应生成**（PP 模块），并把先验模块换成 **DWT 下/上采样 + 谱-空混合注意力**的 Wavelet-Assisted Transformer（WAT）——不额外加高频分支就能补高频细节；在 CAVE/KAIST 上 PSNR 最好的同时，GFLOPs 只有 3DTNet 的 4.1%（p.1、p.7）。

## 核心问题

高光谱超分（HSI-SR）输入是一对 LR-HSI $\mathcal{X}$ 与 HR-MSI $\mathcal{Y}$，观测模型是两个退化算子（p.2 式 (8)）：

$$\mathcal{X} = \mathcal{Z}\times_1 \mathbf{S}_1 \times_2 \mathbf{S}_2,\qquad \mathcal{Y}=\mathcal{Z}\times_3 \mathbf{P}$$

论文指出既有深度展开方法的两个硬伤（p.1）：

1. **数据模块参数是"学出来的固定值"**——每个迭代里 μ/η/γ 共享同一套（可学习）参数，不随当前数据与重建进度变化，导致跨数据/跨退化场景泛化受限。作者在原图上对比了多数展开网络与本文的差别（Fig. 1，p.1）。
2. **先验模块里的 Transformer 抓不住高频**——自注意力天然偏低频；要补高频就得挂额外分支（[24], [37]），结构膨胀、算力上升（p.3 相关工作总结）。

## 关键方法

思路是把 (9) 式的模型正则化问题用 **HQS（half-quadratic splitting）** 拆成 Z-子问题与 M-子问题（p.4，式 (12)–(14)）：

$$\mathcal{Z}^k=\arg\min_{\mathcal{Z}}\ \tfrac{1}{2}\|\mathcal{X}-\mathcal{Z}\times_1\mathbf{S}_1\times_2\mathbf{S}_2\|_F^2+\tfrac{1}{2}\|\mathcal{Y}-\mathcal{Z}\times_3\mathbf{P}\|_F^2+\tfrac{\mu}{2}\|\mathcal{M}^{k-1}-\mathcal{Z}\|_F^2$$

$$\mathcal{M}^k=\arg\min_{\mathcal{M}}\ \tfrac{\mu}{2}\|\mathcal{M}-\mathcal{Z}^k\|_F^2+\lambda\Phi(\mathcal{M})$$

Z-子问题因矩阵求逆代价过大改用**梯度下降**（式 (15)(16)），M-子问题被改写为方差 $\sigma^2=\gamma^2$（$\gamma=\sqrt{\lambda/\mu}$）的**去噪问题**，直接用一个去噪网络 $\mathcal{D}(\cdot)$ 代替正则项（式 (17)(18)，p.4）。整个网络就是把这个迭代展开成 K 个 stage，每 stage 三件套：**PP → GUM → WAT**，外加一个 **IM** 做初始化（p.5）。

![CAUWT 整体架构图 Fig2](CAUWT%20整体架构图%20Fig2.png)

> **怎么读这张图**（p.5 Fig. 2）：(a) 是主流程：$\mathcal{X}$、$\mathcal{Y}$ 先过 **IM** 得到 $\mathcal{Z}^0=\mathcal{M}^0$，然后每个 stage 内依次是青色 **PP**（吃 $\mathcal{Z}^{k-1}$ 和观测，吐出 $\mu^k,\eta^k,\gamma^k$）→ 黄色 **GUM**（把式 (15) 的矩阵运算换成卷积/反卷积，更新 $\mathcal{Z}^k$）→ 蓝色 **WAT**（当式 (18) 的去噪网络 $\mathcal{D}$，输出 $\mathcal{M}^k$），最后 $\mathcal{M}^K$ 就是 HR-HSI。(b) IM 由 3×3 卷积 + 上采样 + Channel Attention 组成，注意 $\mathcal{X}$ 上采样后与 $\mathcal{Y}$ 拼接——这是"别浪费 $\mathcal{Y}$ 的空间信息"的具体落点。(c) PP 由 4 个 1×1 卷积 + CA + **Softplus** + Split 组成，Softplus 保证参数为正。(d) GUM 里 $\mathbf{P}$、$\mathbf{P}^T$ 各用一层卷积、$\mathbf{H}$ 用卷积、$\mathbf{H}^T$ 用**反卷积**代替。

### 1) PP：每阶段重新预测 μ、η、γ（p.5，式 (20)）

$$(\mu^k,\eta^k,\gamma^k)=\xi(\mathcal{Z}^{k-1},\mathcal{X},\mathcal{Y})$$

对比原展开网络学一组固定参数的写法，这里是"输入驱动"的：PP 从观测和当前重建结果评估谱/空恢复程度，实时生成正则参数与噪声水平。$\gamma^k$ 同时作为噪声水平条件拼接进 WAT 的输入（式 (22)）。

### 2) GUM：把大矩阵运算换成卷积/反卷积（p.5，式 (21)）

$$\nabla g(\mathcal{Z}^{k-1}_{(3)})=-\mathrm{Deconv}(\mathcal{X}_{(3)}-\mathrm{Conv}(\mathcal{Z}^{k-1}_{(3)}))-\mathrm{Conv}(\mathcal{Y}_{(3)}-\mathrm{Conv}(\mathcal{Z}^{k-1}_{(3)}))-\mu(\mathcal{M}^{k-1}_{(3)}-\mathcal{Z}^{k-1}_{(3)})$$

原式 (15) 里 $\mathbf{H}=\mathbf{S}_1^T\otimes\mathbf{S}_2^T$ 是 Kronecker 积形式的空间退化矩阵，直接算代价高；GUM 用卷积层顶替 $\mathbf{P}/\mathbf{P}^T/\mathbf{H}$、用反卷积顶替 $\mathbf{H}^T$。这一步是把"可解释的闭式梯度"翻译成"可微的卷积算子"，也是它算力低的主要来源。

### 3) WAT：用 DWT/IDWT 当上下采样器（p.5–p.6，式 (22)–(25)）

![CAUWT WAT与HSSAB结构图 Fig3](CAUWT%20WAT与HSSAB结构图%20Fig3.png)

> **怎么读这张图**（p.6 Fig. 3）：(a) WAT 是 U 形，**下采样位置不放池化/步长卷积，而放 DWT**（橙色），**上采样位置放 IDWT**（橙色），彩色小块就是 {HH, HL, LH, LL} 四个子带；每个 HSSAB 之后接一次小波变换，分辨率减半、通道翻倍，解码侧对称回来。(b) HSSAB 内部两条并行支路：**CA（通道注意力）与 MSA（空间多头自注意力）并联**，都作用在 LN 后的特征上，再各自加残差，最后过 FFN。(c) FFN 是 1×1 → 3×3(GELU) → 1×1 的 GLU 变体。

第 i 级编码用 DWT 把特征拆成四个子带、解码用 IDWT 还原（式 (24)(25)），论文的论证是：**以往用单层卷积做尺度变换**，而 Transformer 本身高频能力弱，换成分解/重构的一对变换后，"不增加额外分支结构"就拿到了多尺度多频信息（p.6）。

### 4) HSSAB：CA 管高频、MSA 管低频（p.6，式 (26)）

$$\begin{cases}\hat{\mathcal{F}}=\mathcal{F}_{\mathrm{In}}+\mathrm{CA}(\mathrm{LN}(\mathcal{F}_{\mathrm{In}}))+\mathrm{MSA}(\mathrm{LN}(\mathcal{F}_{\mathrm{In}}))\\ \mathcal{F}_{\mathrm{Out}}=\mathrm{FFN}(\mathrm{LN}(\hat{\mathcal{F}}))+\hat{\mathcal{F}}\end{cases}$$

注意 CA 与 MSA 是**并联相加**而非串联。作者给的分工解释是：MSA 更擅长低频、CA 更擅长高频，两者结合可缓解"MSA 过滤掉高频"的问题（p.6）。这个分工在 Fig. 11 的频谱图上做了可视化验证（p.11）：MSA 特征图偏低频、CA 特征图偏高频。

### 5) 损失：重建项 + 逐阶段约束项（p.6，式 (27)(28)）

$$\mathfrak{L}_{\mathrm{rec}}=\|\mathcal{M}^K-\hat{\mathcal{Z}}\|_1,\qquad \mathfrak{L}_{\mathrm{con}}=\sum_{k=0}^{K-1}\|\mathcal{M}^k-\hat{\mathcal{Z}}\|_1,\qquad \mathfrak{L}=\mathfrak{L}_{\mathrm{rec}}+\alpha\mathfrak{L}_{\mathrm{con}},\ \alpha=0.1$$

$\mathfrak{L}_{\mathrm{con}}$ 的作用是拴住中间阶段，别让某个 stage 的输出跑飞。

## 结论与局限

- **模拟数据（CAVE / KAIST，R=4 与 8，指标 PSNR/SSIM/SAM/ERGAS）**：CAUWT 全面最优，3DTNet 次之（p.7、Table I/II）。同样 R=4 的 CAVE 上，**CAUWT-3stg 比 MoGDCN / 3DTNet 分别高 0.47 dB / 0.14 dB，而 GFLOPs 只有它们的 6.2% / 4.1%**；最大的 CAUWT-7stg 的 GFLOPs 仅为 3DTNet 的 8.8%（p.7）。相对 LRTN / SRLF，CAUWT-3stg 的 ERGAS 低 0.19 / 0.25（p.7）。
- **效率-精度权衡**：Fig. 8（p.9）的 PSNR-GFLOPs 曲线上 CAUWT 位置最好——BDT/PSRT/FeINFN/LRTN/SRLF 算力相当但精度更低，3DTNet/MoGDCN 精度接近却算力高一个量级。
- **真实数据（UH，无 GT，用 $D_S$/$D_\lambda$/QNR）**：CAUWT 的空/谱失真都最低，MoGDCN 第二；作者同时指出 BDT/PSRT/LRTN 结果模糊有伪影、MoGDCN/3DTNet/KNLNet/FeINFN/SRLF 过度平滑丢细节（p.12，Table V）。
- **消融（CAVE，R=8，CAUWT-3stg，p.10–p.11）**：拆 DWT 掉 **0.31 dB（最大）** > 跨 stage 共享权重 0.30 dB > 拆 IM 0.29 dB > 拆 PP 0.17 dB。也就是说**小波是贡献最大的组件，自适应参数 PP 反而是四项里最小的**——这一点与论文摘要的叙事重心略有落差，值得注意（p.10）。
  - WAT 内部：去掉 **MSA 掉 0.53 dB**、去掉 CA 只掉 0.14 dB（p.10）——空间自注意力是主力，通道注意力是补充。
  - PP 单独在 Harvard 的 imgb1 上验证：去掉 PP **掉约 0.66 dB**、ERGAS +0.06（p.8 数据描述、Table IV、p.11）。注意这个数比 CAVE 上的 0.17 dB 大得多，说明 PP 的收益**强依赖数据/退化分布**。
  - 小波层数：I=1 抓不住细纹理；**I=3 只多 0.04 dB 却多 11.82 M 参数**，故默认 I=2（p.11、Fig. 12）。
  - 损失：α=0 掉 0.06 dB、α=0.2 掉 0.05 dB，且 α=0 时中间迭代出现严重失真、破坏展开过程的可解释性（p.11、Fig. 13）。
- **局限（我的判断，论文未系统讨论）**：
  1. PP 只输出 **3 个标量**（μ, η, γ），而 GUM 里的卷积核仍是跨 stage 不共享但固定的权重——"内容自适应"的粒度其实很粗。消融里 PP 只值 0.17 dB，某种程度上印证了这一点（p.10）。
  2. 论文声称"几乎不增加参数量/分支"，但代价是把**尺度变换本身**替换成了 DWT——这只在 U 形且层数少（I=2）时成立；I=3 时参数暴涨 11.82 M（p.11），说明小波带来的通道翻倍并不便宜。
  3. 光谱维度与空间维度的耦合只在 GUM 用卷积近似，没有对波段数做扩展性实验（C 固定为 31/48）。
  4. 数据集都是**可见光 400–720/1050 nm**，未涉及红外/更多波段；UH 只有无参考指标。
  5. 论文全文未提及开源代码，**未找到可用实现**。

## 我的疑问

- [ ] PP 只预测三个标量，能否扩展到**逐通道/逐 stage 的调制向量**（比如 channel-wise 参数生成）？涨点空间可能比现在 0.17 dB 大。
- [ ] 消融里"共享权重"（CAUWT-S）只掉 0.30 dB，说明 K 个 stage 的差异其实不大——那把 K 增大到 9 带来的收益，有多少是"更多参数"而非"更多迭代"？
- [ ] DWT 用的是**固定 Haar**（PyWavelets 实现，p.7）。换成可学习小波（LWTformer 那类）是否还能保持"不加分支"这一卖点？
- [ ] CA 与 MSA 并联相加，能否改成"CA 只作用于高频子带、MSA 只作用于 LL 子带"？现在的做法并没有把子带信息显式喂给两支。
- [ ] GUM 用卷积替代 $\mathbf{H}$ 与 $\mathbf{P}$，那么学到的算子是否还保持物理可解释性（比如是否接近真实的高斯模糊核）？论文没有可视化。
- [ ] 与 [[DTWSR 扩散Transformer建模多级小波谱的单图超分]] 对比：同样是"小波 + Transformer"，一个是扩散链在谱域迭代，一个是展开网络在空域迭代——在小波域里做展开会更好吗？

## 关键摘录

> "Parameters in the data module are fixed (though learnable) at each iteration, i.e., lacking the adaptivity to comprehensive data; ... The Transformer in the prior module cannot effectively capture high-frequency information."（p.1，摘要中点出的两个问题）

> "where ξ is a parametric prediction network that takes Z^{k−1} and the observed X and Y as inputs, and adaptively generates the regularization parameters μ^k, η^k and the noise level γ^k by analyzing the characteristic performance of the reconstruction result Z^{k−1} in the spatial and spectral domains."（p.5）

> "In GUM, the matrices P, P^T and H are replaced with one convolutional layer, respectively, while H^T is substituted with a deconvolution layer."（p.5）

> "Compared with the previous transformer structure, the proposed representation structure can better explore the high-frequency information of HSI without adding additional branch structures."（p.6）

> "This design can effectively resolve the issue of MSA filtering out high-frequency features."（p.6）

> "Among these different architectural modifications of CAUWT, the removal of DWT components leads to the largest PSNR drop of 0.31 dB, followed by the inclusion of weight-share setting and the removal of IM and PP components of 0.30 dB, 0.29 dB, and 0.17 dB, respectively."（p.10）

> "The removal of MSA and CA modules in the WAT resulted in a PSNR drop of 0.53 dB and 0.14 dB, respectively."（p.10）

> "Although a slightly better reconstruction performance can be obtained when I = 3 (with PSNR increase by 0.04 dB), the cost of associated parameters becomes prohibitively high (with the number of parameters increased by 11.82 M)."（p.11）

> "To the best of our knowledge, this is the first work utilizing DWT (embodied in WAT) for effectively capturing the high-frequency components for HSI-SR."（p.12）

## 相关笔记

- [[小波变换]] —— 本篇是"小波当尺度变换器 + Transformer 主干"这一支的最新样本，与 LWTformer（可学习小波）、DTWSR（小波谱域扩散）并列，可在该页对照三种用法
- [[DTWSR 扩散Transformer建模多级小波谱的单图超分]] —— 同样把多级 DWT 塞进 Transformer；差别是 DTWSR 在**扩散链**里逐采样步做谱域去噪，本篇在**展开网络**里逐 stage 做空域去噪
- [[LWTformer 可学习小波变换的古汉字图像修复]] —— 反面取向：本篇用固定 Haar 求效率、且"不加分支"是卖点；LWTformer 把低通/高通核也学出来
- [[Restormer 高效高分辨率图像修复 Transformer]] —— 同为"高效 Transformer"路线，本篇的通道注意力与 FFN 设计可与之对照
- [[各种Norm层的摆放]] —— HSSAB 用的是 Pre-LN 形式（LN 在 CA/FFN 之前），见式 (26)
