---
created: 2026-11-07
source: E:\.dsh-paper-reader\data\小波变换\Yang 等 - 2026 - HDW-SR High-Frequency Guided Diffusion Model based on Wavelet Decomposition for Image Super-Resolut.pdf（CVPR 2026 Open Access，pp. 23462–23472）；代码 https://github.com/Baty2023/HDW-SR（本次未联网验证可达性）
type: 论文
tags: [论文, 小波变换, 超分辨率, 扩散模型, Attention]
---

# HDW-SR：小波分解的高频引导扩散模型做单图超分

> Chao Yang*、Boqian Zhang*、Jinghao Xu、Guang Jiang（西安电子科技大学）。下文页码均指本机 PDF 页码（原文印刷页 23462 起）。

## 一句话总结

不换扩散范式、只换**骨干与引导方式**：把去噪网络从 U-Net 换成"小波版 U 型网"（HDW-Net），**下采样/上采样全部用 Haar DWT/IDWT** 代替池化与转置卷积，再用一张 PreSR 图的高频子带（`LH/HL/HH`）当 **Q 之外的另一半** 做稀疏交叉注意力，去引导噪声图的低频子带；阈值不再取 Top-k，而是用 Otsu 式类间方差自适应定一个 $k^*$（DTB）。结果是感知指标（LPIPS/DISTS/MUSIQ）在三个数据集上基本全赢，PSNR 略低于 ResShift。

## 核心问题

- 扩散做 SISR 的老问题：**高频域缺引导**，细纹理被抹平（p.1 摘要）。
- 现有两条路线都不够：多步条件扩散从纯噪声起跑，收敛慢且不聚焦细节；StableSR/SeeSR 借助预训练扩散先验、ResShift/SinSR 走残差学习，但**依赖全局图像特征、缺高频先验**，仍难处理多尺度细节（p.1–p.2）。
- 小波域的两条近邻路线各有硬伤（p.3）：Lifting Wavelet-Guided Diffusion 在去噪时**分不清细节与噪声**；DiWa 把扩散整体搬到小波域，但**频域扩散依赖输入图质量、且没有残差学习**。
- 作者的落点是：**残差扩散 + 小波骨干**，把多尺度高频先验"注"进去，最后仍在**图像域**出结果（p.3）。

## 关键方法

### 0. 整体：在残差图上做扩散（p.3，Fig. 2）

对图像对 $(X_i,Y_i)$：先用轻量 CNN **RepRFN** 把 LR 预超分成 $	ilde X_i$（PreSR），取残差 $Delta X_i = Y_i - 	ilde X_i$ 缩小待学信号的动态范围；对 $Delta X_i$ 加噪到 $x_T$，反向去噪时把 **PreSR $	ilde X_i$ 当引导信号**与 $x_t$ 一起喂给 HDW-Net 预测 $Delta X_{	heta,i}$，最后 $Z_i = 	ilde X_i + Delta X_{	heta,i}$。也就是说：**扩散只负责残差（高频残部），主体结构由 PreSR 兜底**。

### 1. HE-Net：把 PreSR 拆成多级高频当"先验"（p.3–p.4，式(1)(2)）

3 级 Haar 小波向下分解：

$$\mathrm{2D\text{-}DWT}(\tilde x_{j-1}) = \{\tilde x_{j,LL},\ \tilde x_{j,LH},\ \tilde x_{j,HL},\ \tilde x_{j,HH}\}$$

- $\tilde x_{j,LL}$ 是低频（结构），经卷积调通道后继续下一级分解；
- 三个高频子带合记 $\tilde x_{j,H} = \{\tilde x_{j,LH},\tilde x_{j,HL},\tilde x_{j,HH}\}$，**逐级送给 HA-Net 对应层**当引导；
- 最深的 $\tilde x_{3,LL}$ 过 FC 得到 $\tilde x^{3,LL}_\theta$，作为上采样阶段的低频起点；之后逐级 $\tilde x_\theta^{j-1} = \mathrm{2D\text{-}IDWT}(\tilde x_\theta^{j,LL}, \tilde x_{j,H})$，末端卷积出 $\tilde X_\theta$；
- 用重建损失把 HE-Net 钉住，保证"先验本身是准的"：

$$\mathcal L_{HE} = \lVert \tilde X - \tilde X_\theta \rVert_2 + \lVert \tilde X - \tilde X_\theta \rVert_1$$

### 2. HA-Net：小波上下采样 + 交叉注意力引导（p.4，式(3)(4)）

$x_t$ 先卷积对齐通道，然后 **用 Haar DWT 下采样**（不是池化/步长卷积）拿到 $\{x^{j,LL}_t, x^{j,H}_t\}$。理由写得很直白：小波下采样**不丢高频**，而池化在边缘/纹理区必丢；且小波提供**联合的空–频局部化**表示，比其它频域方法更适合优化高频细节（p.4）。

每级把低频 $x^{j,LL}_t$ 与本级先验高频 $\tilde x_{j,H}$ 一起送进编码模块 $E_j$：$x^{j+1}_t = E(x^{j,LL}_t, \tilde x_{j,H})$。最深一级过 **PFA** 模块（借自 PFT）得 $x^{3,LL}_{t,\theta}$，再由 $n$ 层 Swin-T + MLP（嵌时间步 $t$）解码。

**上采样用一个很关键的替换**：逆小波变换时用**噪声图自己的** $x^{j,H}_t$，而不是先验 $\tilde x_{j,H}$ —— 因为扩散前期先验有用，但 $t\to0$ 时 $x_t$ 本身高频已经很丰富，**继续压先验会放大伪细节、扭曲纹理**（p.4）。这是我读下来最"有内容"的一个设计决策。

损失：$\mathcal L_{HA} = \mathbb E_{\Delta X,t,\epsilon}\lVert \epsilon - \epsilon_\theta(x_t,t,\tilde X)\rVert^2$，总损失 $\mathcal L = \beta\mathcal L_{HE} + (1-\beta)\mathcal L_{HA}$，取 $\beta = 0.2$（p.4，式(4)(5)）。

### 3. DFA 编码器 + DTB 动态阈值（p.5，式(6)(7)(8)，Fig. 4）

Q 来自低频 $x^{j,LL}_t$（先过 2 层 Swin-T 增强），K/V **都来自高频先验** $\tilde x_{j,H}$。两级稀疏：

$$A^l_{oam} = \mathrm{Softmax}\big(\Psi(Q^l,(K^l)^\top,I^{l-1})\big),\qquad I^l = \mathrm{Sign}(A^l)$$

$$A^l = \mathrm{Norm}\big(A^{l-1} \odot A^l_{fam}\big),\qquad O^l = \Psi(A^l,V^l,I^l)$$

- $\Psi$ 是 SMM（稀疏矩阵乘），按上一层索引 $I^{l-1}$ **只算被选中的位置**，把稠密相似度直接跳过；
- $A^{l-1}\odot A^l_{fam}$ 是**跨层连乘继承**（PFT 的 progressive 思路），让跨层一致的高频键值被放大；
- **DTB** 取代 Top-k：把 $A^l_{oam}$ 的元素按 $1/512$ 分箱统计直方图，对阈值 $k$ 做两类划分，用**类间方差最大**定

$$k^* = \arg\max_k \sigma_B^2(k)$$

  再把 $>k^*$ 置 1、其余置 0 得 0/1 mask。作者的依据是：低频与高频差异大，相似度矩阵元素**天然呈双峰分布**（p.5）。

**实现细节备注**：因为 $x^{j,LL}_t$ 与 $\tilde x_{j,H}$ 在同一小波级上**空间尺寸相同**（都是 $H/2^j$），Q 与 K/V 不需要任何重采样/对齐——这是这套跨注意力能"白拿"的前提，论文没点破但很关键。

### 4. 训练配置（p.6）

DIV2K + LSDIR，×4 上采样；DFA 在三级分别重复 [2,4,4] 次，Swin-T 解码 [4,6,6] 层；Adam，lr $1\times10^{-4}$，10 万次迭代；双 RTX 4090。对比方法包含多步（StableSR/ResShift/SeeSR）、单步（SinSR/OSEDiff/AdcSR）与 DiT 系（DiT-SR）。

## 关键图

![[HDW-SR 整体架构 HDW-Net Fig3.png]]

**怎么读 Fig. 3**：左半是 **HE-Net**，右半是 **HA-Net**。左侧从 $\tilde X$（PreSR）出发，Conv→DWT 逐级拆：竖着的蓝色尺寸条是**高频**（$H/2,W/2,3C$ 这样的三通道子带堆），右侧粉色箭头 `High-frequency Guidance` 就是从每一级抽出来、跨到右半对应层的先验；向下的主干是**低频**，一路 Conv→DWT 到 $H/8,W/8,8C$，过 FC 后进入 `PFA Block × n`（图里写 PFA、正文写 DFA，同一模块，命名不一致），再沿蓝色虚线框 `x̃^{j,H} Guided Inverse Wavelet Transform` 逐级 IDWT 回全分辨率，末端与 $\tilde X$ 算 Loss。右半 **HA-Net** 从 $x_t$ 进、Conv→DWT，粉色箭头在每个编码模块 $E_1,E_2,E_3$ 处**汇入先验**；中间 `SwinT × n + MLP(t)` 是解码器（$t$ 在此注入），右侧紫色虚线框 `x_{j,H} Guided Inverse Wavelet Transform` 表示上采样用的是**噪声图自己的高频**（不是粉色先验），逐级 IDWT 后经 $D_1$+Conv 输出 $\epsilon_\theta$。**两色的区别就是全文的题眼：粉=先验往下走，紫=本体往上走。**

![[HDW-SR DFA编码器与DTB Fig4.png]]

**怎么读 Fig. 4**：最上一行是整个编码器——高频先验 $\tilde x^{j,H}$ 与低频 $x^{j,LL}_t$（先过 SwinT ×2）进 `DFA Block × n`，每块是 Pre-LN 的「DFA + 残差」再接 Pre-LN 的「MLP + 残差」，输出端再叠一个注入 $t$ 的 MLP。中间一行把 DFA 拆开：Q 来自低频、$K^\top$ 与 V 来自高频，先由 $I^{l-1}$ 指导 SMM **只算部分位置**，Softmax 得 `Original Attention Map`；进 **DTB** 后被二值化得 `Filtered Attention Map`；再与上一层的 $A^{l-1}$ 做 Hadamard 乘、LayerNorm 得 $A^l$，最后第二个 SMM 用 $I^l$ 乘到 V 上输出。最下一行是 DTB 内部：$a_{i,j}$ 在 $[0,1]$ 上以 $1/512$ 分箱成直方图 → 用 Otsu 的类内/类间方差公式算出 $\sigma_B^2, \sigma_G^2$ → 取 $k^*$（图中红色箭头那条竖线）→ 落成 0/1 Mask。**"噪声/信号"那对小字就是双峰假设的直观表达。**

## 结论与局限

- **主表（Table 1, p.6）**：DIV2K 上 PSNR 24.52 / SSIM 0.6162 / LPIPS 0.2823 / DISTS 0.1934 / NIQE 4.43 / CLIPIQA 0.6937 / MUSIQ 69.68；RealSR 25.71 / 0.7428 / 0.2672 / 0.2044 / 5.39 / 0.6702 / 70.10；DrealSR 28.43 / 0.7824 / 0.2960 / 0.2153 / 6.20 / 0.6970 / 66.36。**LPIPS 与 DISTS 三个数据集全部最优**，MUSIQ 三项也第一。
- **但 PSNR 不是第一**：DIV2K 24.52 低于 ResShift 24.69；RealSR 25.71 低于 ResShift 26.31、SinSR 26.30；DrealSR 28.43 略低于 ResShift 28.45。**RealSR 的 NIQE 5.39 也排在 DiT-SR(5.36)/OSEDiff(5.37)/SeeSR(5.38) 之后**，而正文 (p.7) 写的是"HDW-SR excels in no-reference metrics"，这句在 RealSR 上站不住。真正的结论应表述为：**感知/无参考质量整体占优、保真度小幅退让**。
- **对 GAN 系（Table 2, p.7）**：三数据集 NIQE/CLIPIQA/MUSIQ 全面压过 RealESRGAN/BSRGAN/LDL，PSNR 也都不低（DIV2K 24.52 vs 24.29/24.38/23.83），属于少见的"两头都不掉"。
- **消融：涨点主要来自"有无高频引导"（Table 4, p.8，RealSR ×4）**：四行分别是 `CNN+DFA(HE-Net)` 25.16、`DWT+SwinT(无引导)` 22.15、`DWT+DFA(先验取自 HA-Net 自身)` 24.39、完整版 **25.71**。拆开看：① **换掉引导损失最大**——把先验整个拿掉（DWT + 普通自注意力）掉到 22.15，差 **3.56 dB**；② 先验来源换成 HA-Net 自己的高频（而非独立 HE-Net）掉到 24.39，差 **1.32 dB**，说明"先验必须来自一条被重建损失钉住的外部支路"；③ 把 DWT 换回 CNN 下采样（其余不变）25.71→25.16，**小波采样本身贡献约 0.55 dB**。作者的表述是"DWT significantly improves…"，但表里真正的大头是引导而不是变换，且这四行的注意力类型并不统一，**不是干净的逐项消融**。
- **DTB 不只是涨点，还省算力（Table 5, p.8）**：Top-k 25.37 dB / 172G FLOPs / 113 ms·step⁻¹ / 1.765 s 每图 → DTB 25.71 / **134G（−22.1%）** / **91 ms（−19.5%）** / **1.435 s（−18.7%）**。**精度和速度同向**，这种消融很少见。
- **β 极其敏感（Table 3, p.7）**：β=0.2 最好；0.1 → PSNR 18.39（掉 6 dB）；0.3 → 23.29；0.4 → 22.07；**0.5 直接崩到 10.30**。作者写"最优区间约 0.2–0.3"，但从表看 0.3 已掉 5%，实际是**单点最优**。
- **局限（作者承认 + 我的观察）**：① 只报了三级的配置，四/五级只是"readily supports"（p.8），无数据；② 全程未给参数量、训练开销，也没报采样步数 $T$ 与采样器，"1.435 s/图"无法与单步方法公平对齐；③ 论文多次强调"高频引导"，但**没有对引导子带做逐项消融**（只用/不用 $LH,HL,HH$ 其中之一是否等价？未知）；④ 先验质量取决于 RepRFN，PreSR 变差时的鲁棒性没测。

## 我的疑问

- [ ] 表 3 的 β=0.5 崩塌很难解释：按 $\mathcal L=\beta\mathcal L_{HE}+(1-\beta)\mathcal L_{HA}$，0.5 是最均衡的配比，却掉到 10.30 dB。是不是 $\mathcal L_{HE}$ 的梯度尺度远大于 $\mathcal L_{HA}$（毕竟一个是重建 L1+L2、一个是噪声回归），导致 β 稍大就压垮扩散项？正文没有给两项损失的量级。
- [ ] 第 4.4 节把 β 描述成"controls the trade-off between perceptual quality and fidelity"，但式(5) 里它明明是 HE-Net 重建项的权重——**这两句话不是一回事**，怀疑是笔误或重写了动机。
- [ ] DTB 的直方图是**逐层、逐头**统计还是整张 attention map 一起统计？Otsu 的前提是全局双峰，如果按头做，某些头的分布根本不双峰，$k^*$ 会退化成噪声。
- [ ] 上采样用 $x^{j,H}_t$ 而不是先验，是**按 $t$ 的某个阈值切换**还是**全程都用本体**？正文读起来是全程用本体（先验只走编码器），但 p.4 那句"at the early diffusion stages, the high-frequency prior... provides strong guidance"又暗示早期有先验参与上采样，两处表述可以再核。
- [ ] 与 [[PFT 渐进聚焦 Transformer 单图超分]] 的关系：DFA 的稀疏索引继承几乎照搬 PFT 的 PFA，唯一实质改动就是 Top-k → DTB。作者没有做"PFA 原版 vs DFA"的对照表，**改进量到底来自 DTB 还是来自小波骨架**，无法从表 4/表 5 直接拆开。

## 关键摘录

> "Diffusion-based methods have shown great promise in single image super-resolution (SISR); however, existing approaches often produce blurred fine details due to insufficient guidance in the high-frequency domain."（p.1）

> "at the early diffusion stages, the high-frequency prior of X̃ provides strong guidance for detail restoration in x_t; however, as t gradually decreases toward 0, x_t becomes increasingly rich in high-frequency details. Excessive reliance on X̃ at this point would amplify pseudo-details and distort textures, thereby hindering further refinement of high-frequency quality."（p.4，解释上采样为何改用本体高频）

> "Due to the significant difference between x^{j,LL}_t and x̃^{j,H}, the values in A^l_oam often exhibit a bimodal distribution."（p.5，DTB 的唯一立论依据）

> "Moreover, HDW-Net readily supports wavelet decompositions beyond the three-level configuration, such as four- and five-level decompositions, further enhancing its flexibility."（p.8）

## 相关笔记

- [[PFT 渐进聚焦 Transformer 单图超分]] —— **DFA 的直系祖先**（原文 ref [22]）。我的 PFT 笔记里留的问题"把逐行 Top-K 改成逐行动态阈值能否涨点"，本文正是用 DTB 回答：+0.34 dB 且 FLOPs 172→134G。但本文没做原版 PFA 对照。
- [[DTWSR 扩散Transformer建模多级小波谱的单图超分]] —— 同样"扩散 + 小波"，但那条路是**把去噪链整体搬进小波谱域**；本文是**残差扩散 + 小波骨干**，最后仍在图像域出图。两者可当对照组读。
- [[Light-MambaIR 时频感知稀疏状态空间统一复原]] —— 同样固定 Haar 拆 LL/LH/HL/HH、同样 top-k 稀疏后再 IDWT 重构，但那边用 SSM 主干、无扩散。
- [[AdaptIR 自适应频率增强的一体化图像复原]] —— 级联 Haar 小波只用来扩感受野，可作为"小波当采样器"的最小用法对照。
- [[LWTformer 可学习小波变换的古汉字图像修复]] —— 那条线把 DWT 做成可学习下采样；本文用的是固定 Haar，可对比"可学习 vs 固定基"的取舍。
- [[小波变换]] —— 小波在 SR/复原里的各家用法汇总（本笔记已挂入）
