---
created: 2026-11-07
source: E:\.dsh-paper-reader\data\小波变换\Cui 等 - 2026 - StarIR Convolutional Image Restoration With Spatial-Frequency Fusion.pdf（IEEE TPAMI，2026，DOI 10.1109/TPAMI.2026.3672465）；官方代码 https://github.com/c-yn/StarIR（commit fc5f368，2026-04-09）
type: 论文
tags: [论文, 图像复原, 频域, 多尺度]
---

# StarIR：空间-频率融合的卷积图像复原

> Cui, Zamir, Yang, Knoll, F. S. Khan, S. Khan，*IEEE TPAMI* (2026)。下文页码均指本机 PDF 页码。

## 一句话总结

用**纯卷积**的 encoder-decoder，把「频域中尺度建模（FMB）」与「空间细节建模（SMB）」两条分支用**逐元素相乘（star operation）**融合，再加一个通道调制单元（CMU），在接近 CNN 的开销下拿到 Transformer 级的大感受野——21 个数据集、6 类单退化任务上全面 SOTA（p.1）。

## 核心问题

- 图像复原是不适定逆问题（p.1）。ViT 靠自注意力拿大感受野，但**复杂度随分辨率二次增长**，高分辨率图吃不消；CNN 高效却被**局部感受野**限制（p.1）。
- 已有的双域方法两类都不理想：**显式**类（傅里叶/小波变换 + 精心设计的细化模块、任务专属频谱先验）架构复杂；**隐式**类（池化、自适应卷积）又难刻画复杂频谱（p.3）。
- 作者的取向是**做减法**：只做**一次**傅里叶投影 + 一个简单可学习调制，配极轻的 star 融合，不做重频谱精修、不做交叉注意力（p.3）。

## 关键方法

![StarIR 架构图 Fig4](StarIR%20架构图%20Fig4.png)

**怎么读 Fig.4**：上路是主流程 —— 3×3 卷积提浅层特征 F₀ → 3 级 encoder（每级 n₁/n₂/n₃ 个 Star Block，下采样为**双线性插值 + 3×3 卷积升维**）→ 3 级 decoder，encoder/decoder 同级特征 **concat + 1×1 卷积减半通道**（Fuse）→ 末尾 4 个 refinement block → 3×3 卷积出残差 Ĩ，与输入相加得 Î = Ĩ + I（p.3）。左下蓝框是 Star Block 内部两条残差：`X → Norm → Star Module → ⊕`，`→ Norm → DFFN → ⊕`。右下红框是 Star Module 的四个部件：**FMB（上）、SMB（下）、star 相乘、CMU（右）**。图例：⊗ 逐元素乘、⊕ 逐元素加、DW 深度卷积、GAP 全局平均池化。

### 1) 整体单元：Star Block（p.3，式(1)）

$$\hat X = \mathrm{DFFN}(\mathrm{LN}(\tilde X)) + \tilde X,\qquad \tilde X = \mathrm{Star}(\mathrm{LN}(X)) + X$$

### 2) FMB：在 8×8 窗口里做频域调制（p.4，式(2)(3)）

1×1 → dw3×3 得 $X_i$ → 切 k×k 窗口 → FFT → 与**可学习滤波器** $A\in\mathbb{R}^{k\times(k/2+1)\times C}$ 逐元素相乘 → IFFT → 恢复形状：

$$X'_i = A\odot\mathcal F(\mathcal R_{k\times k}(X_i)),\qquad X_f = \mathcal R^{-1}(\mathcal F^{-1}(X'_i))$$

窗口大小 $k=8$（p.5）。它负责**中尺度、结构性**线索，对应大范围退化。

### 3) SMB：空间注意力保细节（p.4，式(4)）

1×1 → dw3×3 得 $X_j$ → **再用一个 dw3×3 生成空间注意力** → Sigmoid → 乘回 $X_j$：$X_s = \sigma(W^{dw}_{3\times3}X_j)\odot X_j$。负责细纹理与边缘。

### 4) star operation：相乘而不是相加/拼接（p.4，式(5)）

$$X_c = X_f \odot X_s$$

关键主张：逐元素相乘把两域特征**隐式投影到高维非线性空间**，因此在**不增加宽度和深度**的前提下提升表达力（p.4）。消融里相乘确实优于 concat 和相加（p.13）。

### 5) CMU：通道维全局调制（p.5，式(6)）

GAP → 1×1 卷积得通道缩放因子 $d_{scale}$ → 逐元素乘 → 末层 1×1 卷积：$\hat X = W^{final}_{1\times1}(d_{scale}\odot X_c)$。补上全局、通道间的交互。

### 6) DFFN：双域前馈（p.5，式(7)(8)）

空间侧用 GEGLU（$\delta(W^{dw}_{3\times3}W_{1\times1}X)\odot(W^{dw}_{3\times3}W_{1\times1}X)$），频域侧再用一组可学习滤波器做一次窗口 FFT 调制。

### 7) 训练（p.5）

双域 L1 损失 + Adam，lr 1e−3 → 1e−7 余弦退火，窗 $k=8\times8$，随机裁剪 + 水平翻转；不用任何任务专属先验。不同数据集只改通道数 C 与 block 数（Table I，p.5，FLOPs 按 256×256 测）。

> 实现佐证：官方 `StarIR_arch.py` 里默认 `dim=48, num_blocks=[7,8,15], num_refinement_blocks=4, ffn_expansion_factor=3`，FFT 滤波器形状 `(dim,1,1,8,5)`（8×8 窗口 rfft 后 5 列）。代码与论文的三处差异见 [[StarIR 的 StarModule 与 DFFN 实现]]。

![StarIR 双分支设计 Fig2](StarIR%20双分支设计%20Fig2.png)

**怎么读 Fig.2**：左边是自注意力（三个 Linear 投影 → 点积 → Softmax → 再点积），右边本文把它换成 **Fourier Block（上下文）⊗ Conv Block（局部）**，中间那个红圈星号就是逐元素乘。作者想说的是：**同样的"两路上下文×局部"结构，把昂贵的点积注意力换成一次乘法即可**（p.1–p.2）。

## 结论与局限

**主要结果**（3–5 条关键数字，不抄整表）：

- **单退化 21 个数据集**（p.5–p.9）：散焦去模糊比次优 +0.28 dB、比 Mamba 类 +0.53 dB；去雨 AGAN 上比 AST-B **+1.12 dB**、比频域方法 FPro +1.48 dB；去雾 SOTS 室外比 MTFormer-B V2 +1.17 dB、Haze4K 比 DEA-Net-CR **+1.58 dB**；去雪 SRRS 比 MTFormer-L V2 +0.54 dB。
- **效率是主卖点**：GoPro 去模糊比 LoFormer-L **少 62% 参数**；RealBlur-J 比 LoFormer-B +0.77 dB 且参数少 33%；低光增强比 MambaLLIE +0.37 dB 而参数仅 **22%**（p.8、p.9）。
- **一体化（all-in-one）**：五任务平均比 InstructIR **+0.96 dB**；三任务平均 +0.24 dB，Rain100L 38.50 dB（p.10）。
- **复合退化**：LOL-Blur 比 VQCNIR +0.57 dB 且参数少 **82%**；CDD-11 三退化比 OneRestore **+2.89 dB / +0.015 SSIM** 且参数少 41%（p.11）。
- **领域任务**：UHD 去雾 +0.96 dB 且参数不到一半；遥感薄雾 +1.21 dB；水下 LSUI **+3.53 dB**；医学 PET 合成 / CT 去噪均优于专门设计的 Restore-RWKV-light 且参数更少（p.12）。

**消融（GoPro，128×128 patch、100k iter、batch 32，p.13）**：

- 基线换成 ResBlock：**32.46 dB**；
- **只加 SMB +0.17 dB，只加 FMB +0.16 dB** —— 单看都很小；
- **两者 + star 相乘**才有显著提升 —— 说明增益来自**互补性**而非单个模块；
- 融合方式对比：**相乘 > 拼接 > 相加**（拼接多花算力却没换来相应精度）；
- 再加 CMU 达到最佳；
- 特征可视化（Fig.18）：star 之后结构线索更清晰、边缘更锐（p.13）。

**为什么相乘能省深度**（p.13–p.14，式(9)–(12)）：取单输出单通道，简化后有 $y=(w_1^\top x)(w_2^\top x)=x^\top Ax$，$A=w_1w_2^\top$ 记录通道两两关系，可产生 $\frac{d(d+1)}{2}$ 个不同非线性交互项 —— 等于把 d 维隐式投到 d(d+1)/2 维；堆叠 n 层后输出写成 $X_n=\alpha_1h_1(X_0)+\alpha_2h_2(X_0^2)+\cdots+\alpha_nh_n(X_0^{2^n})$，即**少数几层就够**。

**复杂度**（p.14）：FFT 为 $O(CHW\log M)$，远低于自注意力 $O(C(HW)^2)$；实测 256×256×64 特征图上 FFT 调制 0.151 ms vs 7×7 深度卷积 0.143 ms —— 几乎无损效率。

**作者自陈的局限**（p.14）：

1. 真实去雾（Fig.9）复原结果与 GT 仍有可见差距，建议靠更多真实数据或域适应；
2. CMU 只靠 GAP，而 GAP 本质是**最低频分量**，缺高频建模，未来可换多频谱通道表示。

**我观察到的局限**：

3. "一体化"仅指 all-in-one/复合退化那几组；单退化与领域任务**每个数据集各训一个模型**（p.5 原文），不是单模型通吃。
4. 消融只在 GoPro 一个任务、一组配置上做，FMB/SMB 的贡献**随任务类型应当不同**（雾主要动低频、雨/噪动高频，p.14 自己也这么说），但没给分任务消融。
5. 频率调制是**固定 8×8 窗口 + 与通道共享的可学习滤波器**，本质是逐通道固定的频响，不具备输入自适应性——与 AdaptIR 的逐像素动态核路线正好相反，两者可以互相补。

## 我的疑问

- [ ] 8×8 窗口是超参，论文只给了 $k=8$；窗口大小 vs 感受野/延迟的曲线在哪？更大窗口（16/32）会不会在去雾这类低频退化上更好？
- [ ] 可学习滤波器 $A$ 的形状是 $(C, k, k/2+1)$，**跨空间位置共享**。这与 AdaptIR「逐像素动态核 + 频谱分组调制」是两种极端，能否用轻量的低秩/分组方式折中？
- [ ] star 相乘与 GLU/GEGLU 家族的关系：DFFN 里已经用了 GEGLU，等于主分支又叠了一次同类算子，这两处的分工论文没拆开讲。
- [ ] CMU 的 GAP 换成多频带池化（作者展望）能涨多少？这几乎是免费的改动。

## 关键摘录

> "StarIR incorporates two key innovations: 1) a dual-domain representation learning framework, with one branch processing spatial details and the other focusing on mesoscale interactions in the frequency domain; and 2) a high-dimensional feature fusion mechanism, the Star operation, which fuses information from both domains through element-wise multiplication, thereby enhancing representational capacity without increasing network width and depth."（p.1）

> "Instead of relying on heavy spectral refinement blocks, complex frequency manipulation, or task-specific spectral priors, the Star module conducts only a single Fourier projection and applies a simple learnable modulation in the frequency domain."（p.3）

> "our current channel attention unit relies solely on global average pooling to capture global context, which essentially corresponds to the lowest frequency component and thus lacks modeling of higher-frequency information."（p.14）

## 相关笔记

- [[AdaptIR 自适应频率增强的一体化图像复原]] —— 同一个作者（Cui）的前作/姊妹作，走的是"逐像素动态核 + 频谱分组调制 + Haar 小波扩感受野"的另一条双域路线，可正面对照
- [[Restormer 高效高分辨率图像修复 Transformer]] —— StarIR 全文的主要对照基线之一（Rain100L 只领先它 0.14 dB），通道注意力/前馈的对照
- [[StarIR 的 StarModule 与 DFFN 实现]] —— 官方代码逐行对照与三处论文/代码差异
- [[各种Norm层的摆放]] —— Star Block 里两次 Norm 都是 pre-norm，与 Transformer 块同构
- [[transformer的2道残差连接]] —— 可对照 Star Block 的双残差结构
