---
created: 2026-11-07
source: E:\.dsh-paper-reader\data\小波变换\Ruan 等 - LWTformer A Detail-Aware, Learnable Wavelet-Transformer for Ancient Chinese Character Image Restora.pdf（CVPR Findings 开放获取版，CVF 出版；代码 https://github.com/INWLY/LWTformer）
type: 论文
tags: [论文, 图像复原, 小波变换, Transformer]
---

# LWTformer：可学习小波变换的古汉字图像修复

> Ruan, Li, Cheng, Fan, Tian, Lv（安徽大学），CVPR Findings。下文页码均指**本机 PDF 页码**（共 11 页）。

## 一句话总结

把 2D 离散小波变换做成**可学习**的下采样模块，把古汉字图像拆成"低频轮廓 + 三支方向性高频笔画"，再用两级针对性注意力（SEA 守整体结构、WACGA 守高频笔画）做修复；在甲骨文与战国楚简 WSC41K 两个数据集上四项指标全部领先，且轻量版 LWTformer-B 只有 12.44M 参数（p.1、p.6）。

## 核心问题

古汉字（甲骨、楚简）因侵蚀、磨损、断裂、噪声，图像笔画模糊、边缘缺失。作者认为难点有两条（p.2）：

1. **噪声与细笔画深度纠缠** —— 去噪与保细节难以两全；基于骨架的方法（CharFormer）对噪声敏感、生成骨架会失真，扩散方法（DocDiff）可控性差、易出笔划伪影（p.2，Fig. 1(a)(b)）；
2. **传统频域方法的基函数是周期的** —— FFT / DCT 无法刻画古汉字的方向性笔画与局部不规则性，"解耦笔画与背景噪声"上特别低效（p.2）。

第二点正是论文的立论支点，Fig. 2 给了直观证据（见下方第一张图）。

![[LWTformer 频率分解对比 Fig2.png]]

> **怎么读这张图**（p.2 Fig. 2）：每列一组，从左到右是 LQ 低质图、FFT、DCT、本文的 DWT、GT 真值。每种变换给出三格：**频谱 + 低频重建 + 高频重建**。关键看第三格（高频重建）：FFT 与 DCT 重建出的高频基本是满屏噪点，笔画结构看不出来；DWT 的高频子带里却能清楚看到**笔画走向**（水平/垂直/对角各成一支）。这就是"小波能把笔画和噪声分开、FFT/DCT 不能"的直接证据。

## 关键方法

整体是 Transformer 编解码器（p.3 Fig. 3(a)）：3×3 卷积提浅层特征 → Encoder（STB + WTB 交替，两次 WaveDown 降采样）→ Latent（WACGA + GDFN）→ Decoder（STB 上采样）→ Refinement（SEA + GDFN）→ 3×3 卷积输出，配一条**跨整个网络的残差连接**。

![[LWTformer 架构图 Fig3a.png]]

> **怎么读这张图**（p.3 Fig. 3(a)）：灰绿箭头是 WaveDown 下采样（注意它同时出现两次，通道翻倍、分辨率减半），蓝色箭头是上采样。STB 内部是 SEA，WTB 内部是 WACGA——也就是说**注意力机制跟着小波频带走**：浅层用 SEA 管结构，深层用 WACGA 管笔画。最后 Refinement 段回到 SEA，属于"先结构-后细节-再结构"的夹心设计。

### 1) WaveDown：可学习小波下采样（p.4，式 (3)–(8)）

双分支。主路是 `Pixel-Unshuffle(W_3×3(X))`；辅路用**可学习低通 W_L / 高通 W_H 卷积核**沿行、列先后卷积，得到四个 H/2×W/2 子带：

$$y_a,\ (y_h, y_v, y_d) = \mathrm{L\text{-}DWT2D}(X)$$

其中 $y_a$ 是低频近似，$y_h/y_v/y_d$ 分别是水平、垂直、对角高频细节。四支拼成 $I_{all}\in\mathbb{R}^{4C\times H/2\times W/2}$，经 DWConv → GroupNorm → GELU → 1×1 Conv → Sigmoid 生成注意力图 $A$，再**残差式调制**主路：

$$X_{fused} = X_{down}\cdot A + X_{down}$$

作者的动机很直白：池化、stride 卷积这类传统下采样会丢细节，而古汉字修复恰恰依赖细笔画与复杂结构（p.4 原文）。

### 2) SEA：给 MDTA 补空间门控（p.4，式 (1)–(2)）

STB 是对 Restormer **MDTA 的增量改进**——MDTA 只建模通道维依赖，缺细粒度空间建模。SEA 的做法是把 V 转置成 $(HW)\times C$ 后过一层线性 + GELU 得到**空间门控信号**：

$$V_{gate} = \left(\mathrm{GELU}(\mathrm{Linear}(V^T))\right)^T,\qquad Y = W_p(\mathrm{AttentionMap}\odot V_{gate}) + X$$

### 3) WACGA：按小波子带方向加权注意力（p.5，式 (9)–(11)）

WTB 里的模块。先 L-DWT2D 分出四个子带，然后**每个子带用不同的卷积精修**：

- $y_a$：1×1 + 3×3 depthwise（常规）
- $y_h$：1×3 非对称卷积 + **Prewitt 类核**增强水平边缘
- $y_v$：3×1 非对称卷积 + **Prewitt 类核**增强垂直边缘
- $y_d$：tanh 门控 + **Laplacian 类核**强调对角与高频笔画

拼成 $I_{sub}$ 后经 DWConv + Sigmoid 得到调制图 $A_{wave}$，双线性上采样 2× 后与自注意力输出逐元素相乘：

$$Y = Y_{SA}\odot I^{2\times}_{Bilinear}(A_{wave})$$

### 4) 损失：CharFreqPerceptualLoss（p.5）

三路融合，动机是"纯像素损失忽略高频→模糊，纯感知损失过平滑低频"：

$$\mathcal{L}_{total} = \mathcal{L}_{pix} + \lambda_1\mathcal{L}_{freq} + \lambda_2\mathcal{L}_{percep}$$

- $\mathcal{L}_{pix}$：Charbonnier，$\sqrt{\|I_{pred}-I_{GT}\|^2+\epsilon^2}$，$\epsilon=10^{-3}$
- $\mathcal{L}_{freq}$：FFT 谱一致性 $\|\mathcal{F}(I_{pred})-\mathcal{F}(I_{GT})\|_1$
- $\mathcal{L}_{percep}$：VGG19 感知损失
- 权重 $\lambda_1=0.01$、$\lambda_2=0.05$

![[LWTformer SEA与WaveDown Fig3bc.png]]

> **怎么读这张图**（p.3 Fig. 3(b)(c)）：左半边 SEA——注意那支从 V 分叉、过 Linear 得到 $h_{image}$、再接回注意力图的那条路，就是把"通道注意力"变成"空间门控"的关键改动。右半边 WaveDown——上半支是 Pixel-Unshuffle 主路，左中是 L-DWT2D 拆出 $y_a/y_h/y_v/y_d$ 四支，右下经 3×3 DWConv + GNorm + 1×1 + Sigmoid 出注意力图，最后乘回主路再补一个 3×3+1×1 精修。看图时盯住**只有 WaveDown 那支有可学习的小波核**。

## 结论与局限

- **WSC41K（p.6 Table 1，Mix 划分）**：LWTformer-L 25.266 dB / SSIM .9320 / LPIPS .0920 / FID 13.52，在 Easy/Medium/Hard/Mix **四个难度档上四项指标全为最优**；次优 Concertormer 25.13 dB、LoFormer-L 25.03 dB。
- **甲骨文（p.6 Table 2）**：LWTformer-L 23.477 dB / .9537 / .0599 / 26.55；LWTformer-B 23.357 dB / .9535 / .0610 / 26.91，**用 12.44M 参数打败了 50M 级的 LoFormer-L（23.05 / .9462 / .0844 / 46.76）**——性价比是这篇最硬的卖点。
- **组件消融（p.8 Table 3，甲骨文，LWTformer-B）**：基线 22.879 → +SEA 22.997 → +WaveDown 23.132 → +WACGA 23.357 dB。三者**逐级单调递增**，FID 从 34.553 一路降到 26.907；参数只从 11.73M 涨到 12.44M。其中 **WaveDown(+0.135) 与 WACGA(+0.225) 贡献最实**，SEA 只 +0.118 dB 但 FID 掉得最多（−5.4）。
- **损失消融（p.8 Table 4）**：only $\mathcal{L}_{pix}$ 22.654 / FID 29.91；w/o $\mathcal{L}_{freq}$ 22.677 / FID 24.86；w/o $\mathcal{L}_{percep}$ **23.543（PSNR 最高）** 但 LPIPS .0782 / FID 44.16 崩掉；完整版 23.357 / .9535 / .0610 / 26.91。作者据此强调"**单一指标不能代表修复质量**"——去掉感知项 PSNR 反而更高，正是 PSNR 与感知质量脱钩的典型例子。
- **局限（我的观察，非原文结论）**：① 只在甲骨文、楚简两种载体上验证，跨书体/跨载体的泛化没做；② 依赖**成对**的破损-完好训练数据，真实文物的配对获取成本是落地前提；③ 论文明确因 DiffACR、DiffHDR 未完整开源而**排除对比**（p.6），与扩散类方法的比较仍是空白；④ WaveDown 只做 2×、最多两级，更深频带（$H/8$）下小波子带还能不能提供有效方向先验，论文未讨论。

## 我的疑问

- [ ] $y_d$ 那支用 `tanh(W_p(y_d))` 做门控、$y_h/y_v$ 用非对称卷积 + Prewitt，为什么$y_d$ 单独走 tanh？是抑制对角噪声还是稳定训练？论文只给了"directional stroke features 被精修"这种笼统说法（p.5）。
- [ ] WACGA 的 $A_{wave}$ 是**上采样 2× 后**才乘到自注意力输出上——这意味着注意力是在全分辨率做的、而小波门控在半分辨率生成。这个分辨率错配是否损失了对角细节的定位精度？
- [ ] 与 [[AdaptIR 自适应频率增强的一体化图像复原]] 的 RFEM 相比：那边是"级联 Haar 小波扩感受野"，这边是"可学习小波做下采样"。同样是"小波 + 卷积"，什么时候该用**固定 Haar 求效率**、什么时候值得付**可学习核**的代价？
- [ ] 把 WaveDown 换掉 PixelUnshuffle 后，编码器通道翻倍、分辨率减半，与 [[像素重排：PixelUnshuffle 与 PixelShuffle]] 说的"算力平移而非爆炸"相比，可学习核的额外开销到底落在哪？Table 3 显示 MACs 只从 16.12G 涨到 16.86G，但那是 WaveDown 与其他组件一起开的结果。

## 关键摘录

> "This often leads to loss of stroke detail or edge artifacts, rendering them particularly ineffective at decoupling ancient character from background noise."（p.2，批评 FFT/DCT 的周期基函数）

> "Traditional downsampling methods (such as pooling and strided convolution) often lead to the loss of fine details, which is particularly detrimental for ancient character restoration that depends on subtle strokes and complex structures."（p.4，WaveDown 动机）

> "Notably, ablating certain components sometimes yields higher scores in individual metrics than the full CharFreqPerceptualLoss. This highlights our critical insight that individual metrics alone cannot fully reflect the final restoration quality"（p.8）

> "LWTformer-B 23.3568 0.9535 0.0610 26.9066 | 12.44 | 16.86"（p.6 Table 2，轻量版数据行）

## 相关笔记

- [[AdaptIR 自适应频率增强的一体化图像复原]] —— 同属"小波 + 图像复原"路线：那边用**固定 Haar** 级联扩感受野求效率，这篇用**可学习小波核**求细节，正好构成"学不学小波核"的一组对照
- [[像素重排：PixelUnshuffle 与 PixelShuffle]] —— WaveDown 主路就是 PixelUnshuffle，本篇等于问"这个零参数算子够不够用"，答案是换成可学习小波后 PSNR +0.135
- [[Restormer 高效高分辨率图像修复 Transformer]] —— STB 的 SEA 明确是对 MDTA 的增量改进；GDFN 直接沿用
- [[预测LF残差提升HF，LF协同]] —— 同样是"低频承载主体、高频承载细节"的分解思路，可对照它"LF 里其实含部分 HF、需要继续分解"的观察
- [[关于联合小波变换的增强]] —— 小波分解 → 信息增强 → 逆变换这个通用范式的手写片段
- [[小波变换]] —— 库内小波主题的入口页