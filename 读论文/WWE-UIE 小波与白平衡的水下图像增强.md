---
created: 2026-11-07
source: E:\.dsh-paper-reader\data\小波变换\Cheng 等 - 2026 - WWE-UIE A Wavelet & White Balance Efficient Network for Underwater Image Enhancement.pdf（CVF 开放获取版，正文页码 2135–2145；代码 https://github.com/chingheng0808/WWEUIE）
type: 论文
tags: [论文, 图像复原, 小波变换, 水下图像增强]
---

# WWE-UIE：小波 + 白平衡的轻量水下图像增强

> Cheng, Lee, Lee, Hsu（成功大学 / 阳明交通大学），WACV。下文页码均指**本机 PDF 页码**（共 11 页，正文脚注 2135–2145）。

## 一句话总结

不碰 Transformer、不加先验估计分支，只用三个"便宜的领域先验"——**Gray-World 白平衡的逐通道可学习融合**、**固定 Haar 滤波器组的多频带分解**、**Sobel 梯度门控**——塞进一个 U-Net，就用 **0.734 M 参数 / 6.25 G FLOPs / 7.6 ms（256×256，RTX 3090）** 在 UIEB 上拿到 24.32 dB / 0.9196，与 12 M 级 SOTA 打平或更好（p.1、p.6）。

![[WWE-UIE 架构图 Fig2.png]]

> **怎么读这张图**（p.4 Fig. 2）：整个网络只有两条输入支路——最上面一条是原始输入 $X$，左中那条经过 Simple White Balance Prior 得到 $X_{wb}$，右下角"$1-\gamma$ / $\gamma$"的加权和就是式 (1)。中间主干是 **由 WGSRB 组成的 U-Net**（左列编码、右列解码，分辨率从 $H\times W\times C$ 降到 $H/2\times W/2\times 2C$、再到 $H/4\times W/4\times 4C$ 的瓶颈），每个 WGSRB = **WEB → SGFB** 串联，末尾再补一个 Scale。左下、中下、右下三个绿框分别放大 WEB（Haar 四子带 → 拼接 → 1×1 压缩 → DWS-HINB → 上采样 → 残差）、SGFB（Sobel 梯度 → DWS-Conv+Sigmoid 出细粒度边权 $M$，再用标量 $\alpha$ 与原始特征加权）和 DWS-HINB（depthwise 可分离卷积 + InstanceNorm + LeakyReLU，1×1 收口）。看图重点：**没有任何 attention、没有任何可学习的小波核**，全部算力花在卷积上——这就是它能压到 0.7 M 参数的原因。

## 核心问题

水下成像里红光衰减远快于蓝绿，导致强烈的蓝绿偏色 + 散射造成的边缘模糊。作者把现有路线的问题归结成三条（p.1–p.2）：

1. **物理/先验方法**（DCP、Sea-Thru、Retinex、白平衡+融合）可解释但脆弱，依赖深度、水体类型、光照的准确估计；
2. **CNN** 感受野有限，严重散射下不够；**Transformer**（UDAFormer、SMDR-IS、UShape）全局建模强但 9–23 M 参数、40+ GFLOPs、40–80 ms，没法实时；
3. **混合方法**（WF-Diff 用傅里叶先验 + 扩散、TCTL-Net 用颜色迁移）效果好但要么昂贵，要么**根本没做显式频域分解**，细结构恢复不可靠。

作者的立论：**用"便宜且可解释的先验"替代"昂贵的容量"**（p.1 Introduction 末段）。

## 关键方法

### 1) 自适应白平衡先验（p.3，式 (1) + Algorithm 1）

先做标准 Gray-World 白平衡，但比教科书版多两步：

$$g = \mu_g/(\mu_c+\epsilon),\quad X\leftarrow X\cdot g,\quad X_{wb}=\exp\big(\log(X+\epsilon)-\overline{\log(X+\epsilon)}\big)$$

即**对数域去均值再指数还原**（抑制增益造成的过曝），最后 min-max 归一到 $[0,1]$（Algorithm 1 第 1–8 行）。然后**不是替换输入，而是可学习融合**：

$$X_{\mathrm{fused}} = \gamma\cdot X_{wb} + (1-\gamma)\cdot X,\qquad \gamma=\{\gamma_R,\gamma_G,\gamma_B\}\in[0,1]$$

作者特意强调：与 UColor（用逆向介质传输图）、ReX-Net（再估一张反射率图）不同，这里**不引入新的估计分支**，因此没有误差传播、也没有额外复杂度（p.3 原文）。

### 2) WEB：固定 Haar 小波的多频带增强（p.3–p.4，式 (2)–(5)）

用 $h=\tfrac12[1,1]$、$g=\tfrac12[1,-1]$ 的外积构造四个固定 2×2 卷积核：

$$H_{LL}=h^\top h,\ H_{LH}=g^\top h,\ H_{HL}=h^\top g,\ H_{HH}=g^\top g$$

拆出 $\{F_{LL},F_{LH},F_{HL},F_{HH}\}$ → **拼接** → 1×1 卷积压缩 → DWS-HINB 精修 $D(\cdot)$ → 上采样 → 与输入残差相加：

$$F_{\mathrm{fused}}=\mathrm{Conv}_{1\times1}(\mathrm{Concat}(\cdot)),\qquad F_{\mathrm{out}}=\mathrm{Upsample}(D(F_{\mathrm{fused}}))+F_{\mathrm{in}}$$

> 我注意到的一点：**这里没有逆小波变换（IDWT）**。四支子带在半分辨率下拼起来、1×1 压回 $C$ 通道，再靠 Upsample 回到原分辨率，等于用 1×1 卷积隐式学了重构。这和 [[AdaptIR 自适应频率增强的一体化图像复原]]（小波→增强→**IDWT**）、[[Light-MambaIR 时频感知稀疏状态空间统一复原]]（稀疏后 IDWT 重构）的处理方式不同，可能是省算力的关键，也没做消融验证。

![[WWE-UIE WEB与SGFB特征图 Fig3.png]]

> **怎么读这张图**（p.4 Fig. 3）：上排是 WEB 内部——$F_{LL}$ 保留整体亮暗与远景轮廓，$F_{LH}/F_{HL}$ 分别是水平/垂直方向纹理，$F_{HH}$ 几乎全是高频噪点与颗粒（水下悬浮物就堆在这里），$F_{out}$ 看得出亮度被拉回、边缘更利。下排是 SGFB 内部——先看 $G(F_{in})$ 那张 Sobel 梯度图，亮的地方就是粒子散射后仍然残留的边缘方向；$F_{out}$ 相对 $D'(F_{in})$ 明显更锐。这张图直接说明"为什么选 Sobel 不用 Laplacian"：**要的是方向，不是二阶过零点**。

### 3) SGFB：Sobel 梯度融合块（p.4–p.5，式 (6)–(10)）

$$F_0=D'(F_{in}),\qquad G(F_0)=\sqrt{(F_0*S_x)^2+(F_0*S_y)^2}$$

$$M=\sigma\big(\mathrm{DWS\text{-}Conv}_{3\times3}(G(F_0))\big),\qquad F_1=\alpha\cdot(M\odot F_0)+(1-\alpha)\cdot F_0,\qquad F_{out}=D''(F_1)$$

即**两级门控**：$M$ 是像素级细粒度边权（Sigmoid），$\alpha\in[0,1]$ 是可学习的标量粗控。作者对 Laplacian 的批评原文很直接：Laplacian 对噪声敏感、会放大粒子造成的波动，"Sobel 强调边缘方向，恰好与小波分解出的子带对齐"（p.4）。

### 4) 损失：五项加权，HVI 色彩空间是新增项（p.5，式 (11)–(16)）

$$L_{total}=1\cdot L_C+0.1\cdot L_{VGG}+0.1\cdot L_{SSIM}+0.1\cdot L_{Edge}+0.5\cdot L_{HVI}$$

$L_C$ 是 Charbonnier；$L_{Edge}=\|G(Y)-G(Y')\|_2$ 用的正是式 (7) 那个 Sobel 幅值；$L_{HVI}=L_C(\mathrm{HVI}(Y),\mathrm{HVI}(Y'))$，理由是 HSV 有红色色相不连续、CIELab 有非线性色相畸变，而 HVI 把色度与强度解耦，反向传播梯度更平滑（p.5）。

## 结论与局限

- **全参考（p.6 Table 1）**：UIEB **24.3209 / 0.9196 全场最高**；LSUI 31.02/0.9310（PSNR 输给 PhaseFormer 的 32.12，但 SSIM 最高）；UFO-120 28.28/0.8668（PSNR 第二，SSIM 最高）；EUVP-Scene 26.59/0.8243（PSNR 第二、SSIM 最高）；EUVP-Dark 22.53（PSNR 最高，SSIM 0.9083 微输 SMDR-IS 的 0.9085）。**没有一项掉出前二**。
- **效率**：0.734 M / 6.251 G / 7.602 ms，对比 SMDR-IS 的 12.25 M / 48.32 G / 36.59 ms（参数少 16.7×，快 4.8×）、PhaseFormer 的 1.78 M / 13.04 G / 40.66 ms。轻量变体 Ours-E 降到 0.471 M / 3.603 G / 7.111 ms，PSNR 只掉 0.40 dB（23.9176）。
- **非参考（p.7 Table 2）**：Challenging-60 上 NIQE **5.9575 最低（最好）**、UCIQE 0.5784；U45 上 UCIQE **0.6110 最高**、NIQE 4.1057。但 URanker 在**两个数据集上都不是第一**（CDF-UIE 1.4501 / NU2Net 1.9096），作者的解释是"别人指标在数据集间波动大，我们更均衡"。
- **颜色保真**：Color-Check7 上 CIEDE2000 **10.0979，全场最低（最好）**，优于 CDF-UIE 的 10.3654 与 UDAFormer 的 10.4503（p.7）。
- **消融，模块（p.8 Table 4）——这个顺序值得记**：去掉 **SGFB 的梯度分支掉得最多**（24.3209 → 23.9047，−0.416 dB）；去掉 **WEB 掉 SSIM 最多**（0.9196 → 0.9134，−0.0062，PSNR 只掉 0.10 dB）；去掉白平衡先验 −0.139 dB / −0.0021 SSIM；把 WGSRB 里的顺序换成 **SGFB→WEB 反而更差**（24.06），证明"先多频带提特征、再梯度修边界"的顺序有意义。作者总结：白平衡贡献结构相似度，梯度分支贡献像素精度。
- **消融，损失（p.8 Table 3）**：去掉 $L_{HVI}$ PSNR 掉最多（−0.436 dB，24.32→23.885），去掉 $L_{SSIM}$ 掉 SSIM 最多；**Fig. 8 显示 $L_{Edge}$ 与 $L_{HVI}$ 合用收敛更快，而换成 $L_{HSV}$ / $L_{CIELab}$ 不但不够、甚至比只用 $L_C$ 还差**——这是全文最实用的一条工程结论。
- **作者承认的局限**：只有一句"更均衡"来解释 URanker 不是第一；跨域能力（LOL-v2 低照、Foggy Cityscapes 去雾）只在 Fig. 9 给了视觉效果，**没有任何数字**（p.8）。
- **我观察到的局限（非原文）**：① 所谓"实时"只在 RTX 3090 上测（p.6），没有 Jetson/手机端实测，而 UIE 的实时场景恰恰在移动平台；② 全部实验都是 256×256，高分辨率水下视频能不能维持 7.6 ms 未知；③ Haar 核是**固定**的，与 [[LWTformer 可学习小波变换的古汉字图像修复]] 的可学习 DWT 相比谁更划算，论文没做这个消融；④ 白平衡支路本身是个手工先验，若融合权重 $\gamma$ 真的只有 3 个标量，那"自适应"的含金量有限（见下方疑问）。

## 我的疑问

- [ ] 式 (1) 写的是 $\gamma=\{\gamma_R,\gamma_G,\gamma_B\}$（3 个逐通道标量），但 Fig. 2 里画的是 "Adaptive Weights Generation ∈ (0,1)" 且输入标着 $H\times W\times 3$ —— $\gamma$ 到底是 3 个标量还是一张空间权重图？如果只是 3 个标量，那 Table 4 里"去掉白平衡先验"掉的那 0.14 dB，主要该归功于对数域增益本身，而不是"可学习融合"。
- [ ] WEB 为什么不还原 IDWT？直接 Upsample 出来的分辨率存在混叠，作者没有给出"上采样 vs IDWT"的对比。
- [ ] $\alpha$ 是**每个 SGFB 一个标量**还是全网络共享？论文只说"learnable scalar"，没说初始化值。
- [ ] Sobel 梯度图是**从输入特征算**的，随着训练走它会自我强化——会不会在严重浑浊场景里把浮游颗粒也当成"边缘"放大？Fig. 3 的 $G(F_{in})$ 看起来颗粒确实很亮。
- [ ] UCIQE / NIQE / URanker 三个非参考指标在这篇和对手论文里给出的"最优"互相矛盾（例如 U45 的 NIQE 最好的是 PhaseFormer 3.9684），这套指标的可信区间到底多大？

## 关键摘录

> "Instead of introducing a new estimation branch or replacing the input entirely, we apply a learnable fusion of the original image $X$ and its white-balanced counterpart $X_{wb}$."（p.3）

> "Unlike Laplacian filters that are noise-sensitive and prone to amplifying particle-induced fluctuations, Sobel emphasizes edge orientation and just perfectly aligns with wavelet-decomposed subbands, making it better suited for underwater images where scattering blurs edges but preserves directional cues."（p.4）

> "Extensive experiments on both reference and non-reference benchmarks show that our method achieves competitive restoration quality with substantially fewer parameters and faster inference."（p.8，结论段）

## 相关笔记

- [[小波变换]] —— 整个小波主线的方法汇总，本文用的是**固定 Haar 四子带、且不做 IDWT**：
- [[Light-MambaIR 时频感知稀疏状态空间统一复原]] —— 同样固定 Haar 拆 LL/LH/HL/HH，但用低频出系数去调制高频、再稀疏重构，与本文"拼接后 1×1 压回 + 上采样"形成对照
- [[AdaptIR 自适应频率增强的一体化图像复原]] —— 级联小波扩感受野 + **完整 IDWT** 闭环；本文省掉逆变换换效率
- [[LWTformer 可学习小波变换的古汉字图像修复]] —— 把 DWT 变成可学习下采样，是"固定核 vs 可学习核"这条线的另一极
- [[StarIR 空间-频率融合的卷积图像复原]] —— 同为轻量空间-频率卷积路线，其在水下 LSUI 上 +3.53 dB，是可以直接对表的对象
- [[各种Norm层的摆放]] / [[PointWise和DepthWise]] —— WEB/SGFB 里反复出现的 DWS-HINB 就是"depthwise 可分离卷积 + InstanceNorm"的组合，可对照这两篇看摆放方式
