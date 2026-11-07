---
created: 2026-11-07
source: arXiv:2511.01175v2 (4 Nov 2025)；E:\.dsh-paper-reader\data\小波变换\Du 等 - 2025 - Diffusion Transformer meets Multi-level Wavelet Spectrum for Single Image Super-Resolution.pdf
type: 论文
tags: [论文, 小波变换, 超分辨率, 扩散模型, Transformer]
---

# DTWSR 扩散 Transformer 建模多级小波谱的单图超分

## 一句话总结
把**扩散 Transformer（DiT）的整条去噪链搬进小波谱域**：每个采样步都把噪声图转成 3 级 Haar 小波谱，用金字塔 tokenization + 双解码器（LEDec / HDDec）联合去噪 LF 与各级 HF 子带，再逆小波变回像素图——是第一个在真正 3 级小波谱上跑 DiT 做 SISR 的工作，动机是"现有 DWT-SR 方法把各级 HF 子带当独立任务，忽略了多尺度频率之间的相互关系"（p.1、p.3）。

## 核心问题
单图超分要同时要保真度（LF 主导）和感知质量（HF 细节）。已有小波域方法（Wavelet-SRNet、WaveFace、WaveDM、WFEN 等）把多级 HF 系数**各自独立**预测，导致重建图出现不一致与非自然伪影（p.1、p.2）。作者的核心论点是：空间拓扑之外，**子带之间的相互关系**同样决定感知质量，而这恰好是 Transformer 的长程建模能力擅长的（p.3）。

> 原文（p.1）："existing methods generally treat each level of HF coefficients independently, without considering the interrelations among the multi-scale HF sub-bands."
>
> 原文（p.3）："to our knowledge, we are the first to model image's multi-scale wavelet spectra by using the basic transformer architecture for SR task."

## 关键方法

### 整体：条件扩散在小波谱上做
条件去噪 $p_\theta(I_{t-1}|I_t, I_{lr}) = \mathcal{N}(I_{t-1}; \mu_\theta(I_t,t,I_{lr}), \Sigma_\theta(I_t,t,I_{lr}))$（式 5，p.3）。关键差别在于：**不是**在像素域给 $I_t$ 去噪，而是先做 J 级 MDWT 得到 $I^{fre}_{t,3}$，在谱域去噪后 IMDWT 回像素域（p.3、图 2a）。
- MDWT 记法：$x^1_L,x^1_V,x^1_H,x^1_D = \mathrm{DWT}(I)$，对 $x^1_L$ 递归分解（式 1–2），把各级子带重排成 J 级小波谱 $I^{fre}_J = \mathrm{MDWT}(I,J)$（式 3，p.3）。
- 优化沿用 DDGAN：加时间相关判别器做对抗训练（式 12），再加 $L_1$ 的**像素 + 频率双域重建项** $\mathcal{L}_{pixel}=\|\tilde I_0-I_0\|$、$\mathcal{L}_{fre}=\|\tilde I^{fre}_{0,J}-I^{fre}_{0,J}\|$，总目标 $\mathcal{L}^G=\alpha\mathcal{L}^D_{adv}+\beta\mathcal{L}_{pixel}+\gamma\mathcal{L}_{fre}$（式 13–14，p.5）。

![[DTWSR 架构图 Fig2.png]]

**怎么读这张图**：上半 (a) 是采样链——最左边纯噪声 $I_T$，中间灰框就是"每步把图转成小波谱、去噪、再转回来"的那一步，右侧逐步回到 $I_0$；下半 (b) 是去噪网络 WSDT 的内部：左侧 Pyramid tokenizer 把 LR、LF、三级 HF 分别切成 token，中间 LEDec 只处理 LF 主体，HDDec 处理所有 HF 与 LF 残差，右侧 Pyramid detokenizer 拼回谱图。

### 1. 金字塔 tokenization（p.4，式 6–7，图 3a）
HF 系数稀疏，所以**不要**全用同一 patch 大小。LF 用最小 patch $p_{min}$，第 j 级 HF 用 $p^j = p_{min}\times 2^{J-j}$（式 6），这样各级子带的**感受野保持一致**，token 数还大幅下降：704 token vs 像素域 DiT 的 1040 token，只有三分之二（表 6，p.8）。
- 各子带拼接后分别过 Conv2d 嵌入：$f_{lr}=\mathrm{Conv2d}_{lr}(I_{lr})$、$f^J_L=\mathrm{Conv2d}_L(x^J_L)$、$\mathbf F^j_H=\mathrm{Conv2d}^j_H(\mathbf X^j_H)$（式 7）。

![[DTWSR 金字塔tokenization与4D位置编码 Fig3.png]]

**怎么读这张图**：(a) 左边是 LR 与四种 patch 大小示意——红/绿框是 $p$（LF 与最细 HF），黄框是 $2p$（更粗一级），越稀疏的高频子带框越大；(b) 是位置编码，token 除了二维绝对位置还被额外标上级号 $j$ 与子带号 $d \in \{x_L{=}0,x_V{=}1,x_D{=}2,x_H{=}3\}$，凑成四元组 $[j,d,Pos_h,Pos_w]$。

### 2. 4D 位置编码（p.4，图 3b）
金字塔切分让各子带 token 落在**相同的二维绝对坐标**上，这天然方便建模跨子带关系；为了还区分"第几级""哪条子带"，加上 $j$ 与 $d$，得到 4D 位置 $[j, d, Pos_h, Pos_w]$，用标准 ViT 的 sine-cosine 编码后加到 patch embedding 上。

### 3. 双解码器 LEDec / HDDec（p.5，式 8–9，图 4）
LF 平滑高能量、HF 稀疏——方差分布差异太大，**统一解码器会互相拖累**（消融里 Freq-DiT 的 Deg./FID 明显更差，p.8）。
- **LEDec**（LF Elementary Decoder）：输入 $[f_{lr}, f^J_L]$，用 in-context conditioning 注入 LR，配合掩码 $M_{low}$ 防止噪声 LF token 污染 LR 条件，时间步经 AdaLN-Zero 注入：$\tilde f_{lr},\tilde f^J_{Le}=\mathrm{LEDec}([f_{lr},f^J_L],M_{low},t)$（式 8）。
- **HDDec**（HF Detail Decoder）：输入 $[\tilde f_{lr},\tilde f^J_{Le},\mathbf F_H]$，同时去噪多级 HF 与 **LF 残差** $\tilde f^J_{Lr}$——注意 $x^J_L$ 里仍残留少量 HF 成分，所以 LF 主体（LEDec）≠ LF 子带（p.2）：$\hat f_{lr},\tilde f^J_{Lr},\tilde{\mathbf F}_H=\mathrm{HDDec}([\tilde f_{lr},\tilde f^J_{Le},\mathbf F_H],M_{high},t)$（式 9）。
- $M_{high}$ 的两条刻意约束（p.5）：① **LR token 对 LF token 不可见**，逼 LF 靠 HF 残差对齐、而不是靠 LR 走捷径；② **LF token 对 HF token 不可见**，防止 LF 压过 HF 的细节生成。
- 反 tokenize：LF 侧 $\tilde f^J_{Le}+\tilde f^J_{Lr}$ 相加后经 $\mathrm{FC}_L$ 解码，HF 各支经 $\mathrm{FC}^j_H$ 解码（式 10），重排回谱图后 IMDWT 得 $\tilde I$。

![[DTWSR TransBlock与注意力掩码 Fig4.png]]

**怎么读这张图**：(a) 是 DiT 的 TransBlock——两个残差分支，注意力与 MLP 前各挂一组 AdaLN 的 Scale / Shift 调制（参数由时间步 MLP 给出），另外还有一组可学的 (bias, scale) 直接作用在输入 token 上；(b)(c) 的两张黑块图分别是 LEDec 的 $M_{low}$ 与 HDDec 的 $M_{high}$，**黑=被 mask 掉**：$M_{low}$ 里 LR 可以看 LF、LF 看不到 LR；$M_{high}$ 里 LF 看不到 LR、HF 看不到 LF。

## 结论与局限

- **8× 人脸超分（16²→128²，CelebA-HQ 100 张，表 1，p.6）**：PSNR 24.09 / SSIM 0.71 / Cons. 0.50 / Deg. 53.85 / FID 56.77。Cons./Deg./FID 三项全场最优；PSNR 最高的是 WFEN 25.53，但其 FID 高达 106.34——作者的解释是 WFEN 只用 MSE 训练、估计的是后验均值而不考虑数据分布，输出过平滑。
- **4× 通用场景（DIV2K 验证集）**：PSNR 28.18 / SSIM 0.79 / LPIPS 0.097 / Cons. 1.151，在生成式方法里 LPIPS 与 WGSR(0.096) 基本持平、Cons. 优于 SRDiff/WGSR（表 2，p.7）。
- **Manga109 / Set14 / Set5（4×，表 3，p.7）**：PSNR 27.79 / 26.58 / 29.47 dB，三者都是同表生成式方法里最高的；SSIM 在 Manga109（0.865）与 Set14（0.725）最优，Set5 的 0.846 略低于 ResDiff 的 0.854。
- **大倍率 12×/16×（16²→192²、16²→256²）**：Cons. 从 IDM 的 8.16/9.57 降到 0.64/0.89，FID 60.37（16×）优于 IDM 的 63.99（表 4，p.7）。
- **真实世界（RealSR）**：MUSIQ 64.04、LIQE 3.79、NIQE 3.55、PI 3.49 均最优，NRQM 6.70 略低于 SinSR 的 6.72（表 5，p.7）。

**消融（8× 人脸，表 6/表 7，p.8）——哪些设计真有用：**
- 把去噪搬到谱域（Freq-DiT vs Pixel-DiT）：token 从 1040 → 704，PSNR 23.94 → 24.02，FID 61.37 → 60.35。
- 双解码器 DTWSR(a)：FID 立刻从 60.35 → 58.53，但 Cons. 反而变差（0.582 → 2.35）——**拆开解码器必须配 LF 残差来对齐**。
- 加 LF 残差 DTWSR(b)：Cons. 修回 0.549，但 HF 被 LF 过度影响，Deg./FID 退步到 56.19 / 60.01。
- 再加 $M_{high}$（完整版）：Cons. 0.503、FID 56.77、Deg. 53.85 全项最优——收益来自**"对齐 LF/HF"与"防止 HF 被压"这两件事要一起做**。
- 换成等 patch 大小 DTWSR(c)：token 回到 1040，FID 反而恶化到 66.05——金字塔切分是效率与质量双赢。
- **频率相互关系本身**（表 7）：人为切断 HF 子带之间的交互，PSNR 24.09→23.99、FID 56.77→61.67，掉得最狠的是 FID，直接印证了论文的核心动机。

**局限（作者未正面列出，我按全文判断）：**
- 训练成本不低：人脸 250 epoch（约 0.5M iter，bs=32）、通用 300 epoch（约 0.8M iter，bs=32）（p.6），且基于 DDGAN 的对抗采样，推理也要多步。
- 主结果表里 **PSNR 并不是第一**（如 4× DIV2K 上 HAT 29.83 vs 本文 28.18），作者对保真度指标的追求让位于感知质量，属于 diffusion-SR 的通病。
- 只验了 Haar 小波、J=3 这一配置，小波基与级数与效果的敏感性没有消融。
- 未做真实退化之外的盲复原实验，"泛化能力"实际只用 RealSR 一个数据集佐证。

## 我的疑问
- [ ] 为什么 HF 子带之间"交互"掉点主要体现为 FID 而 PSNR 几乎不动？是否说明相互关系的收益全在纹理感而非几何一致性？
- [ ] $M_{high}$ 让 LR 对 LF 不可见——那 LF 的**内容对齐**完全交给 HDDec 侧的 $\tilde f_{lr}$ 与 LF 残差，是否会在大倍率下丢失身份信息？表 4 的 Deg. 反而更好，值得单独验证。
- [ ] 训练是 DDGAN 的对抗框架，去掉对抗项（纯 $L_{pixel}+L_{fre}$）会掉多少？论文没给这个消融。
- [ ] 金字塔 patch $p^j=p_{min}\times2^{J-j}$ 只保证"名义感受野"一致，但 HF 子带本身是稀疏的高斯状响应，是否最优？

## 关键摘录
> "Despite some DWT-based methods improving SR by capturing fine-grained frequency signals, most existing approaches neglect the interrelations among multi-scale frequency sub-bands, resulting in inconsistencies and unnatural artifacts in the reconstructed images."（p.1）

> "It should be noted that the elementary contents from LEDec is not equal to LF sub-band. The LF sub-band still has HF components, though quite few."（p.2）

> "LR tokens are invisible to LF tokens to avoid LR conditioning in HDDec to be used for LF sub-band denoising and force the realignment between LF and HF sub-bands. LF tokens are invisible to HF tokens to prevent the influence from LF tokens on HF."（p.5）

> "when the model ignores the interrelations among HF sub-bands, performance decreases across all metrics. The drop is particularly notable in FID, indicating significant degradation in image textural details."（p.8）

## 相关笔记
- [[预测LF残差提升HF，LF协同]] —— 同论文的碎片笔记，正好对应本文 LF Residual + $M_{high}$ 那条线
- [[金字塔tokenization对多级小波变换]] —— 同论文的碎片笔记，对应式 6 与图 3
- [[LWTformer 可学习小波变换的古汉字图像修复]] —— 另一条路：把小波核本身做成可学习的下采样
- [[AdaptIR 自适应频率增强的一体化图像复原]] —— 级联小波扩感受野 + 频域调制，非扩散路线
- [[StarIR 空间-频率融合的卷积图像复原]] —— 纯卷积的空间/频域 star 融合
- [[PFT 渐进聚焦 Transformer 单图超分]] —— 同为 Transformer SISR，但走"先选后算"省 FLOPs 的路线
