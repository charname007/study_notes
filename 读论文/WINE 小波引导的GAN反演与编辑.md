---
created: 2026-11-07
source: E:\.dsh-paper-reader\data\小波变换\Kim 等 - 2025 - WINE  Wavelet-Guided GAN Inversion and Editing for High-Fidelity Refinement.pdf（WACV 2025，CVF 开放版，pp. 4523–4532）
type: 论文
tags: [论文, 小波变换, 频域, GAN反演]
---

# WINE：用小波给 GAN 反演去低频偏置

> Kim, Moon, Park，*WACV 2025*。下文页码均指本机 PDF 页码（PDF 第 1 页 = 论文集第 4523 页）。

## 一句话总结

作者指出 GAN 反演里通用的像素损失（$L_1$/$L_2$）本质上**对四个小波子带等权**，而实测低频 LL 的误差比三个高频子带大 30 倍，所以优化几乎全花在低频上；WINE 用**小波损失**（在多级分解的高频子带上做 $L_1$/$L_2$）加上**小波融合**（不经生成器卷积、直接在 SWAGAN 的小波系数上做尺度-偏移调制）把高频细节补回来，是首个从频域解释 GAN 反演的工作（p.1、p.4–p.5）。

## 核心问题

GAN 反演要把一张真实图片压进 StyleGAN 的隐空间再重建出来，用于属性编辑。低码率反演（pSp/e4e/ReStyle）把 $1024\times1024\times3$ 的图压成 $18\times512$ 的 $w$，只有原图维度的约 0.3%，按信息瓶颈理论必然丢细节（p.1）。高码率反演（HyperStyle 调生成器、HFGI/StyleRes 提残差特征）能降失真，但作者发现它们降的只是**低频子带**的失真（p.1–p.2）：

- Fig.2(b) 上，HFGI 的 $L_{2,LL}$ 明显低于 ReStyle，但 LH/HL/HH 三个高频子带"几乎没降甚至上升"（p.2）；
- 原因被归结为两条：**训练上**，$L_2$ 可写成四个子带 $L_{2,f}$ 的等权和（式 1），$L_1$ 同理（式 2，p.4），而实际误差分布严重偏向 LL，等权就等于偏心；**结构上**，HFGI 的特征融合只施加在 64×64 分辨率（$\ell_f=7$），按 Shannon–Nyquist 其可承载的最高频率只有 $f_{nyq}=64\sqrt{2}$，远低于 1024 图像（p.5）。

顺带一个动机证据：Fig.2(c) 用隐向量优化直接比较两个骨干，SWAGAN（在小波域逐级生长生成）的谱密度贴合 GT 到 1.0 的 $f_{nyq}$，StyleGAN2 在半频之后断崖式下跌（p.2）。

## 关键方法

![WINE 训练框架 Fig3](WINE%20训练框架%20Fig3.png)

**怎么读 Fig.3**（p.4）：右上角是冻结的两段——编码器 $E_0$（e4e）出低码率 $w$、SWAGAN $G_0$ 出朴素反演 $\hat X_0$，红框标的是"源图 $X$ 与最终 $\hat X$ 之间算的 $\mathcal L^{K=3}_{2,wave}$"。左下角是残差线：$\Delta = X-\hat X_0$ 先被 RandAugment 人为打散成 $\tilde\Delta$（因为训练时没有"保细节的编辑图"可用，只能自监督造错位），再由可训练的 **ADA** 对齐回 $\hat\Delta$，对齐损失就是橙色框里的多级小波损失 $\mathcal L^{K=3}_{1,wave}(\Delta,\hat\Delta)$——注意画的是三次 $\Delta*LL^{(i)}$ 逐级下采样后各自比高频。$\hat\Delta$ 随后分两路出 scale/shift：$\mathcal F_{feat}$ 走**绿色**的特征融合（在 $\ell_f=7,9$ 两层），$\mathcal F_{wave}$ 走**橙色**的小波融合（在 $\ell_w=11$ 层，DWT 拆 LL/LH/HL/HH → $\otimes g_{wave}$ → $\oplus h_{wave}$ → IWT 再回特征）。图例：红箭头是加小波损失的位置，锁头是冻结模块，$\otimes/\oplus$ 是逐元素乘/加。

### 1) 小波损失：把"等权"改成"只算高频、且覆盖更宽的频带"（p.4–p.5）

先证明偏置的来源。$L_2$ 可拆成子带和，且 $\lambda_f=1$ 对所有 $f$ 相同：

$$L_2(I_1,I_2)=\sum_{f\in\mathbb F_l\cup\mathbb F_h}\lambda_f L_{2,f}(I_1,I_2),\qquad \lambda_f=1,\ \forall f \tag{1}$$

$L_1$ 在像素差 i.i.d. 服从 $\mathcal N(\mu\approx0,\sigma^2)$ 的假设下也有对数形式的等权分解（式 2）。既然等权、而 $L_{2,LL}$ 实际大 30 倍，作者主张"让高频的 $\lambda_f$ 更大"，于是只取高通滤波器集合：

$$\mathcal L_{p,wave}(I_1,I_2)=\sum_{f\in\mathbb F_h}\mathcal L_{p,f}(I_1,I_2),\qquad p\in\{1,2\} \tag{3}$$

但一次分解只覆盖 $f_{nyq}/2$ 以上的频段，而"很大一部分图像细节其实落在半频以下"，所以再用多级小波分解把 LL 反复下分，得到覆盖 $[f_{nyq}/2^K,\ f_{nyq}]$ 的版本（式 4），训练时取 $K=3$（p.5）：

$$\mathcal L^K_{p,wave}(I_1,I_2)=\sum_{i=1}^{K}\sum_{f\in\mathbb F_h}\mathcal L_{p,f}\big(I_1*LL^{(i)},\ I_2*LL^{(i)}\big) \tag{4}$$

用在哪两处：最终重建 $\mathcal L^K_{2,wave}(X,\hat X)$；以及对 ADA 的对齐损失加权（$\lambda_{wave,ADA}=0.1$）：

$$\mathcal L_{wave,ADA}(\Delta,\hat\Delta)=\mathcal L_1(\Delta,\hat\Delta)+\lambda_{wave,ADA}\mathcal L^K_{1,wave}(\Delta,\hat\Delta) \tag{5}$$

总损失 = 式 (5) + 以 $L_2$、$\mathcal L^K_{2,wave}$、ID 损失、LPIPS 加权和的图像重建项（式 6，p.5）。整篇训练只训 ADA + 两个融合模块，$E_0$ 与 $G_0$ 全程冻结（p.4 Fig.3 注）。

### 2) 小波融合：绕开生成器卷积，直接改小波系数（p.5）

HFGI 式的特征融合是 $F_{\ell+1}=g_{feat}\cdot \mathrm{ModConv}(F_\ell,w_\ell)+h_{feat}$（式 7），信息必须穿过预训练的 StyleGAN 层，细节仍会退化。WINE 先把融合层从 64 提到 64 与 128 两处（$\ell_f=\{7,9\}$，理论上把 $f_{nyq}$ 翻倍），但承认"单纯提分辨率解决不了问题"，于是借 SWAGAN 逐级上采样时**显式生成小波系数**的结构，在 $\ell_w=11$ 层直接调制：

$$\hat W_w=g_{wave}\cdot \mathrm{tWavelets}(F_w)+h_{wave} \tag{8}$$

分工是经验性的：**特征融合管粗形状（早层 $\ell_f=7,9$），小波融合管细纹理（晚层 $\ell_w=11$）**（p.5）。

## 结论与局限

- **反演指标全面领先**（CelebA-HQ 全测试集，1024²，p.6 Table 1）：$L_2$ 0.011 vs StyleRes 0.018 / HFGI 0.023；高频 $L_{1,wave}$ 0.230 vs 0.301 / 0.351；LPIPS 0.051 vs 0.076 / 0.091；SSIM 0.753 vs 0.708 / 0.661；ID 0.906 vs 0.894 / 0.864。作者强调"学到了最接近 GT 的频率分布且不掉身份"（p.6）。
- **消融是关键设计都不可省**（Table 2，p.7）：把 HFGI 换成 StyleGAN2 基线是 0.124 / 0.023 / 0.653；只把小波损失加到 StyleGAN2 上（A）SSIM 就涨到 0.703；只把骨干换成 SWAGAN（B）几乎无收益（0.659），说明"换生成器"本身不够；C（SWAGAN+小波损失）把 ADA 的 $L_1(\Delta,\hat\Delta)$ 拉低约 19% 到 0.100；D（再加小波融合、但不带小波损失）SSIM 虽到 0.717 却出现**鬼影**，作者归因于 ADA 没对齐好；E（全上，即 WINE）才同时把 $L_2$ 从 0.024 砍到 0.011、SSIM 到 0.753 —— 即**小波损失与小波融合是协同关系，缺一不成立**。
- **编辑场景**用 InterFaceGAN 与 StyleCLIP 各跑六种属性，定性上"重建质量与可编辑性的平衡"最好（p.6 Fig.5），并额外在 AFHQ 动物域验证泛化（p.6 Fig.6）。
- **局限（作者未单列 limitation，以下多为我的观察，已标注）**：(a) 编辑结果的**定量**对比不在正文，作者指向附录（p.6 原文）；(b) 方法绑定 SWAGAN 骨干，而消融 B 说明现有反演结构是为 StyleGAN 拟合的，换骨干本身要付代价，推广到别的小波生成器需重训；(c) 只在人脸（FFHQ/CelebA-HQ）与动物（AFHQ）两个域验证，没有通用自然图像或文本图像实验（我的观察）；(d) ADA 依赖 RandAugment 造错位的自监督，D 配置的鬼影说明对齐质量对小波融合是硬前提（我的推断）。

## 我的疑问

- [ ] 式 (1)(2) 的"等权"是分析工具而非实现——优化器在等权损失下自然会多分配梯度给 LL，这是否等价于"$\lambda$ 相同导致偏置"的因果？换成按子带能量归一化的 $L_2$ 会不会有同样效果？
- [ ] 小波损失本质是重加权高频，和 focal frequency loss / 直接 FFT 加权（HFGI 的改进版）差在哪？附录里应该有谱损失对照，待补。
- [ ] $K=3$、$\lambda_{wave,ADA}=0.1$、融合层 $\{\ell_f=7,9;\ \ell_w=11\}$ 都是经验值，附录的敏感性曲线怎么走？
- [ ] 小波融合在 $\ell_w=11$ 层做，此时空间分辨率已较低，为什么仍能救回 1024 级别的细节？是否因为 SWAGAN 的系数本身跨分辨率耦合？

## 关键摘录

> "we look into the widely used pixel loss in GAN inversion, revealing its predominant focus on the reconstruction of lowfrequency features."（p.1，Abstract）

> "While the high-rate inversion models apparently decrease $L_{2,f}$ for $f\in\mathbb F_l$, they marginally decrease or even increase $L_{2,f}$ for $f\in\mathbb F_h$, compared to ReStyle."（p.2）

> "we empirically find that $L_{2,LL}$ of the previous GAN inversion models have around 30 times larger value than $L_{2,f\in\mathbb F_h}$, on average. This indicates that $L_2$ predominantly focuses on the low-frequency information."（p.4）

> "According to the Shannon-Nyquist theorem, the image $I\in\mathbb R^{H\times W}$ ($H=W=l$), cannot store information with the frequency range higher than $f_{nyq}=\sqrt{H^2+W^2}=l\sqrt{2}$. Consequently, the upper bound of information frequency by the feature fusion of HFGI is $f_{nyq}=64\sqrt{2}$, which is relatively lower than the image size, 1024."（p.5）

> "we empirically find that feature fusion is helpful for reconstructing the coarse shape, while wavelet fusion is for the fine details."（p.5）

## 相关笔记

- [[AdaptIR 自适应频率增强的一体化图像复原]] — 同样从"小波子带"入手，但落在复原网络的感受野与动态核频谱调制上
- [[StarIR 空间-频率融合的卷积图像复原]] — 空间分支 ⊗ 频率分支的 star 相乘融合，可与本文"特征融合 vs 小波融合"分工对照
- [[DTWSR 扩散Transformer建模多级小波谱的单图超分]] — 多级小波谱 + 高低频双解码器，同样是"低频主体 / 高频残差"的分工
- [[预测LF残差提升HF，LF协同]] — 频带间协同的另一条论证路径
- [[关于联合小波变换的增强]] — 小波域增强的零散整理
- [[LWTformer 可学习小波变换的古汉字图像修复]] — 把小波变换本身做成可学习下采样
