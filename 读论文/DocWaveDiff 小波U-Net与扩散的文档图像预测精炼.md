---
created: 2026-11-07
source: E:\.dsh-paper-reader\data\小波变换\Marulli和Bertini - 2026 - DocWaveDiff A Predict-and-Refine approach for Document Image Enhancement with Wavelet U-Nets and Di.pdf（WACV 2026 Open Access，pp. 8511–8520）
type: 论文
tags: [论文, 小波变换, 扩散模型, 图像复原, Attention]
---

# DocWaveDiff：小波 U-Net + 扩散的文档图像"预测–精炼"增强

> Matteo Marulli、Marco Bertini（佛罗伦萨大学）。下文页码 = 本机 PDF 页码；原文印刷页从 8511 起（p.1 = 8511）。

## 一句话总结

把 DocDiff 式的"残差扩散"整条搬进**小波 U-Net** 作骨干：前级 Early Predictor 只吃低频、先出一张"不糊但不锐"的图，后级 Denoiser Refiner 用**确定性 DDIM 式反向链**专门预测高频残差，两者相加得最终图；在文档去模糊与手写擦除（inpainting）上都刷到 SOTA，且直接提升下游 OCR（p.1 摘要、p.8 结论）。

## 核心问题

- OCR 与版面分析模型基本只在**干净图**上训练，遇到模糊扫描件、纸面手写批注，性能断崖式下降（p.1）。
- 传统阈值/卷积核方法与 GAN 系（含 DeblurGAN、DocEnTr）在**保留文本细节与高频**上不够，边缘易出伪影；纯 L1/L2 训练只对 PSNR 友好，视觉发虚；GAN 又容易在高分辨率下模式崩塌（p.1–p.2）。
- 小波 + predict-and-refine 这个组合在文档复原里"基本没人做"，本文补这个空（p.2 末）。

## 关键方法

### 0. 整体：两段式 predict-and-refine（p.3，Fig. 2）

$$\hat I = E_\theta(I),\qquad I_{res} = I_{GT} - \hat I,\qquad \hat I_{res} = D_\theta(I_t, t, E_\theta(I)),\qquad \text{out} = \hat I + \hat I_{res}$$

- **Early Predictor $E$**：负责低频、整体结构与抹掉不想要的内容（重建背景、填充字符内部）。它自己恢复不了高频，去模糊时边缘发糊（p.3 §3.1、式(1)）。
- **Denoiser Refiner $D$**：条件扩散模型，学的是**残差的分布**而不是图，只负责高频（边缘、笔画、纹理）（p.3 §3.2、式(2)）。
- 两网**端到端联合训练**，不是先训 E 再冻结。可选输入一张先验特征（inpainting 时就是 mask）（p.3、Fig. 2）。

![[DocWaveDiff 整体 predict-refine 框架 Fig2.png]]

**怎么读 Fig. 2**：上支路是 Early Predictor，输入为退化图 +（可选）先验（黑底白字的 mask 图），出灰底纹的初版；下支路是 Refiner Denoiser，输入为**噪声** + 初版图的低频，出黑白的残差图；右下那个"圆圈加号"节点把两者相加才是最终输出。关键就在这个相加——后面消融会看到，inpainting 上这一支几乎不贡献（p.8）。

### 1. 扩散部分：确定性反向链（p.3–p.4，式(3)–(12)）

前向加噪是标准 DDPM 马尔可夫链，用重参数化直接跳到任意 $t$：$I_t = \sqrt{\bar\alpha_t}I_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$（式(6)）。反向把方差置零 $\beta_t = 0$（式(8)），即走 **DDIM 的确定性路径**：

$$I_{t-1} = \sqrt{\bar\alpha_{t-1}}\hat I_{res} + \sqrt{1-\bar\alpha_{t-1}}\cdot\frac{I_t - \sqrt{\bar\alpha_t}\hat I_{res}}{\sqrt{1-\bar\alpha_t}} \qquad \text{(12)}$$

好处是**不要随机采样**：推理更快、结果稳定（p.4）。注意这里的"确定性"指采样阶段不加噪声，不是单步直出——仍然要跑满 $T$ 步。

### 2. 小波：只当上下采样器用（p.4，式(13)–(16)）

2D Haar 一层分解出 $I_{LL}$（低频近似）与 $I_{LH}, I_{HL}, I_{HH}$（水平/垂直/对角高频），四个 2×2 固定正交可分离核（原文直接给出矩阵形式）：

$$LL=\begin{bmatrix}1&1\\1&1\end{bmatrix},\ LH=\begin{bmatrix}-1&-1\\1&1\end{bmatrix},\ HL=\begin{bmatrix}-1&1\\-1&1\end{bmatrix},\ HH=\begin{bmatrix}1&-1\\-1&1\end{bmatrix}$$

卖点写得很直白：**DWT 可逆无损、IDWT 无参数**，替掉池化/步长卷积和转置卷积，降分辨率时不丢高频（p.4、p.5）。

### 3. Wavelet U-Net 的两处非模板设计（p.5，Fig. 3）

![[DocWaveDiff Wavelet U-Net 架构 Fig3.png]]

**怎么读 Fig. 3**：左端输入是"初版图 + 噪声"在通道维拼接（图中那个 "c" 圆圈 = level-channel concatenation）；橙色块 DWT(Haar) 是下采样、粉色块 IDWT(Haar) 是上采样，整个 U 形靠这两者而不是池化/转置卷积；蓝色 Res-Block 每级都有，暗红 AttentionBlock **只出现在最后两级分辨率**（此时特征图已经很小，省显存）；U 底是 MiddleBlock，黄色 Dilated Conv2D 连续四个 = **膨胀率 2/4/8/16** 的堆叠，用来不加深度地扩感受野，后面还接一个注意力 + 一个残差单元；灰色点线是 skip connection，紫色点线是从左下 Time embedding 逐级注入时间步条件。

- 编码器把四子带**按通道拼接**后整体送入 DownBlocks，让低频结构与高频细节一起建模（p.5）。
- 自注意力只在最后两级开，其余层纯残差（p.5）。
- 解码器逐级 IDWT 上采样 → 与同级编码特征 concat → UpBlock（p.5）。

### 4. 损失：频域分离训练（FST）（p.5–p.6，式(17)–(21)）

$$\mathcal L_{pixel} = \mathbb E\lVert I_{GT}-\hat I\rVert_2,\quad \mathcal L_{low} = \mathbb E\lVert\phi_L*(\hat I - I_{GT})\rVert_2,\quad \mathcal L_{high} = \mathbb E\lVert\phi_H*(I_0 - D_\theta(I_t,t,\hat I))\rVert_2$$

$$\mathcal L_{total} = \beta_1(\mathcal L_{pixel}+\beta_0\mathcal L_{low}) + (\mathcal L_{DM}+\beta_0\mathcal L_{high}),\qquad \beta_0 = 2,\ \beta_1 = 0.5$$

即：低频损失逼 E 管结构，高频损失逼 D 管细节，两边各自加权。$\beta$ 取值直接沿用 DocDiff（p.6）。训练/采样完整流程见 **Algorithm 1**（p.6），采样最后一行 `return Î + I0` 就是"初版 + 残差"。

**实现**：论文 p.6 脚注写明代码与权重 available_upon_acceptance，**未找到可用实现**，故本节无代码对照。

## 结论与局限

- **Inpainting（EnsExam，手写擦除）**：patch 级 MSE 0.0003 / MS-SSIM 0.9817 / PSNR 40.37 dB，全面超过 NetRefine(0.06/0.9647/35.44)、EnsNet、EraseNet、MTRNet++（p.7 Table 1）；**整图推理**也保持优势（MSE 0.0002 / 0.9849 / 39.54 dB，p.7 Table 2）。
- **Deblurring（TDD）**：LPIPS 0.0211、SSIM 0.9733、PSNR 26.28 dB，参数仅 **8.2M**（与 DocDiff 同量级，p.7 Table 3）。作者在结论里也承认 PSNR 只是"competitive"——**次于 DocRes 的 27.35 dB**，赢在 LPIPS/SSIM 与参数量（p.8）。
- **OCR 收益**：Tesseract 上 CER 从模糊图的 0.7874 → 0.3554（DocDiff 0.3710），WER 0.9349 → 0.6495（DocDiff 0.6623）；深度学习 OCR（db-resnet50 + crnn-vgg16bn）CER 0.6422 → 0.1316（DocDiff 0.1320）、WER 0.8071 → 0.3204（p.8 Table 5）。
- **消融给了很诚实的结论，也是全文最有信息量的部分**（p.8 Table 4、§4.3.1）：
  - Inpainting 上**只有 mask 是命门**：去掉 mask，MS-SSIM 0.9849→0.9427、PSNR 39.54→27.22 dB、MSE 0.0003→0.0026；而**去掉 Refiner 结果几乎不变**（0.9849 / 39.53 dB / 0.0002）——说明 EnsExam 上 Early Predictor 一个人就吃下了任务，分数已近天花板；小波换成步长卷积也几乎无差（0.9831 / 39.53 dB）。
  - Deblurring 上**小波与注意力都是关键**：去小波 SSIM 0.9733→0.9608、PSNR 26.28→24.63 dB、LPIPS 0.0211→0.0299；去注意力掉最多（SSIM 0.9366、PSNR 22.44 dB、LPIPS 0.0468）。
  - 去 Refiner 时 **PSNR 反而升到 27.09 dB，但 LPIPS 从 0.0211 恶化到 0.0447** —— 教科书级的感知–失真权衡：残差扩散买的是"看起来锐"，代价是像素对齐略差。
- 我的观察（非原文）：Table 3 里 DocEnTr 一行数值（LPIPS 0.9130 / SSIM 0.2225）与常识量级明显对调；p.8 正文描述 OCR 时把"0.1316 CER（深度学习 OCR）"与"0.6495 WER（Tesseract）"混在一句里比较，属笔误；式(17)(18) 前把 $\phi_H$ 称作低通、$\phi_L$ 称作拉普拉斯高通，与随后公式里的用法恰好相反——按公式用法理解即可。

## 我的疑问

- [ ] $\hat I$ 在 $\mathcal L_{DM}$ 里被 detach，但 $I_t$ 又由 $I_{GT}-\hat I$ 构造，梯度到底通不通到 $E$？p.6 的表述自相矛盾，需看代码确认。
- [ ] Inpainting 上 Refiner 几乎无用，那这篇在擦除任务上的实际贡献等于"小波版 U-Net + mask"；作者未解释为何仍保留残差扩散分支。
- [ ] mask 由 $|I-I_{GT}|$ 过 Otsu 生成（p.6 式(22)），推理时必须**已知 GT**——真实场景从哪来这张 mask？文中未讨论。
- [ ] 参数量都是 8.2M，但推理步数 $T$ 没报，实际耗时无法与 DocDiff/DocRes 比较。

## 关键摘录

> "In contrast, wavelets are important: without them, SSIM and PSNR decrease and LPIPS worsens... Finally, removing the denoiser-refiner increases the PSNR (26.28→27.09 dB) but significantly worsens perception (LPIPS 0.0211→0.0447), revealing the usual trade-off between L2 and perception."（p.8）

> "Notably, removing the denoiser-refiner leaves results essentially unchanged (MS-SSIM 0.9849; PSNR 39.53dB; MSE 0.0002), suggesting that on the EnsExam dataset the early predictor already solves most of the task and the refiner offers little headroom at these near-ceiling scores."（p.8）

## 相关笔记

- [[HDW-SR 小波高频引导扩散的单图超分]] — 同一套"Haar 当上下采样 + 残差扩散"的组合拳，但那边把高频子带拿去做**跨注意力引导**，这边只是当骨干的池化替代
- [[DTWSR 扩散Transformer建模多级小波谱的单图超分]] — 另一种把扩散搬进小波域的做法，可与本文"仍在像素域做残差扩散"对照
- [[LWTformer 可学习小波变换的古汉字图像修复]] — 同为文档/古文字图像复原，但那篇把 DWT 做成可学习下采样
- [[小波变换]] — 本库小波类论文的总览页
- [[预测LF残差提升HF，LF协同]] — 低频先出、高频靠残差的同构思路
