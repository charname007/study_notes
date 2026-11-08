---
created: 2026-11-08
source: ICCV 2025 (arXiv:2508.09824v2)；E:\.dsh-paper-reader\data\默认专题\Huang 等 - 2025 - Reverse Convolution and Its Applications to Image Restoration.pdf；https://github.com/cszn/ConverseNet
type: 论文
tags: [论文, 图像复原, 卷积算子, 频域, 逆问题]
---

# ConverseNet 反向卷积算子与图像复原

## 一句话总结

Huang 等（南京大学 + 香港理工 + OPPO 研究院，ICCV 2025，第 1 页）指出**转置卷积并非卷积的数学逆**（它只是"插零上采样 + 标准卷积"，且会带来棋盘伪影，第 1–2 页），于是首次提出 **Converse2D**——把"从深度卷积输出 Y 反推输入 X"写成带二次正则的最小二乘问题，在**圆形边界假设下推出 FFT 闭式解**（式 6，第 3–4 页），单步、非迭代、支持任意通道数与 s 倍上采样；再配 LayerNorm + 1×1 卷积 + GELU 组成类 Transformer 的 **Converse Block**，直接替换 DnCNN / SRResNet / USRNet 的对应层得到 ConverseNet 三变体，在去噪、×4 超分、去模糊上验证其作为通用基础模块的有效性（第 1、6、8 页）。

## 核心问题

卷积用于特征提取/下采样，转置卷积用于上采样，二者是神经网络的基础算子（第 1 页）。但：

- **转置卷积不是真逆**：数学上等于"在输入元素间插零，再做标准卷积"，常因核重叠不均产生 checkerboard artifacts；PyTorch 等框架也没有原生"反卷积算子"（第 1–2、3 页）。
- **去模糊反卷积（deconvolution）不够通用**：多需迭代优化、通常只作用于单通道灰度或 3 通道 RGB 图；Wiener 反卷积虽有闭式解但同样假设已知模糊核（第 1、3 页）。
- **可逆卷积（invertible conv）是受约束的标准卷积**：要求输入输出同形、Jacobian 可算，主要用于流模型/可逆网络，并非新算子（第 1–2 页）。

因此作者的目标是：求 $X = F(Y, K, s)$，即从 $Y=(X\otimes K)\downarrow_s$ 反解原始特征图 $X$——这是一个典型的**病态逆问题**（式 1–2，第 3 页）。

## 关键方法

### 1. 正则化最小二乘 + FFT 闭式解（第 3–4 页）

目标函数（式 3–4），第二项是稳定病态解的先验：

$$X^*=\arg\min_X \|Y-(X\otimes K)\downarrow_s\|_F^2 + \lambda\|X-X_0\|_F^2,\qquad X_0\in\{0,\;\mathrm{Interp}(Y,s)\}$$

在**圆形边界条件**下有闭式解（式 6）：$X^*=\mathcal{F}^{-1}\big(\frac{1}{\lambda}(L-\overline{F_K}\odot_s \frac{(F_K L)\Downarrow_s}{|F_K|^2\Downarrow_s+\lambda})\big)$，其中 $L=\overline{F_K}\,F_{Y\uparrow_s}+\lambda F_{X_0}$。$s=1$ 时退化为经典 Wiener 形式 $X^*=\mathcal{F}^{-1}\big(\frac{\overline{F_K}F_Y+\lambda F_{X_0}}{|F_K|^2+\lambda}\big)$。**核 K 可端到端学习，也可作为已知先验直接条件注入；$\lambda$ 也可联合优化**（第 4 页）。算子按 Conv2D 命名习惯叫 **Converse2D**，论文 Algorithm 1 给了 PyTorch-like 伪代码（第 4 页）。

💻 官方实现见 [[代码细节笔记/ConverseNet 的 Converse2D 算子实现]]。核心一行（`cszn/ConverseNet@368aa39 models/util_converse.py`，真实读取）：

~~~python
self.biaseps = torch.sigmoid(self.bias - 9.0) + self.eps   # 式(9) 的 lambda，逐通道可学
FX = (FR - FCBinvWBR) / self.biaseps                       # 式(6) 的 (L - ...)/lambda
out = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))         # 回空域
~~~

### 2. 实现四要素（第 4–5 页，消融表 1–3）

| 设计 | 结论 | 锚点数字（Set12 / BSD68 PSNR dB） |
|---|---|---|
| **核初始化** | Softmax 归一化（非负、和为 1）> 高斯 > 均匀 | 30.31 → 30.60 → **30.70** / 28.92 → 29.30 → **29.36**（表 1，第 5 页） |
| **核尺寸** | 5×5 是精度/复杂度折中，7×7 几乎不涨 | 5×5 30.70/29.36 vs 7×7 30.71/29.36（表 1） |
| **Padding** | circular 最好，尺寸几乎无影响 | Zero 30.63 → **Circular 30.70**；0×0 30.67 → 4×4 **30.70**（表 2，第 5 页） |
| **$\lambda$ 与 $X_0$** | $\lambda=\mathrm{Sigmoid}(b-9)+\epsilon$，b 逐通道可学、初值 0 使 $\lambda$ 很小（初期强数据保真）；$X_0=\mathrm{Interp}(Y,s)$ 明显优于 0 | 去噪 0→Y：30.66→**30.70**；超分 0→Interp：32.22→**32.25**（表 3，第 5 页） |

**消融最有料的洞察**：padding **模式**比 padding **尺寸**重要得多（circular 与闭式解的圆形边界假设自洽）；$X_0$ 用 $Y$ 的插值不仅稳住优化，还保住去噪需要的高频细节；核从 5×5 加到 7×7 只 +0.01 dB， justify 了默认 5×5。

### 3. Converse Block 与 ConverseNet（第 4、6 页）

Converse2D 是 **depthwise** 的、不含跨通道交互，故用 1×1 卷积补通道混合：块结构 = LayerNorm → 1×1 Conv → GELU → **Converse2D（只管空间依赖）** → GELU → 1×1 Conv，配残差；再串一个 LayerNorm → 1×1 → GELU → 1×1 子块（Fig 2，第 4 页）。这种"空间/通道解耦"正是 Transformer 块的模块化思路。

![[ConverseNet 反向卷积 Fig1 算子结构对比.png]]

> **怎么读 Fig 1**（p.1 图注 + 第 2 页）：三个子图各画一张输入特征图（橙）与输出特征图（蓝）。(a) 标准卷积、(b) 转置卷积都是**把核作用在输入上**；(c) 反向卷积反过来**把核作用在输出上以重建输入**——这就是全文"reverse"的字面含义，也是它与转置卷积最直观的区别。

![[ConverseNet 反向卷积 Fig2 Converse Block.png]]

> **怎么读 Fig 2**（p.4 图注）：绿色虚框是 s=1 时一个 Converse2D 算子的三组输入 Y、核 K、输出 $X^*$；灰色虚框是重卷积结果 $X^*\otimes K$ 与误差图 $Y-X^*\otimes K$——误差小才说明式(3) 被真正求逆了，输出 $X^*$ 带有增强的结构细节。主干那串 LayerNorm / 1×1 Conv / GELU / Converse2D / GELU / 1×1 Conv 就是上面说的双残差块。

三个应用网络（Fig 3，第 6 页）：

![[ConverseNet 反向卷积 Fig3 三网络架构与表4.png]]

> **怎么读第 6 页整页（含 Fig 3 与表 4）**：Fig 3(a) Converse-DnCNN = 1×1 Conv → 20 个 Converse Block → 1×1 Conv；(b) Converse-SRResNet = 1×1 Conv+GELU → 若干 Converse Block 与 Converse2D+GELU 上采样块交替 → 1×1 Conv + Interpolate；(c) Converse-USRNet = 迭代模块里 **Conv 1×1 → Converse2D（数据模块，吃 KernelNet 给的核）→ Converse Block 去噪先验 → 1×1**，K 经 KernelNet 嵌入成 64 通道核并在特征域条件化。同页表 4 是去噪主结果（见下）。默认配置：5×5 核、circular padding 4、$X_0=\mathrm{Interp}(Y,s)$（第 6 页）。⚠️ 注：官方 released 去噪代码实际用的是 7×7 核、padding 6，与论文 §5 的"默认 5×5"不一致，见 [[代码细节笔记/ConverseNet 的 Converse2D 算子实现]]。

## 结论与局限

- **去噪（表 4，第 6 页）**：$\sigma=25$，Set12/BSD68 上 Converse-DnCNN **30.70 / 29.36 dB**，一致超过把 Converse2D 换回 depthwise 卷积的 Conv-DnCNN（30.64/29.30）与换成转置卷积的 ConvT-DnCNN（30.61/29.29）；原版 DnCNN 30.43/29.23 作参考（参数更少，作者明说重点比的是算子而非刷 SOTA，第 6 页）。视觉上 butterfly 图 23.53 dB 略高于 DnCNN 23.49（Fig 4，第 7 页），且**端到端训练不引入反卷积常见的边界伪影**（第 6 页）。
- **×4 超分（表 5–6，第 7 页）**：Set5/Set14/BSD100/Urban100 = **32.25/28.72/27.62/26.24**，与 Conv-/ConvT-SRResNet 及原版 SRResNet（32.21/28.60/27.59/26.09）相当；把上采样模块换成 Converse2D 也与 PixelShuffle / 最近邻 / 转置卷积打平（表 6 最优 32.26/28.78/27.66/26.36）。作者结论是"**可替换而不掉点**"，而非超越（第 7 页）。
- **去模糊（表 7，第 8 页）**：这是优势最明显的任务。非盲设定 Converse-USRNet 在 BSD100 五核平均 **34.62 dB** vs Conv-USRNet 34.33；盲设定（去掉 KernelNet）ConverseNet **27.31** vs ConvNet 23.98，**+3.33 dB**，作者归因于 Converse2D 内在的反卷积能力（第 8 页）。Fig 6：ConvNet 把直线墙壁弄弯，ConverseNet 保持结构（第 8 页）。关键设计差异：USRNet 在**图像域**做反卷积+去噪，Converse-USRNet 在**特征域**、且能在特征层直接条件化模糊核（第 8 页、第 2 页贡献 4）。
- **作者承认/我观察到的局限**：① 全文定位是"验证算子可作为基础模块"，实验用**简化设置**（去噪只 $\sigma=25$、超分只 ×4 bicubic、模糊核只 7×7 运动/高斯 + 噪声 2.55），目标是 comparable/superior 而非 SOTA（第 6、8 页）；② 闭式解依赖**圆形边界假设**，真实边界仍需 padding 策略补救（第 3、5 页）；③ 只做了深度（depthwise）反向卷积，未推广到普通/分组卷积（第 1、3 页）；④ 只测了三个复原任务，未报告相对 DnCNN/SRResNet 的推理耗时/显存同口径对比（参数从 557k 涨到 735k，第 6 页表 4）；⑤ 未来工作才计划扩到大规模视觉模型与图像合成/复杂退化生成任务（第 8 页）。

## 我的疑问

- [ ] 式(6) 的 $\Downarrow_s$ 是"对 $s\times s$ 块取平均"，对应的是标准 s 倍下采样（取左上角像素）——若实际网络用的是别的下采样（如 stride 卷积带重叠），这个闭式解还精确吗？论文第 3 页只声明了"standard s-fold downsampler"。
- [ ] 核 K 用 Softmax 强制非负且和为 1，等于把逆问题限制成"模糊核"族；这会不会反而限死算子的表达力？若放开到一般实值核，$\lambda$ 的逐通道学习还稳吗？
- [ ] Converse Block 里两次残差、1×1 升维到 2C 再压回 C，与 Transformer FFN 同构；但 Converse2D 本身是"求逆"，把它放在 GELU 之后是否破坏式(3) 的线性卷积假设？论文没讨论非线性夹在中间的数学含义。
- [ ] 论文 §5 说默认 5×5 核，官方 `converse_dncnn.py` 却用 `kernel_size=7, padding=6`；超分/去模糊代码又是 3×3/padding 2。哪个才是表 4–7 的真实配置？复现时应以哪份为准？
- [ ] 盲去模糊 +3.33 dB 很亮眼，但 ConvNet/ConverseNet 是"去掉 KernelNet"的简化版；与真正的 blind deblurring SOTA 比还有多少差距，论文没给。

## 关键摘录

> "transposed convolution is not the true inverse of convolution; rather, it can be described as performing upsampling by inserting zeros between input elements, followed by a standard convolution."（p.1）

> "while standard and transposed convolution apply the kernel to the input, reverse convolution applies the kernel to the output to reconstruct the input."（p.2，解释 Fig 1）

> "By formulating the problem as a quadratic regularized minimization, we obtain a closed-form solution that enables non-iterative computation."（p.2，贡献 1）

> "This involves inverting the combined effect of convolution and downsampling, which is a typical ill-posed problem."（p.3）

> "Softmax-based initialization outperforms both uniform and Gaussian initialization in the Gaussian denoising task ... a 5 × 5 kernel offers a favorable trade-off between performance and model complexity."（p.5）

> "Our Converse-SRResNet achieves performance comparable to the other models. This indicates that Converse2D can effectively replace its counterpart operators without reducing performance."（p.7，作者自己给的定调）

> "USRNet performs deconvolution and denoising directly in the image domain, whereas Converse-USRNet operates in the feature domain."（p.8）

> "This work provides a principled alternative to transposed convolution and opens new directions for designing learnable inverse operators in deep architectures. Future work includes extending this operator to large-scale vision models and generative tasks such as image synthesis and restoration under complex degradations."（p.8）

## 相关笔记

- [[PointWise和DepthWise]] —— Converse2D 是 depthwise 的（缺跨通道交互），1×1 Conv 补通道混合，正是 DW 管空间 / PW 管通道的翻版
- [[感知损失 实时风格迁移与超分辨率]] —— 同用"步幅卷积/上采样"讨论网络内分辨率变化，可对照转置卷积的替代方案
- [[像素重排：PixelUnshuffle 与 PixelShuffle]] —— 表 6 里与 Converse2D 对比的上采样基线之一
- [[StarIR 空间-频率融合的卷积图像复原]] —— 同样在频域（FFT）做算子、且同为"纯 CNN 复原"路线，可对照频域算子的两种用法
- [[Wave-ViT 小波可逆下采样的多尺度ViT]] —— 另一条"可逆/可逆下采样"思路（DWT 而非求逆卷积）
- [[代码细节笔记/ConverseNet 的 Converse2D 算子实现]] —— 官方 `cszn/ConverseNet@368aa39` 逐行对照与论文/代码差异
