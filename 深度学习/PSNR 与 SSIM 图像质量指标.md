---
created: 2026-11-26
updated: 2026-11-26
source: 综合 Wang et al. 2004 (SSIM 原文)、Understanding SSIM (2006.13846)、Hitchhiker's Guide to SSIM (2101.06354)、Hore & Ziou 2013 等整理
type: 概念
tags: [概念, 图像复原, 超分辨率]
---

# PSNR 与 SSIM 图像质量指标

## 是什么

两者都是**全参考（full-reference）**图像质量指标：必须有一张「标准答案」参考图，才能跟待评价图比较。[6]

| 指标 | 衡量什么 | 取值范围 |
|---|---|---|
| **PSNR** | 逐像素误差的**绝对值大小**（基于 MSE） | 0 ~ ∞ dB，越高越好 |
| **SSIM** | 亮度、对比度、**结构**的相似程度 | $[-1,1]$，越接近 1 越好 |

一句话区别：**PSNR 只看「差多少」，SSIM 还看「差在哪」。**

## 一、PSNR：从 MSE 来的对数比值

### 定义

先算均方误差（MSE）：两张 $M\times N$ 图逐像素差值的平方平均。[6]

$$\mathrm{MSE}=\frac{1}{MN}\sum_{i=1}^{M}\sum_{j=1}^{N}\big(f_{ij}-g_{ij}\big)^2$$

PSNR 是「最大可能像素值的平方」与 MSE 的比值，取对数换成分贝：[5]

$$\boxed{\;\mathrm{PSNR}=10\log_{10}\!\left(\frac{\mathrm{MAX}^2}{\mathrm{MSE}}\right)\;}$$
$$\mathrm{PSNR}=20\log_{10}\!\left(\frac{\mathrm{MAX}}{\sqrt{\mathrm{MSE}}}\right)$$

$\mathrm{MAX}$ 是像素最大值：8 bit 图取 **255**，10 bit 取 1023，归一化到 $[0,1]$ 则取 1。[5]

### 三个关键性质

**① 用对数是因为动态范围太大。** 实际 MSE 跨越好几个数量级，线性刻度看不出差别。[5]

**② MSE 为 0 时 PSNR 无定义（趋于 ∞）。** 两张图完全相同时没有有限值，代码里通常要加极小量防止除零。[5]

**③ 经验刻度（8 bit）**：好的有损压缩通常在 **30–50 dB**；**高于 40 dB 多数人肉眼看不出差别**。[5]

### 为什么它不够

PSNR 有个致命弱点：**它假设所有像素误差同等重要**。

这意味着下面三种失真可能得到相同的 PSNR：
- 整体亮度偏移 1
- 均匀模糊
- 结构被破坏但均值方差不变

而人眼对它们的感受完全不同——人眼对**结构**变化最敏感。[6]

## 二、SSIM：拆成亮度 + 对比度 + 结构

### 核心思想

Wang 等人的出发点是：**图像的结构信息应当独立于亮度和对比度**。同一物体在不同光照下结构不变。[1]

所以 SSIM 把相似性拆成三个独立比较：[1]

$$\mathrm{SSIM}(x,y)=\big[l(x,y)\big]^{\alpha}\big[c(x,y)\big]^{\beta}\big[s(x,y)\big]^{\gamma}$$

三个分量分别是：

**① 亮度比较**（用均值 $\mu$）

$$l(x,y)=\frac{2\mu_x\mu_y+C_1}{\mu_x^2+\mu_y^2+C_1}$$

**② 对比度比较**（用标准差 $\sigma$）

$$c(x,y)=\frac{2\sigma_x\sigma_y+C_2}{\sigma_x^2+\sigma_y^2+C_2}$$

**③ 结构比较**（用协方差，本质是相关系数）

$$s(x,y)=\frac{\sigma_{xy}+C_3}{\sigma_x\sigma_y+C_3}$$

几何上，结构项就是两个去均值向量**夹角的余弦**。[1]

### 合并后的实用形式

取 $\alpha=\beta=\gamma=1$ 且 $C_3=C_2/2$（这样对比度项与结构项可以合并），得到最常用的公式：[1][2]

$$\boxed{\;\mathrm{SSIM}(x,y)=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}\;}$$

### 常数怎么取

$$C_1=(K_1L)^2,\qquad C_2=(K_2L)^2$$

其中 $L$ 是像素动态范围（8 bit 取 255），$K_1=0.01$、$K_2=0.03$。[2]

这些常数只为**防止分母为零**，不影响量级。[1]

### 局部窗口与 MSSIM

SSIM 不是全图算一个数，而是**逐局部窗口**算（原文用 $11\times 11$ 高斯窗，$\sigma=1.5$），再取平均：[2]

$$\mathrm{MSSIM}=\frac{1}{wh}\sum_{x}\sum_{y}\mathrm{SSIM}(x,y)$$

这也是为什么 SSIM 能给出「**质量图**」来定位失真位置——这在 PSNR 上做不到。[3]

## 三、两者关系与对比

### 有解析关系，但不等价

Hore & Ziou 推导了灰度图上 SSIM 与 PSNR 的解析关系，证明二者**相关但不等价**：给定 PSNR 无法唯一确定 SSIM，反之亦然。[4]

### 对比表

| | PSNR | SSIM |
|---|---|---|
| 基础 | MSE | 局部均值/方差/协方差 |
| 与人眼感知 | 相关性弱 | 相关性更强 |
| 是否逐像素 | 是（全图统一） | 否（局部窗口） |
| 能否定位失真 | 否 | **能**（质量图） |
| 计算成本 | 极低 | 低（高斯卷积） |
| 对模糊/结构破坏 | 不敏感 | 敏感 |
| 对亮度偏移 | 敏感 | 相对不敏感 |

### 为什么会给出相反的排序

这是实际做实验最常遇到的困惑：**PSNR 高的模型，SSIM 或人眼效果可能更差**。

原因在于 PSNR 惩罚所有像素误差，而**过度平滑**的输出（模糊）会让 MSE 很小、PSNR 很高，但结构信息全丢了——SSIM 会掉下来。[6]

所以生成式模型（GAN、扩散）常常"PSNR 不占优但看起来更好"，这是**指标与感知目标不一致**，不是模型有问题。

## 四、在超分/复原领域的具体问题

### 它们衡量的是「像素保真度」，不是「任务可用性」

在场景文本超分（STISR）里这个问题被反复指出：

- **GLYPH-SR** 指出 PSNR/LPIPS/MUSIQ 等指标**对字符级错误不敏感**，等于把文本当普通纹理
- **Reading the Unreadable** 计算相关系数发现：PSNR/SSIM 与文本复原质量**无正相关**，LPIPS/DISTS 甚至方向相反

### 论文里的典型矛盾

STISR 论文中常见「PSNR/SSIM 更高但识别率更低」或反之。例如 DPMN 在 TATT 上加模块后 PSNR/SSIM 略降，但识别率提升——作者通常解释为「识别率才更有意义」。

### 实践建议

| 场景 | 该看什么 |
|---|---|
| 压缩、去噪等保真任务 | PSNR + SSIM 够用 |
| 自然图像超分 | 加 LPIPS / DISTS（感知指标） |
| 生成式复原（GAN/扩散） | 必须加感知指标，PSNR 会误导 |
| **文本图像复原** | **必须看识别率（OCR Acc）**，PSNR/SSIM 只能作参考 |
| 需要定位失真 | 用 SSIM 质量图，不用 PSNR |

## 五、常见误区 / 注意事项

**① PSNR 无上界，不能说「PSNR 95%」。** 它是 dB 值，不是百分比。

**② SSIM 可以是负数。** 理论上 $[-1,1]$，负值需要刻意构造，实践中罕见。[4]

**③ 报告 PSNR 必须说明在哪算。** 彩色图可以在 RGB 三通道平均算，也可以只算亮度通道 Y——两者数值不同，必须写明。[5]

**④ 不同论文的 PSNR 不可直接横比。** 取决于：是否裁剪边界、是否只算 Y 通道、是否用同一参考图、评测集切分是否一致。

**⑤ SSIM 对参数敏感。** 窗口大小、高斯 $\sigma$、$K_1/K_2$ 都会影响数值，虽然主流实现已统一，但自定义实现容易出偏差。[2]

**⑥ 「PSNR 更高」不等于「更好」。** 尤其在生成式任务上，这可能是过度平滑的结果。

## 判断口诀

- 想要**快速、可复现、可横比**的基线指标 → **PSNR**
- 想要**更贴近人眼**、还能**看到哪里坏了** → **SSIM**（质量图）
- 任务是**感知质量**（生成、超分观感） → 必须补 **LPIPS / DISTS**
- 任务是**下游可用性**（文本识别、检测） → 直接看**下游指标**，别用 PSNR/SSIM 代替

## 相关笔记

- [[文本聚焦损失]] — STISR 中同时报告 PSNR/SSIM 与识别率的典型场景
- [[感知损失 实时风格迁移与超分辨率]] — 感知指标（LPIPS 一类）的由来，正是为了补 PSNR/SSIM 的短板
- [[低分辨率场景文本识别 LR-STR 方法谱系]] — 该领域 PSNR/SSIM 与识别率矛盾的具体案例
- [[低分辨率 STR 论文清单 2024-2026]] — 含对 PSNR/SSIM 失效的实证
- [[特征融合方法谱系]] — 同为「指标/方法如何选」类概念笔记

### 待建笔记

以下概念本文涉及但库中尚无独立笔记：LPIPS、DISTS、MUSIQ、MANIQA、CLIP-IQA（感知/无参考指标族）。

## Sources

[1] https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf — Wang et al. 2004 — Image Quality Assessment: From Error Visibility to SSIM
[2] https://arxiv.org/pdf/2006.13846 — Nilsson & Akenine-Möller 2020 — Understanding SSIM
[3] https://arxiv.org/pdf/2101.06354 — Hitchhiker's Guide to Structural Similarity 2021
[4] https://digital-library.theiet.org/doi/pdf/10.1049/iet-ipr.2012.0489 — Hore & Ziou 2013 — PSNR vs SSIM analytical relationship
[5] https://technav.ieee.org/topic/psnr/ — IEEE Technology Navigator — PSNR definition
[6] https://www.mathworks.com/help/images/image-quality-metrics.html — MATLAB — Image Quality Metrics