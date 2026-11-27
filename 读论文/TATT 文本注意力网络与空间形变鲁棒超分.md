---
created: 2026-11-26
updated: 2026-11-26
source: E:\Zotero\storage\GLZ458GB\Ma 等 - 2022 - A Text Attention Network for Spatial Deformation Robust Scene Text Image Super-resolution.pdf（CVPR 2022, arXiv:2203.09388）
type: 论文
tags: [论文, 场景文本识别, 超分辨率, Transformer, Attention]
---

# TATT：文本注意力网络与空间形变鲁棒超分

## 一句话总结

**问题定位很精确**：现有 STISR 全是 CNN，而卷积是**局部算子**，无法应对旋转/弯曲文本造成的**大范围位置变化**。TATT 用 Transformer 的**全局注意力**把文本先验对齐到图像空间位置，再用一条**三元结构一致性损失**逼模型在形变下保持稳定。

## 核心问题

先前的 STISR（如 TSRN、TPGSR）在规则文本上已经不错，但**遇到旋转和弯曲文本就失效**——输出的字模糊且语义错误。[1][3]

作者把原因归到算子性质上：CNN 的卷积核是**locality-based**，感受野有限，捕捉不到形变带来的**大位置偏移**。[1]

对 TPGSR 的批评尤其具体：它**只是用卷积把文本先验和图像特征合并**，这使先验只能在很小的局部范围内起作用，**先验的价值被浪费了**。[1][3]

所以作者的判断是：需要一个**globality-based** 算子（即注意力）来建长程关联。[1]

## 关键方法

### 整体流程

```
LR 图 ─┬─→ TPG（文本先验生成器）→ f_P ─┐
       │                                ├─→ TP Interpreter → f_TM ─┐
       └─→ 9×9 Conv → f_I ──────────────┘                          ├─→ 5×TPGB → PixelShuffle → SR
                                                                   f_I ┘
```

文本先验 $f_P$ 来自识别模块，字符集为 0–9、a–z 加 blank，共 **37 类**。[1]

### ① TP Interpreter：全文核心

它的任务是把**语义域**的文本先验，对齐到**空间域**的图像特征上，产出调制图 $f_{TM}$。[1]

$$f_{TM}=\mathrm{TPI}(f_P,f_I)$$

分两部分：

**Encoder（编码器）**：对文本先验做**多头自注意力（MSA）**，让每个字符感知其它字符，得到上下文增强特征 $f_E$。[1]

关键细节：因为文本先验是**并行**处理的，模型不知道字符顺序，所以要先加**固定正弦位置编码（FPE）**。[1]

**Decoder（解码器）**：做**多头交叉注意力（MCA）**，让语义域去查询空间域。[1]

$$CA_i(f_{Ei},f'_{Ii})=\mathrm{SM}\!\left(\frac{(f'_{Ii}W_i^{\alpha})(f_{Ei}W_i^{\beta})^T}{\sqrt{d_k}}\right)(f_{Ei}W_i^{\gamma})$$

注意 query 来自**图像特征** $f'_{Ii}$，key/value 来自**文本先验** $f_{Ei}$——即「每个图像位置去文本里找自己该是什么字」。[1]

多头结果拼接后投影：

$$\mathrm{MCA}=\odot(CA_0,\dots,CA_{n-1})W^o$$

**解码器用了循环位置编码 RPE** 而非固定编码，因为图像特征在**水平方向有序列依赖**，RPE 能更好编码这种相邻上下文。[1]

**为什么必须用注意力**：MCA 让文本先验的**每一个语义元素**与空间域**每一个位置**建立关联，从而强化语义有意义的区域。[1]

消融给出的对比最有说服力（Table 1）：[1]

| 对齐/引导策略 | 识别率 | PSNR | SSIM |
|---|---|---|---|
| 无文本先验 | 41.4% | 21.42 | 0.7690 |
| Pixel-Shuffle + 相加 | 46.2% | 20.58 | 0.7683 |
| Pixel-Shuffle + SFT | 47.9% | 20.72 | 0.7560 |
| 反卷积 + 相加 | 50.6% | 21.10 | 0.7819 |
| 反卷积 + SFT | 49.6% | 20.87 | 0.7783 |
| **TP Interpreter** | **52.6%** | **21.52** | **0.7930** |

作者的解读很关键：那些策略**都没有做「文本语义与图像特征的全局关联」**，所以先验引导是弱的；只有 TPI 能同时拿到好的语义上下文和准确的区域对齐。[1]

**注意力热图验证**：对应前景字符的区域权重最高，且邻域也被高亮——说明它能感知邻近上下文。[1]

### ② TSC Loss：三元结构一致性

**动机**：即使有 TPI，CNN 对形变文本特征的表达能力仍不如规则文本，重建结果**字符结构较弱、对比度偏低**。[1]

**做法**：构造三张图，强迫它们结构一致：[1]

- $D\mathcal{F}(Y)$：先把 LR 超分，再施加随机形变
- $\mathcal{F}(DY)$：先施加形变，再超分
- $DX$：把 HR 真值施加同样的形变

其中 $D$ 是随机形变，包括**旋转 ±10°、切变、长宽比 0.5–2.0**。[1]

**关键的数学创新是把 SSIM 从二元扩展成三元（TSSIM）**：

$$\mathrm{TSSIM}(x,y,z)=\frac{(\mu_x\mu_y+\mu_y\mu_z+\mu_x\mu_z+C_1)(\sigma_{xy}+\sigma_{yz}+\sigma_{xz}+C_2)}{(\mu_x^2+\mu_y^2+\mu_z^2+C_1)(\sigma_x^2+\sigma_y^2+\sigma_z^2+C_2)}$$

对照二元 SSIM $\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}$，可以看到把「两两配对」推广成了「三对求和」。[1][5]

$$\mathcal{L}_{TSC}(X,Y;\mathbf{D})=1-\mathrm{TSSIM}\big(D\mathcal{F}(Y),\mathcal{F}(DY),DX\big)$$

**TSC Loss 是通用的，这点很实用**——它不只对 TATT 有效（Table 2）：[1]

| 模型 | 无 TSC | 有 TSC | 增益 |
|---|---|---|---|
| TSRN | 41.4% | 45.7% | **+4.3** |
| TBSRN | 48.1% | 49.4% | +1.3 |
| TPGSR | 49.8% | 50.6% | +0.8 |
| TATT | 51.6% | 52.6% | +1.0 |

特别值得注意的是：**TBSRN 声称自己就针对多方向文本鲁棒，但加上 TSC 仍能提升 1.3%**。[1]

### ③ 总损失

$$\mathcal{L}=\mathcal{L}_{SR}+\alpha\mathcal{L}_{TP}+\beta\mathcal{L}_{TSC}$$

$\mathcal{L}_{SR}$ 是 L2，$\mathcal{L}_{TP}$ 是 L1 + KL 散度，$\alpha=1$、$\beta=0.1$。[1]

## 结论与局限

### 主结果（TextZoom）

评测集为 TextZoom 的三个子集（easy 1,619 / medium 1,411 / hard 1,343），训练集约 17,367 对。[4][1]

| 指标 | 数值 |
|---|---|
| PSNR | **21.52**（对比 PCAN 21.49、TSRN 21.42） |
| SSIM | **0.7930**（对比 PCAN 0.7752） |
| ASTER 识别 | 63.6% avg（easy 78.9 / medium 62.7 / hard 44.5） |
| CRNN 识别 | **52.6%** avg（对比 TPGSR-3 的 51.8%） |

**一个亮点**：TATT 用**单阶段**就超过了 TPGSR 的**三阶段**（52.6% vs 51.8%）。[1]

**速度**：960 fps，与 TBSRN（982）和 TPGSR（1085）相当，但识别率高 2.7% / 3.6%。[1]

**参数量**：14.9M，代码已开源。[1][2]

### 空间形变专项评测（Table 6）

作者从 TextZoom 手工挑出 **804 个旋转/弯曲样本**单独评测：[1]

| 模型 | ASTER | MORAN | CRNN |
|---|---|---|---|
| TSRN | 46.6% | 43.8% | 35.2% |
| TBSRN | 48.5% | 45.1% | 37.3% |
| TPGSR | 46.6% | 45.3% | 40.2% |
| **TATT** | **51.7%** | **47.3%** | **43.8%** |

在形变样本上**领先幅度比常规样本更大**——这正是论文的核心主张。[1]

### 退化鲁棒性（Table 7）

在对比度降低（CO）、高斯模糊（GB）、高斯噪声（GN）下，TATT 也全面领先。最极端的是 **GN 下 CRNN**：Bicubic 仅 5.5%、TSRN 11.5%、TBSRN 15.8%、TPGSR 18.1%，TATT **21.1%**。[1]

### 诚实点：论文自曝的局限

**① 极端模糊仍失败。** 原话说 "has limitation on recovering extremely blurry texts"，此时**笔画互相粘连、难以分离**。[1]

**② 计算复杂度随文本长度指数增长。** 因为用了全局注意力，这是 transformer 路线的固有代价；作者把降低复杂度列为未来工作。[1]

> 这两条很关键——第②点直接解释了为什么后来很多工作转向更轻量的方案。

## 我的疑问

- [ ] RPE 的具体实现细节放在补充材料里，它和标准可学习位置编码的本质差别是什么？论文只给了「+1.8%」这个结果。
- [ ] TSSIM 的三元扩展在数学上是否是 SSIM 的严格推广？还是只是「三对配对求平均」的工程写法？
- [ ] 既然 TSC Loss 对 TSRN 能提 +4.3%，为什么作者的消融里 TATT 自己只提 +1.0%？是因为 TATT 已部分解决了形变，还是 TSC 与 TPI 的作用有重叠？
- [ ] 全局注意力导致的复杂度增长，在实际最长文本（15 字符以上）时到底有多严重？论文未给量化。

## 关键摘录

> "This is because the current CNN-based methods adopt locality-based operations, which are not effective to deal with the variation caused by deformations." [1]

> "it merely merges text prior with image feature by convolutions. This arrangement can only let the text prior interact with the image feature within a small local range, which limits the effect of text prior on the text reconstruction process." [1]

> "the proposed TATT network has limitation on recovering extremely blurry texts ... In such cases, the strokes of the characters in the text are mixed together, which are difficult to separate." [1]

## 相关笔记

- [[低分辨率场景文本识别 LR-STR 方法谱系]] — TATT 在该谱系中处于「文本先验 + 结构约束」阶段
- [[低分辨率 STR 论文清单 2024-2026]] — TATT 至今仍是该领域最常被引用的强基线
- [[TextZoom 与 TSRN 真实场景文本图像超分]] — 本文使用的基准数据集与对比基线
- [[STT 文本聚焦的场景文本超分]] — 同期工作，同样用字符级监督但走双路损失路线
- [[C3-STISR 三重线索引导的场景文本超分]] — 后续工作，把线索从单一路扩展到三类
- [[PSNR 与 SSIM 图像质量指标]] — 理解 TSSIM 扩展的基础
- [[调制 Modulation]] — TP map 作为调制图作用于图像特征，与调制机制同源
- [[特征融合方法谱系]] — MCA 与「先上采样再相加/SFT」的对比，正是注意力融合优于拼接/相加的实例

### 待建笔记

以下概念本文涉及但库中尚无独立笔记：TP Interpreter、循环位置编码 RPE、三元 SSIM（TSSIM）、文本结构一致性损失。

## Sources

[1] https://arxiv.org/abs/2203.09388 — Ma, Liang, Zhang 2022 — TATT (CVPR 2022)
[2] https://github.com/mjq11302010044/TATT — TATT official code
[3] https://arxiv.org/abs/2106.15368 — TPGSR 2021 — 被对比的文本先验基线
[4] https://arxiv.org/abs/2005.03341 — TextZoom & TSRN 2020 — 基准数据集
[5] https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf — Wang et al. 2004 — SSIM（TSSIM 的扩展基础）