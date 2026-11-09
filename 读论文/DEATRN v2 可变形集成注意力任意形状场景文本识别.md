---
created: 2026-11-09
source: Xu, Zhong, Zhuang, Li, Su — "DEATRN v2: A Robust Arbitrary-Shaped Scene Text Recognition Network with Deformable Ensemble Attention", Pattern Recognition Letters (Elsevier, © 2026)；南京大学 计算机软件新技术国家重点实验室；本机 PDF：E:\.dsh-paper-reader\data\组\PRL.pdf
type: 论文
tags:
  - 论文
  - 场景文本识别
  - Attention
  - Transformer
  - 可变形卷积
---

# DEATRN v2：可变形集成注意力 + Bezier 中心线的鲁棒任意形状场景文本识别

> 页码均指 PDF 页码（共 8 页）。前作 DEATRN 见参考文献 [6]（ICPR 2024 / LNCS vol.15331, 2025, pp.237–253，第 8 页）。

## 一句话总结

用「局部可变形集成注意力（LDED）+ 全局形状约束（Bezier 中心线）」替代以往 STR 里"矫正 / 2D 注意力各管一段"的做法：自回归解码时，在每个字符周围预测一组可变形采样点去聚合判别性特征，同时用一条三次 Bezier 文本中心线把字符中心**钉在曲线上**，从几何上抑制注意力被背景、邻近文字带偏（第 1–3 页）。在 Union14M / WordArt 这类难集上 8 项里 6 项最优（第 6 页），是前作 DEATRN 的 v2 升级版。

## 核心问题

STR 主流是 encoder–decoder：CNN/Transformer 提视觉特征，序列模型（自回归或并行）转写成字符串（第 1 页）。对**任意形状**文字，两条老路都不够（第 1、2 页）：

- **空间变换 / 矫正**（STN、TPS，如 ASTER/MORAN）：先把弯曲文字"拉直"再认，属于预处理，形状信息没进解码器。
- **1D→2D 注意力**：把特征序列扩成 2D 特征图，自适应聚焦相关区域，比 1D 更灵活；但作者指出它们「lack explicit and sufficient utilization of text shape cues」（第 1 页）——没有显式利用文字形状线索，于是在复杂外观、杂乱背景、邻近文字干扰下，字符特征定位出现歧义（attention drift），认错字（第 1、3 页）。

DEATRN v2 的切入点是**字符级空间注意力**：以往 Transformer STR 多建模 word-level 全局依赖，本文额外引入"显式建模文字形状、按字符自适应取特征"的机制，专门解决上下文干扰下的注意力漂移（第 3 页）。

## 关键方法

整体 = 视觉编码器 + 混合解码器（**局部可变形集成解码器 LDED** + **Transformer 全局解码器**），并行挂一个 **Bezier TCL（文本中心线）预测模块**；解码**双向**进行，取词级概率更高的一个（第 3 页，Fig. 1 在第 2 页）。

![DEATRN v2 架构图 Fig1](DEATRN%20v2%20架构图%20Fig1.png)

<small>裁自论文第 2 页 Figure 1，仅取图本体与标注，已去掉页眉、正文与图注。</small>

**怎么读这张图：**

- 最左「text image」进 **Encoder**（浅 CNN → 12 层 Transformer，§3.1 第 3 页），出 2D 特征图 fe。
- **上方支路**：「Bezier TCL Prediction」从 fe 回归 4 个控制点，画出「Bezier text centerline & control points」；「Text Geometry Label Generation」从合成数据的字符框离线生成真值，用 L_cline 监督（第 3 页）。
- **中间支路**：「Local Deformable Ensemble Decoder」吃 encoder 输出 + 字符 embedding，输出「center & sampling points」（字符中心 + 一圈可变形采样点），用 L_center、L_disp 监督；箭头上的 Forward 'A' / Backward 'W' 即双向解码（第 2、4 页）。
- **下方支路**：「Global Transformer Decoder」再抓字符间全局关系，「Linear & Softmax」出字符，示例词 "WISTRRIA"，用 L_rec 监督（第 2、5 页）。
- **一句话**：全局形状（Bezier）负责"字符大致该在哪"，局部集成注意力负责"在这个字周围具体取哪些特征"，两个尺度用 L_cline / L_disp 的权重比来平衡（§4.6 第 7 页）。

### 1. Encoder（§3.1，第 3 页）

浅 CNN 把输入 X（H×W×3）下采样成 H/4 × W/4 × C 的 Fi，展平后过一串标准 Transformer encoder block（MHSA + FC），输出 fe，维度 (HW/16)×C。

### 2. Bezier 中心线：把形状变成可回归的参数（§3.2，第 3 页）

用**三次 Bezier 曲线** B(t) = (1−t)³P₀ + 3(1−t)²tP₁ + 3(1−t)t²P₂ + t³P₃（式 1）描述文本中心线，TCL 预测模块（卷积 + 线性层）回归控制点 P₀…P₃。它作为字符注意力的**全局位置约束**，专门治"杂乱背景 / 邻近文字 / 噪声"引起的注意力漂移。

**真值标签怎么来**（离线、只用合成数据自带字符框）：K-means(K=2) 对像素聚类出二值文本分割图 → 对文本像素再做 K-means(K=L，L=字符数) 得 L 个"中心轴向点" → 最小二乘把这些点拟合成 Bezier 曲线，回归出的控制点即真值 P̄ᵢ（第 3 页，Fig. 3）。监督用中心线近似损失 L_cline = (1/4)·Σᵢ₌₀³ ‖Pᵢ − P̄ᵢ‖²₂（式 2）。

### 3. 局部可变形集成解码器 LDED（§3.3，第 3–5 页；Fig. 4 在第 4 页）

三个机制叠在一起（第 4 页）：

1. **集成采样点**：一个字符可能由特征图上多处共同决定，故在每个字符中心周围预测 **K=16** 个采样点，加权聚合它们的特征（式 4：fₘ = Σₖ₌₁ᴷ wₖ · fe[pos_samp⁽ᵏ⁾]）。
2. **可变形采样网格**：采样位置不是固定规则网格，而是相对中心点预测偏移 pos_off（pos_samp = pos_cen ⊕ pos_off），自适应找判别性特征。
3. **形状约束**：字符中心**必须落在 Bezier 中心线上**——MDEA 层用线性层预测曲线参数 t_cen，由 pos_cen = B(t_cen)（式 3）算出中心坐标，而不是让中心自由漂移。

结构：N 个相同 decoding block，每块 = masked MSA → **MDEA（多头可变形集成注意力）** → FC（Fig. 4）。MDEA 用 **M=8** 个头各预测 K 个偏移与权重 wₖ，8 头特征 concat 后投影成 g ∈ R^C，C=512（式 5）。

**监督（§3.3.1–3.3.2，第 4–5 页）**：

- 字符中心真值用字符框两角取平均 (x*, y*) = ((x₁+x₂)/2, (y₁+y₂)/2)（式 6）；另训一个**字符中心点预测网络**提供标注（L_reg，式 7）。
- **center loss** L_center：约束解码器预测的中心贴合上述真值，减少漂移。
- **dispersion loss** L_disp（式 8）：约束采样点别离中心太远，超参 τ 控制松紧；τ→∞ 时等于关掉该损失。

### 4. 全局解码器与损失（§3.4–3.5，第 5 页）

全局解码器 = 多个 Transformer block（self-attn + cross-attn，cross-attn 的 K/V 来自 encoder、Q 来自 self-attn 输出）+ FFN，再 FC + softmax 出字符；取前/后向**词级概率最高**的序列为结果。

混合损失（式 9）：

    L = λ₁·L_rec + λ₂·L_cline + λ₃·L_center + λ₄·L_disp
    λ = (1.0, 10.0, 1.0, 1.0)

L_rec 是字符分类交叉熵。注意 λ₂=10 —— 全局形状约束被刻意加权到最重（§4.6 消融印证）。

### 5. 相对前作 DEATRN 的 5 处改动（第 2 页）

① 多项式中心线 → **Bezier 曲线**（描述能力更强，标签生成与 L_cline 随之改）；② 字符中心**约束在中心线上**、不再自由移动；③ 新增专用网络生成字符中心点监督标注；④ 删掉 bidirectional alignment loss（被新 center loss 取代）；⑤ 补了关键超参实验。

> 未找到本文官方开源实现，故「关键方法」不挂代码（笔记规范：找不到就写"未找到可用实现"，不臆造）。

## 结论与局限

**设置（§4.1–4.2，第 5–6 页）**：训练只用合成集 **MJSynth(MJ) + SynthText(ST)**，不在目标集微调；评测 **Union14M**（真实任意形状，子集 CUV/MOR/ART/CTX/SAL/MWD/GEN + WordArt）与 6 个常规集（IIIT5k/SVT/IC13/SVT-P/IC15/CUTE）。指标 = word accuracy。PyTorch、4×V100、AdamW、batch 384、6 epoch、warm-up 3000 iter 后 cosine annealing；输入统一 **32×128**，90 类字符（大小写 + 数字 + 28 符号）。

**关键数字（Table 1，第 6 页；不抄全表）**：

- DEATRN v2 在 **8 项测试里 6 项最优**（第 6 页原话）：CUV **69.8**、CTX **61.1**、SAL **74.1**、MWD **66.0**、GEN **62.2**、WordArt **73.7**；MOR 22.4、ART 58.6 不是该列最优。
- 对**非任意形状专用**的老方法优势巨大：CRNN 的 CUV 仅 7.5、ASTER 34.0，而 DEATRN v2 69.8；WordArt 上 SATRN 65.7、ABINet 67.4、CornerTransformer 70.8，DEATRN v2 **73.7**（比前作 DEATRN 72.1 还高 1.6）。
- **Table 2（第 6 页）**：6 个常规集上只是"comparable"——IIIT5k 96.1 / SVT 94.3 / IC13 97.7 / SVT-P 90.2 / IC15 85.7 / CUTE 94.1，普遍略低于 ABINet/MATRN/LISTER 等。作者解释：常规集里不规则形状与复杂上下文样本少，本方法的优势**没被充分体现**（第 6 页）。这是本文最该记住的"边界"。

**消融（§4.4–4.6，第 6–7 页，信息密度最高，保留洞察）**：

- **τ（dispersion 松紧，Table 3）**：τ=0.5 最佳；τ 太小把采样点锁死、太大（=∞ 等于关掉约束）注意力会漂到邻近物体/噪声，精度反而掉——"约束与灵活要平衡"。
- **局部 vs 全局 vs 混合解码器（Table 4）**：只用 6 个 local block 已**全面超过**只用 6 个 global block（如 CUV 64.5 vs 61.8）；3 local + 3 global 的 hybrid 在多数集再涨（CUV 68.6、MWD 65.7、WordArt 72.9）→ 局部可变形注意力是主力贡献，全局是补充。
- **字符中心监督（Table 5）**：w/o 68.6 / 72.9（CUV / WordArt）→ w. **69.8 / 73.7**，验证"专用网络生成中心标注 + L_center"确实抬点。
- **采样点数 K（§4.5 / Table 6）**：**K=16 最优**；K=4 漏视觉线索明显掉点，K=8~24 接近，K=32 反而引入噪声特征退化。
- **λ₂ : λ₄（全局约束 : 局部搜索，Table 7）**：**10:1 最好**（即式 9 里 λ₂ 给 10 倍权），多词(MWD)场景尤其受益——全局形状约束能防止采样器抓到相邻词的特征；但自由形态/风格化文字上灵活局部搜索同等重要（第 7–8 页）。

**局限与失败案例（§4.7，第 8 页）**：作者给出 Fig. 6，错误集中在 ① 风格化/畸变字符混淆（'l' vs 'd'、'4' vs 'A'）、② 文字朝向歧义、③ 图像模糊。**未来工作**自述：利用字符间空间相关性（相对位置、邻域依赖）作更强上下文，以及引入语言先验/语义建模来消解形近字（第 8 页）——等于承认当前**纯视觉、无语言模型**。

**我的观察**：优势几乎全部集中在 Union14M/WordArt 的"形状/上下文干扰"子集；常规集打平甚至略输，说明这是为复杂场景特化的方案，不是通用精度提升。另外 Bezier 真值依赖合成数据的字符级框标注，其"专用中心点预测网络"也只在合成数据上训练，真实域的形状标注噪声是隐含前提（论文未讨论）。

## 我的疑问

- [ ] L_disp 里 τ=0.5 是在归一化坐标 (o_x/(W/4), o_y/(H/4)) 上取的，换输入分辨率（本文固定 32×128）时是否要重调？
- [ ] "字符中心必须落在中心线上"对**多词 / 两行**文字（MWD 子集）是否过强？Table 1 里 MWD 66.0 已最高，但作者仍把"字符间空间相关"列为未来工作。
- [ ] 与语言模型路线（ABINet / PARSeq / MATRN）结合能否补上 §4.7 的形近字短板？
- [ ] 未开源，K=16 / M=8 / λ=(1,10,1,1) / τ=0.5 这组超参的可复现性待验证。

## 关键摘录

> "We propose a deformable ensemble attention model to improve the accuracy of feature alignment in arbitrary-shaped text recognition. The attention model adaptively selects and aggregates discriminative features of a character at a set of deformable attention positions during autoregressive decoding of the text…"（第 2 页，贡献第一条）

> "Although existing 2D attention mechanisms have demonstrated effectiveness in recognizing irregularly shaped text, they often lack explicit and sufficient utilization of text shape cues. As a result, these methods struggle with the ambiguities in text feature localization under complex text appearances and image contexts, leading to erroneous recognition results."（第 1 页，动机）

> "DEATRN v2 achieves the highest recognition accuracy in six out of eight benchmark tests."（第 6 页，§4.3）

> "The main causes of the errors in text recognition are confusion between stylized or distorted characters ('l' vs. 'd' and '4' vs. 'A' in the first two examples), ambiguous text orientation (3rd example), and blurred image (last example)."（第 8 页，§4.7）

> "Future work will investigate leveraging spatial correlations among characters, such as relative positions and neighborhood dependencies… Besides, we will also explore the integration of semantic modeling mechanisms, such as language priors…"（第 8 页，§5）

## 相关笔记

- [[SVTR 场景文本识别]] — 另一条"单视觉模型"STR 路线，Table 2 里是本文的对比基线
- [[ASTER 注意力场景文本识别器]] — TPS/STN 矫正路线的代表，本文动机里点名的老办法
- [[MORAN 多目标矫正注意力场景文本识别]] — 同为矫正 + 注意力，Table 1/2 的对比基线
- [[多头注意力+proj_out]] — MDEA 里 8 头 concat 后投影（式 5）的结构参照
