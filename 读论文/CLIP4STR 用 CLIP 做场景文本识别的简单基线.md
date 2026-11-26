---
created: 2026-11-26
source: E:\.dsh-paper-reader\data\STR\Zhao 等 - 2024 - CLIP4STR A Simple Baseline for Scene Text Recognition with Pre-trained Vision-Language Model\Zhao 等 - 2024 - CLIP4STR A Simple Baseline for Scene Text Recognition with Pre-trained Vision-Language Model.pdf（arXiv:2305.14014v4 [cs.CV]，2024-12-24；代码 https://github.com/VamosC/CLIP4STR）
type: 论文
tags: [论文, 场景文本识别, CLIP, 视觉语言预训练, 多模态]
---

# CLIP4STR：把 CLIP 直接改造成场景文本识别器

> Shuai Zhao、Ruijie Quan（通讯）、Linchao Zhu、Yi Yang（Senior Member, IEEE）；ReLER Lab, Australian AI Institute, UTS + ReLER Lab, CCAI, 浙江大学。用户在阅读器选中的标题位于第 1 页。下文页码均指本机 PDF 页码（共 12 页）；事实与数字均来自原文，不补原文没有的内容。

## 一句话总结

把 CLIP 的图像/文本编码器**原样搬来**做场景文字识别，用「视觉分支先猜、跨模态分支再按语义纠错」的双分支结构 + 排列序列建模（PSM）解码器，直接在 13 个 STR 基准上刷到 SOTA——核心论点是**图文对比预训练出来的多模态神经元天然就是优秀的文字感知器**，比 ImageNet 单模态预训练甚至从零训练都更合适（p.1 摘要、p.6–7、p.9 结论）。

## 核心问题

STR 方法长期偏好**单模态（视觉）预训练骨干**（ImageNet 或合成数据），尽管 STR 本身就是图像+文本的双模态任务（p.1–2）。作者的切入点是两个观察：

1. **排版攻击（typographic attacks）**：在 CIFAR-10 自然图上贴文字贴纸后做零样本分类，CLIP-ViT-B/32 的 Grad-CAM 注意力死死盯住贴纸文字而忽略自然物体；无论水平、旋转、弯曲、模糊还是遮挡，都能读出单词语义（p.1，Fig.1）。这来自 CLIP 训练数据中天然包含的场景文字，以及能同时响应同一概念的视觉与文本信号的**多模态神经元**（p.2，§II-A）。
2. **对真实 STR 图像也成立**：把 Fig.1 的自然图换成从自然图裁剪的单词条（cocacola / delocated / finish / bridgestone / marriott / starbucks），文字被遮挡、弯曲、模糊、旋转，CLIP 依然能感知到（p.2，Fig.2）。

于是问题变成：能否**不做蒸馏、不借全局图像特征**，直接把 CLIP 变成一个 STR 阅读器？此前 CLIP-OCR 走特征蒸馏、CLIPTER 用全局 CLIP 特征当辅助上下文，本文明确与这两条路线划清界限（p.3）。

## 关键方法

### ① 双分支框架：视觉预测 → 跨模态精炼

![CLIP4STR 架构图 Fig3](CLIP4STR%20架构图%20Fig3.png)

*怎么读这张图*：左半边是**视觉分支**（CLIP image encoder → image decoder），右半边是**跨模态分支**（image encoder + CLIP text encoder 的特征拼接 → cross-modal decoder）。视觉分支出初始预测 `briiad`，文本编码器把这份预测编成语义特征，与图像特征在联合嵌入空间比对、纠正差异后由跨模态分支输出 `bread`。两个 decoder 结构相同、输入不同。关键细节图上标了：**text encoder 部分冻结**（partial）、`C` 表示拼接、跨模态解码器到图像编码器之间有 **stop gradient** 阻断（p.3，Fig.3）。

跨模态分支的角色被作者定义为 **semantic-aware spell checker（语义感知拼写检查器）**：由于 CLIP 预训练时图像与文本特征已对齐到同一联合空间，视觉特征与预测文本语义之间的「不一致」很容易被识别出来（p.3，§III-B）。

### ② 编码器：继承 CLIP，但要「拆掉 CLASS 捷径」

- 图像编码器用 CLIP ViT，**返回全部 token 的特征**而非只返回 `[CLASS]`——因为字符级识别需要 patch 级的细粒度局部信息。消融证实：只用 `[CLASS]` + `[EOS]` 掉到 90.6%，而全 token 是 91.2%（p.3、p.7）。
- 文本编码器用 CLIP Transformer，BPE 小写词表 49,152，取所有 token 的语言特征；**冻结一半层数**，既保留语言理解能力又省训练成本。
- 图像编码器**完全可训练**：STR 数据是裁剪单词条，CLIP 预训练数据是网页自然图，域差距大；同时阻断跨模态解码器→图像编码器的梯度，让视觉分支自主学习（p.3）。
- 特征计算：`Ft = g(t) ∈ R^{Lt×D}`、`Fi = h(x) ∈ R^{Li×D}`、`Fc = [Fi^T Ft^T]^T ∈ R^{Lc×D}`，`Lc = Li + Lt`（p.3，式 1–3）。

### ③ 解码器：Transformer decoder + 排列序列建模（PSM）

![CLIP4STR 解码器 Fig4](CLIP4STR%20解码器%20Fig4.png)

*怎么读这张图*：输入的 `[B]` 是占位上下文，位置 query `[P]` 与目标位置绑定；第一层 MHA 做 **context-position attention**（随机注意力掩码 `M` 在这里生效），第二层 MHA 做 **feature-position attention**，接 MLP + Linear 出字符 logits，与 GT 算交叉熵。随机掩码让每个字符可以依赖任意位置的上下文，摆脱「必须从左到右」的假设——例如预测 `model` 里的 `o`，只需 `m_de` 而不必是 `m_` 或 `led_`（p.3–4，§III-A2、Table I、Fig.4）。

- 训练时每个输入上下文用 **K=6 个掩码**：前两个是左→右、右→左 AR 掩码，其余随机生成（沿用 PARSeq）。
- 两次注意力（省略输入输出线性变换）：`m1 = softmax(pc^T/√D + M)c + p`（式 4）、`m2 = softmax(m1F^T/√D)F + m1`（式 5）、`y = Linear(MLP(m2) + m2)`（式 6）。
- 联合损失：`L = CE(yi, ŷ) + CE(y, ŷ)`，两个分支一起优化（式 7，p.4）。

### ④ 推理：dual predict-and-refine 三段式（Alg.1，p.5）

1. 视觉分支 **自回归解码**出 `yi`（后续字符依赖前面预测）；
2. 把 `yi` 送进 CLIP 文本编码器，与图像特征拼成 `Fc`，跨模态分支再**自回归解码**出 `y`；
3. 用 `yi` 和 `y` 当上下文，换成 **cloze 掩码**做迭代精炼（可重复 `Ti` 次），最终输出跨模态分支的结果。

实践里 `Ti = 1`：Table IX 显示迭代 2/3 次精度不再涨（p.9）。

### ⑤ 训练配方（p.5，§IV-A）

- 模型：CLIP-ViT-B/16（158M）、ViT-L/14（446M）、ViT-H/14（1B）；训练 8× RTX A6000，B 12.8h / L 23.4h / H 48h。
- 数据：`MJ+ST`（合成 9M+6.9M）、`Real(3.3M)`（COCO-Text/RCTW17/UberText/ArT/LSVT/MLT19/ReCTS/TextOCR/Open Images 共 330 万真图）、`RBU(6.5M)`（Real + 基准训练集 + 部分 Union14M-L）。**主用真实数据**，因为已有工作表明真实数据优于合成数据（p.2–3、p.5）。
- 图像 224×224，文本序列 16，字符最大 25（`N=26` 含 `[B]`/`[E]`）；训练 94 类（大小写+标点），推理 36 类（小写+数字）；RandAugment depth 3 / magnitude 5（去掉 sharpness 和 invert）；AdamW wd 0.2，warmup + cosine，batch 1024，混合精度。
- CLIP 编码器学习率 `8.4e-5 × batch/512`，从零训练 decoder 的学习率再 ×19（p.5，Table II 注）。

> 💻 官方实现在 `https://github.com/VamosC/CLIP4STR`（p.5 明写）。本次**未逐行读码**，故不臆造代码片段；如需代码级对照另建 `代码细节笔记/` 伴生篇。

## 实验与消融：哪些设计真正值钱

### 主结果（p.6，Table III / IV）

- **13 个 STR 基准 SOTA**：常规 10 个（IIIT5K/SVT/IC13/IC15/SVTP/CUTE/HOST/WOST/OCTT）+ 3 个大基准（COCO-Text / ArT / UberText）。
- 最强配置 CLIP4STR-H（DFN-5B 预训练 + RBU 6.5M）：IIIT5K **99.5**、SVT **99.1**、IC13 **98.9**、IC15 **91.7/91.0**、SVTP **98.0**、CUTE **99.0**、HOST **82.6**、WOST **90.9**、OCTT **86.5**；大基准 COCO **83.0** / ArT **86.4** / Uber **91.7**。
- **遮挡是最大亮点**：HOST / WOST 上最佳分别超此前 SOTA **+7.8 / +5.4 点**（p.6 正文），归功于预训练文本编码器 + 跨模态解码器能用语义或视觉特征推断缺失字符。
- 直接迁移 > 蒸馏/辅助特征：全面碾压同方向的 CLIP-OCR 与 CLIPTER（p.6）。
- **规模收益递减**：1B 的 H 与 L 在多数基准相当，H 只在遮挡类（WOST/HOST/OCTT）更好；作者明说 “scaling CLIP4STR to 1B parameters does not bring much improvement”（p.6–7）。

### 消融（p.7，Table V，9 个基准均值）

| 逐步加料 | Avg. |
|---|---|
| 基线：无预训练 ViT-S 视觉分支 | 89.2 |
| + PSM（并沿用 PARSeq 训练配方） | 89.9 |
| + CLIP ViT-B/16 骨干但**不改配方** | 无显著增益 |
| + 新配方（16×16 patch、编码器小 LR、解码器大 LR、16 epochs） | **90.8** |
| 同上但去掉 PSM | 90.0 |
| 再加跨模态分支 | **91.2**（+0.4） |
| 只用 `[CLASS]`+`[EOS]` 特征 | 90.6（掉 0.6） |
| 换 CLIP ViT-L/14 | **91.9**（+0.7，且只要 10 epochs） |

要点：① **PSM + 适配训练配方缺一不可**——CLIP 骨干直接套旧配方没用；② 跨模态分支只贡献 +0.4，不算大；③ 全 patch 特征比 `[CLASS]` 重要；④ 大模型收敛更快（10 vs 16 epochs）。

### 冻结策略（p.7，Table VI）

- 冻文本编码器影响小：即使只用固定的预训练 token embedding 仍有 90.7；作者解释 STR 的文本只是词/短语，语义理解比通用语言任务容易。
- 冻图像编码器影响大：冻结 6 层 → 89.5，9 层 → 87.8，全冻结 → **61.2**。域差距（网页自然图 vs 裁剪单词）必须靠可训练的图像编码器弥合。

### VLM vs 单模态预训练（p.8，Table VII）

只换视觉编码器、其余相同（仅视觉分支结果）：Scratch **90.1**、ImageNet-1K **89.7**、ImageNet-21K **89.3**、图文对预训练 **90.3**。反直觉结论：**ImageNet 预训练甚至不如从零训练**。作者解释：分类目标不教模型读图中文字——两张都写着 “park” 的猫图和狗图，会让模型学到关于同一段文字的矛盾信息（p.8，§V-C）。

### 参数高效微调（p.8，Table VIII）

Frozen 60.9 → CLIP-Adapter（262K 参数）63.6 → LST r=4（4.1M）88.2 → LST r=2（13.1M）89.6 → 全量微调（86M）**90.3**。结论：加几层线性适配不足以跨越域差距；LST 参数量上去后逼近全量微调，可作为算力受限时的备选。

### 速度与定性（p.9）

- 单张 A100 40GB：ABINet 1.30ms / PARSeq-S 1.32ms / CLIP4STR-B 视觉分支 AR 3.03ms / 跨模态分支 3.52ms（仅 **+0.49ms**，因为文本序列只有 16）/ ViT-L 跨模态 6.52ms。用 Eq.8 复用视觉预测当上下文可降到 3.41ms，但精度略掉（91.1 vs 91.2）。
- 定性（Fig.6）：弯曲/遮挡/模糊/旋转文字都能读；**会主动补全残缺字符**（最后两例多补了个 `n`），可能源自 CLIP 的语义理解，但**补全正确与否不确定、也无法控制**（p.9，§V-F）。
- 在 Yang 等清理过标注的基准上仍是 SOTA（p.9，Table X）。
- 与自回归预训练的 DTrOCR（6B textlines）对比（p.10，Table XI）：DTrOCR 在 IC13/IC15/SVTP 略胜，但差距很小，CLIP4STR 在 IIIT5K/SVT/CUTE 更好；CLIP4STR 的两点实用优势是**现成 VLM 免去预训练成本**且**开源可复现**（DTrOCR 闭源）。

## 结论与局限

- 结论：CLIP 是强大的场景文字阅读器，视觉语言预训练确实有益于 STR；CLIP4STR 希望成为后续 VLM-based STR 研究的简单强基线（p.9，§VI）。
- 局限（作者承认 + 我观察到）：
  - **速度代价**：ViT-B 3.52ms 是 PARSeq-S 1.32ms 的约 2.7 倍，ViT-L 6.52ms；精度-速度权衡明显，且 1B 模型收益有限（p.9）。
  - **字符补全不可控**：会补出不存在的字符，作者明说 “the accuracy of this complement is uncertain, and we currently cannot control this behavior”（p.9）。
  - **依赖真值训练配方**：CLIP 骨干必须换配方才能变现（p.7），PSM/全 token 等设计各自只有零点几个点，整体是「堆小胜」而非单一杀手锏。
  - **规模上限存疑**：H（1B）相对 L 提升微弱，继续放大是否还有收益未验证（p.6–7）。
  - 训练/评测字符集为小写字母数字（36 类），中文等语言未覆盖（p.5）。

## 我的疑问

- [ ] 多模态神经元是 CLIP 特有，还是所有图文对比模型（ALIGN/SigLIP/FLIP）都有？换骨干后「图像编码器必须可训练」的结论是否仍成立？
- [ ] 字符补全不可控这一点，与近期 OCR 大模型「幻觉」问题（见 [[低分辨率 STR 论文清单 2024-2026]] 中 HR-Diffusion 的语义引导幻觉）是否为同一机理？
- [ ] 若把 PSM 换成 PARSeq 完整 PLM（对所有排列求期望）而非 K=6 个掩码，跨模态分支还能再涨吗？
- [ ] 全量微调 vs LST 的差距（90.3 vs 89.6）在更难集（HOST/WOST）上是否更大？

## 关键摘录

> “CLIP can perceive and understand text in images, even for irregular text with noise, rotation, and occlusion. CLIP is potentially a powerful scene text recognition expert.”（p.1，Fig.1 后）

> “The cross-modal branch acts as a semantic-aware spell checker.”（p.3，§III-B）

> “ImageNet classification pre-training does not align well with STR. Classifying objects in an image does not help the model learn specific information about the text within the image.”（p.8，§V-C）

> “CLIP4STR achieves state-of-the-art results on 13 STR benchmarks, showing that CLIP is a powerful scene text reader and that vision-language pre-training benefits STR.”（p.9，§VI）

## 相关笔记

- [[CLIP 大规模图文对比预训练]] — 本文直接继承的骨干：4 亿图文对对比预训练、联合嵌入空间
- [[PARSeq 用排列语言建模统一场景文本识别]] — PSM / K=6 掩码 / cloze 迭代精炼的来源，本文解码器的技术母体
- [[ABINet 自主双向迭代的语言建模]] — 同为「初始视觉预测 + 语义纠错」思路，对照跨模态分支的角色
- [[SVTR 场景文本识别]] — 单视觉模型路线，与本文「VLM 骨干」形成对照
- [[多模态对齐与对比学习 主题地图]] — 本文所属的图文对齐大主题索引
