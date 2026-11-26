---
created: 2026-11-26
updated: 2026-11-26
source: E:\.dsh-paper-reader\data\STR\PARSeq\PARSeq.pdf（arXiv:2207.06966, ECCV 2022）
type: 论文
tags: [论文, 场景文本识别, 自回归, Transformer]
---

# PARSeq：用排列语言建模统一场景文本识别

## 一句话总结

用**排列语言建模（PLM）**训练一个内部语言模型，使**同一个模型、同一套权重**，只靠切换注意力掩码就能分别做无上下文并行解码、自回归解码与双向迭代精炼——从而用一个模型取代「视觉模型 + 外部语言模型 + 融合层」的两阶段拼装。

## 核心问题

带语言先验的 STR 方法通常内建自回归（AR）语言模型，但 AR 约束带来两个固有局限：[1]

1. **单向性**：只能在训练时的方向（通常是左到右）上学习 token 依赖，导致模型偏向单一阅读方向，出现无意义的后缀添加或方向相关预测；
2. **串行解码**：推理时只能按训练方向逐 token 输出（monotonic AR decoding），无法并行。

另一条路是**两阶段集成**：先用无上下文模型出初始预测，再用一个**外部 LM** 做纠错（ABINet 属此类）。[1]

PARSeq 对这一路线的批评很尖锐，也构成了本文的出发点：[1]

- 外部 LM **与输入图像条件独立**，因此会在预测看起来像拼写错误时**错误地「纠正」本来正确的预测**；
- 更糟的是，相似的、似然更高的词会诱使 LM 改错；
- 证据：SRN 与 ABINet 的 LM 用作拼写纠正器时，**词准确率分别只有 27.6% 与 41.9%**。[1]

作者进一步指出 ABINet 的 LM「相对其参数量而言未被充分利用」，且占用了整体算力的很大一块。[1]

## 关键方法

### ① 用 PLM 替代标准 AR 训练

标准 AR 只用规范顺序 $[1,2,\dots,T]$ 做链式分解；PLM 则对**所有 $T!$ 种排列**求期望：[1]

$$\log p(y\mid x)=\mathbb{E}_{z\sim\mathcal{Z}_T}\left[\sum_{t}\log p_\theta(y_{z_t}\mid y_{z_{<t}},x)\right]$$

**实现上不需要真的打乱标签**，只需按给定排列构造**注意力掩码**即可表达不同的分解顺序。[1]

关键洞察：**标准 AR 训练只是 PLM 中「只取一个排列」的特例**。[1]

论文给了一个三元素序列的例子，四种排列对应四种不同的信息流掩码（1 表示可依赖，0 表示不可）：[1]

| 排列 | $y_1$ 可见 | $y_2$ 可见 | $y_3$ 可见 |
|---|---|---|---|
| $[1,2,3]$ | B | B,$y_1$ | B,$y_1$,$y_2$ |
| $[3,2,1]$ | B,$y_2$,$y_3$ | B,$y_3$ | B |
| $[1,3,2]$ | B | B,$y_1$,$y_3$ | B,$y_1$ |
| $[2,3,1]$ | B,$y_2$,$y_3$ | B | B,$y_2$ |

**实践中不训练全部 $T!$ 个排列**（计算量指数增长），只取 $K$ 个。采样策略是刻意设计的：[1]

- 用 $K/2$ 对排列；
- 前半 = 左到右排列 $[1,\dots,T]$ + $K/2-1$ 个随机排列；
- 后半 = 前半的**翻转版本**。

作者称这种采样让训练更稳定。[1]

训练损失是 $K$ 个排列对应掩码的交叉熵均值：[1]

$$\mathcal{L}=\frac{1}{K}\sum_{k=1}^{K}\mathcal{L}_{ce}\big(y_k,\hat y\big),\qquad y_k=\mathrm{Dec}(z,p,c,m_k)$$

### ② 位置 token 解耦「目标位置」与「输入上下文」

这是 PARSeq 能统一多种解码的**技术核心**。[1]

位置 token $p$ 同时充当 query 向量与位置嵌入；上下文 $c$ 只作为 K/V。第一个 MHA 做上下文-位置注意力，第二个做图像-位置注意力：[1]

$$h_c=p+\mathrm{MHA}(p,c,c,m),\qquad h_i=h_c+\mathrm{MHA}(h_c,z,z)$$

最后 $h_{dec}=h_i+\mathrm{MLP}(h_i)$，输出 $y=\mathrm{Linear}(h_{dec})\in\mathbb{R}^{(T+1)\times(S+1)}$（多出的一维是 `[E]` 结束符）。[1]

作者明确指出成败关键：

> 如果没有位置 token（像标准 Transformer 那样让上下文 token 自己当 query），**模型从 PLM 里学不到任何有意义的东西，会退化成普通的 AR 模型**。[1]

这一参数化来自 two-stream attention 的 query stream，但 PARSeq 做了简化：**decoder 不用自注意力、也不交错两路 stream**，因此单层 decoder 只需 query stream，省掉了未使用的 content stream 开销。[1]

### ③ 架构：深编码 + 浅解码

- **编码器**：12 层 ViT（去掉分类头与 `[CLS]`），图像切成 $p_w\times p_h$ patch 后线性投影并加可学习位置嵌入；**所有输出 token** 都送入 decoder。[1]
- **解码器**：仅 **1 层**，但注意力头数翻倍（$n_{head}=d_{model}/32$）。[1]

这种「深-浅」配置是刻意选择：把计算量压到最低而性能损失可忽略。[1]

### ④ 一个模型，三种解码

三种解码**只改掩码，不改权重**：[1]

| 解码 | 掩码 | 上下文 | 位置 query |
|---|---|---|---|
| **AR** | 左到右 lookahead | `[B]` + 已生成 token | $[p_1,\dots,p_i]$ 逐个增 |
| **NAR** | 无掩码（等价全 1） | 恒为 `[B]` | 全部 $[p_1,\dots,p_{T+1}]$ |
| **迭代精炼** | cloze（全 1 再遮掉自身位置） | 上一轮输出（截到 `[E]`） | 全部 |

cloze 掩码由「全 1 再 mask 掉对应的 token 位置」构造，这与 ABINet 的 BCN 思路相同，但**用途不同**：ABINet 训练与推理都用 cloze 上下文，PARSeq **只在迭代精炼时用**。[1]

最关键的区别：**PARSeq 的精炼过程同时考虑输入图像与语言上下文**，而 ABINet 的精炼模型与图像条件独立。[1]

## 结论与局限

### $K$ 的消融（最有信息量）

| $K$ | AR 准确率 | NAR 准确率 | cloze 准确率 | 训练时长 |
|---|---|---|---|---|
| 1 | 93.04 | **0.01** | **71.14** | 5.86h |
| 2 | 93.48 | 22.69 | 94.55 | 7.30h |
| **6** | 93.34 | **92.22** | 94.81 | 8.48h |
| 12 | 92.91 | 91.71 | 94.59 | 10.10h |
| 24 | 92.67 | 91.72 | 94.36 | 13.53h |

三点结论：[1]

1. $K=1$（退化为标准 AR）时，**NAR 解码完全失效**（0.01%），cloze 也只有 71.14%——**证明统一性确实依赖 PLM 训练，不是白捡的**；
2. 所有解码方式要到 $K\ge 6$ 才表现良好，$K=6$ 是精度与训练时间的平衡点；
3. cloze 准确率高达约 94%，说明**联合使用图像特征与语言上下文**的精炼很有效——毕竟 STR 的主要输入信号是图像，不是语言上下文。

### 主结果（字符集维度）

| 训练数据 | 方法 | 36 字符 | 62 字符 | 94 字符 |
|---|---|---|---|---|
| 合成 | PARSeqA | **91.9** | **75.5** | **73.0** |
| 合成 | ABINet | 89.8 | 68.5 | 66.4 |
| 合成 | TRBA | 90.6 | 71.9 | 69.9 |
| 真实 | PARSeqA | **96.0** | **94.6** | **93.3** |
| 真实 | ABINet | 95.2 | 93.7 | 92.4 |

PARSeqA 在所有字符集规模上都最高；合成数据下从 36 到 62/94 字符有陡降，说明**合成数据缺乏大小写多样性**。[1]

### 更难的数据集（更见真章）

| 方法 | 训练数据 | ArT | COCO | Uber | 总计 |
|---|---|---|---|---|---|
| PARSeqA | 合成 | 70.7 | 64.0 | 42.0 | 51.8 |
| ABINet | 合成 | 65.4 | 57.1 | 34.9 | 45.2 |
| PARSeqA | 真实 | 84.5 | 79.8 | 84.5 | **84.1** |
| ABINet | 真实 | 81.2 | 76.4 | 71.5 | 74.6 |

作者指出：**方法间的差距在这些更大更难的数据集上才真正显现**（这些集含遮挡与任意方向文本）。[1]

### 成本-质量

PARSeq-S 在三个指标（参数量 / FLOPs / 延迟）上都是 **Pareto 最优**；相比 ABINet 与 TRBA 用显著更少的参数与 FLOPs。[1]

PARSeq-S 的 AR 解码延迟略慢于 TRBA，但仍显著快于 ABINet。[1]

## 我的疑问

- [ ] $K=6$ 是针对 94 字符集选的。换成中文这样的大字符集，$K$ 是否要重新调整？直觉上字符集越大、需要的排列越多。
- [ ] 单层 decoder 是刻意设计，但它对 PLM 的表达能力是否构成上限？加深 decoder 与增加 $K$ 哪个更划算？
- [ ] 迭代精炼时用 cloze 掩码，与 AR 解码产物作为上下文，两者是如何配合的？论文说精炼可用在任一初始解码后，但两种起点的增益差异没细说。
- [ ] PARSeq 批评 ABINet 的 LM 词准确率仅 41.9%，但 ABINet 的 LM 是**故意**做到与图像独立的。这是「设计缺陷」还是「可解释性的代价」？

## 关键摘录

> "The conditional independence of the external LM on the input image may cause it to erroneously rectify correct predictions, leading to significant inefficiencies." [1]

> "PLM can be considered a generalization of AR modeling, and a PLM-trained model can be seen as an ensemble of AR models with shared architecture and weights." [1]

> "Without the position tokens, i.e., if the context tokens are used as queries themselves like in standard Transformers, the model will not learn anything meaningful from PLM and will simply function like a standard AR model." [1]

> "Unlike ABINet which uses the cloze context for both training and inference, PARSeq uses it for iterative refinement only." [1]

## 相关笔记

- [[ABINet 自主双向迭代的语言建模]] — 本文主要批评对象，两篇的分歧点在于「精炼时是否看图像」
- [[SVTR 场景文本识别]] — 同期纯视觉骨干路线，可作为「不用语言模型」的上界参考
- [[ASTER 注意力场景文本识别器]] — 早期注意力 Seq2Seq，属本文批评的「monotonic AR」一类
- [[曝光偏差 Exposure Bias]] — AR 的误差累积问题，本文用迭代精炼缓解
- [[调制 Modulation]] — 本文的注意力掩码机制可与条件注入机制对照理解

### 待建笔记

以下概念本文涉及但库中尚无独立笔记：Permutation Language Modeling（PLM）、two-stream attention 的 query stream、cloze 掩码、unaligned-length 问题。

## Sources

[1] https://arxiv.org/abs/2207.06966 — Bautista & Atienza 2022 — PARSeq (ECCV)