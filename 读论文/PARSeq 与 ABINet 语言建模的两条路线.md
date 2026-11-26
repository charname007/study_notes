---
created: 2026-11-26
updated: 2026-11-26
source: E:\.dsh-paper-reader\data\STR\PARSeq\PARSeq.pdf; E:\.dsh-paper-reader\data\STR\ABINet\ABINet.pdf（arXiv 2207.06966 / 2103.06495）
type: 论文
tags: [论文, 场景文本识别, 自回归, 大语言模型]
---

# PARSeq 与 ABINet：语言建模的两条路线

## 一句话总结

两篇都在解决「如何用语言模型纠正视觉识别错误」。ABINet 走**解耦 + 双向 + 迭代**路线，把 LM 做成独立可预训练模块；PARSeq 走**统一 + 排列建模**路线，用一个模型同时支持无上下文、自回归与迭代精炼三种解码。

## 核心问题

STR 的难点在于：图像特征不足时（遮挡、模糊、噪声），必须靠语言语义补足。[1][2]

两篇论文都在批判同一类前作：把 LM 与视觉模型**耦合**在一起隐式学习语言，导致「模型到底学到了什么不可知」，且无法利用大规模无标注文本预训练。[1][2]

## 关键方法

### ABINet 的三条原则

作者从人类阅读的心理学观察出发，提出 autonomous、bidirectional、iterative 三条设计原则。[2]

**① Autonomous（自治）—— 阻断梯度流 BGF**

在视觉模型 VM 与语言模型 LM 之间**阻断梯度**，强制 LM 独立学习语言规则。[2]

思想上，这把 STR 拆成两个可独立预训练的单元：VM 从图像学、LM 从纯文本学。[2]

消融显示，若放开梯度流（AGF），性能平均**下降 0.9%**，且训练 loss 骤降——说明 LM 在靠 VM「作弊」而过拟合。[2]

**② Bidirectional（双向）—— BCN 双向填空网络**

作者用信息论量化了动机：双向表示可用熵为 $H_y=(n-1)\log c$，而单向表示只捕获 $\tfrac{1}{n}\sum_{i=1}^{n}(i-1)\log c=\tfrac{1}{2}H_y$。[2]

也就是说，此前把 LTR 与 RTL 两个单向模型做集成，**本质上仍是单向表示，只拿到一半信息**。[2]

**为什么不用 BERT 的 MLM？** 最直接的想法是照搬 MLM：把 $y_i$ 换成 `[MASK]` 让它填。[2]

但论文指出这**不可接受**：一条文本有 $n$ 个字符，每个位置都要单独 mask 一次、各跑一次前向，也就是 **$n$ 次前向**，效率极低。[2]

BCN 的关键转向是：**不 mask 输入字符，而是 mask 注意力**。[2]

**结构上的三处改动。** BCN 是 $L$ 层 Transformer decoder 的变体，与 vanilla Transformer 有三点不同：[2]

1. **字符向量直接喂入多头注意力块**，而不是先过第一层网络；
2. 多头注意力里用掩码**阻止「看到自己」**；
3. **完全不施加自注意力**，以避免跨时间步泄漏信息。

第 3 点尤其关键：没有自注意力，每个位置就无法从「自己的概率」反推自己，只能从「别人的概率」推断，这才构成真正的填空。[2]

**Q 的来源是分层的。** 第一层 $Q$ 取位置编码，之后各层取上一层输出：$Q\in\mathbb{R}^{T\times C}$ 在首层是字符顺序的位置编码，其余层则沿用上一层的输出。[2]

而 $K,V$ 始终来自字符概率向量 $P(y_i)\in\mathbb{R}^{c}$ 经线性映射 $W_l\in\mathbb{R}^{c\times C}$ 得到，$c$ 为字符集大小。[2]

$$M_{ij}=\begin{cases}0,&i\ne j\\[4pt] -\infty,&i=j\end{cases},\qquad K_i=V_i=P(y_i)W_l,\qquad F_{mha}=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{C}}+M\right)V$$

注意 $K_i=V_i$ 同源：字符概率既做 key 又做 value，意味着**视觉错误是以「噪声」的形式直接进入 LM 的**——这正是后面必须做迭代纠错的原因。[2]

**收益与代价。** 相比集成两个单向模型，**参数与计算量只需一半**，且比 SRN 快 20%–25%。[2]

作为独立拼写纠正器评测时，BCN 比 SRN 高 **4.5% 字符准确率、14.3% 词准确率**（82.8/41.9 对 78.3/27.6）。[2]

**它是怎么工作的：一个可视化例子。** 论文以 “today” 为例展示 BCN 的 top-5 概率行为：[2]

- 输入 `-oday`、`tod-y` 时，BCN 能**高置信度**补出 `t` 与 `a`，并对最终融合预测有实际贡献；
- 但当输入里混入错误字符（如 `l`、`o`）时，BCN 会**变得不自信**，对最终预测影响很小。

也就是说，BCN 会判断自己的输入可不可信：可信就大声纠，不可信就少说话。[2]

**它的失效边界。** 论文明确指出：若同时存在多个错误字符，上下文不足以支撑恢复，BCN 很难纠正回来。[2]

这也正是 PARSeq 后来攻击它的基础——词准确率 41.9% 意味着它经常改错，而它又看不到图像，无从判断该不该改。[1]

**③ Iterative（迭代）—— 逐步纠错**

LM 重复执行 $M$ 次：第 1 次用 VM 的概率预测，之后每次用上一轮融合模型的预测。[2]

这解决两个问题：一是**噪声输入**（VM 的错误会污染 LM），二是 Transformer 的 **unaligned-length** 问题（预测长度与真值不符时无法纠正）。[2]

消融：迭代 3 次平均提升 0.4%/0.3%，在 SVT 上可达 **1.3%**；超过 3 次即饱和。[2]

**④ 融合与自训练**

融合用门控机制 $G=\sigma([F_v,F_l]W_f)$，$F_f=GF_v+(1-G)F_l$。[2]

自训练部分用「迭代预测的集成」构造高质量伪标签，以最小置信度 $C=\min_t e^{\mathbb{E}[\log P(y_t)]}$ 过滤。[2]

### PARSeq 的三层创新

**① 用 PLM 替代标准 AR 训练**

标准 AR 只在单一顺序 $[1,2,\dots,T]$ 上做因式分解；PLM 则对所有 $T!$ 种排列求期望：[1]

$$\log p(y\mid x)=\mathbb{E}_{z\sim\mathcal{Z}_T}\left[\sum_{t}\log p_\theta(y_{z_t}\mid y_{z_{<t}},x)\right]$$

实现上不需要真的打乱标签，只要**换注意力掩码**就能表达不同分解顺序。[1]

关键洞察：**标准 AR 训练只是 PLM 中只取一个排列的特例**。[1]

**② 位置 token 解耦「目标位置」与「输入上下文」**

这是 PARSeq 实现统一解码的技术核心：位置 token $p$ 同时充当 query 与位置嵌入，而上下文 $c$ 只作 K/V。[1]

$$h_c=p+\mathrm{MHA}(p,c,c,m),\qquad h_i=h_c+\mathrm{MHA}(h_c,z,z)$$

作者明确指出：如果没有位置 token（像标准 Transformer 那样让上下文自己当 query），**模型从 PLM 里学不到任何有意义的东西，会退化成普通 AR 模型**。[1]

**③ 一个模型，三种解码**

- **AR 解码**：左到右掩码，逐 token 生成
- **NAR 解码**：不加掩码，一次并行输出全部 token
- **迭代精炼**：cloze 掩码（全 1 再遮掉自身位置），在任一初始解码后反复修正

三者的切换**只改掩码，不改权重**。[1]

## 结论与局限

### ABINet 的关键数字

| 设置 | IC13 | SVT | IIIT | IC15 | SVTP | CUTE |
|---|---|---|---|---|---|---|
| ABINet-LV（合成数据） | 97.4 | 93.5 | 96.2 | 86.0 | 89.3 | 89.2 |
| ABINet-LVest（+自训练） | 97.7 | 95.5 | 97.2 | 86.9 | 89.9 | 94.1 |

优势集中在 SVT、SVTP、IC15 这类**低质量图像**多的数据集——正是视觉模型无法自信识别、需要语言补足的场景。[2]

ABINet-LV 相比同配置 SRN-LV 提升：IC13 +0.6%、SVT +1.2%、IC15 +1.8%、SVTP +1.4%、CUTE +1.0%。[2]

### PARSeq 的关键数字

| 训练数据 | 36 字符 | 62 字符 | 94 字符 |
|---|---|---|---|
| PARSeqA（合成） | **91.9** | 75.5 | 73.0 |
| ABINet（合成） | 89.8 | 68.5 | 66.4 |
| PARSeqA（真实） | **96.0** | 94.6 | 93.3 |
| ABINet（真实） | 95.2 | 93.7 | 92.4 |

在更大更难的数据集（ArT / COCO / Uber）上，PARSeqA 用合成数据达 51.8%，用真实数据达 **84.1%**，均为第一。[1]

### 两篇的核心分歧（本笔记重点）

**关键差异在于「精炼阶段是否看图像」**：ABINet 的 LM 与图像**条件独立**，PARSeq 的精炼**同时看图像与语言上下文**。[1]

PARSeq 对 ABINet 的批评很具体：ABINet 的 LM 词准确率只有 **41.9%**，却在整体算力中占很大比重，属于「相对参数量严重未被充分利用」；而且它与图像条件独立，可能**错误地「纠正」本来正确的预测**。[1]

一个佐证：PARSeq 复现 ABINet 时，在更短的训练日程下准确率**下降约 1.4%**——说明 ABINet 的成绩依赖长日程与 WikiText-103 预训练等额外设置。[1]

但反过来看，ABINet 的**可解释性**是 PARSeq 没有的：因为 LM 是独立单元，可以直接把它当拼写纠正器评测（这正是上面 41.9% 这个数字的来源）。[2]

### 各自的局限

**ABINet**：
- 迭代使推理时间**线性增长**（1→2→3 次：22ms→27.3ms→33.9ms）。[2]
- LM 与图像条件独立，存在误纠风险，必须靠额外融合层兜底。[1]
- 多个错误字符同时出现时，BCN 因上下文不足难以恢复。[2]

**PARSeq**：
- 需训练 $K$ 个排列，$K=6$ 时训练时间从 5.86h 增至 8.48h。[1]
- $K=1$（退化为标准 AR）时 **NAR 解码完全失效**（0.01%），说明统一性依赖 PLM 训练。[1]
- 单层解码器是刻意设计，但也意味着解码端容量有限。[1]

## 我的疑问

- [ ] ABINet 的 BGF 与 SRN 的 argmax（不可导，天然等价于 BGF）究竟差在哪？论文说 argmax 是 BGF 的特例，那增益是否全来自「不取 argmax 而保留概率分布」？
- [ ] PARSeq 的 $K=6$ 是针对 94 字符集选的，换成中文等大类字符集，$K$ 是否要重新调？
- [ ] 若把 PARSeq 的「精炼时看图像」思想反哺 ABINet，BCN 的词准确率能提升多少？

## 关键摘录

> "The conditional independence of the external LM on the input image may cause it to erroneously rectify correct predictions, leading to significant inefficiencies." [1]

> "we argue that the limited capacity of language models comes from: 1) implicitly language modeling; 2) unidirectional feature representation; and 3) language model with noise input." [2]

> "Without the position tokens ... the model will not learn anything meaningful from PLM and will simply function like a standard AR model." [1]

## 相关笔记

- [[SVTR 场景文本识别]] — 同期 STR 骨干网络方案，可与两篇的解码头对比
- [[ASTER 注意力场景文本识别器]] — 早期注意力式 STR，是本文批判的「隐式语言建模」代表
- [[DEATRN v2 可变形集成注意力任意形状场景文本识别]] — 处理不规则形状文本
- [[STIRER 统一框架的低分辨率场景文本恢复与识别]] — 同样是「统一框架」思路
- [[曝光偏差 Exposure Bias]] — PARSeq 的 AR 与迭代精炼正是缓解该问题的工程实践

### 待建笔记

以下概念本文涉及但库中尚无独立笔记：Permutation Language Modeling、双向填空网络 BCN、门控融合、Ensemble Self-training。

## Sources

[1] https://arxiv.org/abs/2207.06966 — Bautista & Atienza 2022 — PARSeq (ECCV)
[2] https://arxiv.org/abs/2103.06495 — Fang et al. 2021 — ABINet (CVPR)