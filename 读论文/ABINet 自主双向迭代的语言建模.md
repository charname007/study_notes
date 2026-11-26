---
created: 2026-11-26
updated: 2026-11-26
source: E:\.dsh-paper-reader\data\STR\ABINet\ABINet.pdf（arXiv:2103.06495, CVPR 2021）
type: 论文
tags: [论文, 场景文本识别, 自回归, Attention]
---

# ABINet：自主、双向、迭代的语言建模

## 一句话总结

把语言模型从「与视觉耦合的隐式学习」中拆出来，做成一个**独立、可单独预训练、可单独评测**的拼写纠正单元；再用**双向填空**与**迭代纠错**把它的能力推到集成式单向模型的水平之上，同时只要一半参数。

## 核心问题

STR 的困难在于：遮挡、模糊、噪声会让视觉特征不足以判定字符，此时必须靠语言语义补足。[1]

但此前主流的注意力式方法（R2AM、ASTER 等）采用 encoder-decoder 结构，把 LM 与视觉模型**耦合**在一起隐式学习语言。作者认为这带来三个具体缺陷：[1]

1. **隐式语言建模**：模型到底有没有学到字符间关系、学到多少，无法知晓；[1]
2. **单向特征表示**：语言上下文只从左到右，信息只用了一半；[1]
3. **语言模型接收噪声输入**：自回归把上一步的错误当输入，错误会累积。[1]

作者从人类阅读的心理学观察引出三条原则，作为整个设计的骨架：[1]

| 原则 | 心理学依据 | 技术落点 |
|---|---|---|
| **Autonomous** 自治 | 聋人与盲人可分别拥有完好的语言与视觉 | VM / LM 解耦，阻断梯度流 BGF |
| **Bidirectional** 双向 | 推理语境像完形填空，左右线索可同时用 | BCN 双向填空网络 |
| **Iterative** 迭代 | 人会在困难时反复修正以提高置信度 | 迭代纠错 |

## 关键方法

### ① Autonomous：阻断梯度流 BGF

在视觉模型 VM 与语言模型 LM 之间**阻断梯度**，强制 LM 独立学习语言规则。[1]

这把 STR 拆成两个可独立预训练的单元：VM 从图像学、LM 从纯文本学。[1]

消融显示，若放开梯度流（AGF），性能平均**下降 0.9%**，且训练 loss 骤降——说明 LM 在靠 VM「作弊」而过拟合。[1]

论文还指出一个细节：SRN 在 VM 之后用了 argmax，而 argmax 不可导，**本质上就是 BGF 的一个特例**。[1]

解耦带来的直接好处是**可解释性**：LM 可以作为独立单元单独评测，看它到底会不会拼写纠正（见下方「BCN 独立评测」）。[1]

### ② Bidirectional：BCN 双向填空网络

作者用信息论量化了动机：双向表示可用熵为 $H_y=(n-1)\log c$，而单向表示只捕获 $\tfrac{1}{n}\sum_{i=1}^{n}(i-1)\log c=\tfrac{1}{2}H_y$。[1]

也就是说，此前把 LTR 与 RTL 两个单向模型做集成，**本质上仍是单向表示，只拿到一半信息**。[1]

**为什么不用 BERT 的 MLM？** 最直接的想法是照搬 MLM：把 $y_i$ 换成 `[MASK]` 让它填。[1]

但论文指出这**不可接受**：一条文本有 $n$ 个字符，每个位置都要单独 mask 一次、各跑一次前向，也就是 **$n$ 次前向**，效率极低。[1]

BCN 的关键转向是：**不 mask 输入字符，而是 mask 注意力**。[1]

**结构上的三处改动。** BCN 是 $L$ 层 Transformer decoder 的变体，与 vanilla Transformer 有三点不同：[1]

1. **字符向量直接喂入多头注意力块**，而不是先过第一层网络；
2. 多头注意力里用掩码**阻止「看到自己」**；
3. **完全不施加自注意力**，以避免跨时间步泄漏信息。

第 3 点尤其关键：没有自注意力，每个位置就无法从「自己的概率」反推自己，只能从「别人的概率」推断，这才构成真正的填空。[1]

**Q 的来源是分层的。** 第一层 $Q$ 取位置编码，之后各层取上一层输出。[1]

而 $K,V$ 始终来自字符概率向量 $P(y_i)\in\mathbb{R}^{c}$ 经线性映射 $W_l\in\mathbb{R}^{c\times C}$ 得到，$c$ 为字符集大小。[1]

$$M_{ij}=\begin{cases}0,&i\ne j\\[4pt] -\infty,&i=j\end{cases},\qquad K_i=V_i=P(y_i)W_l,\qquad F_{mha}=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{C}}+M\right)V$$

注意 $K_i=V_i$ 同源：字符概率既做 key 又做 value，意味着**视觉错误是以「噪声」的形式直接进入 LM 的**——这正是后面必须做迭代纠错的原因。[1]

**收益。** 相比集成两个单向模型，**参数与计算量只需一半**，且比 SRN 快 20%–25%。[1]

**它是怎么工作的：一个可视化例子。** 论文以 "today" 为例展示 BCN 的 top-5 概率行为：[1]

- 输入 `-oday`、`tod-y` 时，BCN 能**高置信度**补出 `t` 与 `a`，并对最终融合预测有实际贡献；
- 但当输入里混入错误字符（如 `l`、`o`）时，BCN 会**变得不自信**，对最终预测影响很小。

也就是说，BCN 会判断自己的输入可不可信：可信就大声纠，不可信就少说话。[1]

**失效边界。** 若同时存在多个错误字符，上下文不足以支撑恢复，BCN 很难纠正回来。[1]

### ③ Iterative：迭代纠错

LM 重复执行 $M$ 次：第 1 次用 VM 的概率预测，之后每次用上一轮**融合模型**的预测。[1]

这解决两个问题：[1]

- **噪声输入**：VM 的错误会污染 LM，反复用融合结果喂回可逐步净化；
- **unaligned-length 问题**：Transformer 类方法在预测长度与真值不符时无法纠正，因为 padding mask 是固定的；多次融合后预测长度也会被逐步修正。

消融：迭代 3 次平均提升 0.4%/0.3%，在 SVT 上可达 **1.3%**；超过 3 次即饱和，说明迭代数不必大。[1]

论文还区分了训练期与测试期迭代：测试期直接迭代也有效，训练期迭代则相当于给 LM 提供额外训练样本。[1]

### ④ 融合与自训练

视觉与语言来自不同模态，用**门控机制**做最终决策：[1]

$$G=\sigma([F_v,F_l]W_f),\qquad F_f=GF_v+(1-G)F_l$$

其中 $W_f\in\mathbb{R}^{2C\times C}$，$G\in\mathbb{R}^{T\times C}$。[1]

**半监督自训练**把「迭代预测」当作一个集成来构造高质量伪标签，用最小置信度过滤：[1]

$$C=\min_{1\le t\le T}e^{\mathbb{E}[\log P(y_t)]},\qquad P(y_t)=\max_{1\le m\le M}P_m(y_t)$$

即：先取各次迭代的最大概率，再取整条文本的最小值作为「这条文本的确定度」，低于阈值 $Q$ 的丢弃。[1]

### 视觉模型（配套）

VM 由 ResNet + Transformer + **位置注意力**组成：$F_b=\mathcal{T}(R(x))$，再用 query 范式把视觉特征转成字符概率：[1]

$$F_v=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{C}}\right)V$$

其中 $Q$ 是字符顺序的位置编码，$K$ 经 mini U-Net 得到，$V$ 为恒等映射。[1]

论文的 VM 消融结论：升级 VM 能带来明显精度提升但有参数与速度代价；推荐的升级方向是「特征提取用位置注意力 + 序列建模用更深的 Transformer」。[1]

## 结论与局限

### 主结果

| 设置 | IC13 | SVT | IIIT | IC15 | SVTP | CUTE |
|---|---|---|---|---|---|---|
| ABINet-LV（MJ+ST） | 97.4 | 93.5 | 96.2 | 86.0 | 89.3 | 89.2 |
| ABINet-LVest（+Uber-Text 自训练） | 97.7 | 95.5 | 97.2 | 86.9 | 89.9 | **94.1** |

优势集中在 SVT、SVTP、IC15 这类**低质量图像**多的数据集——正是视觉模型无法自信识别、需要语言补足的场景。[1]

相比同配置的 SRN-LV，ABINet-LV 提升：IC13 +0.6%、SVT +1.2%、IC15 +1.8%、SVTP +1.4%、CUTE +1.0%。[1]

### 关键消融

**BCN 独立评测**（把它当纯拼写纠正器，输入是人工加噪文本）：[1]

| 语言模型 | 字符准确率 | 词准确率 |
|---|---|---|
| SRN（集成式单向） | 78.3 | 27.6 |
| **BCN** | **82.8** | **41.9** |

**+4.5% 字符、+14.3% 词准确率**，参数与算力却只需集成方案的一半。[1]

**双向表示的公平对比**（同配置、仅换网络）：[1]

| VM | LM | 平均准确率 | 参数 | 时间 |
|---|---|---|---|---|
| LV | SRN（单向） | 90.6 | 45.4M | 24.2ms |
| LV | SRN（双向集成） | 91.9 | 49.3M | 26.9ms |
| LV | **BCN** | **92.3** | 36.7M | 22.0ms |

BCN 以**更少参数、更快速度**超过双向集成。[1]

**迭代纠错**（LV + BCN）：1 次 92.3 → 2 次 92.4 → 3 次 **92.6**，但延迟 22 → 27.3 → 33.9ms 线性增长。[1]

## 我的疑问

- [ ] BGF 与 SRN 的 argmax 究竟差在哪？论文说 argmax 是 BGF 特例，那么增益是来自「不取 argmax 而保留完整概率分布」，还是来自别的？
- [ ] BCN 的 $K=V$ 同源设计是否有必要？让 $K,V$ 各自独立投影会不会更好？
- [ ] 若把 PARSeq 的「精炼时同时看图像」思想迁移过来，BCN 的 41.9% 词准确率能提升多少？

## 关键摘录

> "we argue that the limited capacity of language models comes from: 1) implicitly language modeling; 2) unidirectional feature representation; and 3) language model with noise input." [1]

> "as both deaf-mute and blind people could have fully functional vision and language separately, we use the term autonomous to interpret the independence of learning between vision and language." [1]

> "we notice that this is unacceptable as MLM should be separately called n times for each text instance, causing extreme low efficiency. Instead of masking the input characters, we propose BCN by specifying the attention masks." [1]

## 相关笔记

- [[PARSeq 用排列语言建模统一场景文本识别]] — 同期竞争方案，批评 ABINet 的 LM 与图像条件独立导致误纠
- [[SVTR 场景文本识别]] — 纯视觉骨干路线，可与本文的语言补足思路对照
- [[ASTER 注意力场景文本识别器]] — 早期注意力式 STR，是本文批判的「隐式语言建模」代表
- [[STIRER 统一框架的低分辨率场景文本恢复与识别]] — 同为「统一框架」思路的对照
- [[曝光偏差 Exposure Bias]] — 自回归的误差累积正是本文「迭代纠错」要解决的问题

### 待建笔记

以下概念本文涉及但库中尚无独立笔记：双向填空网络 BCN（可考虑在此笔记基础上拆出）、门控融合、Ensemble Self-training、unaligned-length 问题。

## Sources

[1] https://arxiv.org/abs/2103.06495 — Fang et al. 2021 — ABINet (CVPR)