---
created: 2026-11-26
updated: 2026-11-26
source: 综合 Bengio et al. 2015 (Scheduled Sampling)、Ranzato et al. 2016 (MIXER)、Schmidt 2019、He et al. 2021、Arora et al. 2022 等原文与 2025 综述整理
type: 概念
tags: [概念, 自回归, 大语言模型]
---

# 曝光偏差 Exposure Bias

> 自回归模型训练时读的是「标准答案前缀」，推理时却只能读「自己刚生成的前缀」。两者分布不一致，早期一个小错会作为下一步输入被不断放大——这就是 exposure bias。它几乎是所有「改进自回归训练」工作的出发点，但 2019 年之后，它到底有多严重、是否被夸大，一直有争议。这篇把定义、数学、缓解谱系，以及正反两方的实证证据一次理清。

## 是什么

### 1. 先分清两个词：Teacher Forcing 与 Exposure Bias

- **Teacher forcing**（Williams & Zipser, 1989）：训练第 $t$ 步时，输入用真实前缀 $y_{t-1}$，而不是模型自己的预测 $\hat y_{t-1}$。它是最大似然 + 链式法则的直接结果：$\log p(y_{1:T}\mid x)=\sum_{t}\log p(y_t\mid y_{<t},x)$。优点：各步可并行、训练稳定、误差不在训练中累积。
- **Exposure bias**：由 teacher forcing 造成的**训练/推理分布错位**这个「症状」。两者常被混用，但严格说 teacher forcing 是（可能的）因，exposure bias 是果；Schmidt 2019 特别强调了这一区分。

### 2. 形式化：错在「前缀分布」不同

MLE 训练（前缀来自数据分布 $d_{\text{data}}$，每个 token 都正确）：

$$\mathcal L_{\text{MLE}}=-\sum_{t=1}^{T}\log p_\theta\big(y_t \mid y_{<t}, x\big), \qquad y_{<t}\sim d_{\text{data}}$$

推理 / free-running（前缀来自模型自身分布 $d_{\pi_\theta}$，可能含错）：

$$\hat y_t \sim p_\theta\big(\cdot \mid \hat y_{<t}, x\big), \qquad \hat y_{<t}\sim d_{\pi_\theta}$$

模型从没在「带错的前缀」上练过，于是落入训练未见过的状态空间：一个错 token → 下一步输入偏移 → 更容易再错，即 **compounding errors / 雪球效应**。「曝光」二字即指训练时模型只被 *exposed* 到正确前缀、没被 *exposed* 到自己的错误（Ranzato et al. 2016 命名原话：「only exposed to the training data distribution, instead of its own predictions」）。

## 为什么重要

误差会累积，而且不是线性累积。

- Ranzato et al. 2016 在提出 MIXER 时，把 exposure bias 与「词级损失 vs 序列级指标」并列为文本生成的两大缺陷。
- **模仿学习视角**（DAgger, Ross et al. 2011）：设策略在训练分布上单步误差为 $\epsilon$，则在自身状态访问分布上，整条长度 $T$ 轨迹的累计误差可达 $O(\epsilon T^2)$（二次放大），而非独立误差假设下的 $O(\epsilon T)$。直观例子：10 步推理链、每步 95% 正确率，整体正确率仅 $0.95^{10}\approx 60\%$。⚠️ 该界源于交互式模仿学习/蒸馏，套到 LLM 上要谨慎——教师模型在 OOD 前缀上自身也可能失准。
- 它也被用来解释**幻觉、重复、文本退化**等生成病态。

## 怎么做 / 关键细节

### 4. 缓解方法谱系（一张表）

| 方法 | 核心思想 | 解决什么 | 代价 / 局限 |
|---|---|---|---|
| **Scheduled Sampling**（Bengio 2015） | 训练时以概率 $\epsilon_i$ 掷硬币决定喂真值 $y_{t-1}$ 还是模型预测 $\hat y_{t-1}$，$\epsilon_i$ 随训练按 linear / exp / inverse-sigmoid curriculum 衰减 | 让模型提前适应自己的前缀 | 目标不再是严格似然（上下文混合、监督却仍是原始 gold）；Huszár 2015 指出它是有偏/不一致估计，可能推向不连贯模式 |
| **DAgger / DAD**（Ross 2011；Venkatraman 2015） | 迭代把「learner 访问到的状态 + 专家纠正动作」并入训练集，直接对齐状态分布 | 从根上对齐 $d_{\text{data}}$ 与 $d_{\pi}$ | 需要能给出最优续写的**动态 oracle**；对 BLEU/ROUGE 等不可分解指标只能近似（beam search），no-regret 保证随之失效 |
| **序列级训练 / RL**（MIXER, Ranzato 2016） | REINFORCE 直接优化 BLEU/ROUGE，训练即用模型自身预测；从 XENT 最优策略逐步「退火」到自由生成 | 同时解决 exposure bias 与 metric mismatch | 大词表下动作空间 $\mathcal O(\mathcal W^T)$ 巨大，随机策略起步学不动，需 curriculum |
| **Professor Forcing**（Goyal 2016） | 对抗训练，让 free-running 隐藏状态分布匹配 teacher-forced 分布 | 不改动输入 token，改隐状态分布 | 引入判别器 + GAN 式不稳定 / mode collapse |
| **最小风险训练 MRT**（Wang & Sennrich 2020） | 用 risk loss 从「曝光偏差 ↔ 幻觉 ↔ 域偏移」角度缓解 | 序列级目标 | 需采样、训练成本高 |
| **对比 / 逆RL / on-policy 蒸馏**（2025 综述） | 把「让模型在自己的输出分布上练」搬到蒸馏与 LLM | 当代 LLM 长链推理的误差累积 | 仍需稳定教师动力学，否则 OOD 前缀下教师也失准 |

### 5. 争议：它到底有多严重？（本笔记的重点）

这是最容易被忽略、也最能体现深度的一节——**证据是混合的，结论并不统一**。

- **Schmidt 2019《Generalization in Generation》**：标准训练是自回归 + 最大似然的直接推论；只要以 log-likelihood 为准，用真值前缀「无可厚非」。exposure bias 本质上描述的是相对于**另一个（往往隐含、任务相关）指标**的**泛化不足**。很多「缓解方法」其实只是在做正则化，且它们只报告 test log-likelihood 的微小提升，从未把 exposure bias 与一般泛化分离开；一旦专门去测显式效应，往往没有显著发现。
- **He et al. 2021（EMNLP）「Exposure Bias vs Self-Recovery」**：设计「前缀切换」实验，给同一个 MLE 语言模型分别喂真值前缀 / 模型前缀 / 打乱前缀 / 随机前缀，量化质量、多样性、一致性的下降。结论反直觉：前缀不一致带来的退化仅约 **1–2%**，且**不随前缀变长而递增**（用到长度 100 的前缀也没加剧），与「incremental distortion」的预测矛盾；模型存在 **self-recovery（自恢复）**能力，能从被扰动的历史里恢复回来。
- **Arora et al. 2022（Findings of ACL）「Why Exposure Bias Matters」**：从模仿学习视角重新检验，认为 exposure bias **确实**导致误差累积，并指出 **perplexity 捕捉不到这种累积**，最终拉低生成质量——支持「它确实重要」。
- **现状**：He 2021 与 Arora 2022 看似对立，但任务设定不同（开放端生成 vs 模仿学习分析）、度量不同，且都承认原始定义不够严谨、量化只能算 proxy。不可简单说「已被证伪」或「铁证如山」。到了大 LLM 时代，这场争论借 **on-policy 蒸馏 / 长链推理误差累积**重新活跃。

## 常见误区 / 注意事项

### 6. 一句话与心智模型

- **一句话**：exposure bias = 训练只看正确前缀、推理要读自己可能写错的前缀，两者分布错位，使早期错误沿序列被放大。
- **因果链**：MLE + teacher forcing → 前缀分布 $d_{\text{data}}\ne d_{\pi_\theta}$ → 训练未见状态 → compounding errors。
- **判断口诀**：谈缓解前先问「以哪个指标为准」。指标若就是 log-likelihood，teacher forcing 没问题；若是序列级/任务级指标、幻觉、或长链推理可靠性，才需要 scheduled sampling / DAgger / RL / on-policy。
- **与相邻概念的边界**：它是**训练—推理分布偏移**的一种，不等于数据层面的「曝光不足」，也不同于 decoding 策略——beam search 只是缓解症状、不治本。

### 7. 追问：自回归是不是「必然」有这个问题？大模型也逃不掉吗？

把问题拆成三层，争论才不会错位：这个「错位」本身是否定义性存在、它是否必然造成严重退化、现在的大模型是否必然受害。

#### 1. 错位是结构性的 —— 这是必然的

只要训练读真值前缀、推理读自己前缀，两个前缀分布就必然不同，这与模型多大、数据多少无关，是 teacher forcing 的定义直接决定的。

关键在于：MLE 训练最小化的是 $\arg\min_\theta D_{\mathrm{KL}}(P_D \,\|\, P_\theta)$，即让模型分布逼近数据分布。

因此 teacher forcing 本身并不是「有偏的目标函数」，而只是链式法则 + 最大似然的直接结果。真正的「偏差」出现在**有限模型、有限数据、有限长度**的组合里：只要单步误差 $\epsilon>0$，$O(\epsilon T^2)$ 的累积界就成立。

#### 2. 错位存在 ≠ 危害严重 —— 三个反驳

- **最优解处会消失**：若容量足够、训练充分使 $P_\theta=P_D$，则 $\epsilon\to0$，exposure bias 随之消失。它是「近似误差与泛化不足」的问题，不是数学上不可消除的固有缺陷（Schmidt 的 Proposition 1）。
- **自恢复能力**：He et al. 2021 发现真值前缀与模型前缀的质量差距只有 1%–2%，且不随前缀变长到 100 而放大；模型甚至能从打乱、乃至随机前缀中恢复出合理文本。
- **KL 方向本身不对称**：训练最小化 forward KL $D_{\mathrm{KL}}(P_D\|P_\theta)$，倾向覆盖所有模式（mode-covering），会过度泛化；而评测指标与人类偏好更像 reverse KL，倾向集中到最大模式（mode-seeking）。两个方向的行为本就不同，所以「训练—评测错位」并不全由 teacher forcing 造成。

反方证据同样硬：Arora et al. 2022 直接测量发现误差累积呈**近二次增长**，且 perplexity 完全捕捉不到这种累积。两者不矛盾——自恢复压制了平均意义的退化，但长程关键决策点上的错误仍会累积。

#### 3. 大模型必然吗？—— 架构必然，工程在治

- **架构上必然**：只要还是自回归 + teacher forcing，错位就还在，不会因参数量变大而消失。
- **工程上已被当作一等问题**：现代 LLM 的后训练几乎全是 **on-policy** 的。RLHF / RLVR / GRPO 都在模型自己采样的轨迹上算梯度，这正是 **DAgger 的大规模工程版**；拒绝采样微调（STaR / RAFT）用模型自己生成并通过验证的数据再训练；on-policy 蒸馏则明确以消除「Off-Policy Exposure Bias」为目标（学生采样自己的轨迹，教师给稠密监督）。这也解释了为什么 chat 模型比 base 模型抗造得多。

**规模为什么会帮忙？** 流行的悲观论证是：每 token 独立错误率 $e$，长度 $n$ 时正确率 $(1-e)^n\to0$。但这个论证脆弱：实测只有约 5%–10% 的 token 真正依赖长程上下文，其余随上下文累积反而更好预测；LLM 嵌入呈分层低维流形，模型一旦处在某语义区域就倾向留在那里；而且错误是**样本间特异**而非系统性的，这正是多数投票能挽回大量准确率的原因。于是两速率模型 $P\approx(1-e_{\text{key}})^{k}(1-e_{\text{non}})^{n-k}$ 预测的衰减至多是拉伸指数、常是幂律，有时甚至恒定。

**但长输出依然危险**：小模型在长 CoT 上会崩——仅 8k 长 CoT 样本即可让某些模型损失最多 75% 原始性能，即使给到 220k 样本，最小模型仍回不到基线。大模型恢复得更快更彻底（14B 在 16k 样本后即超过基线），说明规模 + 数据确实在压制该效应。

#### 4. 想从结构上绕开：非自回归与扩散 LM

自回归的根本问题是单向、从左到右、不可回头，早期错误被永久固化。扩散语言模型一次性对整块 token 并行去噪，可迭代修正，天生具备纠错能力。块扩散（BD3-LM）在块间自回归、块内扩散，兼顾变长生成与 KV 缓存；LLaDA 把掩码扩散推到 8B 并配合块扩散采样，Gemini Diffusion 以 256 token 为块去噪并主打迭代精修。

代价是真实的：似然建模弱于自回归，块大小为 1 时仍有约 2 点 perplexity 差距，且梯度方差偏大。所以它只是「用不同方式绕开」，不是免费午餐。

### 一句话结论

**错位必然存在，危害并非必然。** 短输出、封闭任务几乎无感；长链推理与开放生成显著且随长度放大。大模型不是豁免，而是把它纳入了训练闭环——on-policy 后训练让模型在自己会犯的错上练习恢复，本质是 DAgger 的规模化。要从结构上摆脱它，就得换生成范式，代价是似然与效率上的让步。

## 相关笔记

- [[DVAR 动态视觉自回归单图超分]] — 视觉自回归生成的论文实例，可直接观察自回归解码的误差累积
- [[DVAR 的动态尺度与采样调度器实现]] — 自回归推理中尺度与采样调度的代码实现
- [[RDA 残差解码适配器 保持ID的自回归文字渲染]] — 自回归文字渲染，长序列误差累积的另一实例
- [[对比损失谱系 InfoNCE 到 Sigmoid]] — 另一条「训练目标与评测目标错位」的线索，可与本文的 reverse KL 讨论对照

### 待建笔记

以下概念本文反复引用但库中尚无独立笔记，暂以纯文本记录：Teacher Forcing、Scheduled Sampling、DAgger 与模仿学习、束搜索 Beam Search、神经文本退化、on-policy 蒸馏。

## 参考来源

[1] Bengio, Vinyals, Jaitly, Shazeer (2015). *Scheduled Sampling for Sequence Prediction with RNNs*. https://arxiv.org/abs/1506.03099
[2] Ranzato, Chopra, Auli, Zaremba (2016). *Sequence Level Training with RNNs (MIXER)* — 术语出处. https://arxiv.org/abs/1511.06732
[3] Schmidt (2019). *Generalization in Generation: A closer look at Exposure Bias*. https://arxiv.org/abs/1910.00292
[4] He, Zhang, Zhou, Glass (2021). *Exposure Bias versus Self-Recovery*. EMNLP. https://aclanthology.org/2021.emnlp-main.415/
[5] Arora et al. (2022). *Why Exposure Bias Matters: An Imitation Learning Perspective*. Findings of ACL. https://aclanthology.org/2022.findings-acl.58/
[6] QuiddityML — *Teacher forcing and exposure bias explained*. https://quiddityml.com/blog/teacher-forcing-and-exposure-bias/
[7] mbrenndoerfer — *Teacher Forcing / Scheduled Sampling*. https://mbrenndoerfer.com/writing/teacher-forcing-seq2seq-training-exposure-bias-scheduled-sampling
[8] AI Compendium — *Sequence Models*（含 Huszár 对 scheduled sampling 的批评）. https://theaicompendium.com/5-DL/sequence-models.html
[9] (2025) *Mitigating exposure bias in LLM distillation: an imitation learning approach*（缓解方法综述 / DAgger 界）. https://link.springer.com/article/10.1007/s00521-025-11162-0
[10] (2024) *Improving Autoregressive Training with Dynamic Oracles*（DAgger + 动态 oracle）. https://arxiv.org/html/2406.09393
[11] Goyal et al. (2016). *Professor Forcing*. https://arxiv.org/abs/1610.09038
[12] (2021) *Relating Neural Text Degeneration to Exposure Bias*. https://arxiv.org/abs/2109.08705

### 第七节补充来源（自回归必然性与大模型）

[13] (2025) *Beyond Exponential Decay: Rethinking Error Accumulation in LLMs*. https://arxiv.org/html/2505.24187
[14] (2019) *Rethinking Exposure Bias in Language Modeling*（forward vs reverse KL 不对称）. https://arxiv.org/abs/1910.11235
[15] (2026) *LLM Post-Training: A Unified View of Off-Policy and On-Policy Learning*. https://arxiv.org/html/2604.07941v2
[16] (2026) *A Survey of On-Policy Distillation for LLMs*（Off-Policy Exposure Bias 与 $O(\epsilon T^2)$）. https://arxiv.org/html/2604.00626v4
[17] (2025) *Through the Valley: Path to Effective Long CoT Training for Small LMs*（Long CoT Degradation）. https://arxiv.org/html/2506.07712
[18] (2025) *Block Diffusion: Interpolating Between Autoregressive and Diffusion LMs*. https://arxiv.org/abs/2503.09573
[19] (2026) *How To Build A Modern Diffusion Language Model*（LLaDA / Gemma Diffusion / 块扩散）. https://arxiv.org/abs/2610.33391
[20] (2024) *Preference Fine-Tuning of LLMs Should Leverage Suboptimal, On-Policy Data*. https://arxiv.org/abs/2404.14367
[21] Google DeepMind — *Gemini Diffusion*（256 token 块去噪、迭代精修）. https://deepmind.google/models/gemini-diffusion/
[22] (2025) *A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce*（RAFT / STaR）. https://arxiv.org/abs/2504.11343
