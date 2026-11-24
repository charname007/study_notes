---
created: 2026-11-25
source: 本次对话；参考 Vaswani et al. 2017、Xiong et al. 2020、T5、BART、Whisper、Flamingo、PyTorch 与 Hugging Face 实现
type: 概念
tags: [概念, Transformer, Attention, Norm]
---

# Transformer Decoder 中 self-attention 与 cross-attention 的顺序

## 是什么

在 encoder-decoder Transformer 的 decoder block 中，cross-attention 以 decoder 当前隐藏状态为 Query，以 encoder memory 为 Key/Value。最主流的顺序是：

```text
masked self-attention
→ cross-attention
→ FFN
```

以 Pre-LN 为例：

```text
u = x + SelfAttn(LN(x), causal_mask)
v = u + CrossAttn(LN(u), encoder_memory)
y = v + FFN(LN(v))
```

这不是一个普遍最优的固定定理，而是自回归 encoder-decoder 中形成的主流信息流。

## 为什么需要

### 1. self-first 先构造有上下文的 Query

当前目标 token 常有歧义，例如代词指代、词义选择和目标语言搭配。先做 causal self-attention，可以吸收已经生成的目标端前缀，使 cross-attention 使用的 Query 更有判别力。

直观分工是：

```text
self-attention：我是谁、前文说了什么
cross-attention：根据这些信息，应该从源端读取什么
```

若先做 cross-attention，它使用的 Query 尚未充分吸收目标端上下文，因此更接近“源端先验驱动”的建模。

### 2. 符合目标前缀逐步查询源端的生成过程

自回归解码时，decoder 只能看到当前 token 之前的目标序列：

```text
目标前缀
→ masked self-attention 聚合前缀
→ cross-attention 查询 encoder memory
→ 预测下一个 token
```

masked self-attention 负责目标端的时间因果上下文，cross-attention 负责对源端的非因果检索。两种 mask 的职责因此是先分清的。

### 3. 两条残差分支的信息流不同

在串行结构中，cross-attention 读取的是 self-attention 已更新后的残差流。先执行 self-attention 会保留 token 自身的残差，同时给 cross-attention 增加目标端上下文。

如果改成 cross-first，后面的 self-attention 会处理已经被源端信息增强的序列。它不一定更差，但语义变成：

```text
先用源端信息增强目标 token
→ 再在目标序列内部传播和整合
```

这可能改善 grounding 和一致性，也可能弱化 token 级对齐、造成源端信息被过度平滑。

## 常见变体

### 1. 标准 self-first

适用于机器翻译、语音识别、图像描述等多数 encoder-decoder 任务。T5、BART 和 Whisper 都是典型实例。

### 2. cross-first

```text
cross-attention
→ self-attention
→ FFN
```

可能适合：

- 源端信息应尽早进入模型。
- 目标端上下文很短。
- 任务是强条件生成，要求早期 grounding。
- 条件是主要信息来源，例如检索、语音或视觉条件。

代价是 cross-attention 的 Query 尚未获得充分的当前层目标端上下文。是否更好必须做任务级消融，不能只按直觉判断。

### 3. 并行 self/cross

一种常见写法：

```text
h = x
    + SelfAttn(LN(x), causal_mask)
    + CrossAttn(LN(x), encoder_memory)

y = h + FFN(LN(h))
```

self-attention 和 cross-attention 读取同一个输入，层内互不等待，关键路径更短，推理延迟可能下降。但 cross 的 Query 看不到本层 self 的结果，self 也看不到本层 cross 注入的信息，二者交互被推迟到下一层。

### 4. 跨层控制 cross-attention 的插入位置

除了同一层内的先后顺序，还存在“每层都插”“隔层插”“只在若干层插”的设计：

```text
L1: self → cross → FFN
L2: self → FFN
L3: self → cross → FFN
```

减少 cross-attention 次数可以降低计算量和 encoder KV 读取开销，但也可能削弱 grounding。Flamingo 这类工作更值得从“cross-attention 的插入频率和位置”角度参考，而不是只看同一 block 内的先后。

## 常见误区

1. **把顺序问题与 LayerNorm 顺序混在一起**  
   比较 self/cross 顺序时，必须同时控制 Pre-LN 与 Post-LN。归一化位置会显著改变梯度、训练稳定性和深层可训练性。Pre-LN 与 Post-LN 的差异见 [[各种Norm层的摆放]]。

2. **忽略 mask 语义**  
   decoder self-attention 必须有 causal mask；cross-attention 通常只屏蔽 padding。顺序错了和 mask 错了会产生不同的问题，不能混为一谈。

3. **只看输出质量，不看延迟**  
   并行 attention 主要提供层内并行性，但把 self/cross 的交互延后。串行 self-first 多一次层内依赖，却让 cross 使用更成熟的 Query。

4. **把主流惯例当成普适最优**  
   目前没有一篇论文专门系统研究“所有任务中 self 与 cross 谁先谁后”。现有证据主要来自原始架构、主流模型实现和归一化/并行化研究。

## 选择建议

没有特殊约束时，从标准结构开始：

```text
masked self-attention
→ cross-attention
→ FFN
```

若怀疑顺序是瓶颈，消融时至少控制：

- 总层数、参数量、FLOPs
- Pre-LN / Post-LN
- self/cross 是否共享参数
- 每层 cross 还是隔层 cross
- padding mask 与 causal mask
- 训练吞吐、推理延迟和显存

评价指标除常规任务损失外，还应关注源端覆盖率、token 级对齐、grounding 和 hallucination。

## 参考资料

- Vaswani et al., 2017, [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：标准 decoder 顺序的直接来源。
- Xiong et al., 2020, [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)：Pre-LN 与 Post-LN 的梯度分析。
- Wang et al., 2019, [Learning Deep Transformer Models for Machine Translation](https://arxiv.org/abs/1906.01787)：深层 Transformer 与 Pre-LN。
- Liu et al., 2020, [Understanding the Difficulty of Training Transformers](https://arxiv.org/abs/2004.08249)：残差分支对训练稳定性的影响。
- Raffel et al., 2020, [T5](https://arxiv.org/abs/1910.10683)：标准 decoder block 实例。
- Lewis et al., 2020, [BART](https://arxiv.org/abs/1910.13461)：encoder-decoder 结构实例。
- Radford et al., 2022, [Whisper](https://arxiv.org/abs/2212.04356)：语音 decoder 中 self → cross → FFN 的实例。
- Alayrac et al., 2022, [Flamingo](https://arxiv.org/abs/2204.14198)：周期性插入 gated cross-attention 的多模态实例。
- Medina & Kalita, 2018, [Parallel Attention Mechanisms in Neural Machine Translation](https://arxiv.org/abs/1810.12427)：较早的并行 attention 尝试，不能直接外推为普遍结论。

## 相关笔记

- [[各种Norm层的摆放]] — 比较 self/cross 顺序时，必须先控制 Pre-LN 与 Post-LN。
- [[transformer的2道残差连接]] — 理解 self-attention 与 FFN 两道残差为何不能合并。
- [[多头注意力+proj_out]] — attention 输出之后为何还需要输出投影。
- [[DEATRN v2 可变形集成注意力任意形状场景文本识别]] — 一个采用 self-attn → cross-attn → FFN 的具体 decoder 实例。
