---
created: 2026-11-06
source: E:\dsh-paper-reader\data\默认专题\Seo 等 - 2016 - Progressive Attention Networks for Visual Attribute Prediction.pdf ；官方代码 https://github.com/phseo/PAN @ b2a94f7
type: 论文
tags: [论文, Attention, 属性预测, 多尺度]
---

# PAN：渐进式注意力网络做视觉属性预测

## 一句话总结
不再只在 CNN 顶层做一次注意力，而是**在多个中间层反复施加注意力**：每层算出逐位置的权重 α，把特征乘上 α 后**不求和、直接交给下一层卷积**，层层过滤直到末层才聚合。由此把「注意力能看多细」从单层固定感受野里解放出来，配合**硬注意力 + 似然边际化**（免 REINFORCE）和**局部上下文**，在查询式属性预测上大幅超过单步注意力基线。

## 核心问题

任务是 **reference problem / query-specific attribute prediction**：给一张图和一个查询 q（"这个 7 是什么颜色"），只对 q 指定的那个对象预测属性（第 2 页）。难点是同一张图里目标在**尺度和形状**上差异巨大（MREF 里数字缩放范围 0.5–3.0，第 6 页）。

论文对两类已有注意力各打一枪：

1. **特征类注意力（软注意力 [23]）**：*"it can only model attention to local regions with a fixed size depending on the receptive field of the layer chosen for attention"*（第 2 页）。α 图的分辨率 = 该层特征图分辨率，而 CNN 越深分辨率越低、感受野越大——**注意力精度和语义层次被同一层锁死**，二者不可兼得。
2. **变换类注意力（STN [11]）**：只能覆盖「少数参数定义的单个区域」，形状受限于仿射变换，*"STN is limited to attending a single candidate region defined by a small number of parameters"*（第 3 页）。

补充：软注意力的末层聚合还会把一个区域的特征**加权平均成一个向量**，*"it loses semantic layout of the original feature map"*（第 5 页）。

## 关键方法

整体流程（论文 Fig. 2，第 4 页）：特征图 → 逐层乘 α → 不聚合、继续卷积 → 末层再做软/硬两条分支。

![PAN 整体流程 Fig2](PAN%20整体流程%20Fig2.png)

<small>裁自论文第 4 页 Figure 2，仅取图本体与标注，已去除图注正文。</small>

**怎么读这张图：**

- **左侧主链（自左向右）**：三个立方体依次是 FEATURE → ATTENTION → ATTENDED，最后交给 NEXT CONV。**关键就是 f̂ 之后没有 Σ 求和**，而是原样进下一层卷积——这是与软注意力唯一但致命的差别。
- **右上 (a) Soft attention**：末层把 f̂ 求和池化成一个细长条（attended feature），再进 attribute classifier 出一个分布。**先合并、后分类**。
- **右下 (b) Hard attention with marginalization**：特征图**先**进 attribute classifier（注意是逐位置的），得到一堆小柱状图，再用 α 做 marginalize（圈里那个 Σ）混合成最终分布。**先分类、后合并**。
- **(a)(b) 的差别只在括号位置**：H(Σα·f) 与 Σα·H(f)。

### 1. 逐层注意力的四条式子（§3.1，第 4 页）

**式 (1)** 逐位置门控，α 直接乘在特征上：

    f̂^l_{i,j} = α^l_{i,j} · f^l_{i,j}

**式 (2)** 注意力分数与归一化方式：

    s^l_{i,j} = g^l_att(f^l_{i,j}, q; θ^l_att)
    α^l_{i,j} = softmax_{i,j}(s)  若 l = L ；  σ(s^l_{i,j})  其他

**式 (3)** 加权后的特征图原样进下一层卷积（**不求和**）：

    f^{l+1} = g^{l+1}_CNN(f̂^l; θ^{l+1}_CNN)

**式 (4)** 只在末层求和：

    f_att = Σ_i Σ_j f̂^L_{i,j} = Σ_i Σ_j α^L_{i,j} f^L_{i,j}

**三处值得停下来的设计：**

| 设计 | 原文依据 | 为什么这样 |
|---|---|---|
| 中间层用 **sigmoid**（逐位置独立），只有末层用 **softmax** | 第 4 页："The attention probability at each location is estimated independently in the same feature map" | 中间层是**门控**，位置之间不竞争，可以"都要"也可以"都不要"；末层必须 Σα=1，因为式(6) 要拿它当概率分布用。**这两个选择是配套的** |
| 中间层 f̂ **不求和** | 第 4 页："is not summed up to generate a single vector representation" | 一旦求和就丢空间布局，而且后面没法再分层细化 |
| 前向从 f^0 = I 一路推到 f̂^L | 第 4 页 | 纯前馈，没有循环、没有迭代，推理与训练同构 |

### 2. 硬注意力 + 似然边际化（§3.2，第 5 页）

**式 (5)** 只用单个位置的特征做一次属性预测：

    p(a | f^L, i, j) = H(f^L_{i,j})

**式 (6)** 用注意力概率把逐位置预测边缘化起来（全概率公式）：

    p(a | f^L) = Σ_i Σ_j p(a | f^L, i, j) · p(i, j | f^L)
              = Σ_i Σ_j H(f^L_{i,j}) · α^L_{i,j}

这是**全概率公式**：把「每个位置各自的一个小分布」按 α 混合。**是预测层面的平均，不是特征层面的平均**——混合分布能保住双峰，加权平均特征会把它们糊成一个不存在的中间物。

之所以说"硬"：原始硬注意力 [23] 是**从 α 里采样一个位置**，离散采样不可微 → 只能 REINFORCE。本文改成**算期望**：

   采样 :  E_{i,j~α}[ H(f_{i,j}) ]      ← 单次抽样，需 REINFORCE
   本文 :  Σ_{i,j} α_{i,j} H(f_{i,j})   ← 期望的精确值，处处可微

*"This marginalization eliminates the need for REINFORCE technique used in [23] by removing the hard selection process"*（第 5 页）。所以它**名义上是硬注意力，实现上是完全可微的期望**（文献里常叫 deterministic hard attention / attention marginalization）。直接后果见第 6 页 §3.5：*"Training a PAN is as simple as training a soft attention network ... because every operation within the network is differentiable."*

⚠️ **符号重载坑**：式(6) 那个 H 是特征图**高度**的求和上限 Σ_{i=1}^{H}，只有 H(f^L_{i,j}) 那处的 H 才是分类器。中文译稿极容易读反。

### 3. 局部上下文（§3.4，第 6 页）

**式 (7)** 打分函数的输入从单点扩成 δ 邻域：

    F^l_{i,j} = { f^l_{s,t} | i-δ ≤ s ≤ i+δ, j-δ ≤ t ≤ j+δ }
    s^l_{i,j} = g^l_att(F^l_{i,j}, q; θ^l_att)

逐位置独立判断有个隐患：孤立看一个格子容易判错。于是把邻域也喂进打分函数，*"by comparing centroid features with their surroundings"*（第 6 页）。**末层不需要**上下文，因为末层的激活直接进最终聚合。窗口大小取"下一层 conv+pool 复合操作的滤波尺寸"（第 6 页），**δ=1（3×3）** 用于 MNIST 系（第 7 页）、**δ=2（5×5）** 用于 VG（第 9 页）。

### 4. 注意力插在哪几层（第 5 页）

不必每层都插：*"we can also skip the first few pooling layers and attach the attention module only to the outputs of last K pooling layers"*（第 5 页）。原因是分辨率下降几乎都来自 pooling，插在 pooling 输出上就等价于"层间"。

- **MNIST 系**：4 个 pool 后各插一个（**全部都用**，见代码）；
- **VG**：VGG-16 的 **pool3/pool4/pool5** 之后堆 att1/att2/att3，跳过 pool1/pool2，*"because the features in those layers are not discriminative enough to filter out"*（第 9 页）。

## 深度理解（精读补充）

### 1. 为什么"渐进"能把尺度问题解开（§3.3，第 5 页）

![PAN 注意力逐层演进 Fig1](PAN%20注意力逐层演进%20Fig1.png)

<small>裁自论文第 2 页 Figure 1，仅取图本体与子图标注，已去除图注。</small>

**怎么读这张图**（查询 = "7"，答案 = 红）：

- (a) 输入：一堆 5–9、大小不一、颜色各异的数字撒在 100×100 空图上。
- (b) 第一次注意力：已经压掉了大部分，但仍留有大片亮区（大尺寸干扰还在）。
- (c) 第二次：范围收紧，干扰更少。
- (d) 第三次：基本只剩目标数字的笔画。
- (e) 最终：几乎全黑，只剩 "7" 本身那一小片。

**这张图就是论文那句机制论证的实证**：

> *"In lower layers, features whose receptive fields contain small distractors are suppressed first while the features from a part of large distractors remain intact. In higher layers, features corresponding to these large distractors would have low attention probability as each feature contains information from larger receptive fields..."*（第 5 页）

即：**浅层感受野小 → 只能分辨小干扰 → 先把小的按住；深层感受野大 → 看得见大干扰的全貌 → 再把大的按住。**

给"跨层累积"一个准确的表述（容易误读）：**跨层流动的是被 α 乘过的特征图，不是 α 本身**。论文没有把 α^{l-1} 拼进 g^l_att，也没有做多层 α 的对齐相加；α 是通过特征这条主线"搭车"过去的，且经过 l+1 层卷积后已被核加权模糊，因此图注用的是 "**accumulate**"（累积）而非 "multiply"（相乘）（第 8/10/14 页图注）。

### 2. 主动承认的对手优势 + 未明说的代价

- **软注意力可微、好训练**是它最大的优点，本文没否定它（第 2 页），只是说它的表达受限。
- **式(6) 的代价**：要在**每个空间位置**都跑一遍分类器 H，成本 = H×W 次分类。MNIST 上末层只有 4×4 = 16 个位置（代码里就是 (4,4) 那张 α 图），可接受；VG 上 749 个属性类 × 每个位置，才是真正的开销来源。工程上应把 H(·) 实现成 1×1 卷积（权重在各位置共享）以把 H×W 次前向压成一次。
- **BN 是论文没写但代码里有的**：每层注意力之后接一个 BatchNorm 才进下一层卷积（见代码对照）。这正好回答了"sigmoid 不归一化会不会把特征整体幅值压小"——**用 BN 补偿**。

### 3. 官方实现对照（Theano/Lasagne）

仓库 https://github.com/phseo/PAN ，HEAD b2a94f7（2016-10-25），核心在 libs/theano_util.py 的 satt()：

```python
# libs/theano_util.py @ b2a94f7 -- satt()
l_img = conv(img, num_filters=32, filter_size=context_box, ..., pad='same')  # 局部上下文
l_q   = lasagne.layers.EmbeddingLayer(ref, input_size=10, output_size=32)    # q
l_mix = SpatialElemwiseMergeLayer(l_img, l_q, T.add)                         # 相加融合
l_weight = conv(l_drp, num_filters=1, filter_size=(1,1), nonlinearity=None)  # 出 α
```

**三处与论文措辞不一致，值得记住：**

| | 论文（第 7 页） | 代码 | 说明 |
|---|---|---|---|
| q 的处理 | 与特征**拼接**后过两层 FC | 经 Embedding 层后与特征图**逐通道相加** | one-hot 过 Embedding = 线性投影，故 q 的编码等价；但 **concat+FC 与 add 的表达力不等价**，add 是更强的结构约束 |
| q 是否 one-hot | "q is a one-hot vector representing a target object" | EmbeddingLayer(ref, input_size=10) | 等价（one-hot × W = 查表），不是矛盾 |
| BN | 全文未提 | 每个 satt 之后接 BatchNormLayer | 见 train.py:133-134, 139-140, 145-146 |

另外 train.py 开头的注释写着 "Code for producing the resutls for the FFN-SPN network" —— 这套 satt 的"embedding + 逐通道相加 + 1×1 出权重"结构与 FFN 一脉相承，作者是在其上改造的。

**已发布代码只覆盖软注意力版**：train.py 用 Pool2DLayer(..., (4,4), mode='average_exc_pad') 对 f̂ 求平均再进 Dense（第 151–160 行），即 **PAN[S]**；**硬注意力 / 式(5)(6) 的边际化在发布的 train.py 里没有实现**。所以 Table 1 的 PAN[H] 行无法直接复现。

详细代码解读见 [[PAN 的 satt 注意力层实现]]。

## 结论与局限

**实验设置**（§4，第 6–9 页）：MNIST 合成集三档（MREF / +干扰物 MDIST / +自然背景 MBG）；CNN 为 4 组 3×3 conv@32 + 2×2 maxpool，RMSprop，50 轮无提升即停。VG 用 VGG-16、86,674 图 / 827 物体类 / 749 属性类，Adam。额外在 VOC 2007 上用分割掩码量 TPR。

**主结果**（Table 1，第 8 页，TPR = 注意力落在目标区域的比例）：

- **渐进注意力是主力**：单步 SAN 的 TPR 仅 **13.83**（MREF），PAN[S] 直接 **44.25**，MBG 上更悬殊（**6.92 → 32.07**）。
- **单步时软/硬几乎无差别**：SAN 13.83 vs HAN 13.95 —— **硬注意力的收益只有在渐进结构里才释放**。
- **消融的意外**（MREF TPR）：PAN[S] 44.25 → PAN[H] 62.31（+18.06）；PAN[S] → PAN[S]+CTX 60.45（+16.20）；而 **PAN[H] → PAN[H]+CTX 只从 62.31 到 62.36（+0.05）**。两条单独有效的改进**合起来几乎不叠加**——说明硬注意力与局部上下文触及的是同一个瓶颈（"注意力图太糊、逐位置判断不可靠"），不是可叠加的独立收益。
- **尺度鲁棒性最硬的证据**（第 8 页右表，MBG 按目标尺度分 5 档）：SAN 随尺度剧烈波动（49.1 → 68.9 → 54.6，跨度 19.8 个点），PAN[H]+CTX 是 84.2 / 87.1 / 87.5 / 81.6 / 79.6，**跨度仅 4.6**。
- **VG 上**（Table 2，第 9 页）：PAN[H]+CTX 的 mAP **30.00**（加类别先验 **34.19**），TPR **22.23**（加先验 24.37），均高于 SAN（27.62 / 15.01）。VOC 分割掩码 TPR：SAN 22.01 → PAN[H]+CTX **31.79**。
- **STN 的负面结论很有价值**：*"these models actually do not attend to target objects"*（第 8 页）——STN-M 在 MREF 准确率高达 93.89，但 TPR 只有 0.66，即**它靠类别先验答对了，注意力却根本没落在目标上**。这提醒：**高准确率不等于注意力有效，必须单独量 TPR**。

**局限**：

- TPR 增益在 60 上下饱和（PAN[S]+CTX 60.45 ≈ PAN[H] 62.31 ≈ PAN[H]+CTX 62.36），说明**剩下约 37% 的注意力误差不是靠这两个组件能补的**，论文没有分析这部分误差来自哪里。
- 式(6) 的逐位置分类器开销随分辨率与类别数线性上升，论文未给任何加速手段或复杂度分析。
- 只在 VGG-16 与一个 4 层小 CNN 上验证，未探索更深主干。
- 查询 q 用裸 one-hot，无语言/属性语义，表达能力受限于 FC 层能否解出有意义的投影。
- 硬注意力版本未开源（见上），可复现性打折。

## 我的疑问
- [ ] **PAN[H] 未开源**，式(6) 的逐位置分类器在 VG 上（749 类 × 每个位置）具体怎么算的？1×1 卷积实现？
- [ ] **q 的融合方式究竟哪个是"真"的**？论文写 concat+2×FC，代码写 Embedding→逐通道 add。add 相当于强制 q 与特征共享同一组基底，是更强的约束——作者是为了省参数还是笔误？
- [ ] **硬注意力与 CTX 的增益几乎完全重叠**（PAN[H] 62.31 与 PAN[H]+CTX 62.36，只差 0.05）说明什么？两者是否都在补同一个"注意力图太糊"的问题？
- [ ] **TPR 卡在 ~62%** 的瓶颈在哪？是 attention 分辨率的物理上限（末层只有 4×4），还是训练信号不足？
- [ ] 中间层 sigmoid 不归一化 → 若某层整体 α 偏低，特征幅值被系统性压缩。代码用 BN 兜底，但没有 BN 的版本会掉多少？论文没做这个消融。
- [ ] 与 FFi-Net / 视觉注意力"加性融合"谱系是什么关系？train.py 头注释提到 FFN-SPN，但正文没有引用 FFN。

## 关键摘录

> "One of the most popular attention mechanisms for NNs is soft attention [23], which aggregates responses in a feature map weighted by their attention probabilities. This process results in a single attended feature vector."（第 2 页）

> "However, it can only model attention to local regions with a fixed size depending on the receptive field of the layer chosen for attention."（第 2 页）

> "Unlike the soft attention model [23], the attended feature map in the intermediate attention layers is not summed up to generate a single vector representation of the attended regions."（第 4 页）

> "This marginalization eliminates the need for REINFORCE technique used in [23] by removing the hard selection process, and helps maintain the unique characteristics of feature maps."（第 5 页）

> "In lower layers, features whose receptive fields contain small distractors are suppressed first while the features from a part of large distractors remain intact. In higher layers, features corresponding to these large distractors would have low attention probability as each feature contains information from larger receptive fields enabling the attention module to distinguish between distractors and target objects."（第 5 页）

> "Local context improves attention prediction quality by comparing centroid features with their surroundings and making the estimated attention more discriminative."（第 6 页）

> "Training a PAN is as simple as training a soft attention network [23] because every operation within the network is differentiable."（第 6 页）

> "An important observation regarding to STNs is that these models actually do not attend to target objects."（第 8 页）

## 相关笔记
- [[PAGRN 渐进式注意力引导循环网络]] — 同样叫"渐进式注意力"但路线不同：PAGRN 是**高层指导低层的跨层特征筛选**，PAN 是**同一目标在多层反复过滤**；两篇都很干净地靠消融表讲清每个组件值多少
- [[浅层与深层]] — 本文全部立论都建立在这条事实上：浅层感受野小、深层感受野大。那篇是必备背景
- [[ASTER 注意力场景文本识别器]] — 本文第 3 页重点批评的 STN/变换类注意力，正是 ASTER 的 TPS 矫正所用的那套；对照看"变换式"与"过滤式"注意力的边界
- [[FlashAttention 精确注意力与IO感知]] — 同样在改"注意力怎么算"，但方向相反：PAN 改**注意力的语义结构**，FlashAttention 改**注意力的数据搬运**，后者完全不碰 α 的含义
- [[PAN 的 satt 注意力层实现]] — 本文官方代码的逐行解读

#论文 #Attention #属性预测 #多尺度
