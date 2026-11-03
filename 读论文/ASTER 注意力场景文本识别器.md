---
created: 2026-11-03
source: Shi 等 - 2019 - ASTER An Attentional Scene Text Recognizer with Flexible Rectification（IEEE TPAMI vol.41, no.9, pp.2035–2048, Sept. 2019）；本机 PDF：E:\.dsh-paper-reader\data\默认专题\Shi 等 - 2019 - ASTER An Attentional Scene Text Recognizer with Flexible Rectification.pdf
type: 论文
tags:
  - 论文
  - 场景文本识别
  - Attention
  - 文本矫正
---

# ASTER：带柔性矫正的注意力场景文本识别器

> 页码均指 PDF 页码（印刷页 2035 对应 PDF 第 1 页，即 PDF 页 = 印刷页 − 2034）。

## 一句话总结

把「不规则场景文字识别」拆成**先拉直、再转写**两段：前半段用 **TPS（薄板样条）+ STN 框架**预测 20 个控制点、把弯曲/透视文字重采样成规整条带；后半段用**注意力 Seq2Seq 解码器**（双向）直接把条带读成字符串。两段共用一条反向传播链，**不需要任何控制点标注**，端到端训练。

## 核心问题

场景文字常见**透视**与**弯曲**两类不规则形变，而主流识别器（CRNN/CTC 一路）默认文字是水平规则排列的，形变会直接拖垮识别率（第 1 页）。已有的"先检测字符框再分类"路线在背景噪声下不稳（第 2 页）。

论文点名了当时两条路各自的短板（第 5 页）：

- **CTC 类方法**：可端到端、对水平字符间距不敏感，但「**CTC does not have a mechanism to model the dependencies among its output characters**」，只能外挂词典/语言模型来做语言先验。
- **矫正类前作 RARE [55]**：有 TPS 矫正，但控制点被 tanh 限制在 [0,1]，且定位与采样用同一分辨率，矫正图质量差（第 4、8 页）。

ASTER 的答案：**矫正交给可微的 STN 分支，语言先验交给自回归解码器**，两者用一个损失联合训练。

## 关键方法

### 0. 总体结构

两个网络串起来：矫正网络（第 3.1 节）→ 识别网络（第 3.2 节）。控制点全部由输入图自行预测，所以「the rectification network takes no extra inputs other than the input image」（第 3 页）。

```
输入图 I (resize 到 64×256)
  → 定位网络（在 Id = 32×64 上回归）→ C′（K=20 个控制点）
  → 网格生成器（解 TPS）→ 采样坐标场 P
  → 采样器（双线性 + 越界裁剪）→ 矫正图 Ir (32×100)
  → 45 层 ResNet + 2 层 BiLSTM → H = [h₁…h₂₅]
  → 注意力 LSTM 解码器（双向各一个）→ 字符序列
```

![ASTER 矫正网络结构图 Fig4](ASTER%20矫正网络结构图%20Fig4.png)

<small>裁自论文第 3 页 Figure 4，仅取图本体与标注，已去掉图注。</small>

**怎么读这张图：** 上方横链是"预测 → 变换 → 采样"的主线。左边 `I` 是原图，向上 `Resize` 成 `Id` 喂给 **Localization Network**（只吃缩小的图，为了省参数、也为了"看得全"）；它吐出 **Control Points C′**——图上那些青色十字就是 20 个点，训练后自己长到了文字上下边缘。**Grid Generator** 用 C′ 解出 TPS，产出采样坐标场 `P`（箭头朝下指向 ⊗）。⊗ 是 **Sampler V**，它同时吃 `P` 和原图 `I`（注意是**原图**，不是缩小的 Id，这是本文相对 STN 的第一处改动），输出矫正图 `Ir`。整条链没有标注输入，梯度从最右边的 `Ir` 一路回到最左边的定位网络。

### 1. 矫正网络：固定输出点，预测输入点

这是全篇最容易读反的地方。TPS 需要两组对应点：

- **输出图 Ir 上的 K 个点 → 硬编码常量**：等间距铺在图像上、下边界（第 3 页）。
- **输入图 I 上的 K 个点 C′ → 网络预测**：定位网络回归出来的。

为什么反过来不行：输出侧的点编码的是"规整文字应该长什么样"这一**先验**，天然固定；输入侧的点编码的是"这张图里文字实际在哪"，是唯一未知量。于是只要预测点贴住文字上下边缘，TPS 就必然把那条弧线拉成直线——**"拉直"的保证蕴含在"输出点固定"里**。

| 子模块 | 作用 | 关键数字 |
|---|---|---|
| 3.1.1 定位网络 | 从 Id 回归 C′（2K 个归一化坐标） | 6 个 3×3 卷积（前 5 个各接 2×2 maxpool，通道 32/64/128/256/256/256）+ 2 个全连接（512、2K）；K=20（第 7 页） |
| 3.1.2 网格生成器 | 由 C 与 C′ 解 TPS 系数 T（2×(K+3)），再对每个输出像素求采样位置 | 核函数 `f(r) = r² log r`；线性方程组带 4 个边界条件（第 4 页） |
| 3.1.3 采样器 | `Ir = V(P, I)`，对 p′ 的四邻域做双线性插值 | 越界先做 value clipping（第 4–5 页） |

**三处相对 STN / RARE 的改动**（第 4–5、8 页）：

1. **定位与采样用不同分辨率**：定位在 32×64 的 Id 上做，采样在 64×256 原图上做——避免 STN 一路下采样导致的输出分辨率塌陷。
2. **末层去掉 tanh**：让 C′ 不受 (−1,1) 约束，反传梯度更完整、收敛更快；代价是点可能跑出图外，靠采样器的 value clipping 兜住。
3. **不强制控制点留在图内**：前作 [55] 强行约束，反而"may disturb the sampling grid and causes image distortions"。

### 2. 可微性：梯度怎么流回去

论文只给了一句"refer the readers to [28]"，实质分三层：

- **采样器**：双线性权重 `w = (1−δx)(1−δy), δx(1−δy), …` 是连续偏移 `δ` 的分段线性函数，于是 `∂Ir/∂p′ ≈ 邻域差分`（右邻居减左邻居）。**必须用双线性而非最近邻**，否则梯度恒为 0。
- **网格生成器**：TPS 线性系统的左端矩阵只由**常量** C 构成，故 `T = C′ · DDC` 对 C′ 是**线性**的，反向只是一次常量矩阵乘（Fig. 6 里只有 concat 与 matmul）。
- **定位网络**：标准 CNN；再叠加"去掉 tanh"这一步保住梯度幅度。

三者串起来全是现成算子，所以 Fig. 6 那句 "the computation from p to p̂ … can be cached and reused for images of the same resolution" 才有意义——同分辨率下 lifted vector 可离线算好。**代价**：value clipping 在越界区导数为 0，形成梯度死区（论文未讨论，属观察）。

### 3. 识别网络：注意力解码 + 双向

- **编码器**：45 层 ResNet 把 32×100 压到 **1×25×512**（高度压成 1，宽度只降到 25），沿行轴切开得 25 个向量；再经 **2 层 BiLSTM**（各 256 hidden，输出拼后投影回 256）得 `H = [h₁…h₂₅]`（第 5、7 页）。
- **注意力（式 4）**：`e_{t,i} = wᵀ tanh(W s_{t−1} + V h_i + b)`，`a_{t,i} = exp(e_{t,i}) / Σ_{i′} exp(e_{t,i′})`——加性（Bahdanau 式）注意力，**每步对 25 个列位置做一次 softmax**。
- **glimpse（式 5）**：`g_t = Σ_i a_{t,i} h_i`，硬切分的可微替代品；权重可以同时分给相邻两个字符，容错。
- **循环单元（第 6 页）**：`(x_t, s_t) = rnn([s_{t−1}; g_t; f(y_{t−1})])`，`f` 是**上一字符的 one-hot 嵌入**（第 6 页原话 "the one-hot embedding of y_{t−1}"）。字符类共 94 类（10 数字 + 26 大写 + 26 小写 + 32 标点，第 7 页）+ EOS，故 one-hot 维度应为 95 —— **论文未给嵌入维度，此为按 94+EOS 的推断**。
- **输出（式 6/7）**：`p(y_t) = softmax(W_o x_t + b_o)`，`y_t ~ p(y_t)`。注意 `x_t`（本步输出，供预测）与 `s_t`（新状态，供下一步注意力）角色不同。
- **隐式语言模型**：因为 `y_{t−1}` 进门，RNN 状态累积了 `y₁…y_{t−1}`，等价于建模 `p(y_t | y_{<t}, 图像)`——这就是不必外挂词典的原因（第 6 页）。
- **双向解码器（3.2.3）**：L2R 与 R2L 各训一个；训练用平均损失（式 8），推理**取累积 log-softmax 分数更高的那个**。动机很具体：左到右解码器分不清词首的 `I` 与 `l`，因为它没有后续字母的记忆（第 6 页）。

![ASTER 识别网络结构图 Fig7](ASTER%20识别网络结构图%20Fig7.png)

<small>裁自论文第 5 页 Figure 7，仅取图本体，已去掉图注。</small>

**怎么读这张图：** 自下而上是编码，自上而下是解码。底部 `Rectified image`（"storage"）先经 **ConvNet** 逐级下采样（蓝色堆叠的厚度 = 通道数递增），到最上面时高度已压成 1；`Map to sequence` 把它横切成一根特征序列，交给 **BLSTM**（橙色块，双向箭头表示前后各扫一遍）。解码侧：`Attend` 模块吃编码器输出的 `h_i` 和解码器上一步状态 `s_{t−1}`，算出权重 `α_t`，与特征点乘（⊗）得到 `g_t`，送进 LSTM；每个 LSTM 向上吐一个字符（'S'→'t'→'o'…），直到 `<EOS>` 停下。注意虚线框把整图分成 Encoder / Decoder 两半——这就是"两个网络"的边界。

![ASTER 双向解码器 Fig8](ASTER%20双向解码器%20Fig8.png)

<small>裁自论文第 6 页 Figure 8，仅取图本体，已去掉图注。</small>

**怎么读这张图：** 同一个编码器特征分给两个方向相反的解码器。左到右解码器读出 "FLTNESS"（分数 0.5，把 I 认成了 L），右到左解码器倒着读再 `Reverse` 回来，得 "FITNESS"（分数 0.8）。右边那个双掷开关表示**二选一：取分数高的**。这张图就是"为什么需要双向"的最小反例。

### 4. 训练

- 损失（式 8）：`L = −½ Σ_t [log p_ltr(y_t|I) + log p_rtl(y_t|I)]`，两个方向各算一份再平均（第 6 页）。
- **identity 初始化**：随机初始化会让控制点乱放、Ir 一开始就扭曲，训练不稳；故把末层 fc2 的**权重置零、偏置设成使 C′ = C**，训练起点即恒等变换（第 6 页）。
- 优化：ADADELTA，batch 64（32 张 Synth90k + 32 张 SynthText），100 万次迭代，学习率 1.0 → 0.1（0.6M）→ 0.01（0.8M）；单卡 Titan Xp 约 2 天收敛，推理 20 ms/图（第 7 页）。

### 5. 副产品：用控制点反推定向检测框

训练后的 ASTER 总把点放在文字上下边缘，于是把前 10 个点、后 10 个点各拟合一条直线（最小二乘），取四个端点构成四边形，就能**把水平检测器的框升级成定向框**（第 11 页）。

## 结论与局限

- **矫正有效，且只在不规则文本上明显**：加矫正后在六个数据集上全面上涨，SVTP **+4.7%**、CUTE **+3.1%**（这两个恰是透视/弯曲数据集）；构造的"不规则比例递增"序列显示增益随不规则程度单调上升（第 8 页）。
- **双向解码器一致更好**：Table 5 上 Bidirectional 在 IIIT5k/SVT/IC03/IC13/SVTP 全面领先，SVT 上比 L2R、R2L 分别高 **2.8** 和 **1.6** 个百分点（L2R 91.93 / R2L 89.96 / 双向 91.57）；CUTE 上双向与 R2L 打平（74.31）（第 9–10 页）。
- **主结果 9/12 项最优**：ASTER（ResNet, 90k+ST）在 IIIT5k 达 **99.6/98.8/93.4**（50/1k/0 词典），SVT 99.2/93.6，IC03 98.8/98.0，IC13 94.5，IC15 91.8，SVTP 76.1，CUTE 78.5/79.5；仅少数项落后于 [11]（多用了字符级标注）与 [26]（只在其 90k 词典内输出，属受限识别）（第 10 页）。
- **矫正能反哺检测**：TextBoxes + ASTER 的检测 HMean 从 **0.626 → 0.768**，端到端 IC15 从 0.638 提到 **0.706**（第 11 页）。
- **局限**（论文自述 + 观察）：① 靠近图像边框的文字（"city"、"lights"）矫正会失效，偶有失败案例（第 8 页）；② 词长超过 11 后整词准确率下降，部分归因于长词本身更难（第 9 页）；③ 固定尺寸 resize 引入形变，作者以"训练测试同分布"自辩（第 9 页）；④ 隐式检测能力只覆盖目标文字附近的小范围，扩到全图仍是 open problem（第 11 页）。

## 我的疑问

- [ ] **EOS 与 one-hot 维度**：论文说 94 个字符类，又提到会输出 EOS，那嵌入到底是 94 还是 95 维？全文未给嵌入维度，也没说是否有可训练的 embedding 矩阵（若为纯 one-hot，LSTM 输入应是 256+95=351 维）。
- [ ] **Table 2 与 Table 5 数字对不上**：Table 2 的 "With Rect." 与 Table 5 的 "Bidirectional" 是同一配置，但 IIIT5k 一处 92.67、一处 92.27。是两次独立训练，还是表格笔误？
- [ ] **value clipping 的梯度死区**：越界区导数为 0，点一旦跑出去就再也拉不回来。论文只说这招"显著提升性能与稳定性"，没给为什么会更稳的机制解释。
- [ ] **K 的敏感性**：正文说 "Other values of K lead to similar results"，但没有给出 K=10/20/30 的对比数据。
- [ ] **teacher forcing 未明说**：式 8 在真值序列上求和，默认训练时喂真值；但论文明文没写，推理与训练的 exposure bias 也没讨论。
- [ ] **可微链条的实现细节缺失**：3.1.3 整段只有一句 "refer the readers to [28]"，ASTER 自己有没有对采样器做改动，无从判断。

## 关键摘录

> "The rectification network is based on the Spatial Transformer Network [28]. The central idea of STN is to model spatial transformation as a learnable network layer."（第 3 页）

> "Therefore, the problem of text rectification boils down to predicting the control points on the input image."（第 3 页）

> "The sampler is made differentiable, i.e., it can back propagate the gradients on Ir to P. This is achieved by the differentiable image sampling method. We refer the readers to [28] for more details."（第 4 页）

> "Even without a direct supervision on control point locations, the rectification network learns to place the control points near the upper and lower edges of text."（第 8 页）

> "Despite its effectiveness, CTC does not have a mechanism to model the dependencies among its output characters."（第 5 页）

> "Since y_{t−1} is incorporated in the computation, the decoder learns to capture the dependencies between its output characters. This acts as an implicit language model, which assists the recognition with the language priors it learns."（第 6 页）

> "Concretely, we set the weights of fc2 to zeros and its biases to values such that C′ is identical to C."（第 6 页）

> "Although ASTER's rectification fails occasionally, it generally produces images with much less distortion than those of [55]."（第 8 页）

## 相关笔记

- [[SVTR 场景文本识别]] — 后续工作，直接复用 ASTER 的 TPS 矫正模块（"SVTR uses the rectification module [Shi et al., 2019]"），可对照看"矫正 + 单一视觉主干"的演进
- [[各种初始化方式]] — ASTER 的 identity 初始化（末层权重置零、偏置调成恒等）正属于"恒等初始化"家族，可对照残差分支零初始化
- [[齐次坐标变换]] — TPS 与齐次坐标同属"用坐标映射描述几何变换"，一个走非刚性、一个走仿射/投影，适合并读
- [[PAGRN 渐进式注意力引导循环网络]] — 同样是"注意力权重可视化验证隐式定位"的思路，可比较两者的注意力热图读法

---

#论文 #场景文本识别 #Attention #文本矫正
