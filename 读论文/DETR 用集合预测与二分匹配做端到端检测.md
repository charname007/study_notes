---
created: 2026-11-23
source: 默认专题/Carion 等 - 2020 - End-to-End Object Detection with Transformers.pdf（arXiv:2005.12872v3, ECCV 2020, Facebook AI）
type: 论文
tags:
  - 论文
  - Transformer
  - 目标检测
  - 集合预测
---

# DETR：用集合预测与二分匹配做端到端检测

## 一句话总结

把目标检测从「堆候选框 + 事后 NMS」改写成**直接输出一个无序集合**：用**匈牙利算法**在预测与真值之间求唯一的一对一匹配，构造出**置换不变**的损失，再用**Transformer 编码器-解码器**一次性并行解码 N 个预测——anchor 生成与 NMS 被双双删除，而代价是训练极长、小目标偏弱（第 1 页）。同参数量在 COCO val 上与调优过的 Faster R-CNN 打平（42.0 AP），大物体 **+7.7 AP_L**、小物体 **−6.1 AP_S**；接一个 mask head 就把这套范式推到全景分割 SOTA（PQ 45.1，第 16 页）。

## 核心问题

主流检测器并不直接预测集合，而是在**代理任务**上回归：两阶段的 proposal、单阶段的 anchor 或物体中心网格。作者列出的代价是三处人工先验（第 1–2 页、第 4 页）：

1. **anchor / proposal 集合怎么设计**（尺寸、比例、密度）；
2. **目标怎么分配给 anchor** 的启发式规则——文献 [52] 已经证明最终性能**强烈依赖**这套规则；
3. **NMS 事后去重**：它是不可学习的后处理超参，而它之所以存在，恰恰是因为预测之间**没有通信**。

作者的主张是「直接集合预测（direct set prediction）」。这在此前不是没人试过（Stewart 等用 RNN 做端到端人群检测），但都只在**小数据集**上验证、打不过强基线；本文要补的正是这个 gap（第 2 页）。他们同时把「为什么以前不行」讲清楚了：直接集合预测有两个硬需求——**置换不变的损失**（否则第 3 个 query 该负责谁完全由输出顺序决定）与**能建模元素间关系的全局架构**（否则重复预测无法被抑制，只能靠 NMS 兜底）（第 3 页）。

## 关键方法

### 1. 集合预测损失：先匹配，再算损失（第 5 页）

模型一次前向固定输出 N 个预测，**N 远大于图中物体数**（实现取 N=100）。真值集合 $y$ 用 $\varnothing$ 补到 N 个，预测集合记作 $\hat{y}=\{\hat{y}_i\}_{i=1}^{N}$。

**第一步：匈牙利算法求最优置换**

$$\hat{\sigma}=\arg\min_{\sigma\in\mathfrak{S}_N}\sum_{i=1}^{N}\mathcal{L}_{\text{match}}\left(y_i,\hat{y}_{\sigma(i)}\right)$$

**匹配代价**同时看类别概率与框的重合度（$b\in[0,1]^4$ 是相对图像尺寸的中心点 + 宽高）：

$$\mathcal{L}_{\text{match}}\left(y_i,\hat{y}_{\sigma(i)}\right)=-\mathbb{1}_{\{c_i\neq\varnothing\}}\hat{p}_{\sigma(i)}(c_i)+\mathbb{1}_{\{c_i\neq\varnothing\}}\mathcal{L}_{\text{box}}\left(b_i,\hat{b}_{\sigma(i)}\right)$$

**第二步：在匹配上算匈牙利损失**

$$\mathcal{L}_{\text{Hungarian}}(y,\hat{y})=\sum_{i=1}^{N}\left[-\log\hat{p}_{\hat{\sigma}(i)}(c_i)+\mathbb{1}_{\{c_i\neq\varnothing\}}\mathcal{L}_{\text{box}}\left(b_i,\hat{b}_{\hat{\sigma}(i)}\right)\right]$$

四个必须记住的实现细节：

| 细节 | 做法 | 位置 |
|---|---|---|
| 空类降权 | $c_i=\varnothing$ 的 log 项**乘 1/10**，治正负不平衡（类比 Faster R-CNN 的正负 proposal 采样） | 第 5 页 |
| 空类匹配代价是常数 | 与预测无关，故**不影响匹配** | 第 5 页 |
| 匹配用**概率**而非 log 概率 | 让分类项与框损失**同量级**，作者说经验上更好 | 第 5 页 |
| 框损失 = $\lambda_{L1}\|\cdot\|_1+\lambda_{iou}\mathcal{L}_{\text{GIoU}}$ | 直接回归绝对框，$\ell_1$ 对大框小框**尺度不一致**，必须配尺度不变的 GIoU | 第 6 页 |

> 这一步的角色等价于传统检测器里「proposal/anchor → 真值」的启发式分配规则，区别在于它是**一对一**的（第 5 页）。这正是 NMS 能被删掉的根源：多出来的预测只能去匹配 $\varnothing$，**重复预测在损失里就已经被惩罚**，不需要事后剪枝。

### 2. 架构：CNN → 编码器 → 并行解码器 → FFN（第 6–7 页，Fig. 2）

| 模块 | 输入/输出 | 关键点 |
|---|---|---|
| **CNN backbone** | $x\in\mathbb{R}^{3\times H_0\times W_0}$ → $f\in\mathbb{R}^{C\times H\times W}$ | ResNet-50/101，$C=2048$，$H,W=H_0/32,W_0/32$ |
| **1×1 卷积** | 降到 $d=256$，空间维拉平 | 得到 $d\times HW$ 的序列 |
| **Transformer 编码器** | 6 层 self-attention + FFN | 补**固定正弦位置编码**（无位置信息则置换不变，无法定位） |
| **Transformer 解码器** | N 个 **object queries** → N 个输出 embedding | **非自回归**：每层并行解码 N 个物体，不像 [47] 那样逐词生成 |
| **预测 FFN** | 3 层 MLP（ReLU，隐维 $d$）+ linear | 出归一化框坐标（softmax 出类）；**N 个槽位共享参数** |

两个容易被忽略的设计：

- **为什么需要 object queries**：解码器同样置换不变，「N 个输入 embedding 必须不同才能产生不同结果」，于是用 N 个**可学习的位置编码**充当槽位（第 7 页）。它本质上是**语义空间里的隐式 anchor**——不在图像上铺框，而是在特征空间里占坑。
- **辅助解码损失**：每个解码层后都接预测 FFN + 匈牙利损失（FFN 与 LayerNorm 共享），作者说这对「输出正确数量的物体」帮助很大（第 7 页）。这条设计既加速收敛，也顺带让「逐层 AP」这种分析成为可能。
- **$\varnothing$ 类**：固定输出 N 个框就必然有空槽位，故需要一个特殊的「no object」类，作用类似背景类（第 7 页）。

作者顺带强调：推理代码在 PyTorch 里 **50 行内**能写完，实现不依赖任何自定义层（第 6 页）。

### 3. 训练配方（第 8–9 页）

| 项 | 设置 |
|---|---|
| 数据 | COCO 2017 detection/panoptic，118k 训练 / 5k 验证，平均 7 个实例、最多 63 个 |
| 优化 | AdamW，transformer lr $10^{-4}$、backbone lr $10^{-5}$、wd $10^{-4}$；Xavier 初始化 |
| backbone | ImageNet 预训练 ResNet，**BN 层冻结** |
| 长训练 | 基准 300 epoch（200 后 lr÷10）；对齐比较用 **500 epoch**（+1.5 AP）；16×V100、batch 64、**3 天**（第 9 页） |
| 增强 | scale aug（短边 480–800、长边 ≤1333）+ 随机裁剪 $p=0.5$（**+1 AP**） |
| DC5 变体 | C5 stage 加膨胀、去 stride，分辨率 ×2：小目标变好，但编码器自注意力贵 **16×**，总算力 ×2 |
| 推理技巧 | 空槽位**不丢弃**，用第二高分类别与置信度覆盖（**+2 AP**，第 9 页） |

## 实验与结论

### 与 Faster R-CNN 的公平对比（Table 1，第 9 页）

作者特意把 baseline 加强（加 GIoU、同样的随机裁剪、9× 长训），避免「赢在训练配方」：

| 模型 | GFLOPS/FPS | 参数 | AP | AP50 | AP75 | AP_S | AP_M | AP_L |
|---|---|---|---|---|---|---|---|---|
| Faster R-CNN-FPN | 180/26 | 42M | 40.2 | 61.0 | 43.8 | 24.2 | 43.5 | 52.0 |
| Faster R-CNN-FPN+ | 180/26 | 42M | 42.0 | 62.1 | 45.5 | 26.6 | 45.4 | 53.4 |
| **DETR** (R50) | 86/28 | 41M | **42.0** | 62.4 | 44.2 | 20.5 | 45.8 | **61.1** |
| DETR-DC5 | 187/12 | 41M | 43.3 | 63.1 | 45.9 | 22.5 | 47.3 | 61.1 |
| DETR-R101 | 152/20 | 60M | 43.5 | 63.8 | 46.4 | 21.9 | 48.0 | 61.8 |
| DETR-DC5-R101 | 253/10 | 60M | **44.9** | 64.7 | 47.7 | 23.7 | 49.5 | 62.3 |

- **同参数打平**（42.0 vs 42.0），且 FLOPs 只有一半；但 FPS 28 起、DC5 掉到 12，**推理没有优势**。
- **强弱项极不对称**：作者口径「大物体 +7.8 AP_L、小物体 −5.5 AP_S」（第 10 页）。按表 1 与 FPN+ 行相减是 **+7.7 / −6.1**（对比行不同，数量级一致）。
- 作者对成因的解释值得记住：大物体靠**自注意力的非局部计算**（整图上下文），小物体靠**高分辨率特征**——而 DETR 只有单一 1/32 特征图，于是「FPN 之于 Faster R-CNN」的那套补丁还没被搬过来（第 2 页、第 10 页）。

### 消融：每个部件都在干什么（第 10–13 页）

| 消融 | 结果 | 结论 |
|---|---|---|
| **编码器层数**（Table 2, 第 10 页） | 0/3/6/12 层 → 36.7 / 40.1 / 40.6 / 41.6 AP | 去掉编码器 **−3.9 AP**，大物体 **−6.0 AP**；编码器在做**实例分离** |
| **解码器层数**（Fig. 4, 第 12 页） | 第 1→6 层 **+8.2 AP / +9.5 AP50** | 逐层都涨；**NMS 只在第 1 层有用**，末层反而误删真阳性 |
| **FFN**（第 11 页） | 去掉后参数 41.3M→28.7M | **−2.3 AP**，FFN 不可省（等价 1×1 卷积，提供通道非线性） |
| **位置编码**（Table 3, 第 13 页） | 全去 32.8（**−7.8**）；只在输入加一次 39.2（−1.4）；解码器也只在输入加 39.3（−1.3）；每层注意力都加 = 基线 40.6 | 空间 PE 必需；**加在哪层比怎么编码更重要**（第 12 页） |
| **损失**（Table 4, 第 13 页） | 只留 $\ell_1$（去 GIoU）**35.8（−4.8）**；只留 GIoU（去 $\ell_1$）39.9（−0.7） | **GIoU 承担主要性能**，但 $\ell_1$ 不能删——两者互补 |

> 两条最有信息量的：① **NMS 曲线是「后处理可被架构取代」的直接证据**——第 1 层没有跨槽位通信才会重复预测，所以 NMS 有用；深层自注意力已抑制重复，NMS 开始伤 AP。② **「空间 PE 全去掉还留 32.8 AP」**，说明解码器的 object query 本身携带了可用的定位信息（第 12 页），这一点反过来说明 PR 里「query 是隐式 anchor」的说法是有实据的。

### 机制分析：谁在干什么（第 10–14 页）

- **编码器自注意力把实例「切开」**：对若干个参考点可视化最后一层编码器的注意力，已能分离单个实例，作者据此认为编码器在做场景级去纠缠（Fig. 3, 第 11 页）。
- **解码器注意力很「局部」**：主要落在**物体端点**（头、腿、边缘）。假设是——实例已由编码器分好，解码器只需读边缘就能同时拿到类别与边界（Fig. 6, 第 13 页）。
- **槽位会专化但不绑类别**：可视化 100 个 query 的预测中心分布，每个槽位有若干工作模式（区域 + 长宽比），且几乎所有槽位都保留一个「预测整图大框」的模式（Guessed 与 COCO 分布有关）；合成图里训练集最多 13 只长颈鹿却能检出 24 只，说明**没有强类别专化**（Fig. 7 / Fig. 5，第 14 页）。

### 全景分割：加个 head 就能统一（第 14–16 页）

1. 训练 DETR 同时预测 **things + stuff** 的框（匈牙利匹配用框距离，所以框必须先预测，第 15 页）；
2. mask head 取每个对象在解码器的输出 embedding，对**编码器输出**做 M 头注意力得 M 张低分辨率热图，再用 **FPN 式结构**上采样到 stride 4，DICE/F-1 + Focal loss 监督；
3. 推理时**逐像素 argmax** 合并所有 mask → **天然不重叠**，省掉 Panoptic FPN 那套对齐启发式（第 16 页）。

**Table 5（第 16 页）**：DETR-R101 **PQ 45.1**（things 50.5 / stuff **37.0**），超过 PanopticFPN++ R101 的 44.1、UPSNet-M 的 43.0；COCO test set **46 PQ**。优势集中在 stuff 类（37.0 vs 33.6），作者归因于编码器的全局推理；但 **mask AP 落后基线最多 6.7**（33.0 vs 39.7）——「掩码质量一般、但分得干净 + stuff 强」把 PQ 拉了回来，这个反差在引用结论时不该被省略。

## 结论与局限

**作者承认的**
1. **小物体弱**（AP_S 20.5，明显低于 Faster R-CNN-FPN+ 的 26.6），根因是单尺度 1/32 特征（第 2 页、第 10 页）；
2. **训练极慢**：300–500 epoch、16 卡 3 天，还需要辅助损失帮忙才稳（第 2 页、第 9 页）；
3. 结论段落直接点名「训练、优化、小物体性能」是留给后续工作的挑战（第 17 页）。

**我判断的**
- 编码器自注意力是 $O((HW)^2)$，DC5 把分辨率翻倍换来 16× 的编码器开销——**这套全局注意力的成本随分辨率爆炸**，这正是后来 Deformable DETR（稀疏采样）、多尺度 DETR 的动机；
- 一对一匹配意味着**每个 epoch 的监督目标会随匹配漂移**，这很可能是收敛慢的机理之一（论文没直接讨论，属于我的推测）；
- 全景分割的 25 epoch 微调 + 冻结主干，说明这套表征的**迁移成本很低**，这一点比 PQ 数字更有长期价值。

## 我的疑问

- [ ] 匹配代价用**概率**而非 log 概率（第 5 页）：在 N=100、空槽位占多数时，概率项趋于 0，会不会让匹配对「已经分得很好的样本」几乎不敏感？作者只给了「经验上更好」，没给消融。
- [ ] 一对一匹配是**每个 batch 内**的最优解且逐 epoch 变化，这种「监督目标漂移」与 500 epoch 的长训之间是否有因果？如果把匹配固定几个 epoch 再解冻会怎样？
- [ ] N=100 相对 COCO「最多 63 个实例」是够的，但**密集小目标场景**（如航拍）槽位会成为瓶颈吗？论文没有 N 的消融。
- [ ] 「空间 PE 全去掉仍有 32.8 AP」（第 13 页）——定位信息到底从哪来？是 query 自身，还是卷积 backbone 的隐性位置信息？
- [ ] 槽位对整图大框的偏好（第 14 页）被归因于 COCO 分布，那么换到**长尾/极端尺寸分布**的数据集上，槽位专化会怎么变？

## 关键摘录

> "We present a new method that views object detection as a direct set prediction problem."（第 1 页）

> "The main ingredients of the new framework, called DEtection TRansformer or DETR, are a set-based global loss that forces unique predictions via bipartite matching, and a transformer encoder-decoder architecture."（第 1 页）

> "In practice, we down-weight the log-probability term when $c_i=\varnothing$ by a factor 10 to account for class imbalance."（第 5 页）

> "The most commonly-used $\ell_1$ loss will have different scales for small and large boxes even if their relative errors are similar."（第 6 页）

> "Unlike many modern detectors, DETR can be implemented in any deep learning framework that provides a common CNN backbone and a transformer architecture implementation with just a few hundred lines."（第 6 页）

> "Since the decoder is also permutation-invariant, the N input embeddings must be different to produce different results."（第 7 页）

> "With its set-based loss, DETR does not need NMS by design."（第 11 页）

> "Interestingly, not passing any spatial encodings in the encoder only leads to a minor AP drop of 1.3 AP."（第 12 页）

> "This new design for detectors also comes with new challenges, in particular regarding training, optimization and performances on small objects."（第 17 页）

## 相关笔记

- [[attention+PE（注意力上+位置编码）]] — DETR 的位置编码是本文最反直觉的消融之一：**加在每层注意力里**比「输入时加一次」高 1.4 AP，可直接对照这篇的 RPE 讨论
- [[多头注意力+proj_out]] — 编码器/解码器的 multi-head + 输出投影，是 DETR 全部全局推理的载体；配合「解码器注意力落在物体端点」来看更直观
- [[transformer的2道残差连接]] — DETR 用的是标准 pre-LN block（注意力一道、FFN 一道残差），本文的辅助损失就是挂在每个解码 block 之后的
- [[各种Norm层的摆放]] — 本文在预测 FFN 前加了一个**跨解码层共享**的 LayerNorm，属于「共享归一化」的一个少见用法
- [[FlashAttention 精确注意力与IO感知]] — 同一个「自注意力很贵」的出发点，但走的是**不改语义、只改数据搬运**的路；DETR 的答案是**改语义（稀疏/多尺度）**，两条路线可对照
- [[Restormer 高效高分辨率图像修复 Transformer]] — 同样面对「注意力随分辨率爆炸」，Restormer 把注意力搬到**通道维**；DETR 只能靠降采样到 1/32，这也是它小目标弱的直接原因

#论文 #Transformer #目标检测 #集合预测
