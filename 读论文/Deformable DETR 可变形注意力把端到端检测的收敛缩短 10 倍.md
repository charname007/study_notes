---
created: 2026-11-24
source: 默认专题/Zhu 等 - 2021 - Deformable DETR Deformable Transformers for End-to-End Object Detection.pdf（arXiv:2010.04159v4, ICLR 2021, SenseTime + 中国科大 + 港中文）
type: 论文
tags: [论文, 目标检测, Transformer, 可变形注意力]
---

# Deformable DETR：可变形注意力把端到端检测的收敛缩短 10 倍

## 一句话总结

DETR 的毛病不在「集合预测」，而在**注意力必须看过所有像素**：初始时注意力近似均匀 → 梯度模糊 → 要 500 epoch 才收敛（第 1、4 页）；而编码器自注意力 $O(H^2W^2C)$ 又让高分辨率特征图不可用 → 小物体弱。本文把每个 query 的注意力改成**只在参考点周围采 $K$ 个点**（多尺度下是 $L\times K$ 个）的「可变形注意力」：编码器复杂度降到 $O(HWC^2)$、解码器交叉注意力与特征图尺寸无关，于是 **50 epoch（DETR 的 1/10）在 COCO val 拿到 43.8 AP**（DETR 500 epoch 42.0、DETR-DC5 500 epoch 43.3），小物体 AP_S 从 20.5 升到 26.4，训练 325 GPU 时（DETR 2000、Faster R-CNN 380）（第 8 页 Table 1）；再叠迭代框精修与两阶段，50 epoch 到 46.2 AP。

## 核心问题

DETR 已证明「CNN + Transformer 编码解码器 + 匈牙利损失」可以删掉 anchor、NMS 与分配规则，但它留下两个病灶，作者把它们归到同一处（第 1 页）：

1. **收敛极慢**：COCO 上 DETR 要 500 epoch，比 Faster R-CNN 慢 10~20 倍。原因是初始化时 $U_mz_q$ 与 $V_mx_k$ 近似零均值单位方差，注意力权重 $A_{mqk}\approx 1/N_k$ 几乎均匀（第 3 页），输入特征的梯度因此含糊，训练只能一点点把注意力「收窄」到稀疏的目标位置上。
2. **小物体差**：现代检测器靠高分辨率特征图看小物体，而 DETR 编码器自注意力是 $O(H^2W^2C)$，分辨率翻倍代价 ×16（第 4 页）——DETR-DC5 就是被这条成本卡住，只在 C5 上做膨胀、分辨率 ×2，FLOPs 与训练时间直接翻倍。

已有的高效注意力方案作者认为不合用（第 3 页）：预定义稀疏模式（局部窗口 / 跨步 / 全局 token）会丢全局信息；数据相关的稀疏（LSH、k-means、块置换）与低秩近似在图像域实现后，同 FLOPs 下比卷积至少慢 3×（访存模式所致）。而可变形卷积虽然「只看稀疏点」且高效，却**缺少元素间的关系建模**——那正是 DETR 成功的关键。本文的定位就是补上这个缺口。

## 关键方法

### 1. 可变形注意力：只看参考点周围 $K$ 个点（第 5 页，式 2）

$$\mathrm{DeformAttn}(z_q,p_q,x)=\sum_{m=1}^{M}W_m\Big[\sum_{k=1}^{K}A_{mqk}\cdot W'_mx(p_q+\Delta p_{mqk})\Big]$$

$K\ll HW$：每个头只在参考点 $p_q$ 附近采 $K$ 个点，$A_{mqk}$ 由 softmax 归一化（$\sum_k A_{mqk}=1$），$\Delta p_{mqk}\in\mathbb{R}^2$ 是**无约束实数偏移**，$p_q+\Delta p_{mqk}$ 是分数坐标，故用双线性插值取值（沿用可变形卷积的做法）。两者都由 query 特征 $z_q$ 线性投影得到：论文口径是**一个 $3MK$ 通道的投影**（前 $2MK$ 出偏移、后 $MK$ 过 softmax 出权重）。

复杂度（第 5 页；推导见附录 A.1，第 12 页）：$O(2N_qC^2+\min(HWC^2,N_qKC^2))$。用在编码器（$N_q=HW$）时为 $O(HWC^2)$——**对空间尺寸线性**；用作解码器交叉注意力（$N_q=N$）时为 $O(NKC^2)$——**与特征图尺寸无关**。这是全文省算力与省内存的两个关键式子。

官方实现里这两条投影是拆成两个 Linear 的（等价于把 $3MK$ 通道切开），偏移还会按该层特征图的 $(W_l,H_l)$ 归一化——这些只有读 python 侧封装才能确认，记在 [[Deformable DETR 的 MSDeformAttn 实现]]。

```python
# models/ops/modules/ms_deform_attn.py:55-56（官方仓库 @ 11169a6）
self.sampling_offsets  = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)   # 之后 softmax
```

![Deformable DETR 可变形注意力 Fig2](Deformable%20DETR%20可变形注意力%20Fig2.png)

<small>裁自论文第 5 页 Figure 2，仅取图本体，已去除图注文字。</small>

**怎么读这张图（三条支路）：**

- **左上**：query 特征 $z_q$ → Linear → 每个头、每个采样点的 2 维偏移 $\{\Delta p_{mqk}\}$（图中三个头各画几个方向互不相同的箭头）；
- **右上**：同一条 $z_q$ → Linear → Softmax → 每个头、每个点的标量权重 $\{A_{mqk}\}$（三个头各给出归一化后的 3~4 个数，如 0.5/0.3/0.2）；
- **中下**：输入特征图 $x$ 先经 Linear 得到每头自己的 value 特征 $\{W'_mx\}$（三张彩色特征图），再在 $p_q+\Delta p_{mqk}$ 处**双线性采样**（虚线引出的小方块），按权重 Aggregate 逐头加权求和，最后 Linear 出输出。

> 与标准注意力的区别全在「采样位置」这一格：标准注意力是对整张特征图求权重（$HW$ 个 key），这里是对 $K$ 个**学出来的**位置求权重。

### 2. 多尺度版本：$LK$ 个点，一次跨层（第 6 页，式 3）

$$\mathrm{MSDeformAttn}(z_q,\hat p_q,\{x^l\}_{l=1}^{L})=\sum_{m=1}^{M}W_m\Big[\sum_{l=1}^{L}\sum_{k=1}^{K}A_{mlqk}\cdot W'_mx^l\big(\phi_l(\hat p_q)+\Delta p_{mlqk}\big)\Big]$$

参考点改用归一化坐标 $\hat p_q\in[0,1]^2$，$\phi_l$ 把它**重缩放到第 $l$ 层的特征图**上，于是同一个 query 可以同时在前景该被看的几个尺度上采样；权重在 $L\times K$ 上统一 softmax。

作者给了两个「身份」说明这个模块的位置（第 6 页）：

- 取 $L=1$、$K=1$、$W'_m$ 固定为单位阵时，**退化为可变形卷积**（每个头只看一个点）；
- 采样点遍历所有可能位置时，**等价于 Transformer 注意力**——所以可变形注意力是「用可变形采样做 pre-filter 的注意力」，不是另一种算子。

### 3. 编码器：不做 FPN 的多尺度（第 6 页；构造见附录 A.2，第 12 页）

| 项 | 做法 |
|---|---|
| 输入尺度 | C3/C4/C5 各接 $1\times1$ 卷积，与 C6 一起构成 $L=4$ 层；C6 由 C5 再走 $3\times3$ stride 2 卷积得到（$H/64$） |
| 通道 | 全部 256 |
| 参考点 | 每个 query 像素的参考点**就是它自己** |
| 位置编码 | 固定正弦位置编码 + **可学习的 scale-level embedding $e_l$**（区分像素属于哪一层；与固定编码不同，随机初始化并参与训练） |
| 参数 | 编码器参数在各特征层间共享（第 7 页） |

![Deformable DETR 多尺度特征构造 Fig4](Deformable%20DETR%20多尺度特征构造%20Fig4.png)

<small>裁自论文第 12 页 Figure 4（附录 A.2），仅取图本体，已去除图注文字。</small>

**怎么读这张图：** 左列是 ResNet 的 C3/C4/C5（512/1024/2048 通道，$H/8$、$H/16$、$H/32$）；每个只过一个 $1\times1$ stride 1 卷积降到 256 → 得到三个尺度的输入特征图；C5 额外分出一条 $3\times3$ stride 2 卷积得到 $H/64$ 的第四层。**关键是没有 FPN 的 top-down 上采样与相加**——图里没有任何一条从低分辨率回到高分辨率的箭头。

> 为什么敢去掉 FPN：多尺度可变形注意力本身就跨层采样，信息交换发生在注意力内部。作者在消融里验证了这一点——加 FPN 43.8、加 BiFPN 43.9，与不加（43.8）持平（第 9 页 Table 2）。

### 4. 解码器：只换交叉注意力（第 6 页）

- 解码器里有两种注意力：**自注意力**（query 之间互看，负责「物体之间别重复」）保持标准形式不动；**交叉注意力**（query 从特征图取特征）换成多尺度可变形注意力。
- 每个 object query 的参考点 $\hat p_q$ 由它自己的 embedding 经一个线性层 + sigmoid 预测（**与图像内容相关**，这是相对 DETR 的实质变化）。
- 检测头不再直接回归绝对框，而是**回归相对参考点的偏移**（参考点当框中心的初值，式见附录 A.3，第 12 页）：

$$\hat b_q=\{\sigma(b_{qx}+\sigma^{-1}(\hat p_{qx})),\ \sigma(b_{qy}+\sigma^{-1}(\hat p_{qy})),\ \sigma(b_{qw}),\ \sigma(b_{qh})\}$$

这样「看图的位置」与「预测的框」被绑在一起，作者的观察是这同时加速了收敛（第 6 页）。

### 5. 两个变体：迭代框精修与两阶段（第 7 页；细节在附录 A.4，第 12–13 页）

- **迭代框精修（iterative bounding box refinement）**：受光流里 RAFT 的迭代精修启发，每一层解码器拿上一层的框当前提，预测增量 $\Delta b^d_q$：$\hat b^d_q=\{\sigma(\Delta b^d_{qx}+\sigma^{-1}(\hat b^{d-1}_{qx})),\dots\}$；初始框 $w=h=0.1$（换成 0.05/0.1/0.2/0.5 都不敏感），**梯度在 $\sigma^{-1}(\hat b^{d-1})$ 处截断**，只回传到 $\Delta b^d$；第 $d$ 层的采样偏移还按上一层框的宽高调制（$\Delta p\cdot\hat b^{d-1}_{w,h}$）。
- **两阶段**：第一阶段**去掉解码器**，让每个像素都当 object query 直接预测框（3 层 FFN 回归 + 线性前后景二分类，基准尺度 $s=0.05$ 并按层缩放 $2^{l_i-1}s$），取高分框当 proposal；**送进第二阶段前不做 NMS**，proposal 坐标编码成位置嵌入当对象 query 的初值、框当迭代精修的初始框。

![Deformable DETR 架构图 Fig1](Deformable%20DETR%20架构图%20Fig1.png)

<small>裁自论文第 2 页 Figure 1，仅取图本体，已去除图注文字。</small>

**怎么读这张图（从左往右）：**

- **左下 Image**：一张图 → **Image Feature Maps**（4 级特征图，紫三角是每层的采样点）；
- **中间 Multi-scale Feature Maps（Encoder 虚线框）**：编码器的输入与输出**都是同分辨率的多尺度特征图**——不像 FPN 那样先融合成一个尺度。紫色虚线箭头从一张特征图跨到另外几张，就是「多尺度可变形自注意力」在层间交换信息；图中标着 ×4，对应 $L=4$ 个层级；
- **右侧 Decoder**：底部是 **Object Queries**（彩色小方块），往上每一级都有红色虚线箭头（多尺度可变形交叉注意力，从编码器输出取特征）和黑色虚线箭头（query 之间的标准自注意力）。**只有红色那条被换掉了**，黑色照旧；
- **最右上的 Bounding Box Predictions**：每个 query 落在原图上的框；整张图里没有 NMS，也没有 anchor 网格。


### 6. 框与类别到底怎么生成：检测头 + 迭代精修的完整链（第 4 页 DETR 综述；第 6、7、12–13 页；官方代码确认）

这是把前面「参考点 / 采样 / 逐层精修」串成实际输出的一步。解码器输出 N=300 个 query embedding（第 7 页），每个 embedding 接两个小头（第 4 页：“A 3-layer FFN and a linear projection are added on top of the object query features … The FFN acts as the regression branch … The linear projection acts as the classification branch”）：

| 头 | 结构 | 输出 | 证据 |
|---|---|---|---|
| 分类头 | 一个线性层 | 每类的 logits（含 $\varnothing$ 类） | 第 4 页；代码 `class_embed = nn.Linear(d_model, num_classes)` (`deformable_detr.py:54`) |
| 回归头 | 3 层 FFN（256→256→4） | 框的 4 个未归一化数 $(b_x,b_y,b_w,b_h)$ | 第 4 页；代码 `bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)` (`deformable_detr.py:55`) |

训练时分类用 **Focal Loss（权重 2）**（第 7 页），代码进一步是 `sigmoid_focal_loss(..., alpha=0.25, gamma=2)` + 逐类 sigmoid / one-hot 目标（`deformable_detr.py:238`）；监督信号仍来自 DETR 的匈牙利二分匹配（第 4 页），一对一匹配使推理不需要 NMS。

**生成一条预测的完整链（以第 $d$ 层为例）：**

1. query embedding 经线性层 + sigmoid 得到参考点 $\hat p_q$（第 6 页）；
2. 多尺度可变形交叉注意力**围绕 $\hat p_q$** 从四级特征图采 $L\times K$ 个点并加权聚合，更新 embedding；
3. 分类头出类别 logits；回归头出相对参考点的偏移，按附录 A.3（第 12 页）解码成框：

$$\hat b_q=\{\sigma(b_{qx}+\sigma^{-1}(\hat p_{qx})),\ \sigma(b_{qy}+\sigma^{-1}(\hat p_{qy})),\ \sigma(b_{qw}),\ \sigma(b_{qh})\}$$

$\sigma^{-1}$ 把参考点映到无界空间、加偏移后再 $\sigma$ 压回 $[0,1]$；这一步让「看哪里」和「预测什么」强绑定，加速收敛（第 6 页）。代码对应 `deformable_detr.py:166-174`：`reference = inverse_sigmoid(reference)` → `tmp = bbox_embed[lvl](hs[lvl])` → 中心加偏移 → `tmp.sigmoid()`。

4. **下一层把刚预测的框当新参考点**：$\hat b_q^d=\{\sigma(\Delta b^d_{qx}+\sigma^{-1}(\hat b^{d-1}_{qx})),\dots\}$，初始框 $w=h=0.1$（附录 A.4，第 12 页），每层预测头**参数不共享**，梯度在 $\sigma^{-1}(\hat b^{d-1})$ 处截断；第 $d$ 层交叉注意力的采样偏移还按上一层框的宽高缩放 $\Delta p\cdot\hat b^{d-1}_{w,h}$。代码是 `deformable_transformer.py:341-351`：`new_reference_points = (tmp + inverse_sigmoid(reference_points)).sigmoid()`，随后 `.detach()`。

每一层都挂辅助损失，推理取最后一层的 `pred_logits` / `pred_boxes`（代码 `deformable_detr.py:180`，辅助输出走 `aux_outputs`）。Table 1（第 8 页）显示这条「逐层改写框」的链本身值钱：基础 43.8 → 开迭代精修 **45.4（+1.6）** → 加两阶段 **46.2**。

> **为什么中间层需要框，而不只是最后输出一个框？** 值得澄清：交叉注意力的采样**只需要参考点这一个 2D 坐标**，严格说中间层并不「需要」完整的 4 维框才能采样。但把框喂回参考点有两个论文给出的理由：① 采样偏移按框宽高调制（附录 A.4，第 12 页），使采样点分布**贴合物体尺度与形状**——初始 8 方向小网格对一个大物体显然不够，框给出了尺度信息；② 多尺度下参考点是 4 维 `(x,y,w,h)`（代码 `reference_points.shape[-1] == 4` 分支，`deformable_transformer.py:332-345`），宽高本身参与采样位置计算。所以中间框不是「多余的输出」，而是**给下一层采样的条件信号**。只保留最后一层预测、放弃迭代精修，作者测了：43.8 vs 45.4，**掉 1.6 AP**（第 8 页 Table 1）。

## 实验与结论

设置（第 7 页）：COCO 2017；ResNet-50（ImageNet 预训练）；$M=8$、$K=4$；无 FPN；编码器参数跨层共享；把 DETR 的分类换成 **Focal Loss（权重 2）**、object query 从 100 加到 300（为公平对比也报了同样改动的 DETR-DC5+）；默认 **50 epoch**，第 40 epoch 学习率 ×0.1；Adam，基础 lr $2\times10^{-4}$、wd $10^{-4}$，预测参考点/采样偏移的线性层 lr ×0.1；V100 计时。

### 与 DETR 的对照（Table 1，第 8 页）

| 方法 | epochs | AP | AP50 | AP75 | AP_S | AP_M | AP_L | 参数 | FLOPs | 训练 GPU 时 | 推理 FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Faster R-CNN + FPN | 109 | 42.0 | 62.1 | 45.5 | 26.6 | 45.4 | 53.4 | 42M | 180G | 380 | 26 |
| DETR | 500 | 42.0 | 62.4 | 44.2 | 20.5 | 45.8 | 61.1 | 41M | 86G | 2000 | 28 |
| DETR-DC5 | 500 | 43.3 | 63.1 | 45.9 | 22.5 | 47.3 | 61.1 | 41M | 187G | 7000 | 12 |
| DETR-DC5 | 50 | 35.3 | 55.7 | 36.8 | 15.2 | 37.5 | 53.6 | 41M | 187G | 700 | 12 |
| DETR-DC5+ | 50 | 36.2 | 57.0 | 37.4 | 16.3 | 39.2 | 53.9 | 41M | 187G | 700 | 12 |
| **Deformable DETR** | 50 | **43.8** | 62.6 | 47.7 | 26.4 | 47.1 | 58.0 | 40M | 173G | 325 | 19 |
| + 迭代框精修 | 50 | 45.4 | 64.7 | 49.0 | 26.8 | 48.3 | 61.7 | 40M | 173G | 325 | 19 |
| ++ 两阶段 | 50 | **46.2** | 65.2 | 50.0 | 28.8 | 49.2 | 61.7 | 40M | 173G | 340 | 19 |

三件值得记住的事：

1. **同 epoch 比，差距是量级级的**：50 epoch 时 Deformable DETR 43.8，DETR-DC5 只有 35.3（+8.5 AP），同改动的 DETR-DC5+ 也只有 36.2；而 43.8 已经超过「DETR 500 epoch」的 42.0 与「DETR-DC5 500 epoch」的 43.3。训练 325 GPU 时 vs 2000，**比 Faster R-CNN 的 380 还省**。
2. **FLOPs 不省，省的是访存**：173G 与 DETR-DC5 的 187G 几乎一样，但 FPS 19 vs 12（1.6×）；论文的解释是标准注意力要做大量连续访存，可变形采样虽无序但总量小。代价是仍比 Faster R-CNN 慢 25%（第 8 页）。
3. **AP_L 反而下降**：58.0 vs DETR 的 61.1。多尺度稀疏采样对**大物体**没有 DETR 那种「整图全局注意力」的优势——这是全文最容易被忽略的 trade-off。

![Deformable DETR 收敛曲线 Fig3](Deformable%20DETR%20收敛曲线%20Fig3.png)

<small>裁自论文第 8 页 Figure 3，仅取图本体，已去除图注文字。</small>

**怎么读这张图：** 红线是 Deformable DETR（把不同学习率下降 epoch 的几个方案曲线叠在一起），灰线是 500 epoch 的 DETR-DC5 基线。红线每个**竖直跳变**都是一次 lr ×0.1，图中标注的 41.1 / 43.8 / 44.9 / 45.3 / 45.5 是不同方案的终点 AP；灰线到 400 epoch 才有一跳，最后收敛在 43.6。**读法**：红线 50 epoch 就到 43.8，已与灰线 500 epoch 的终值持平——差距不是「训得更久」，而是「学得更快」。

### 消融：每一块贡献多少（Table 2，第 9 页）

| 多尺度输入 | 多尺度注意力 | $K$ | 额外 FPN | AP | AP50 | AP75 | AP_S | AP_M | AP_L |
|---|---|---|---|---|---|---|---|---|---|
| | | 1 | w/o | 39.7 | 60.1 | 42.4 | 21.2 | 44.3 | 56.0 |
| ✓ | | 1 | – | 41.4 | 60.9 | 44.9 | 24.1 | 44.6 | 56.1 |
| ✓ | | 4 | – | 42.3 | 61.4 | 46.0 | 24.8 | 45.1 | 56.3 |
| ✓ | ✓ | 4 | – | **43.8** | 62.6 | 47.7 | 26.4 | 47.1 | 58.0 |
| ✓ | ✓ | 4 | FPN | 43.8 | 62.6 | 47.8 | 26.5 | 47.3 | 58.1 |
| ✓ | ✓ | 4 | BiFPN | 43.9 | 62.5 | 47.7 | 25.6 | 47.4 | 57.7 |

拆解（第 8 页原文口径）：多尺度输入 **+1.7 AP（AP_S +2.9）**；$K$ 从 1 到 4 **+0.9 AP**；跨层交换（多尺度注意力本身）**+1.5 AP**；而 FPN/BiFPN **一点不涨**。

> 最后两行是全文最有信息量的负结果：**当注意力已经能跨层取特征，金字塔的 top-down 融合就是冗余的**。反过来看，$K=1$ 且单尺度那行（39.7）本质是「退化成可变形卷积」，比完整版差 4.1 AP——关系建模这部分确实值 4 个点。

### 与 SOTA 比（Table 3，第 9 页 test-dev）

| 方法 | backbone | TTA | AP | AP50 | AP75 | AP_S | AP_M | AP_L |
|---|---|---|---|---|---|---|---|---|
| FCOS | ResNeXt-101 | | 44.7 | 64.1 | 48.4 | 27.6 | 47.5 | 55.6 |
| ATSS | ResNeXt-101 + DCN | ✓ | 50.7 | 68.9 | 56.3 | 33.2 | 52.9 | 62.4 |
| TSD | SENet154 + DCN | ✓ | 51.2 | 71.9 | 56.0 | 33.8 | 54.8 | 64.2 |
| EfficientDet-D7 | EfficientNet-B6 | | 52.2 | 71.4 | 56.3 | – | – | – |
| Deformable DETR | ResNet-50 | | 46.9 | 66.4 | 50.8 | 27.7 | 49.7 | 59.9 |
| Deformable DETR | ResNet-101 | | 48.7 | 68.1 | 52.9 | 29.1 | 51.5 | 62.0 |
| Deformable DETR | ResNeXt-101 | | 49.0 | 68.5 | 53.2 | 29.7 | 51.7 | 62.8 |
| Deformable DETR | ResNeXt-101 + DCN | | 50.1 | 69.7 | 54.6 | 30.6 | 52.8 | 64.7 |
| Deformable DETR | ResNeXt-101 + DCN | ✓ | **52.3** | 71.9 | 58.1 | 34.4 | 54.4 | 65.6 |

唯一超过 52 AP 的一行带着 TTA；不带 TTA 的最好成绩是 50.1（ResNeXt-101 + DCN），与 ATSS 的 50.7（同样带 TTA）同档，仍低于 EfficientDet-D7 的 52.2。作者的定位很克制：「without bells and whistles」49.0（第 8 页）。

### 它到底在看哪里（附录 A.5、A.6，第 14–15 页）

![Deformable DETR 梯度范数 Fig5](Deformable%20DETR%20梯度范数%20Fig5.png)

<small>裁自论文第 14 页 Figure 5（附录 A.5），仅取图本体，已去除图注文字。</small>

**怎么读这张图：** 五行分别是最终预测里 $x$、$y$、$w$、$h$ 与类别分数 $c$ 对输入每个像素的梯度范数（红色＝改这个像素最影响该输出）。$x$ 与 $w$ 的梯度集中在物体**左右边界**，$y$ 与 $h$ 集中在**上下边界**；最后一行 $c$ 的梯度**铺满物体内部**。

> 与 DETR 的差别就在这里：DETR 连类别判断也只看端点，而 Deformable DETR 判类别要看内部像素（附录 A.6 原文：focused on the whole foreground instance instead of only extreme points）。作者的解释是稀疏采样点没有足够的全局上下文，只能靠内部纹理定类别。

![Deformable DETR 采样可视化 Fig6](Deformable%20DETR%20采样可视化%20Fig6.png)

<small>裁自论文第 15 页 Figure 6（附录 A.6），仅取图本体，已去除图注文字。</small>

**怎么读这张图：** 每个点的颜色是该采样点的注意力权重（红高蓝低），绿十字是参考点。(a) 编码器的多尺度可变形自注意力：不同参考点的采样点**分别落在不同实例上**（猫、斑马、房间）——实例在编码器阶段已被分开；(b) 解码器的交叉注意力：绿框是预测框，采样点**覆盖整个前景实例**（自行车、人、电视），并随物体尺度与形状调整分布。

## 结论与局限

- **主结果**：50 epoch 43.8 AP（DETR-DC5 同预算 35.3），叠迭代精修 45.4、两阶段 46.2；训练 325 GPU 时（DETR 2000、Faster R-CNN 380），FLOPs 173G、FPS 19（第 8 页 Table 1）。
- **消融的排序**：多尺度输入 +1.7（AP_S +2.9）> 跨层交换 +1.5 > 采样点 $K$ 1→4 +0.9；FPN/BiFPN 零收益，是「多尺度融合可以内化进注意力」的直接证据（第 9 页 Table 2）。
- **相对 DETR 的得与失**：小物体 +5.9 AP_S、大物体 **−3.1 AP_L**（58.0 vs 61.1）——稀疏采样换来收敛与成本，代价是丢掉全局注意力对大物体的那部分优势（第 8 页）。
- **作者承认的局限**：推理仍比 Faster R-CNN 慢 25%，无序访存使它比传统卷积略慢（第 8 页）；两阶段的逐像素首阶段让训练从 325 涨到 340 GPU 时（第 8 页 Table 1）。
- **我看到的局限**：① 采样点被「参考点」绑死，而参考点源自 query / 上一层框，早期错框是否会把采样锁在错误区域，论文没有敏感性实验；② $K$ 只做到 4，没有 $K=8/16$ 的饱和曲线，「采样预算该多大」没答；③ Table 3 里本方最强行带 TTA、对手 EfficientDet-D7 不带，跨 backbone 的横向比较不宜直接引用；④ 对 query 的处理仍是 DETR 那套可学习位置嵌入，没有 query 语义/初始化方面的改进（后来 DINO 才补上）。

## 我的疑问

- [ ] 参考点由**上一层预测的框**决定，采样偏移又按框的宽高调制——这套「以预测为条件」的采样在早期猜错框时，梯度能否把它拉回来？加一组「多假设参考点」会不会更稳？
- [ ] $K=4$、$L=4$ 意味着每头只看 16 个点。若 $K$ 加到 8/16，收益在哪里饱和？论文缺这条曲线，而它直接决定工程上该花多少算力。
- [ ] 编码器参数「在各特征层间共享」（第 7 页）——共享与不共享差多少？这决定它到底是「一个跨层注意力」还是「四个独立注意力」。
- [ ] 附录 A.4 里按层的尺度先验 $2^{l_i-1}s$（$s=0.05$）很像手工先验：所谓「完全端到端」在工程上仍残留多少启发式？
- [ ] 如果把可变形注意力换进解码器的**自注意力**（而非只换交叉注意力），去重能力会变好还是变差？能否对上 DETR 笔记里「NMS 在深层失效」的观察？
- [ ] **中间框到底是不是必要的？** 采样本身只需要参考点（2D 坐标），为什么每层都要回喂完整的 4 维框？论文只给了「按框宽高调制采样偏移」（附录 A.4，第 12 页）这一条理由，消融也只有「开/关迭代精修」的 +1.6 AP，没有拆出「只前传参考点、不回喂宽高」的中间配置——这是最容易做但论文没做的对照。

## 关键摘录

> "Deformable DETR can achieve better performance than DETR (especially on small objects) with 10× less training epochs."（第 1 页，Abstract）

> "At initialization, the attention modules cast nearly uniform attention weights to all the pixels in the feature maps."（第 1 页，Introduction）

> "The proposed attention module will degenerate to deformable convolution (Dai et al., 2017), when L = 1, K = 1, and W′m ∈ RCv×C is fixed as an identity matrix."（第 6 页，§4.1）

> "Note that the top-down structure in FPN (Lin et al., 2017a) is not used, because our proposed multi-scale deformable attention in itself can exchange information among multi-scale feature maps."（第 6 页，§4.1）

> "Top scoring bounding boxes are picked as region proposals. No NMS is applied before feeding the region proposals to the second stage."（第 7 页，§4.2）

> "Our proposed Deformable DETR has on par FLOPs with Faster R-CNN + FPN and DETR-DC5. But the runtime speed is much faster (1.6×) than DETR-DC5, and is just 25% slower than Faster R-CNN + FPN."（第 8 页，§5.1）

> "A 3-layer feed-forward neural network (FFN) and a linear projection are added on top of the object query features … The FFN acts as the regression branch to predict the bounding box coordinates … The linear projection acts as the classification branch."（第 4 页，§3 DETR 综述）

> "Because the multi-scale deformable attention module extracts image features around the reference point, we let the detection head predict the bounding box as relative offsets w.r.t. the reference point to further reduce the optimization difficulty."（第 6 页，§4.1）

## 相关笔记

- [[Deformable DETR 的 MSDeformAttn 实现]] —— 官方仓库里这个模块的 python 封装：两个 Linear、偏移归一化、初始化到底怎么写
- [[DETR 用集合预测与二分匹配做端到端检测]] —— 本文的直接前作；「编码器自注意力 $O((HW)^2)$、DC5 代价 ×16」正是本文要绕开的东西
- [[DEATRN v2 可变形集成注意力任意形状场景文本识别]] —— 可变形采样在场景文本识别里的同构用法（按字符中心预测采样点）
- [[FlashAttention 精确注意力与IO感知]] —— 同一「注意力太贵」命题的另一条路：不改语义，只改访存

#论文 #目标检测 #Transformer #可变形注意力
