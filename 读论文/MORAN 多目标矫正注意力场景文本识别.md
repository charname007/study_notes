---
created: 2026-11-03
source: Luo 等 - 2019 - MORAN A Multi-Object Rectified Attention Network for scene text recognition（Pattern Recognition vol.90, pp.109–118, 2019）；本机 PDF：E:\.dsh-paper-reader\data\默认专题\Luo 等 - 2019 - MORAN A Multi-Object Rectified Attention Network for scene text recognition.pdf
type: 论文
tags:
  - 论文
  - 场景文本识别
  - Attention
  - 文本矫正
---

# MORAN：多目标矫正 + 注意力序列识别的场景文本识别

## 一句话总结

把不规则文字识别拆成「先逐块预测偏移场把图掰直（MORN）→ 再用注意力解码器逐字读出（ASRN）」两段，用一条弱监督损失端到端联合训练；配套两个纯训练期技巧——分数拾取（fractional pickup）加宽注意力视野、三阶段课程学习稳住收敛。IIIT5K 无词典 91.2%，弯曲文本 CUTE80 77.4%。

## 核心问题

场景文字常以透视、弯曲等**不规则**形态出现（Fig. 1，第 2 页），主流识别器的注意力难以对齐到正确字符区域。已有的 top-down 校正方案各有一处硬约束：

- **仿射变换**（STAR-Net [34]）：只能旋转/缩放/平移，复杂变形处理不了，还会引入黑边噪声（第 3 页）。
- **RARE [12] 的 TPS + 基准点**：点数写死在网络结构里，图像一宽点就被摊薄，抓不住局部细节（第 4 页）。
- **可变形卷积**：作者试过，序列到序列优化下有时不收敛，IIIT5K 只有 78.1%（第 3 页）。

MORAN 的答案：**让校正网络直接回归一张稠密位移场，不受几何约束、也不受固定点数约束。**

## 关键方法

### 1. 整体结构：MORN → ASRN，端到端

MORN 输出校正图，ASRN 读字。虚线表示梯度只从文本标签回流，**不需要任何字符位置或像素级标注**（第 2 页 Fig. 2、第 4 页 "weak supervision"）。

![MORAN 整体结构图 Fig4](<MORAN 整体结构图 Fig4.png>)

**怎么读这张图：** 左半是 MORN——输入图先经过 Conv 前的池化（括号标注 Pooling before Conv），进入一组卷积，输出低分辨率的 Offset Maps；Offset Maps 经 Resize（双线性）铺到原图尺寸，再与 Basic Grid 相加，得到 Resized Offset Maps；最后交给采样器（S）。右半是 ASRN——校正图经 CNN 得到序列特征，喂 BLSTM 融上下文，解码端每步用 Attn 取 glimpse、GRU 出字符；红字 "Fractional pickup in training phase" 指出注意力权重在**训练时**被额外扰动。

### 2. MORN：33 个锚点 + 双线性展开

输入 `1×32×100`，最前面就放 `MaxPooling k2×2, s2×2`（第 3 页）：

> "the MORN predicts the position offsets but not the categories of characters. The character details for classification are not necessary. We hence place a pooling layer before the convolutional layer to avoid noise and reduce the amount of calculation."

之后每层卷积接 BN + ReLU，**最后一层除外**（因为输出是带符号的偏移量）。网络把图粗分成 `3×11 = 33` 个部分各预测一个偏移，通道收敛到 2（x/y 方向）：

| 层 | 输出尺寸 |
|---|---|
| Input | 1 × 32 × 100 |
| MaxPooling k2×2 s2×2 | 1 × 16 × 50 |
| Conv 64 / MaxPool s2×2 | 64 × 8 × 25 |
| Conv 128 / MaxPool s2×2 | 128 × 4 × 12 |
| Conv 64 / Conv 16 / Conv 2 | 2 × 4 × 12 |
| MaxPooling k2×2 s1×1 | **2 × 3 × 11** |
| Tanh | 2 × 3 × 11 |
| Resize（双线性） | **2 × 32 × 100** |

（第 3 页 Table 1）

**关键点：33 个偏移不是「33 个刚性平移块」**，而是控制点。经双线性插值后 `32×100` 每个像素各得一个连续变化的偏移量。论文用 "smoothly resize" 和 "no indented edges" 描述这一点（第 3 页）；Fig. 3 的判据是热图上红蓝边界的颜色**渐变**：

![MORAN 仿射变换对比与偏移热图 Fig3](<MORAN 仿射变换对比与偏移热图 Fig3.png>)

**怎么读这张图：** 第一行仿射变换把歪的「22」拉正，但引入了黑边；第二行 MORN 的中间图是预测出的 offset map 热力图（红蓝渐变，过渡处偏移为零、两侧方向相反）；第三行黄色箭头标出每个字符各自的校正方向——两个 "2" 方向不同，这正是仿射做不到的；第四行是校正结果，边缘无锯齿、无黑边。

采样是**反向映射**（第 3 页 Eq. 1–3）：

```
offset'(c,i,j) = offset(c,i,j) + basic(c,i,j),  c = 1,2      (1)
I'(i,j) = I(i', j')                                          (2)
i' = offset'(1,i,j),  j' = offset'(2,i,j)                    (3)
```

`basic` 是归一化到 [−1,1] 的原始网格（左上 (−1,−1)、右下 (1,1)）。因为 `i', j'` 是**实数**，采样必须双线性；也正因为这一式可微，MORN 才能被识别损失反传训练（第 4 页）。

官方实现（`Canjie-Luo/MORAN_v2` @ `410596f`，`models/morn.py`）：

```python
self.cnn = nn.Sequential(
    nn.MaxPool2d(2, 2),                                    # 池化在最前，对应论文设计
    nn.Conv2d(nc, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
    ...
    nn.Conv2d(16, 1, 3, 1, 1), nn.BatchNorm2d(1)           # 末层无 ReLU
)
```

### 3. ASRN：26 步序列 + 加性注意力

CNN 主干把高度压到 1、宽度保留 26，得到 `512×1×26`（第 4 页 Table 2），即长度 `L=26` 的**序列特征向量** `h_i`——注意 `L` 是特征图宽度，不是字符数。解码器每步做一次加权求和（Eq. 5–9）：

```
g_t = Σ_{i=1..L} α_{t,i} h_i                       (7)
α_{t,i} = exp(e_{t,i}) / Σ_j exp(e_{t,j})          (8)
e_{t,i} = Tanh(W_s s_{t-1} + W_h h_i + b)          (9)
s_t = GRU(y_prev, g_t, s_{t-1})                    (5)
```

遇到 EOS 就停，所以**序列长度固定 26、输出长度可变**（第 4 页）。

### 4. 分数拾取（fractional pickup）：训练期加宽注意力

动机：解码器的注意力峰太窄，字符边界不清或有阴影时容易漂到邻居上（第 5 页）。做法是每个时间步随机挑一对**相邻**权重掺混（Eq. 10–12）：

```
α'_{t,k}   = β α_{t,k}   + (1−β) α_{t,k+1}         (10)
α'_{t,k+1} = (1−β) α_{t,k} + β α_{t,k+1}
β = rand(0,1),   k = rand[1, T−1]                  (11)(12)
```

**注意 "fractional" 指小数 β，不是「得分」。** 因为两者互换一部分、总和不变，不需要重新 softmax。三条机制（第 6 页）：① 注意力分布每步都变 → 防过拟合；② 相当于给 BLSTM 造了一条前向捷径（Fig. 7）；③ 梯度耦合从 `δh_k = δg_t α_{t,k}`（Eq. 13）变成 `δh_k = δg_t(β α_{t,k} + (1−β) α_{t,k+1})`（Eq. 14），误差项扯上了 `h_{k+1}`，反向传播的影响范围因此变宽。

**只在训练期使用**——推理不执行 Eq. 10，不增加任何开销。

![MORAN 分数拾取注意力对比图 Fig6](<MORAN 分数拾取注意力对比图 Fig6.png>)

**怎么读这张图：** 三组样本左右对照（左无 FP、右有 FP），每组上面是原图、下面若干条是各时间步的 α_t 热力图（EOS 已删）。有 FP 一侧的亮斑明显更宽、更连续，且识别结果从 hot→hotel、argeh→angels、easer→laser 全部纠正——直观说明「视野变宽 → 少漏字、少错字」。

### 5. 三阶段课程学习

MORN 和 ASRN 会互相拖累（ASRN 太强就不用 MORN 校正；样本已被 MORN 掰直，ASRN 就学不到鲁棒性），所以分三步训（第 6 页）：先单独训 ASRN，再冻结 ASRN 训 MORN，最后联合端到端微调。实现细节：ADADELTA，学习率 1.0 起、第三阶段降到 0.01，batch 64，三阶段分别 600,000 / 20,000 / 300,000 次迭代，总计约 30 小时（第 7 页）。

## 结论与局限

- **主结果**：无词典模式下 IIIT5K 91.2%、SVT 88.3%、IC03 95.0%、IC13 92.4%、SVT-P 76.1%、CUTE80 77.4%、IC15 68.8%（Table 8/9，第 8 页）。IC15 上持平当时最好，其余多项领先。
- **消融（Table 4，第 8 页）**：只 ASRN 84.2 → 加 MORN 89.7 → 再加 FP **91.2**（IIIT5K）。MORN 贡献 +5.5，FP 贡献 +1.5。总编辑距离同步下降：IC03 22.7→19.8、IC13 45.3→42.0、IC15 345.2→334.0（Table 5）。
- **池化位置的消融（Table 3，第 7 页）**：不加池化 85.7，平均池化 s2/s1 为 89.2/89.3，最大池化 s2/s1 为 **90.4/91.2**——「卷积前放最大池化、stride=1」是最优配置，验证了第 3 页那个设计动机。
- **同类对比（Table 7，第 8 页）**：同识别网络下对 RARE 全面领先，差距最大在 CUTE80（59.2 → 72.6，+13.4），正是「固定基准点数」最吃亏的弯曲文本场景。
- **作者承认的局限（第 8 页）**：① 曲线角度太大时会把复杂背景误判成前景而失败（Fig. 9 最后两行）；② 训练集只有水平合成文本，**不适用于竖排文本**——IC15 里大量竖排，所以 FP 在 IC15 上几乎没提升（68.6→68.8）；③ 实验都基于裁剪好的文本图，不带检测器就不是端到端系统。
- **我观察到的**：FP 的收益高度集中在「横向但注意力易漂移」的场景；一旦超出设计前提（竖排、大曲率），收益基本归零。

## 我的疑问

- [ ] **通道数对不上**：论文第 3 页说偏移图"contain two channels, which denote the x-coordinate and y-coordinate"，Table 1 也写 `Convolution maps:2`；但官方代码 `models/morn.py` 末层是 `nn.Conv2d(16, 1, 3, 1, 1)`（**1 通道**），且 `offsets_x = torch.cat([grid_x, grid_y + offsets_grid], 3)` 只把偏移加到 y 上。是 v2 版本改了，还是论文与实现本就有出入？
- [ ] **激活函数对不上**：论文说 `Tanh` 限制到 (−1,1)，代码里没有 Tanh，而是 `relu(offset) - relu(-offset)` 再做 `MaxPool2d(2,1)`——这是对带符号值正确实现 max pooling 的技巧，但因此**偏移量并无理论上界**。论文的 (−1,1) 是简化描述？
- [ ] **β 的取值范围**：论文 Eq. 11 是 `β = rand(0,1)`；代码 `models/fracPickup.py` 写 `beta = npr.rand()/4.`，即 β ∈ [0, 0.25)。另外代码作用在**采样网格坐标** `w_list` 上再 `grid_sample`，而非直接改 α。两者数学上近似等价（改坐标经双线性即等于改权重），但强度只有论文写的 1/4。
- [ ] **Table 1 的 256 还是 512**：表注 `BLSTM hidden unit:256` 输出 `256×1×26`，但双向 LSTM 通常拼接，应是 512。代码 `asrn_res.py` 里是 `nn.LSTM(nIn, nHidden, bidirectional=True)` + `nn.Linear(nHidden*2, nOut)`，即 512 再压回 256——表格写的是**投影后的维度**。
- [ ] 代码里 MORN 前向有一句 `if not test and np.random.random() > 0.5: return upsample(x)`，即训练时有一半概率直接跳过校正。论文没提这个随机跳过，它算正则化还是别的原因？

## 关键摘录

> "Furthermore, the MORN predicts the position offsets but not the categories of characters. The character details for classification are not necessary. We hence place a pooling layer before the convolutional layer to avoid noise and reduce the amount of calculation."（第 3 页）

> "The MORN first divides the image into several parts and then predicts the offset of each part. With an input size of 32 × 100, the MORN divides the image into 3 × 11 = 33 parts."（第 3 页）

> "3) The MORN is more flexible than methods using a specific number of regressing points. Existing method [12] cannot capture the text shape in details if the width of the image is large. Thus the MORN has no limit with respect to the image size, especially the width of the input image."（第 4 页）

> "Both i′ and j′ are real values as opposed to integers so rectified image I′ is sampled from I using bilinear interpolation. Because Eq. (2) is differentiable, the MORN can back-propagate the gradients."（第 4 页）

> "We propose a training method called fractional pickup that fractionally picks up the neighboring features in the training phase."（第 5 页）

> "RARE using fiducial points can only capture the overall text shape of an input image, whereas the MORAN can rectify every character in an image. ... Furthermore, the MORAN without any fiducial points is theoretically able to rectify text of infinite length."（第 7 页）

> "The training of MORAN is more difficult than that of RARE. We thus designed a curriculum learning strategy to enable the stable convergence of the MORAN."（第 7 页）

## 相关笔记

- [[ASTER 注意力场景文本识别器]] — 同为「矫正 + 注意力 Seq2Seq」的 top-down 路线，但 ASTER 走 TPS/STN 固定 20 控制点，MORAN 走无约束偏移场，可对照看两条非刚性矫正分支的取舍
- [[SVTR 场景文本识别]] — 后续工作，直接复用 TPS 矫正模块；与 MORAN 的「校正网络 + 识别主干」是同一思路的不同时代实现
- [[齐次坐标变换]] — MORAN 的 basic grid + offset 求和在数学上就是坐标映射，与齐次坐标描述仿射/投影属同一家族，可并读
- [[PAGRN 渐进式注意力引导循环网络]] — 同样用注意力热图可视化来验证「定位是否对齐」，热图的读法可以互相借用

---

#论文 #场景文本识别 #Attention #文本矫正
