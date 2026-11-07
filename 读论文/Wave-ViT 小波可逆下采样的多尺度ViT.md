---
created: 2026-11-07
source: E:\.dsh-paper-reader\data\小波变换\Yao 等 - 2022 - Wave-ViT Unifying Wavelet and Transformers for Visual Representation Learning.pdf（ECCV 2022；JD Explore Academy + 新加坡管理大学；官方代码 https://github.com/YehLi/ImageNetModel ，commit 0b4302e，2023-05-05）
type: 论文
tags: [论文, 小波变换, Transformer, 多尺度]
---

# Wave-ViT：用小波做可逆下采样的多尺度视觉 Transformer

> Ting Yao, Yingwei Pan, Yehao Li, Chong-Wah Ngo, Tao Mei，*ECCV 2022*。下文页码均指本机 PDF 页码。

## 一句话总结

把多尺度 ViT 里「对 K/V 做平均池化下采样」换成 **DWT → 3×3 卷积 → IDWT** 的可逆下采样，让自注意力在不丢高频细节的前提下把注意力算力降到 1/4，再顺手把 IDWT 支路当作「扩了感受野的局部增强」拼进输出——ImageNet1K 上 Wave-ViT-S 用 19.8M 参数 / 4.3 GFLOPs 拿到 82.7% Top-1，比参数更大的 PVTv2-B2（25.4M / 4.0G / 82.0%）高 0.7（p.9）。

## 核心问题

- 多尺度 ViT（PVT、Swin、MViT）要处理高分辨率特征图，但自注意力代价是 O(H²W²D)，随 patch 数**平方**增长（p.6 式(1)）。主流做法是对 K/V 做空间下采样：PVTv2 用平均池化 [54]，MViT 用池化核 [14]。
- 作者的反驳：**这类下采样不可逆**，必然丢信息，尤其丢物体的高频成分（纹理细节）；而且已有研究表明池化会破坏 CNN 的平移等变性（p.2）。
- 所以真正要的不是「少算点」，而是「**少算但不丢**」。小波分析恰好提供这个性质：DWT 把 2D 数据无损拆成 1 个低频 + 3 个高频子带，IDWT 还能原样还原（p.2 图 1(a)）。
- 此前的空白：小波在 CNN 里被用作跨 stage 的下/上采样，但**从未进入 Transformer block 内部**去降 K/V 的分辨率（p.5）。

## 关键方法

### 1) 整体骨架：四 stage、只在前两个 stage 用小波块（p.8）

输入 224×224，四个 stage 依次输出 H/4 → H/32，通道逐级放大。以 Wave-ViT-S 为例：depth = 3/4/6/3，维度 64/128/320/448，head = 2/4/10/14（p.8 表 1）。每个 stage = patch embedding + 若干 Wavelets block + FFN。

### 2) Wavelets block：一次前向里同时拿「长程」和「局部」（p.6–p.7，图 2(c)）

![Wave-ViT Wavelets块对比 Fig2](Wave-ViT%20Wavelets块对比%20Fig2.png)

> **怎么读这张图**（p.5 Fig. 2）：三块并排比较。(a) 是 ViT 原生自注意力——Q/K/V 全在 H×W×D，Softmax 后注意力图是 H×W×H×W，最贵。(b) 是 PVTv2 那类做法——先用 **DS(2,2)** 把输入池化到一半再出 K/V，注意力图缩成 H×W×H/2×W/2，但**池化不可逆**。(c) 是本篇的 Wavelets block：输入分两条路，左路 `Linear → Query` 保持全分辨率；右路先 `Linear` 降到 D/4 通道，再 `DWT` 拆成四个子带（图上标的 H/2×W/2×D），接 `Conv 3×3` 做局部建模，然后**兵分两路**——一路 `Linear → Key / Value` 参与 Softmax 注意力（H/2×W/2），另一路 `IDWT` 单独上采回 H×W×D/4，最后与注意力输出拼接后过 `Linear`。**关键在右下那根 IDWT 的线**：它让「下采样」变成可逆的，且顺手给了一个大感受野的局部特征。

具体四步（p.7）：

1. **降维**：X̃ = X·W_d，W_d ∈ R^(D×D/4)。
2. **Haar DWT**：低通 f_L = (1/√2, 1/√2)、高通 f_H = (1/√2, −1/√2)，先沿行再沿列，得 X_LL, X_LH, X_HL, X_HH ∈ R^(H/2×W/2×D/4)。X_LL 是粗粒度结构，另三个是细粒度纹理——**四者合起来覆盖全部信息，无丢弃**。
3. **拼接 + 局部化**：通道维拼成 X̂ ∈ R^(H/2×W/2×D)，过 3×3 卷积得 X_c。
4. **两用**：X_c 一路出 K_w, V_w 做注意力 Softmax(Q·K_wᵀ/√D_h)·V_w；另一路做 IDWT 得全分辨率 X_r。最后

   WaveletsBlock(X) = Concat(head₀, …, head_Nh, X_r) · W^O　（式 3，p.7）

### 3) DWT-Conv-IDWT 为什么比单层 3×3 卷积强（p.2，图 1(c)(d)）

![Wave-ViT 架构图 Fig1](Wave-ViT%20架构图%20Fig1.png)

> **怎么读这张图**（p.2 Fig. 1）：(a) 直观展示 DWT——一张街景图被拆成 LL（模糊的轮廓）和三个高频图（只剩边缘与纹理），箭头标了 DWT/IDWT 双向。(b) 是 Wavelets block 的骨架：输入 x 进 DWT→Conv 3×3，分叉给 Multi-Head Attention 的 q/k/v 与 IDWT；注意 **q 走的是原始 x**（全分辨率），而 IDWT 的输出汇回输出端。(c) 与 (d) 是作者想让你对比的：一个 3×3 卷积只能覆盖图中红框那一小块；(d) 先 DWT 降一半、再卷积、再 IDWT 升回来，**同一个 3×3 核在放大一倍的网格上生效**，等效感受野翻倍，代价可忽略。这就是消融里 (c)→(d) 涨 0.2 点的来源。

### 4) 官方代码（本次真实读过，非默写）

```python
# YehLi/ImageNetModel@0b4302e classification/wavevit.py:167-178
self.dwt  = DWT_2D(wave='haar');  self.idwt = IDWT_2D(wave='haar')
self.reduce = nn.Sequential(nn.Conv2d(dim, dim // 4, 1), nn.BatchNorm2d(dim // 4), nn.ReLU(inplace=True))
self.filter = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True))

# wavevit.py:208-221
x_dwt  = self.filter(self.dwt(self.reduce(x)))    # DWT → 四子带按通道拼接 → 3×3 卷积
x_idwt = self.idwt(x_dwt)                         # 逆变换回 H×W×D/4
kv     = self.kv_embed(x_dwt)                     # 额外 sr_ratio 跨步卷积出 K/V
x = self.proj(torch.cat([x, x_idwt], dim=-1))     # 注意力输出 ⊗ IDWT 分支
```

💻 完整张量形状对照、论文写法与代码的三处出入见 [[Wave-ViT 的 WaveAttention 实现]]

## 结论与局限

- **ImageNet1K（p.9 表 2）**：Wave-ViT-S 19.8M / 4.3G / **82.7%**，对比 PVTv2-B2 25.4M / 4.0G / 82.0%；加 Token Labeling + conv-stem 后 -S⋆ 83.9%、-B⋆ 84.8%、-L⋆ 85.5%（57.5M / 14.8G，压过 86.3M / 20.6G 的 VOLO-D3⋆ 85.4%）。
- **COCO（p.11 表 3/表 4）**：RetinaNet 1× 上 AP 45.8（PVTv2-B2 44.6）；Mask R-CNN 上 AP^b 46.6 / AP^m 42.4；四个检测器（GFL / Sparse R-CNN / Cascade / ATSS）一致涨点——**小目标 AP_S 29.2 vs 27.4**，正是「高频没被丢」最该体现的地方。
- **ADE20K（p.12 表 5）**：UPerNet 下 Wave-ViT-S/B 达 49.6 / 51.5 mIoU，比 Twins-SVT-S（47.1）/ Swin-S（49.5）高 **2.5 / 2.0**。语义分割涨得比分类型还多，说明收益确实落在细节上。
- **消融（p.13 图 3，全为 Small 规模）**：平均池化 82.0（18.7M/3.9G）→ 池化核 82.0（20.4M/4.0G）→ **只加 DWT 可逆下采样 82.5**（19.8M/4.3G）→ **再加 IDWT 支路 82.7**（同上）。即：换掉不可逆池化 +0.5，IDWT 局部增强再 +0.2，**参数量几乎不变、FLOPs 只多 0.3G**。这是全文最硬的一条证据。
- **局限（作者未展开，我观察到）**：
  1. **收益幅度不大**。+0.7 Top-1 里真正属于「可逆」的只有 +0.5，且要用 4.3G（比 4.0G）算力换——「comparable FLOPs」的说法略宽。
  2. **只降 K/V，不降 Q**。注意力图仍是 H×W×H/2×W/2，复杂度从 O(n²) 只降到 O(n²/r²)，**并没有跳出二次复杂度**，与线性注意力路线无法叠加讨论。
  3. **只在固定 Haar 基、单级分解上验证**（p.7 明写 choose the classical Haar wavelet … for simplicity），小波基 / 分解级数 / 是否可学习都没做消融——对比 [[LWTformer 可学习小波变换的古汉字图像修复]] 把核做成可学习，这里显然放弃了那部分潜在收益。
  4. **只在前两个 stage 用小波块**（官方代码里 block_type='wave' if i < 2 else 'std_att'），深层已无下采样需求，所以「小波 = 效率工具」的定位很清楚，但也很窄。
  5. 分类与密集预测的对比都在 224² 与 COCO 尺度上，**高分辨率（1024² 检测、超分）没测**，而那才是二次复杂度真正致命的地方。

## 我的疑问

- [ ] 注意力图仍是 H×W×H/2×W/2：如果连 Q 也走 DWT、再 IDWT 回来，代价会掉多少、精度会掉多少？论文没试。
- [ ] Haar 是「无损 + 正交 + 常数核」里最简单的一个；换成双正交小波（如 bior4.4）或可学习核（LWTformer 路线）在**分类**任务上是否还有增益？图像复原那边可学习核明显更强，分类这边未必。
- [ ] 图 3 里 (a) 平均池化与 (b) 池化核都是 82.0——池化核没能赢平均池化，那「不可逆」的锅究竟该记在池化上，还是记在「下采样倍率 2×」上？缺少 4× 下采样的对照。
- [ ] DWT 的滤波器被固定成 float16 buffer（torch_wavelets.py:110-113），训练全程是否引入精度损失？论文只说 negligible。

## 关键摘录

> "In this work, we argue that such over-aggressive down-sampling design is not invertible and inevitably causes information dropping especially for high-frequency components in objects (e.g., texture details)."（p.1）

> "the commonly adopted down-sampling operations like average pooling are irreversible, and inevitably result in information dropping."（p.6）

> "In this way, each wavelet subband can be regarded as the down-sampled version of X̃, and all of them cover every detail of inputs without any information dropping."（p.7）

> "Compared to the most efficient (a) with irreversible down-sampling, the Top-1 score of (c) increases from 82.0% to 82.5%."（p.14，对应图 3 的消融描述）

## 相关笔记

- [[小波变换]] —— 库内小波主题入口页；本篇是「用固定 Haar 做**下采样算子**」这一支最直接的原型
- [[AdaptIR 自适应频率增强的一体化图像复原]] —— 同样是 DWT→卷积→IDWT 的「级联扩感受野」，但用在复原的 RFEM 里且不接注意力；本篇把它搬进了自注意力 block，可正面对照同一结构在分类与复原里的不同落点
- [[LWTformer 可学习小波变换的古汉字图像修复]] —— 反方向取舍：那边牺牲效率换可学习小波核，这边为效率钉死 Haar
- [[Light-MambaIR 时频感知稀疏状态空间统一复原]] —— 另一条「固定 Haar 求效率」的路线（SSM + 小波高频支路并行），可与本篇的「串行进注意力」对照
- [[Restormer 高效高分辨率图像修复 Transformer]] —— 同为降低高分辨率下的注意力成本，那边改的是通道维注意力（在 C 上算而非 H×W），与这里的空间下采样是正交的两条路
- [[StarIR 空间-频率融合的卷积图像复原]] —— 显式频域分支的另一种用法（FFT + 可学习频响），对照本篇「只看小波、不做频谱调制」
