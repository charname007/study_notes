---
created: 2026-11-25
updated: 2026-11-25
source: E:\.dsh-paper-reader\data\STR\Mou 等 - 2020 - PlugNet Degradation Aware Scene Text Recognition Supervised by a Pluggable Super-Resolution Unit.pdf（LNCS 模板，本机 PDF 共 17 页；未见 arXiv / 期刊编号）
type: 论文
tags: [论文, 场景文本识别, 超分辨率, 多任务学习, 低质量文本]
---

# PlugNet：把超分当"训练期的监督信号"，而不是"推理期的前处理"

> 用户选中标题（第 1 页）："PlugNet: Degradation Aware Scene Text Recognition Supervised by a Pluggable Super-Resolution Unit"。作者 Yongqiang Mou、Lei Tan（共同一作）、Hui Yang、Jingying Chen、Leyuan Liu、Rui Yan、Yaohong Huang——广州图普 AI-Labs + 华中师范大学（论文署 Nercel），基金：NSFC 61977027 / 61702208、湖北省技术创新重大项目 2019AAA044（p.1、p.15）。页码均指本机 PDF 页码，页内分布：p.1 标题与摘要，p.2–3 引言与动机（Fig.1），p.3 贡献列表，p.4 相关工作与 §3.1 开头，p.5 Fig.2 总览与四模块，p.6 Table 1 骨干结构与 §3.2 PSU，p.7 §3.3 FSM/FEM 与 §3.4 训练数据，p.8 式 1–4 与 Eq.3 近似，p.9 §4.1 数据集与 §4.2 实现细节（含速度），p.10 Table 2 分辨率消融与 Fig.3，p.11 Table 3 消融与 Fig.4，p.12–13 §4.4 λ 实验（Fig.5/6），p.13–14 Table 4 与 §4.5，p.15 结论，p.16–17 参考文献。

## 一句话总结

给一个 Aster 式识别器**挂一个只在训练期存在的超分解码器（PSU）**：它从共享 CNN 骨干的**高层特征**反推出"干净矫正图"，用像素 L1 逼着骨干把高频与空间信息留在特征里；训练完把 PSU 整个删掉，推理零额外模块。配套还有两处轻量改动——**FSM**（去掉最后三个 block 的下采样，让特征保持 $8\times25$ 再用 $1\times1$ 卷积 + reshape 压成序列）与 **FEM**（五级特征 concat 成 992 通道）。七个无词典基准上 IIIT5K 94.4 / SVT 92.3 / IC03 95.7 / IC13 95.0 / IC15 82.2 / SVTP 84.3 / CUTE80 85.0（p.13–14 Table 4）。

## 核心问题

低质量（糊、抖、低分辨率）文本图让识别器缺细节（p.2 Fig.1(a)）。当时的通行解法是**图像级**：串一个超分网络再识别（TextSR = ESRGAN + Aster）。作者摆出两条反对意见（p.2）：

1. **识别不涨**：TextSR 视觉质量上去了，识别率提升有限；
2. **贵到不可接受**：他们用单张 2080Ti 在 Synth90K 上训 ESRGAN-Aster，**"needs nearly 30 days each epoch"**（p.2）。

于是改走**特征级**（p.2 Fig.1(b)）：不生成图，而是让**骨干特征本身**足以支撑"从特征重建出干净图"这件事，识别头直接吃这份特征。PSU 是"训练期的脚手架"，不是推理链路的一环。

这个立论其实包含一个不太显眼但很关键的判断：**"看清"不必以"生成一张图"为中介**。只要重建所需要的空间/高频信息被保留在特征里，识别就能用上它——所以 SR 的功劳可以完全落在特征上，代价只在训练。这一点是后面所有"单阶段 / 蒸馏 / 多任务"LTR 路线的共同起点（见文末相关笔记）。

## 关键方法

### 0. 总览：四段流水线（p.5 Fig.2）

~~~
输入图 (32×100)
  → 矫正网络（Aster 那套 TPS：Localize → Grid Generator → Sampler）
  → 共享 CNN 骨干（Table 1，Block 0–5；后三个 block stride=1）
       ├─→ PSU (c)：8×25×512 → RCAN 式重建 → SR 图（32×100）  【只在训练期】
       └─→ FEM (b)：五级特征降到 8×25 后 concat → 8×25×992
              → FSM (d)：1×1 conv → 8×25×128 → reshape → 25×1024
              → BiLSTM 编码器（2 层，256 hidden）
              → 注意力 LSTM 解码器（2 层）→ 字符序列
~~~

**怎么读 Fig.2**（p.5）：图按 (a)–(e) 分块画。(a) 是矫正，三个小框依次是定位网络、网格生成器、采样器；(b) 是骨干，块越往右通道越厚（32→512），但高度不再降；(c) 是 PSU，上排 Conv → ReLU → Conv → GP → Conv → ReLU → Conv → Sigmoid 是 RCAB 里的**通道注意力**支路，两个 RCAB 组成一个 RG、两个 RG 组成 PSU，最后 Conv + Upsampling 出图；(d) 是 FSM，图上标的三个张量 8×25×128 → 8×128 → 25×1024 就是"把高度塞进通道"的全过程；(e) 是 BiLSTM + 注意力解码，图上示例输出是 "coffee"。

### 1. 改骨干：为什么必须先把 $8\times25$ 保住（p.5–7）

Table 1（p.6）显示：输入 32×100，Block 1/2 各做一次 stride=2，到 Block 2 之后是 $8\times25$；**Block 3/4/5 全部 stride=1**，通道 128/256/512，所以骨干输出的分辨率一直是 $W/4 \times H/4 = 8\times25$。Aster 的 45 层 ResNet 则一路把高度压到 1，得 $1\times25$。

作者给的理由是"空间锐度（spatial acuity）的丢失同时拖累了识别头和矫正头"（p.3 原话："Due to the loss of spatial acuity makes it difficult for both the recognition part and the rectified part to get effective learning."）。FSM 只需一个 $1\times1$ 卷积（通道压缩）+ 一个 reshape：

$$(8,\,25,\,128)\ \xrightarrow{\text{reshape}}\ (25,\,8\times128=1024)$$

高度被折进通道，序列长度仍是 25（每个字符列一个 token），但每列带 8 行信息。消融时为了口径一致，$1\times1$ 卷积的输出通道随分辨率反着调（$1\times25{\to}1024$、$2\times25{\to}512$、$4\times25{\to}256$、$8\times25{\to}128$），保证四种设定输出都是 $25\times1024$——**分辨率越高，那个 $1\times1$ 卷积反而越窄、参数越少**，所以 Table 2 的提升不可能是"参数变多"换来的，这一点作者没明说，但把通道数写出来就等于做了控制变量。

FEM（p.7）：把 Block 0/1 的特征先下采样到 $W/4\times H/4$，再与 Block 2–5 沿通道 concat。按 Fig.2 标的 $8\times25\times992$ 反推，参与拼接的通道正是 $32+64+128+256+512=992$（Block 0 的通道数 Table 1 未列，此为按合计反推）。

顺带一个历史意义：这篇 2020 年的论文是**"别把高度压成 1"这一做法的早期实证**——Table 2 在只用 90K、没有 PSU/FEM 的条件下，光把分辨率从 $1\times25$ 拉到 $8\times25$ 就涨 SVT +3.8 / SVTP +5.9 / IIIT5K +4.6 / IC15 +4.3（p.10）。后来的 ViTSTR、PARSeq 干脆全程保留 2D 特征，方向是一致的。

### 2. PSU：从"高层特征"重建图（p.6）

结构直接借 RCAN：每个 RG 两个 RCAB，共两个 RG，末端 Conv + Upsampling。**输入是骨干末端的高层特征（$8\times25\times512$），输出是 $32\times100$ 的干净矫正图**——注意这与常规 SR 的"低层特征 → 高分辨率图"恰好相反，也和"输入低质图 → 输出高质图"的图像级 SR 不同：PSU 反推的是**判别性特征里还剩多少像素级信息**。

所以严格说它不是"放大"（输出与输入同尺寸，相对特征才是 4× 上采样），而是**去退化重建**。也正因为如此，它配的是 L1 像素损失（p.8 式 6，作者引 EDSR 的经验：L1 收敛优于 L2），而不是感知/GAN 损失——像素是手段，逼特征才是目的。

训练完成后 **"the PSU will be removed which means no extra computation in the inference stage"**（p.6）。但要注意口径：**PSU 零开销 ≠ 整个方法零开销**。FSM/FEM 留在推理链里，实测 22 ms vs Aster 20 ms（batch=1，p.9），训练 0.97 s vs 0.63 s（batch=128）。也就是说"零额外计算"这句话只对 PSU 成立，对整篇方法约 +10% 推理延迟。

### 3. SR 损失的三个坑（p.7–8）

**(坑一) 没有 LR-HR 配对**。文本识别数据集不带高低分辨率成对样本，于是自造退化（p.7 式 1）：

$$I_{blur}=\begin{cases}
f_{d\text{-}u}(f_{gau}(I)), & p_1\ge\alpha,\ p_2\ge\alpha\\
f_{gau}(I), & p_1\ge\alpha,\ p_2<\alpha\\
f_{d\text{-}u}(I), & p_1<\alpha,\ p_2\ge\alpha\\
I, & p_1<\alpha,\ p_2<\alpha
\end{cases}$$

$f_{gau}$ 高斯模糊、$f_{d\text{-}u}$ 先降后升采样，$\alpha=0.5$，四种组合各约 1/4 概率。退化模型就这两个算子——不像后来的 Real-ESRGAN 系那套模糊核/噪声/压缩的退化建模，所以标题里的 "degradation aware" 其实是"对模糊与降采样 aware"。

**(坑二) 退化的位置**。理想写法是"先矫正、再退化"：

$$L_{sr}=f_{loss}\big(f_{rn}(I),\ f_{blur}(f_{rn}(I))\big)\tag{2}$$

但这样网络输入就是一张"已经被矫正过的退化图"，没法再塞回矫正网络——流水线断了。

**(坑三) 于是做近似**（p.8 式 3）：把退化挪到矫正**之前**，两边各过一次矫正：

$$L_{sr}=f_{loss}\big(f_{rn}(I),\ f_{rn}(f_{blur}(I))\big)\tag{3}$$

作者的措辞是"the data generation strategies should take effect after the Rectification Network which means the input images are high-resolution"，并称这样可以"simplify the whole networks"。

> 我的读法：这是一个**必要但不免费的近似**。TPS 矫正的控制点是由输入内容预测的，$f_{rn}(f_{blur}(I)) \ne f_{blur}(f_{rn}(I))$——糊图上的控制点本来就会跑偏（Fig.3 正是在说分辨率高时控制点更准）。副作用是 Eq.3 里 $f_{rn}$ 出现在两项中，**SR 损失同时在监督矫正网络**，于是"PSU 到底在教骨干抗退化，还是在教矫正头放准控制点"就混在一起了。论文没有任何一项实验把这两者拆开。

**总损失**（p.8 式 4–6）：

$$L=L_{rec}+\lambda L_{sr},\qquad \lambda=0.01$$
$$L_{rec}=-\frac{1}{MN}\sum_{i=1}^{M}\sum_{j=1}^{N}y_{i,j}\log s_{i,j},\qquad L_{sr}=\frac{1}{WH}\sum_{i=1}^{W}\sum_{j=1}^{H}\lVert O_{i,j}-I_{i,j}\rVert$$

$O$ 是 PSU 输出，$I$ 是"输入图"的真值（即 $f_{rn}(I)$ 那张干净矫正图）。

### 4. 训练细节（p.7–9）

- 数据：Synth90K + SynthText；SynthText 裁出 **4M** 单行图（Aster 裁 7M，p.7）。作者特意说明"we do not separate the train data and test data"——指这两个**合成集内部**不划分训练/测试，并非把评测集混进训练。
- batch 128（64 + 64），ADADELTA，lr 1 → 0.1（0.6M iter）→ 0.01（0.8M iter），2× 2080Ti（p.9）。
- 评测 7 个基准（IIIT5K / IC03 / IC13 / IC15 / SVT / SVTP / CUTE80），**全部不用词典**（p.9）。

## 实验结果

### 主结果（p.13–14 Table 4）

| 方法 | 数据 | IIIT5K | SVT | IC03 | IC13 | IC15 | SVTP | CUTE80 |
|---|---|---|---|---|---|---|---|---|
| Aster（基线） | 90K+ST | 93.4 | 89.5 | 94.5 | 91.8 | 76.1 | 78.5 | 79.5 |
| TextSR（ESRGAN-Aster） | 90K+ST | 92.5 | 87.2 | 93.2 | 91.3 | 75.6 | 77.4 | 78.9 |
| Liao -SAM\* | 90K+ST\* | 93.9 | 90.6 | 95.2 | **95.3** | 77.3 | 82.2 | **87.8** |
| Yang\* | 90K+ST\* | 94.4 | 88.9 | 95.0 | 93.9 | 78.7 | 80.8 | 87.5 |
| **PlugNet** | 90K+ST | **94.4** | **92.3** | **95.7** | 95.0 | **82.2** | **84.3** | 85.0 |

- 真正拉开差距的是**两张糊图集**：SVT 92.3（比含 \* 的最佳 Liao-SAM 90.6 高 1.7、比 Aster 高 2.8）、SVTP 84.3（比非 \* 最佳 Zhan 79.6 高 4.7、比 Aster 高 5.8）、IC15 82.2（比 Wan 的 79.4 高 2.8）。这正是论文的靶心，结果与主张自洽。
- **"6 个数据集全 SOTA"（p.14）只在排除字符级标注方法后成立**：含 \* 方法时 IC13 是 95.3 > 95.0、CUTE80 是 87.8 > 85.0；IIIT5K（94.4）与 SVTP（84.3）是打平而非超过。作者用一段话解释了 CUTE80（弯曲文本受益于字符级标注），但没解释 IC13 的 0.3 点——那段"6 个数据集"的措辞偏宽。

### 消融：把 PSU 的功劳重新拆一遍（p.11 Table 3）

Table 3 的箭头是逐行相减，我把关键四行摊平：

| 配置 | SVT | SVTP | IC15 | CUTE80 |
|---|---|---|---|---|
| Aster 基线 | 89.5 | 78.5 | 76.1 | 79.5 |
| +FSM | 90.0 | 80.8 | 78.2 | 82.6 |
| +FSM +FEM | 90.6 | 81.6 | 80.2 | 83.7 |
| +FSM +FEM +退化数据（**无 PSU**） | 89.8 | 82.2 | 79.8 | 81.6 |
| SR-PlugNet（ESRGAN 串联，**无 PSU**） | 90.6 | 80.8 | 79.4 | 82.6 |
| +FSM +FEM +退化数据 +**PSU** | **92.3** | **84.3** | **82.2** | **85.0** |

三处值得单独记：

1. **单看 FSM 就已经很强**：89.5→90.0 / 78.5→80.8 / 76.1→78.2 / 79.5→82.6（CUTE80 +3.1），与 PSU 的量级相当。所以这篇的增益不是 PSU 一个人的功劳，"别把特征压扁"本身就贡献了一半。
2. **裸喂退化数据是有害的**：第三行→第四行，SVT −0.8、IC15 −0.4、CUTE80 **−2.1**。也就是说 Table 3 里 PSU 对比的那一行基线，本身比"用干净数据训练"更差。若改用第三行当基线，PSU 的净增益是 **+1.7 / +2.7 / +2.0 / +1.3**；而按论文用的第四行算是 **+2.5 / +2.1 / +2.4 / +3.4**。两种口径都为正，但**故事完全不一样**：前一种说法下 PSU 是"锦上添花的正则"，后一种说法下 PSU 是"给退化训练擦屁股"。论文正文（p.11）用的是后者，而 Table 3 最后一行的 ↑ 箭头（+1.7/+3.5/+2.8/+2.4）又是相对 **SR-PlugNet 那一行**算的——三种口径混在一张表里。
3. **图像级 SR 真的没用**：SR-PlugNet（ESRGAN + PlugNet 识别）对第四行是 +0.8 / −1.4 / −0.4 / +1.0，与 TextSR 的结论一致，也和 30 天/epoch 的成本形成对照。

### 分辨率消融（p.10 Table 2，仅 90K、无 PSU/FEM）

| 分辨率 | SVT | SVTP | IIIT5K | IC03 | IC13 | IC15 | CUTE80 |
|---|---|---|---|---|---|---|---|
| 1×25 | 85.2 | 76.1 | 80.7 | 91.8 | 89.3 | 69.3 | 66.3 |
| 2×25 | 87.0 | 78.1 | 82.7 | 92.3 | 89.4 | 69.3 | 68.4 |
| 4×25 | 87.9 | 79.5 | 82.2 | 92.7 | 89.8 | 71.4 | 69.1 |
| 8×25 | **89.0** | **82.0** | **85.3** | **94.3** | **91.0** | **73.6** | 69.1 |

整体单调上涨，只有两处例外：IIIT5K 在 4×25 掉 0.5、CUTE80 到 69.1 饱和。作者的解释是分辨率高时控制点定位更准（Fig.3）——CUTE80 是弯曲文本，靠的是矫正质量而不是像素量，饱和也说得通。

### λ：0.01 是峰，0.5 会崩（p.12–13 Fig.5）

从 Fig.5 读出的四条曲线（λ=0.01 一格与 Table 3/4 完全对得上，无 PSU 一格与 Table 3 第四行完全对得上，可互证）：

| λ | SVT | SVTP | IC15 | CUTE80 |
|---|---|---|---|---|
| 无 PSU | 89.8 | 82.2 | 79.8 | 81.6 |
| 0.0001 | 90.0 | 81.7 | 80.4 | 81.9 |
| 0.001 | 91.5 | 83.1 | 81.6 | 83.3 |
| **0.01** | **92.3** | **84.3** | **82.2** | **85.0** |
| 0.1 | 91.0 | 82.2 | 81.0 | 78.8 |
| 0.5 | 88.0 | 76.0 | 73.4 | 68.8 |

两点观察（读图，可能有一格误差）：① SVTP 在 λ=0.0001 时（81.7）**低于**无 PSU（82.2），"λ 越小越单调"这句话在 SVTP 上不严格成立；② λ 一旦超过 0.01 就全线下滑，0.5 时 CUTE80 掉到 68.8（−12.8），说明这一项辅助损失**极不宽容**——多任务权重没调好会直接反噬。

作者的解释（p.13）：λ 增大带来两种相反效应——特征更少噪声/模糊（利于识别），但骨干被拉去"重建低层图像"（伤识别）。Fig.6 用通道平均图展示了这件事：λ 越大，CNN out 的特征图越干净、SR out 越锐，但识别掉点。

## 结论与局限

**作者承认的**：CUTE80 上不如用字符级标注训练矫正网络的方法（p.14）；λ 敏感（p.13）；SR-PlugNet"如果用更好的训练策略/参数/数据可能更高"（p.11）。

**我观察到的**：

- **一个内部数字矛盾**：IC15 在 Table 3 里是 **82.2**，但同一行的 ↑ 箭头写 **↑2.8**（79.8+2.8=82.6），正文（p.11）也写 **82.6%**，Table 4 又是 **82.2**。三处口径不一致，无法判断真实值；从"箭头与正文一致"看，表格格子更像是笔误。
- **"零额外计算"是局部真话**：PSU 确实被删掉了，但 FSM/FEM 让推理 20→22 ms、训练 0.63→0.97 s/batch（p.9）。对工业落地这 10% 可接受，但摘要的措辞容易让人以为是严格零开销。
- **30 天/epoch 这个论据没有闭环**：既然作者自己训了 SR-PlugNet（ESRGAN 版），完全可以报它的实际耗时做对照，却没有；只留一句"30 days each epoch"（p.2）。
- **"degradation aware" 名不副实**：退化模型只有高斯模糊 + 降升采样两件套，没有噪声/压缩/运动模糊/光照；而 SVT/SVTP 的真实退化远比这复杂。α=0.5 意味着训练时一半样本是干净图，与推理时"图都糊"的分布也不一致（没有做 domain 分析）。
- **PSU 的结构没有消融**：只测了 λ，没测"把 RCAN 换成普通反卷积解码器会怎样"、"从低层特征而不是高层特征重建会怎样"。所以"有效的是额外监督信号"还是"RCAN 的通道注意力先验"，论文无法回答。这是我认为最该补的一项。
- **评测面窄且无方差**：七个集全是英文词、全无词典；SVT 只有 647 张，92.3 对次优 90.6 的 1.7 点领先没有置信区间或多次运行。弯曲文本 CUTE80 只有 69.1（Table 2 口径）/85.0（最终），仍是弱项。
- **未在 TextZoom 上评测**（同年 ECCV'20 才发布，情有可原），但这也意味着它的"低质量"主张从未在**真实成对**数据上被验证过——后来的 KD-LTR 把它搬到 TextZoom 上，平均只有 68.60%（见下）。

## 我的疑问

- [ ] **Eq.3 的近似到底亏多少**？把 $f_{blur}$ 直接作用在矫正后的张量上（实现上只是算子顺序）就能得到 Eq.2 的真值对，为什么不做这个对照？如果两者差很多，说明 PSU 学到的东西里混了"矫正头抗模糊"的成分；如果差不多，那 "approximate" 这个措辞就没必要。
- [ ] **PSU 到底在教谁**？Eq.3 里 $f_{rn}$ 出现在两项，SR 损失同时回传到定位网络。做一个"PSU 损失只回传到骨干、不回传到矫正头"的对照，就能把"抗退化"与"放准控制点"分开。
- [ ] **为什么从高层特征重建**？$8\times25\times512$ 只剩 25 列、每列 8 个位置——这个分辨率恰好"一个 token 一列字符"。如果换成从 $1\times25$ 重建（Aster 原配置），SR 任务会退化成"按列猜整幅图"，还能涨点吗？这直接决定 FSM 是"PSU 的前提"还是"独立的收益"。
- [ ] **λ 的双效应只有定性证据**（Fig.6 通道平均图）。若做梯度范数/特征频谱的统计，"重建 vs 识别"的拉扯会更有说服力；顺带还能解释为什么 SVTP 在 λ 极小时反而低于无 PSU。
- [ ] **换个重建目标会怎样**？现在的目标是"干净矫正图"的像素。如果把目标换成 HR 图的**识别器特征**（即 KD-LTR 那条路），或换成梯度/边缘图（STT 的 $L_{GP}$ 思路），是否比像素更省、更准？这是这篇与后续工作最自然的分叉点。
- [ ] **推理时保留 PSU 会怎样**？理论上 SR 图能帮识别，但误差会级联——不做这个实验，"删掉 PSU"就只能算设计选择，不能算被验证的结论。

## 关键摘录

> "we train the ESRGAN-Aster with the Synth90K dataset [11] with a single NVIDIA 2080Ti, it needs nearly 30 days each epoch."（p.2 §1）

> "Due to the loss of spatial acuity makes it difficult for both the recognition part and the rectified part to get effective learning."（p.3 §1）

> "we removed the down-sampling layers in last three CNN blocks so that the feature maps after the backbone have the dimension as W/4 × H/4 × C"（p.5 §3.1）

> "we use two Residual Channel Attention Block(RCAB) to construct each Residual Group(RG). Then, two RGs are used to build the final PSU. After training, the PSU will be removed which means no extra computation in the inference stage."（p.6 §3.2）

> "the data generation strategies should take effect after the Rectification Network which means the input images are high-resolution."（p.8 §3.4，解释为什么 Eq.2 要改成 Eq.3）

> "the increase of λ, let the sharing CNN backbone focus more on low-level images to rebuild the super-resolution images which makes a negative impact on the text recognition."（p.13 §4.4）

> "our PlugNet outperforms all the previous state-of-the-art performance in 6 datasets and achieves competitive accuracy to the state-of-the-art techniques in the remain CUTE80 datasets."（p.14 §4.5）

## 追问补编：逐段精读（伴读问答整理）

> 这一节是把阅读器里选中段落逐段追问的结果整理成速查，页码仍指本机 PDF 页码。与上文有少量重复，但按"问题→答案"组织，便于回看。

### Q1. PSU（可插拔超分单元）的组成、原理、功能？（选中于 p.5，Fig.2(c) 图注）

**组成**（p.6 §3.2 原文）："we use two Residual Channel Attention Block (RCAB) to construct each Residual Group (RG). Then, two RGs are used to build the final PSU."

```
骨干末端高层特征 8×25×512
  → 3×3 Conv（通道映射）
  → RG ×2（每个 RG = 2 个 RCAB）
       RCAB：Conv→ReLU→Conv 的主路 + 通道注意力支路
             通道注意力 = 全局池化 → Conv→ReLU→Conv → Sigmoid → 逐通道加权
             外层残差相加
  → Conv → Upsampling → 输出 32×100 干净矫正图
```

Fig.2(c) 上排那串 Conv/ReLU/Global Pooling/Conv/ReLU/Conv/Sigmoid 就是 RCAB 的 CA 支路（p.5）。4× 上采样是从 8×25 → 32×100 反推的，论文没写通道数、卷积核细节和上采样方式（pixel shuffle 还是转置卷积）——别当成原文数字。

**原理**：常规 SR 是"低清图→高清图"，PSU 是"**语义特征→干净图**"（p.6 原话 "build the super-resolution images from the high-level features"）。要从一个只有 25 列、每列 8 行的特征重建 32×100 的图，特征里必须留着字符位置、笔画宽度、边缘——**重建不出来就说明信息丢了**，于是 L1 重建损失反过来把空间信息逼回特征。RCAN 的通道注意力在这里做的是"挑出还有用的频率/笔画成分再拼回去"。GT 就是矫正网络的输出本身（p.8 式 3），不需额外标注。

**功能**：训练期作为多任务头逼骨干抗退化（p.6 "help the sharing CNN backbone better represent the features of degradation images"）；推理期整个删除，零额外模块。严格说它是"**去退化重建**"而非放大，配 L1 而非感知/GAN 损失——像素是手段，逼特征才是目的。

### Q2. 识别部分那段在说什么？怎么把 (B,C,H,W) 变成 (B,L,C)？（选中于 p.5 §3.1 Recognition Part）

**段落解读**：① 沿用 ESIR/Aster 的 CNN+LSTM 路线而非 CTC（"在解决完整序列方面有优势"）；② LSTM 要一维向量，所以中间必须有 FSM（Fig.2(d)）；③ 编码器 = 2 层 BiLSTM、每层 256 hidden，输出 Hs 与输入 Vs **等长**；④ 解码器 = 2 层注意力 LSTM，自回归吐字符；⑤ 那句 Vs 形状 W×(H×C) 是"通道在后"写法：**序列长度 = W（字符排列方向），每步维度 = H×C**。本文具体是 25×(8×128)=25×1024（8 是特征高、128 是 FSM 的 1×1 输出通道）。

**核心：选谁当 L**。文本水平排列，所以序列轴必须是 W；H 与 C 才是"这一步的特征"。正确路径只有一条：

```python
def fsm(x, conv1x1):                    # x: (B,C,H,W)
    x = conv1x1(x)                      # (B,992,8,25) → (B,128,8,25)
    B,C,H,W = x.shape
    x = x.permute(0,3,2,1)              # (B,25,8,128) ← W 提到序列维
    return x.reshape(B,W,H*C)           # (B,25,1024)  ← H 折进通道
```

两种等价写法 `x.permute(0,3,2,1).flatten(2)` 与 `x.permute(0,3,1,2).flatten(2)` 只差通道置换，后续 LSTM 输入权重可学，能吸收这个置换。

**三种做法对比**：
| 做法 | 结果 | 评价 |
|---|---|---|
| 直接 flatten | (B,H·W,C)，L=200 | ❌ 行也被当序列，字符顺序打散 |
| 池化掉高度（Aster 一路） | (B,W,C) | ⭕ 能用但丢空间 |
| **折进通道（本文）** | (B,W,H·C') | ✅ 保空间又保长度 |

**控制变量**（p.10）：Table 2 把 1×1 通道数反着调（1024/512/256/128），让输出恒为 25×1024——四种分辨率下 BiLSTM 输入维度一致、参数近似不变，所以涨点不是参数变多。

**验证技巧**：用坐标追踪确认元素没跑错位，别只打印 shape：

```python
x = torch.arange(1*3*4*5).float().reshape(1,3,4,5)
y = x.permute(0,3,2,1).reshape(1,5,12)
assert y[0,3,4*3+1] == x[0,1,4,3]
```

### Q3. "注意力 BLSTM"是什么？（p.5–6）

不是单个模块名，是**编码器+解码器**这一对的俗称：Fig.2(e) 的 BiLstm-Encoder → Attentional LSTM-Decoder。

- **BiLSTM 编码器**：普通 LSTM 只看过去，BiLSTM 正反向各扫一遍、每步拼接两个方向的隐状态。输入 25 步×1024 维 → 输出仍 25 步，但每步揉进了左右上下文。**为什么必须双向**：识别第 3 个字符时歧义可能来自右侧（I vs l、连写 rn vs m），只看左边不够。
- **注意力解码器**：生成第 t 个字符时，拿上一步状态 s(t-1) 与编码器 25 个位置逐一算相似度 → softmax 成权重 alpha(t) → 加权求和 glimpse = Σ alpha(t,i)·h(i)（软对齐）→ 预测字符。自回归直到 EOS（p.6 明确序列末尾补 EOS）。**它让解码器不必死板地"第 t 步读第 t 列"**，这是相对 CTC 的核心优势（CTC 要求单调对齐）。

**诚实点**：这篇**没有给注意力公式**，只画了图、写了"带注意力的 seq2seq"。上面 5 步是同族 attentional decoder 的通用结构（见 [[ASTER 注意力场景文本识别器]] 的加性注意力逐式推导），属合理推断，非原文逐条写出。

### Q4. FSM 那段逐句解释（选中于 p.7 §3.3）

矛盾背景（p.3 讲得更全）：LSTM 要一维向量，传统做法用下采样卷积把高度压到 1（8×25×512→1×25×512），代价是丢掉 spatial acuity——旋转、位移、笔画上下结构，作者说这 "makes it difficult for both the recognition part and the rectified part to get effective learning"。FSM 的方案是**不下采样，换种方式把二维变一维**：

1. **1×1 卷积**只压通道（992→128），不动 H/W → 空间分辨率原样保留；
2. **reshape**把高度 8 折进通道 → (B,25,1024)：25 步不变，每步从 512 维变 1024 维，**那 8 行信息没丢，只是搬到了通道维**。

"只增加少量额外计算"的潜台词：1×1 卷积参数量 992×128、reshape 零参数，比"再用三层 stride=2 卷积压高度"还便宜。

最后一句才是重点——**FSM 是 PSU 能挂上去的前提**：PSU 要从高层特征重建 32×100 的图，你不可能从已被压成 1×25 的特征里重建出空间细节（p.5 "Benefit from the FSM ... It helps the PSU"）。所以 FSM 一举两得：识别更准（p.11 Table 3：+FSM 单点 SVT +0.5 / SVTP +2.3 / CUTE80 +3.1），且为 PSU 提供必要条件。

### Q5. PSU 的作用 & 为什么推理时删掉？（p.2/6/11/13）

**作用**：训练期的"特征教练"——用重建辅助任务逼共享主干把空间/高频信息留在特征里；训完即删。见 Q1。

**为什么删**，三层原因：
1. **价值已内化**：PSU 逼出来的能力固化在骨干权重里了；教练练完不必背上赛场。且 SR 图**从不在识别链路上**（Fig.2 里是旁支），留着跑纯属浪费。
2. **图像级 SR 路线已被自己否掉**：p.2 批 TextSR"视觉变好但识别提升有限 + 单卡 30 days/epoch"；p.11 自训的 SR-PlugNet（ESRGAN 串联）对识别几乎无益甚至负（SVT +0.8 / SVTP −1.4 / IC15 −0.4）。推理时留 PSU 等于把自己批评的东西捡回来。
3. **重建目标与识别目标冲突**：p.13 λ 实验显示两种相反效应——特征更少噪声（利识别）vs 骨干被拉去重建低层图像（伤识别）；λ=0.5 时 CUTE80 从 85.0 崩到 68.8。删掉是彻底切断冲突源。

"pluggable"是设计承诺（p.1 摘要/p.3 贡献）：训练期付代价、推理期零负担，这是相对超分流水线的核心卖点。**诚实点**：零开销只对 PSU 成立，FSM/FEM 留在推理链里，整网 22ms vs Aster 20ms（p.9）；且论文**没做"推理时保留 PSU"的对照**——删掉是基于设计哲学与成本的选择，不是被实验证伪的最优解。

### Q6. 损失函数逐项解析（p.8 式 4–6）

$$L=L_{rec}+\lambda L_{sr},\quad\lambda=0.01$$
$$L_{rec}=-\frac{1}{MN}\sum_{i=1}^{M}\sum_{j=1}^{N}y_{i,j}\log s_{i,j},\qquad L_{sr}=\frac{1}{WH}\sum_{i=1}^{W}\sum_{j=1}^{H}\lVert O_{i,j}-I_{i,j}\rVert_1$$

- **Lrec**：交叉熵（识别本质是每步一个字符的分类）。M=batch 样本数、N=标签字符长，y 是 one-hot 真值、s 是 softmax 概率。双重求和 = 对所有样本、所有字符位取平均。**没有对齐项**——注意力 seq2seq 自己学对齐，不像 CTC 要显式求和所有路径。
- **Lsr**：逐像素 L1（MAE）。O 是 PSU 重建输出，I 是真值图（= f_rn(I)，干净图的矫正结果，见 Q9）。用 L1 不用 L2，原文理由是 "L1 loss provides better convergence than L2 loss"（引 EDSR 经验）——L2 对大误差平方惩罚易被 outlier 带偏。
- **lambda=0.01**："balance the weight in two different tasks and keep the recognition performance"（p.8）——辅助任务只占 1% 话语权。梯度量级不同，1:1 相加会让重建盖过识别。这个值是实验甜点（p.12–13 Fig.5）：lambda<0.01 随权重上升而涨，之后全线下滑，0.5 时崩盘。
- **训练谁**：Lsr 的梯度不止到 PSU，还顺主干回传（甚至到矫正网络，见 Q7）；Lrec 只经识别头回传。两者在**共享主干处汇合**——主干被迫同时满足"识别准"和"能重建"，后者就是正则。

### Q7. 为什么 SR 真值是矫正网络的输出？矫正不是管形状的吗？（p.5、p.7–8）

**第一层：矫正不只是管形状**。Aster 式 TPS = 定位网络（预测 n 控制点）→ 网格生成器（解变换）→ **采样器（双线性插值重采样）**。它重新计算了每个输出像素的位置和取值——p.7 原话 "the Rectification Network ... will change distribution of each pixel which makes a huge difference between the output image and input image"。输出 f_rn(I) 是一张全新的规整条带（32×100），不再是原坐标的弯曲文字。

**第二层：逐像素损失要求坐标系一致**。O 是从"矫正图上的特征"重建的，处在矫正后坐标系；若拿原图 I（弯曲、原坐标）当真值，第 (i,j) 像素根本不对应同一个字符位置，L1 失去意义。所以真值只能是 f_rn(I)。p.13 说"可视化时去掉矫正网络才好比较"也是同一原因。

**第三层：退化位置的近似**（p.7–8）：
- 式 2（理想）：Lsr = f_loss( f_rn(I), f_blur(f_rn(I)) ) —— 先矫正、后退化。问题：退化发生在矫正之后意味着网络拿到"已矫正的图"，**矫正网络被绕出链路**。
- 式 3（近似）：Lsr = f_loss( f_rn(I), f_rn(f_blur(I)) ) —— 退化挪到矫正之前，两边都过矫正。坐标系一致 ✓、链路不断 ✓。代价：TPS 控制点由内容预测，f_rn(f_blur(I)) ≠ f_blur(f_rn(I))。

**SR 与矫正的真实关系（被式 3 绑定）**：f_rn 出现在损失两项中，所以 Lsr 的梯度**回传到矫正网络**——它被逼着"在模糊输入下也放准控制点"，即作者说的 "learn about solving low-quality images"（p.8）。矫正管"摆正位置"、PSU 管"补足细节"，但共用坐标、共用梯度，式 3 让两者一起学会抗模糊。**诚实点**：论文没做"SR 损失只回传骨干、不回传矫正头"的消融，两者各占多少无法回答。

### Q8. 矫正网络的结果会输入共享主干吗？（p.5–6）

**会，严格串行**。p.5 图注四段顺序："rectification network, sharing CNN backbone, recognition network and pluggable super-resolution unit"；正文：采样器 "generate the rectified images"，紧接着 "Sharing CNN Backbone ... extract the feature maps"。Table 1（p.6）Block 0 输出 32×100，正是矫正图尺寸。完整链路：

```
输入图 I → 矫正网络（Localize→Grid Generator(TPS)→Sampler）→ 矫正图 Ir (32×100)
  → 共享 CNN 主干（Block 0–5，8×25×512）
       ├─ 识别支路：FEM→FSM→BiLSTM→注意力解码 → 字符序列
       └─ PSU 支路：RCAN → SR 图（仅训练期）
```

"sharing"的含义：同一份主干特征被两条路共用。由此推出：PSU 的重建目标必然是矫正图（Q7）、退化也要绕道矫正（Q9）、矫正质量直接决定后续一切（Fig.3/p.10：分辨率高→控制点准→特征好→重建好，形成闭环）。

### Q9. PSU 重建的 SR 图，要和 LR 的矫正图算损失？（p.7–8）

**纠正：真值不是"LR 的矫正图"，是"干净原图的矫正图" f_rn(I)**。符号表：

| 符号 | 是什么 | 角色 |
|---|---|---|
| I | 原始清晰图 | — |
| f_blur(I) | 人为造退化（LR） | — |
| f_rn(I) | **清晰图**经矫正 | **真值** ✅ |
| f_rn(f_blur(I)) | **退化图**经矫正 | PSU 的输入来源（主干吃它） |
| O | PSU 重建输出 | 预测值，逼近 f_rn(I) |

式 6 原文说 I 是 "the ground truth of input image"，结合式 3 即 f_rn(I)。式 3 第二项是"SR 的输入从哪来"，不是"要拟合成它"。**反证**：若真值是 LR 矫正图，PSU 输入（同图特征）与输出都是糊的，L1 一开始就近 0，辅助任务直接空转。所以真值只能是干净矫正图——让糊的输入逼近干净的目标，重建监督才有意义。

### Q10. PSU 和 OCR 是两条不同的支路吗？（p.5–7）

**结构上：是**（共享主干末端的两个并列任务头）；**数据流上：PSU 输出不进识别支路**（这是推理期能删的前提）；**训练上：不是独立网络**——共享主干权重与梯度，Lrec 与 Lsr 在骨干处汇合，这才是多任务学习的实质。

精确说法：

```
矫正图 → 共享主干 8×25×512
   ├─ 支路 A（识别头/OCR）：FEM(992) → FSM(25×1024) → BiLSTM → 注意力解码 → 字符
   └─ 支路 B（PSU）：8×25×512 → RCAN → 32×100 SR 图【训练期专属】
```

识别支路严格叫 recognition network / recognition head（OCR 是更宽泛的说法）；FSM 算识别支路的"入口"，同时也是 PSU 得以挂上的前提（保住 8×25）。比喻：一个人（主干）同时练两门手艺（识别、重建），练习分开，但长在同一份能力上——练重建让观察力变强，反过来帮识别。

## 相关笔记

- [[ASTER 注意力场景文本识别器]] — 本文的直接基线：矫正网络（TPS + 20 控制点）、骨干、注意力解码器全部沿用，改动集中在"最后三个 block 不下采样 + FSM/FEM + PSU"。对照着读能看清 PlugNet 只动了**特征形状**和**训练目标**两处。
- [[KD-LTR 用高分辨率知识蒸馏做单阶段低分辨率文本识别]] — **同构且更彻底**：都是"训练期挂教师、推理期扔掉"的单阶段思路；区别是 PlugNet 的"教师"是输入图自身（自监督重建，只需合成退化），KD-LTR 的教师是冻结的 HR 识别器（跨模型蒸馏，需要 HR 教师）。而 KD-LTR 批"只在像素空间学"的靶子，正是 PlugNet 这一类多任务方法；其 Table 7 把 PlugNet 列为对照，TextZoom 平均 **68.60%**，低于 IFR 69.04 与 KD-LTR 的 72.45——这是对 PlugNet 路线最直接的后验评价。
- [[STIRER 统一框架的低分辨率场景文本恢复与识别]] — 在相关工作里点名批评 PlugNet："共享 CNN 主干提特征再做 STR + STISR，但特征里可能自带错误信息"。两者都反对"串行 SR"，但 STIRER 选择"共享编码器 + 双解码器、识别不读 SR 图"，PlugNet 选择"SR 只当训练信号"。
- [[TextZoom 与 TSRN 真实场景文本图像超分]] — 同年 ECCV'20 的 STISR 开山作，提供了真实成对评测；PlugNet 没赶上，也就从没在真实 LR-HR 上被验证。
- [[STT 文本聚焦的场景文本超分]] — **方向相反的一对**：STT 是"用识别器的注意力去监督 SR"，PlugNet 是"用 SR 的重建去监督识别特征"。同一枚硬币的两面，值得并排记。
- [[C3-STISR 三重线索引导的场景文本超分]] — 也是"拿识别线索引导 SR"，但把线索扩到识别/视觉/语言三条；与 PlugNet 的"SR 引导识别"构成互逆的知识流向。
- [[LRSTRN 恢复与识别迭代协同增强的低分辨率场景文本识别]] — 把"SR ↔ 识别互相注入"做成多阶段迭代，可看作 PlugNet 那种"单向、只在训练期"耦合的后继升级版。
- [[SVTR 场景文本识别]] — 后续 backbone 演化：同样拒绝把高度压成 1，改走 patch embedding + merging 的 2D 特征路线，与本文 Table 2 的结论一脉相承。
