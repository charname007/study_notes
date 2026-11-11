---
created: 2026-11-11
source: Zhao, Xuyang, Guan, Zhou - STIRER: A Unified Model for Low-Resolution Scene Text Image Recovery and Recognition（复旦 + 同济；本机 PDF：E:\.dsh-paper-reader\data\SR\STIRER\STIRER.pdf；代码 https://github.com/zhaominyiz/STIRER）
type: 论文
tags: [论文, 场景文本识别, 超分辨率, Transformer]
---

# STIRER：统一框架下的低分辨率场景文本恢复与识别

> 用户选中标题（第 1 页）："STIRER: A Unified Model for Low-Resolution Scene Text Image Recovery and Recognition"。作者 Minyi Zhao、Shijie Xuyang、Jihong Guan、Shuigeng Zhou（通讯），复旦大学计算机学院 / 同济大学计算机科学与技术系（p.1）。ACM MM '23（Ottawa），DOI 10.1145/3581783.3612488（p.1）。本机 PDF 共 10 页，页码均指 PDF 页码：p.1 摘要/引言/两处硬伤，p.2 Fig.1/贡献/相关工作 2.1/方法总览 3.1，p.3 相关工作 2.2–2.3/模型 3.1–3.2，p.4 Fig.2/3.3–3.5/算法 1，p.5 4.1 数据集/4.2 实现细节/4.3.1 讨论，p.6 Table 1–2/Fig.3，p.7 Fig.4/4.4 可视化/4.5 消融/Table 3–4，p.8 Fig.5–6/Table 5/结论，p.9–10 参考文献。基金：NSFC U1936205（p.8 致谢）。

## 一句话总结

用一个**共享的「渐进式场景文本 Swin Transformer」编码器 + 两个各自独立优化的解码器**，同时做超分恢复与文本识别；最关键的取舍是**让 STR 解码器直接读 LR 图（而不是读可能出错的 SR 图）**，靠编码器给的文本特征补信息，从而打破 STISR 里「识别精度 vs 图像保真」的零和博弈（p.2、p.3）。TextZoom 平均 **74.8%** / PSNR **22.46 dB**，只需 **0.74G FLOPs / 4.03M 参数**（p.6 Table 1）。

## 核心问题

低分辨率（LR）场景文本识别的两条既有路线各有硬伤（p.1 引言、p.2 Fig.1）：

1. **直接识别 LR 图**（多数早期 STR）：像素信息不足 → 误读。文中实测 CRNN 只有 27.3%（p.6 Table 1 第 2 行）。
2. **先 STISR 超分再识别**（TSRN / TG / TPGSR / C3-STISR 等）：
   - **SR 图本身可能就错**：Fig.1(b) 把 "rn" 恢复成 "m"，直接误导后续识别（p.1）；
   - **保真度被牺牲**：STISR 与 STR 联合优化时，为了识别精度会改字体颜色、拉对比度去「凸显文字」，与 HR 风格不一致，保真度下降，反过来又伤识别（p.1）。

结果是**两条路线都不可取**：识别精度和 SR 保真度至少丢一个。作者要的是两者兼得、且要算得起（p.1）。

作者还把「已有人也把两件事一起做」的路线做了切割（p.3 §2.3）：PlugNet 用共享 CNN 主干提特征再做 STR + STISR，但特征里可能自带错误信息；IFR 迭代地识别与恢复，但只用文本信息助超分、识别模块本身没被强化。

## 关键方法

### 总览：一个编码器 + 两个解码器（p.3 §3.1、p.4 Fig.2）

输入 LR 图 $I_{LR}\in\mathbb{R}^{C\times N}$（$N=H\times W$），目标是输出 SR 图 $I_{SR}\in\mathbb{R}^{C\times 4N}$（每边 2×，像素 4×），**同时**读出 LR 图上的文字（p.3 §3.1）：

$$h_L=\phi(I_{LR}),\qquad I^{L+1}_{SR}=\Phi_{SR}(I_{LR}, h_L),\qquad p_L=\Phi_{Rec}(I_{LR}, h_L)$$

三个组件：编码器 $\phi$（PSTST）、STISR 解码器 $\Phi_{SR}$、STR 解码器 $\Phi_{Rec}$。

两个解码器都以**原始 LR 图**与**编码特征 $h_L$** 为条件——这是全文最关键的一处设计：**STR 分支的输入里没有 SR 图**，所以 SR 出错不会通过串行通路污染识别；反过来 SR 也不必为「让识别器看懂」而扭曲像素（p.3 §2.3、p.4 式 4–5）。

![[STIRER 架构图 Fig2.png]]

> **怎么读这张图**（Fig.2，p.4）：紫色大虚线框是编码器 PSTST $\phi$（Conv layer + $L$ 个 Basic PSTST block）；蓝色细虚线框是第 $i$ 个 basic block 的内部——$N_i$ 个 STL 层 → Bi-GRU → 分叉出 Mini STISR / Mini STR 两个小解码器。**黑线是推理路径**：左边 LR image → Conv → blocks 得到 $h_L$，右侧 STISR decoder 吃 LR image + $h_L$ 出 SR image，**STR decoder 吃的是 LR image（右下那个小图）+ $h_L$，没有 SR image**——这就是全文的立身之处，看图时务必顺着右侧两条黑线各走一遍。**红线只在训练时生效**：顶部把 $I^1_{SR}, I^i_{SR}, I^L_{SR}, I^{L+1}_{SR}$ 汇成 $\mathcal{L}_{SR}$，底部把 $p_1, p_i, p_L, p_{L+1}$ 汇成 $\mathcal{L}_{Rec}$，一眼能看出「每个 block 都被监督」。

### PSTST 编码器：局部窗口 + 全局上下文 + 逐块监督（p.3 §3.2、p.4 Fig.2）

动机两条（p.3）：既要**同时**拿到局部与全局上下文（估计模糊细节需要），又要拿到**富含文本信息的特征**。作者批评了两条既有做法：TBSRN 用全像素多头自注意力，抓得全但跑得慢；SwinIR 用移位窗口注意力，快但抓不住全局上下文，堆层也不划算。

PSTST = 1 个卷积层 + $L$ 个 basic block（p.3）：

- 卷积层出初始特征 $h_0\in\mathbb{R}^{C'\times H\times W}$；
- 每个 basic block $\phi_i$：先 $N_i$ 个 **Swin Transformer 层（STL）**做窗口大小 $[H, W/32]$ 的移位窗口注意力（抓相邻像素），再进 **Bi-GRU** 学上下文，得到 $h_i$；
- 每个 block 额外挂**两个 mini 解码器**（mini STISR = 几层简单卷积；mini STR = 6 层 Transformer encoder + 全连接），产出中间 SR 图 $I^i_{SR}$ 与中间概率分布 $p_i$，用来**直接监督 $h_i$**。

于是训练时共有 $L+1$ 张 SR 图 $\{I^1_{SR},\dots,I^{L+1}_{SR}\}$ 与 $L+1$ 个分布 $\{p_1,\dots,p_{L+1}\}$（p.3）。这就是这里「progressive」的实际含义：**深度监督 / 逐块监督**，而不是 [[渐进式注意力]] 那种「高层注意力指导低层」的渐进注意力——两者只是中文都译作「渐进」，机制无关，别混。

### STISR 解码器（p.4 §3.3）

$$\text{concat}(I_{LR}, h_L)\ \text{(通道维)} \rightarrow \text{3 个 SRB 块 [45]} \rightarrow I^{L+1}_{SR}$$

非常轻：仅 0.13G FLOPs / 0.13M 参数（p.8 Table 5）。SRB 块与 $L_{GP}$ 梯度轮廓损失同源，都借自 [[TextZoom 与 TSRN 真实场景文本图像超分]]。

### STR 解码器（p.4 §3.4）

关键反常规点：**坚决不用下采样层**。典型 STR（CRNN、SVTR）靠下采样压高度，但 LR 图像素本就稀缺，再下采样等于把仅有的信息扔掉（p.4 原话见摘录）。因此改为：

- 把输入图与编码特征裁成 $[H, W/32]$ 的 patch；
- 用 **6 层 encoder + 6 层 decoder 的 Transformer** 自回归建模
  $$p_L(y\mid[I_{LR},h_L])=\prod_{t=1}^{T}p_L(y_t\mid y_{<t},[I_{LR},h_L])$$

即「像素级 patch 全保留 + 自注意力 + 自回归解码」。与 [[SVTR 场景文本识别]] 的分野正在这里：SVTR 是单一视觉模型（无自回归解码器）且靠 merging 压高度。

### 损失与训练（p.4 §3.5–3.6、算法 1）

$$\mathcal{L}_{SR}=\sum_i i\cdot L_{GP}(I^i_{SR}, I_{HR}),\qquad \mathcal{L}_{Rec}=\sum_i i\cdot L_{CTC}(p_i,y)+(L+1)\cdot L_{CE}(p_{L+1},y)$$

$$\mathcal{L}=\mathcal{L}_{SR}+\mathcal{L}_{Rec}$$

- 系数 $i$ 是**渐进因子**：越深的块权重越大、最终输出 $I^{L+1}_{SR}$ 权重最大（$L+1$），把监督压向深层（p.4）；
- 识别侧两种损失分工：**CTC 监督中间层像素特征** $\{h_1,\dots,h_L\}$，**交叉熵监督最终 STR 解码器**（p.4）；
- 训练（算法 1，p.4）：前向跑完 $L$ 个 block + 两个 final 解码器，再统一算损失；
- **推理**：直接丢掉中间结果（算法 1 第 5–6 行），只跑编码器 + 两个 final 解码器（p.5）。

> 代码：官方仓库 https://github.com/zhaominyiz/STIRER 由作者在摘要给出（p.1），但本机未克隆、未读源码，故此处不附代码片段（避免臆造）。解码器结构可对照 [[SVTR 场景文本识别]] 的视觉编码器与 [[TextZoom 与 TSRN 真实场景文本图像超分]] 的 SRB / $L_{GP}$ 实现来读。

## 结论与局限

主结果（TextZoom，p.6 Table 1）：

- **识别**：Easy 87.9 / Medium 74.2 / Hard 59.6 / **Average 74.8**，比当时 SOTA 的 IFR（69.6）高 **+5.2**（p.5 正文、p.6 第 20/23 行）；比同组前作 C3-STISR+ABINet（67.7）高 +7.1。为公平起见另做「STR 解码器换成 CRNN」的变体，平均 **68.6**，比 C3-STISR+CRNN（53.7）高 **+14.9**（p.5）。
- **保真**：PSNR **22.46**（LR 20.35、最强 STISR 基线 C3-STISR 21.51，即 +0.95 dB）；SSIM **0.7706**。
- **效率才是真优势**：0.74G FLOPs / 4.03M 参数，对比 C3-STISR+ABINet 的 10.92G / 123.02M——**算力约 1/15、参数约 1/30**（p.5 正文 "about 15 times lower"、p.6 Table 1）。Fig.3 的 accuracy–FLOPs 图里 STIRER 落在左上角（p.6）。
- **泛化**（p.6 Table 2）：IC15-352 83.0 / SVT 67.4 / IIIT 65.8 / IC15 72.3 / SVTP 55.0，五项全面高于 SVTR、ABINet 及其 C3-STISR 组合。值得记的反例是 **C3-STISR 在 SVT 上把 SVTR 从 49.8 拉低到 47.4、在 SVTP 上从 39.4 拉到 37.5**——作者以此论证「STISR 生成的坏图会反噬识别」（p.6）。

消融（p.7 Table 3，TextZoom）：

| 变体 | Accuracy | PSNR |
|---|---|---|
| STIRER | 74.8% | 22.46 |
| 去掉整个 PSTST | 67.5%（−7.3） | 21.76（−0.70） |
| 去掉 PSTST 里的 mini 解码器 | 74.1%（−0.7） | 22.34（−0.12） |
| 去掉 STL 层 | 72.7%（−2.1） | 22.13（−0.33） |
| 去掉 Bi-GRU | 70.6%（−4.2） | 22.00（−0.46） |
| 只留 STISR 解码器 | 73.1%（−1.7） | — |
| 只留 STR 解码器 | — | 22.14（−0.32） |

贡献排序很清楚：**整个 PSTST 最关键（−7.3）**，其中 **Bi-GRU 比 STL 更要紧（−4.2 vs −2.1）**，而 mini 解码器的逐块监督只值 −0.7。两个解码器缺一都掉点，说明「同一框架里协同做」确实优于两个独立模型（p.8 §4.5.2，原话见摘录）。

模块开销（p.8 Table 5）：编码器 0.52G / 0.52M、STISR 解码器 0.13G / 0.13M、STR 解码器 0.09G / 3.38M。**参数几乎全堆在 STR 解码器（3.38M，占 84%），算力却有 70% 花在只有 0.52M 参数的编码器上**。

局限（作者承认 + 我观察到的）：

- **作者自认**：像素信息极端不足时恢复与识别会一起失败（p.7 Fig.4 Case 8），原文称 "the recovery and recognition of LR images is still a challenging task"（p.7）。
- **SSIM 其实不是最优**：正文只拿 PSNR 说「保真最好」，但 SSIM 0.7706 低于 TPGSR 的 0.7762 与 C3-STISR 的 0.7721（p.6 Table 1）——「保真全面最优」只在 PSNR 上成立。
- **PSNR 增益有限**：20.35→22.46 相对 LR 是 +2.11 dB，但相对已有 STISR 只 +0.95 dB，说明保真提升主要来自「不再为识别扭曲图像」，而非重建质量本身的跃升。
- **只做 2×**：$I_{SR}\in\mathbb{R}^{C\times 4N}$ 即每边 2×；TextZoom 也是 64×16 → 128×32 的设定，更高放大倍数未验证（p.3）。
- **消融粒度偏粗**：mini 解码器只测「全去」、没测个数；$L=3$、$N_i=[4,5,6]$、$C'=48$ 等超参均无敏感性分析（p.5 §4.2）。
- **训练开销未报**：Table 5 只统计推理（Encoder + final decoders 合计 0.74G 正好对齐 Table 1），mini 解码器的训练算力与总训练时长（4× RTX 3090）都未量化。

## 我的疑问

- [ ] **Table 4 的「渐进」并不单调**：中间层 $i=1/2/3/4$ 的 accuracy 是 65.3 / 68.3 / **67.6** / 74.8，PSNR 是 20.84 / **20.77** / 21.20 / 22.46，两列都在中途掉过一次，正文却写 "basically gradually improved"（p.7）。这只是中间监督的噪声，还是第 3 个 block 的表征真退化了？
- [ ] 两个解码器**互不通信**，只共享编码器特征。那「协同」到底来自哪里——共享编码器的多任务正则，还是仅仅数据效率？Table 3 的两个「只留一个解码器」变体同时也少了另一任务的监督信号，没法干净地分离「特征共享」与「任务互补」的贡献。
- [ ] STR 解码器 6+6 层且在 $[H, W/32]$ patch 上做全注意力、还要自回归解码——**推理延迟**会不会成为瓶颈？论文只报 FLOPs 与 PSNR，没报实际耗时。
- [ ] 编码器仅 0.52M 参数却承担 70% 算力，而 Bi-GRU 是消融里第二关键组件；既然它这么吃重，换成 1D 卷积或 Mamba 会怎样？（[[LRSTRN 恢复与识别迭代协同增强的低分辨率场景文本识别]] 沿用了同样的 Swin + Bi-GRU 结构，这看起来是个可继承的改进点）
- [ ] 36 类字母数字、去标点转小写（p.5）意味着中文与符号场景完全未覆盖；同一作者的 C3-STISR 也是这个设定，是 TextZoom 生态整体卡在英文小词汇上的表现吗？

## 关键摘录

> "existing methods have two major drawbacks: 1) STISR models may generate imperfect SR images, which mislead the subsequent recognition. 2) As the STISR models are optimized for high recognition accuracy, the fidelity of SR images may be degraded."（p.1 Abstract）

> "Here the two decoders are optimized separately to avoid malign competition, thus both recovery and recognition performance can be well optimized with the text-specific features from the encoder."（p.2）

> "compared with STISR models that directly recognize texts from SR images that may carry noisy or erroneous information, STIRER develops an STR decoder to read texts from LR images with the help of text-specific feature."（p.3 §2.3）

> "Note that pixel information is valuable for the recognition of LR images. Therefore, utilizing down-sampling layers as in typical STR works [12, 40] is inappropriate for LR images."（p.4 §3.4）

> "Doing STISR and STR collaboratively in a unified framework is better than doing STISR and STR with two independent models."（p.8 §4.5.2）

> "in some tough situations where pixel information is extremely insufficient, the recovery and the recognition may fail. This indicates that the recovery and recognition of LR images is still a challenging task."（p.7）

## 相关笔记

- [[LRSTRN 恢复与识别迭代协同增强的低分辨率场景文本识别]] — **直接继承者**（ICME'26）：同样 Swin + Bi-GRU 骨干、同样逐阶段 mini 解码器 + $L_{GP}$，把本文的单向双解码改成 N 阶段双向迭代注入；TextZoom 78.5，超本文 +3.7（Hard +6.2）
- [[C3-STISR 三重线索引导的场景文本超分]] — **同组前作**（复旦，Minyi Zhao 一作）：三重线索引导 STISR，本文在 Table 1/2 中把它当最强 STISR 基线；两条路线正好对照「线索喂给 SR」vs「线索喂给并行双解码」
- [[TextZoom 与 TSRN 真实场景文本图像超分]] — 本文主数据集 TextZoom 与 $L_{GP}$ 梯度轮廓损失、SRB 块的出处
- [[VTCTSRN 视觉文本线索迁移的场景文本超分]] — 另一条「视觉/字形线索引导 STISR」路线，可与本文「线索只进编码器」对照
- [[TiGeSR 先文本后图像的两阶段场景文本超分]] — 最新的「先文本后图像」解耦思路，与本文「恢复/识别并行解耦」是两种不同的解耦轴
- [[SVTR 场景文本识别]] — Table 1/2 的 STR 基线；其 merging 压高度的做法正是本文 §3.4 明确反对的
