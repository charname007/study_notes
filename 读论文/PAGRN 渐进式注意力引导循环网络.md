---
created: 2026-10-30
source: Zotero 条目 7V7ALNY8 / PDF 附件 FT5JCDB5 — E:\Zotero\storage\FT5JCDB5\Zhang 等 - 2018 - Progressive Attention Guided Recurrent Network for Salient Object Detection.pdf
zotero_key: 7V7ALNY8
type: 论文
tags:
  - 论文
  - Attention
  - 显著性检测
  - 循环网络
---

# PAGRN：渐进式注意力引导循环网络用于显著性目标检测

## 一句话总结
用**渐进式注意力**筛选多级卷积特征（空间注意力 SA + 通道注意力 CA 串联，高层指导低层），再用**多路循环反馈**把顶层全局语义回流到浅层，解决 FCN 类显著性检测方法「多层特征不加区分直接用」和「浅层缺全局语义」两大痛点。

## 核心问题

显著性目标检测（Salient Object Detection, SOD）要模拟人眼判断图像区域重要性，并做像素级分割（第 1 页）。传统方法靠颜色/亮度/对比度等手工特征，复杂场景下失效；CNN 方法破了这个瓶颈，但引出两个新问题：

1. **多级特征不加区分**：当时 SOTA 都把多层卷积特征直接融合，但论文指出 *"not all features are of equal importance to saliency detection and some even cause interference"*（第 1 页）——背景细节特征反而干扰判断。作者用图 1 说明动机：空间注意力能突出前景、抑制背景干扰；通道注意力能给「前景响应强」的通道更大权重（第 1 页）。
2. **浅层缺全局语义**：基于 VGG/ResNet 的 side-output 方法里，浅层只有局部纹理，*"lack the ability to obtain global semantic information thus generate messy results"*（第 2 页），需要一种**从网络内部**改进特征学习能力的办法。

已有的 recurrent 方案（RFCN）只是把上一时刻的 saliency map 拼回原图作为下一时刻输入，但 *"effects of saliency prior is greatly weakened due to the concatenation with raw images"*（第 2 页）——先验被原图稀释了。

## 关键方法（三个创新点）

整体框架（论文 Fig. 2，第 3 页）：VGG19 主干 → 每个 stage 后接注意力模块 → 高层 attentive feature 逐步指导低层 → 最后多路循环反馈从顶层回流浅层。

![PAGRN 架构图 Fig2](PAGRN%20架构图%20Fig2.png)

<small>裁自论文第 3 页 Figure 2，仅取图本体与图例，已去除该页正文与图注文字。</small>

**怎么读这张图：**

- **上半（主干特征金字塔）**：白色堆叠块是 VGG19 各层卷积特征。红色方框标出被取作 side-output 的三个层 —— `Conv3_4`、`Conv4_4`、`Conv5_4`，右端浅紫色块是顶层 `Conv5_5`。
- **虚线黑箭头 = Multi-Path Recurrent Connections**：从 `Conv5_5` 出发同时回流指向 conv3/conv4/conv5 三个位置（对应 §3.3 的 `ℓ ∈ R`，`R = {3,4,5}`）。
- **下半（注意力级联，自右向左）**：浅紫色三角块是逐级生成的 attentive feature —— 右端 `S₅^csa` → 中间 `S̃₄^csa` → 左端 `S̃₃^csa`；紫色粗箭头是特征流向，**绿色箭头 = 2× 上采样**（公式 7 的 `UP(·)₂`）。
- **圆形图标 = Attention Module**：蓝色 × 是通道注意力、黄色 × 是空间注意力，两者串联（见下方 Fig.3）。
- **虚线框 + 图例 "Supervised by GT" = 中间监督**：每级的 saliency 预测都被 GT 监督，最左端输出最终显著性图。

### 1. 空间注意力（SA）+ 通道注意力（CA），串联使用（§3.1，第 3–4 页）

**空间注意力**：不把所有空间位置同等看待，对每个位置学权重，聚焦前景、抑制背景干扰。先用一层卷积把 C 个通道压成一张空间底图，再对图上每个位置做 Softmax（公式 1、2）：

$$m = W_s * f + b_s$$

$$a_s(l) = \frac{e^{m(l)}}{\sum_{l' \in \mathbb{L}} e^{m(l')}}$$

其中 `W_s` 是卷积核、`b_s` 是偏置，`m ∈ R^{W×H}` 把全部通道信息整合成空间注意力底图，`L` 是空间位置集合。

**通道注意力**：把特征 `f` 展开成 `[f₁,…,f_C]`，对每个通道做**平均池化**得到通道特征向量 `v ∈ R^C`，再用一层卷积学聚合特征（公式 3），Softmax 得到通道权重（公式 4）：

$$u = W_c * v + b_c$$

$$a_c(i) = \frac{e^{u(i)}}{\sum_{j=1}^{C} e^{u(j)}}$$

**两者串联**：先用 `a_c` 对各通道做 Hadamard 加权得 `f^ca`（公式 5），再用 `a_s` 对 `f^ca` 做空间加权得 `f^csa`（公式 6）：

$$f_i^{ca} = a_c(i) \times f_i$$

$$f^{csa} = a_s \otimes f^{ca}$$

![PAGRN 逐层注意力 Fig3](PAGRN%20逐层注意力%20Fig3.png)

<small>裁自论文第 4 页 Figure 3，仅取图本体，已去除图注。</small>

**怎么读这张图**：`f` 是输入特征 → 先按**通道维**加权（蓝色 ⊗，权重 `a_c`，由左下长条给出）→ 得 `f^ca` → 再按**空间维**加权（黄色 ⊗，权重 `a_s`，即中间那张灰度图）→ 得 `f^csa`。右端 `f^csa` 用**虚线框**画，对应图注说的「该特征生成的显著性图受 GT 监督」。注意顺序是 **CA 在前、SA 在后**。

论文图 1(d) 给了一个直观实例：conv5_5 中 channel-56（前景响应强）被赋权 **0.3289**，而 channel-36（背景响应强）只得到 **0.0756**（第 1 页）。

### 2. 渐进式注意力指导模块（§3.2，第 4 页）

出发点是深浅层特性互补：深层语义强但分辨率低（池化/卷积反复下采样导致边界模糊），浅层细节丰富但语义弱。

做法是 **stage-by-stage 级联**：先由 conv5 生成 attentive feature，把它作为「指导信号」引导 conv4 生成新的 attentive feature，再引导 conv3。原文描述为 *"attentive features from high-level stage guide the low-level stage to generate new attentive features"*（第 3 页 Fig.2 说明）。中间层 `S₅^csa`、`S̃₄^csa`、`S̃₃^csa` 的预测都受 GT 监督（middle supervision），保证每级注意力真学到东西。

### 3. 多路循环反馈（§3.3，第 5 页）

受 MPF-RNN 启发，把**顶层全局语义**通过可学习的反馈连接 `U_ℓ` **同时回流到多个浅层**（不是只连一条）。前向过程为：

$$H_\ell = f_\ell(W_\ell * H_{\ell-1} + b_{W_\ell})$$

加入多路循环连接后（t 为时间步，`N(·)` 为 L2-norm 归一化，`R` 为带循环连接的层集合）：

$$H_\ell^t = f_\ell\big(N(W_\ell * H_{\ell-1}^t + b_{W_\ell}) + N(U_\ell^t * H_L^{t-1} + b_{U_\ell}^t)\big),\quad \ell \in R$$

网络展开 **T = 2** 个时间步。效果是浅层同时「看到」局部细节与全局语义，**等效于获得更深网络的学习能力**，但参数量没有爆炸。

![PAGRN 多路循环 Fig4](PAGRN%20多路循环%20Fig4.png)

<small>裁自论文第 5 页 Figure 4，仅取图本体，已去除图注。</small>

**怎么读这张图（网络被展开成 2 个时间步）：**

- **t = 1 行（上半）**：正常前向 `H₁¹ → H₂¹ → … → H₅¹`，顶层经 **PAG**（渐进式注意力指导模块）输出一张 saliency map（右上，较粗）。
- **中间一行（自右向左）**：从顶层特征出发，经 `Conv` + 4 个 `Deconv` 逐级生成各层的**反馈参数** `(U_ℓ², h_ℓ²)`——这就是公式 9 里那个「时间可变、自适应学习」的反馈项。
- **t = 2 行（下半）**：把反馈按 ⊕ 加回各层，重新前向，得到更精细的 saliency map（右下）。
- **关键点**：反馈**不是**把上一时刻的 saliency map 拼回原图（RFCN 的做法），而是把**顶层特征**变成逐层可学习的参数注入——这正是本文与 RFCN 的本质差别。

**代码情况**（GitHub 检索结果）：官方仓库 [zhangxiaoning666/PAGR](https://github.com/zhangxiaoning666/PAGR) **只有 README** —— 提供九个数据集（DUTS / ECSSD / HKU-IS / DUT-OMRON / PASCAL-S / SED1 / SED2 / SOD / THUR）saliency maps 的百度网盘（提取码 `tpdz`）与 Google Drive 链接，**未开源训练代码**。第三方 `TracyMok/PAGRnet`（1 star，2021 最后更新）仅 README + LICENSE，无实现。故本篇无代码可对照，实现细节需依公式 (1)–(9) 与 Fig.2 自行复现。

## 结论与局限

**实验设置**（§4.1，第 6 页）：VGG19 主干、Caffe 实现、输入 353×353、训练集 DUTS-train（10,553 张）、固定学习率 1e-10、weight decay 0.0005；在 6 个基准数据集上评测（ECSSD / HKU-IS / THUR15K / PASCAL-S / DUT-OMRON / DUTS-test），指标为 PR 曲线、F-measure（β²=0.3）、MAE、S-measure。

**主要结果**（Table 1，第 7 页，对比 13 个方法）：6 个数据集 F-measure 全面领先 —— ECSSD **0.891**、HKU-IS **0.886**、PASCAL-S **0.803**、DUTS **0.788**、THUR15K **0.729**、DUT-OMRON **0.711**；MAE 同步最低（HKU-IS **0.048**、DUTS **0.055**、ECSSD **0.064**）。论文原话 *"ranks first on almost all datasets"*（第 7 页）。

**消融**（Table 2，第 8 页）：
- CA 与 SA 单独都有效，**组合（CSA）最佳**
- 循环连接加在 **conv3/4/5（记 "345"）** 效果最好；加到 conv1/conv2 无效，原因是 *"these two layers are too shallow to possess ability of effective feature learning"*（第 8 页）
- 循环次数 T=2 已饱和，T=3 不再提升
- 把各时间步结果 merge 反而下降，说明**最好结果就在最后一步**
- 中间监督有效：THUR 上 CSA 0.716 vs 无中间监督 0.706（F-measure）

**局限**：
- 作者承认循环连接对最浅两层无效（第 8 页）
- 主干仅 VGG19，未探索 ResNet 等更深/更强主干
- 学习率固定 1e-10（第 6 页），调参空间讨论不足
- 2018 年的 Caffe 实现，复现成本较高

**启发**（我的判断）：
- 「渐进式注意力指导」可迁移到 STR 任务：让 decoder 的高层语义逐级指导低层视觉特征
- 「多路循环反馈」本质是**用时间换深度**，对参数受限的端侧模型有借鉴意义
- 中间监督是级联注意力的必要 trick，否则中间层注意力可能学到噪声

## 我的疑问
- [ ] 顺序为何定为 CA→SA（公式 5→6）而非反过来？论文只对比了「单独 vs 组合」，没做顺序对调的消融
- [ ] T=2 饱和的真正原因是什么？是参数量、还是梯度回传路径的限制？
- [ ] 循环反馈里 `U_ℓ^t` 是「时间可变」的，那么推理时的时间步数能否动态决定（简单图少走几步）？
- [ ] 若换成 ResNet 主干，浅层语义能力更强，是否还需要多路反馈？

## 关键摘录

> "Effective convolutional features play an important role in saliency estimation but how to learn powerful features for saliency is still a challenging task. FCN-based methods directly apply multi-level convolutional features without distinction, which leads to sub-optimal results due to the distraction from redundant details."（第 1 页）

> "However, not all features are of equal importance to saliency detection and some even cause interference."（第 1 页）

> "However, shallower layers of backbone feature extraction network such as VGG and ResNet lack the ability to obtain global semantic information thus generate messy results."（第 2 页）

> "Saliency map of the previous stage is transmitted to the next stage for refinement. But effects of saliency prior is greatly weakened due to the concatenation with raw images."（第 2 页）

> "Through multi-path recurrent connections, global semantic information from the top convolutional layer is transferred to shallower layers, which intrinsically refines the entire network."（第 1 页）

> "Our algorithm ranks first on almost all datasets."（第 7 页）

> "It is observed that introducing recurrent connections to layer conv2 1 and conv1 1 can not make improvements. The main reason is that these two layers are too shallow to possess ability of effective feature learning."（第 8 页）

## 相关笔记
- [[Restormer 高效高分辨率图像修复 Transformer]] — 同样在「注意力该作用在哪个维度」上做文章：Restormer 把注意力搬到通道维，本文把注意力做成级联的门控筛子
- [[transformer的2道残差连接]] — 本文的多路反馈与残差在思想上同源：都是把「另一处算出来的信息」当成增量补进来
- [[Restormer 的 MDTA 与 GDFN 实现]] — 通道注意力的实现对照：本文用「分类式 Softmax 打分」，MDTA 用「通道协方差」
- [[关于联合小波变换的增强]] — 那篇想用交叉注意力做增强，本文的「高层指导低层」正是一种跨层级注意力

#论文 #Attention #显著性检测 #循环网络
