---
created: 2026-11-07
source: E:\.dsh-paper-reader\data\默认专题\Zhao 等 - 2022 - C3-STISR Scene Text Image Super-resolution with Triple Clues.pdf（arXiv:2204.14044；代码 https://github.com/zhaominyiz/C3-STISR）
type: 论文
tags: [论文, 场景文本识别, 超分辨率, 可变形卷积]
---

# C3-STISR：把线索从一条扩到三条（识别 / 视觉 / 语言）

> Minyi Zhao, Miao Wang, Fan Bai, Bingjia Li, Jie Wang, Shuigeng Zhou（复旦 + 字节跳动，字节实习期工作）。页码均指本机 PDF 页码（7 页正文 + 参考文献）。

## 一句话总结

在"拿识别器反馈当先验"的 STISR 路线上补两条线索——把识别串**渲染成字形图**当视觉线索、用**字符级语言模型**纠错当语言线索——三条线索各自投影成像素级特征图后，用门控融合成统一的 $h_t$ 去指导 TSRN 骨架。TextZoom 上 CRNN 平均识别率 53.7%（前最佳 TG 48.9%，+4.8），ASTER 64.1%。

## 核心问题

已有的 clue-guided 方法只用识别器的**概率分布（PD）**当线索，作者指出两个硬伤（p.1 摘要、p.2）：

1. **模态不兼容（Compatibility）**：PD 是字符概率分布，与 STISR 这个像素级低层视觉任务之间有明显的模态鸿沟；
2. **信息不可靠（Inaccuracy）**：识别器本身就会错——在 LR/HR 图上识别率只有 **26.8% / 72.4%**（p.2）——把错的 PD 喂回去会反过来误导超分。

Fig. 1(c) 就是把这件事画出来：识别器读成 "blrd"，真值是 "bird"，用这份 PD 引导出来的 SR 仍然读 "blrd"。作者的类比是"人怎么补糊字"：除了看图，还会靠字形结构和语言知识去猜。

## 关键方法

### 0. 总览（p.3，Fig. 2）

![C3-STISR 架构图 Fig2](C3-STISR%20架构图%20Fig2.png)

<small>裁自论文第 3 页 Figure 2，仅取图本体与标注，已去掉图注。</small>

**怎么读这张图：** 上半部分黄框是**超分骨架** $f_{SR}$：$I_{LR}$ 走 STN → 5 个 TSRN block → pixel shuffle → $I_{SR}$，注意每个 TSRN block 下方都有一支箭头指向它，那就是线索 $h_t$ 从每层的卷积特征处按通道拼进去（不是只喂第一层）。下半部分紫框是**线索生成器** $f_{CG}$，分两支：抽提支 $f_{CE}$ 和融合支 $f_{CF}$。

抽提支里三条路都从黄色的识别器 $R$ 出发：
- **识别线索** $h_{rec}$：PD → Projection network → DSTA（绿线）；
- **语言线索** $h_{ling}$：PD → Language model（纠错）→ Projection → DSTA；
- **视觉线索** $h_{vis}$：PD → 解码成字符串 → Skeleton painter 画出 "TRIPLE" / "triple" 两张黑白字图 → **Deformable convolution**（蓝框，用 $h_{rec}$ 去对齐补偿）→ DSTA。

三条线索最后在 "Gated fusion"（那个像方向盘/雨刷的图形）处融合成 $h_t$。图下方两支红箭头是两条微调损失：$L_{rec}$（微调识别器）、$L_{ling}$（微调语言模型）。

> 一个易读错的地方：图上 Projection network 和 DSTA 各只画了一个紫框，**看起来像三条线索共享**；但正文说 $f_{ling}$"similar to $f_{rec}$, consists of a projection network and a DSTA block"（p.4），消融里也写"we stack **three** DSTA blocks in our clue extraction branch"（p.6）——**每条线索各一套**，图是为了省地方。

### 1. 超分骨架（p.4）

四点设计：① TextZoom 的 LR-HR 对是人手工裁的，有像素级偏移，所以先接 STN；② 5 个改造过的 TSRN block，线索 $h_t$ 在每个 block 的卷积特征处按通道 concat；③ pixel shuffle 上采样（结构同 [[像素重排：PixelUnshuffle 与 PixelShuffle]]）；④ 两个损失分工——$L_2$ 像素损失管保真、text-focused loss [Chen et al., 2021a] 管识别：

$$L_{pix}=\|I_{HR}-I_{SR}\|_2,\qquad L_{txt}=\lambda_1\|A_{HR}-A_{SR}\|_1+\lambda_2\mathrm{KL}(p_{SR},p_{HR})$$

$A$、$p$ 是**冻结**的 transformer 识别器给出的 attention map 与概率分布，$\lambda_1=10$、$\lambda_2=0.0005$。

### 2. 三条线索怎么抽（p.3–4）

三条线索的目标一致：都从 $R(I_{LR})\in\mathbb{R}^{L\times|A|}$ 出发，最终变成 $\mathbb{R}^{C'\times N}$ 的像素级特征图（$C'=32$），再融合。

| 线索 | 从哪来 | 加工 | 表达的语义 |
|---|---|---|---|
| $h_{rec}$ | 识别器 PD | projection network（4 层转置卷积 + BN + 双线性插值）+ DSTA | 字符在哪里、是什么（但含错） |
| $h_{vis}$ | PD 解码成字符串 → 逐字渲染成黑白字图 | deformable conv（用 $h_{rec}$ 对齐补偿）+ DSTA | 字形结构（更贴近像素任务） |
| $h_{ling}$ | PD 经字符级 LM 纠错后的 $p_{LM}$ | projection network + DSTA | 语言上"应该是什么字" |

三个细节值得记：

- **$h_{rec}$ 靠微调提质**：$L_{rec}=k_1\|R(I_{LR})-R(I_{HR})\|_1+k_2\mathrm{KL}(R(I_{LR}),R(I_{HR}))$（Eq.3），把 HR 图的识别结果蒸馏给 LR 分支——因为后面两条线索全都建在它上面。
- **$h_{vis}$ 的两个缺口与补法**：① 解码过程把置信度丢了（可能放大错误传播）；② 字图是水平、固定字体渲染的，而识别线索是从 PD 插值来的，二者有**运动/形状错位**。所以用 deformable conv 以 $h_{rec}$ 为条件去做对齐补偿（p.4）。
- **$h_{ling}$ 是有预训练的**：用预训练的**双向 cloze 网络**（[Fang et al., 2021b]，即 SRN "Read like humans"）做字符级纠错，先在语料上做"拼写破坏—还原"预训练，再用 $L_{ling}$（Eq.5，形式同 Eq.3）微调到超分任务。

### 3. 门控融合（p.4，Eq.6）

三条线索已是同尺寸同模态的像素特征图，所以融合可以很简单：各自过几层空洞卷积 → 与 $I_{LR}$ 按通道拼接 → 一组卷积出掩码 $M\in\mathbb{R}^{3\times C'\times N}$ → 沿第一维 softmax → 逐像素加权求和：

$$h_t=M[0,:]\otimes h_{rec}\oplus M[1,:]\otimes h_{ling}\oplus M[2,:]\otimes h_{vis}$$

作者的理由也讲得直白（p.6）：经过 projection + deformable conv 之后**模态鸿沟已经没了**，此时"谁对就取谁"的 gated fusion 比多头注意力、DCN 融合都好（表 4：gated 53.7 vs MHA 51.3 vs DCN 51.5）。

### 4. 总损失与三阶段训练（p.4）

$$L=\alpha_1L_{pix}+\alpha_2L_{txt}+\alpha_3L_{rec}+\alpha_4L_{ling}$$

取值：$\alpha_1=20,\alpha_2=20,\alpha_3=1,\alpha_4=0.2$，$k_1=k_2=1.0$，$C'=32$（p.4）。训练分三步：① 语言模型"拼写破坏—还原"预训练；② 识别线索 + 视觉线索抽提模块预训练；③ 合起来整体微调。**目的是别把语言模型的预训练知识洗掉**。

实现细节：PyTorch 1.8、8×V100 32G、Adam lr 1e-3、batch 48、识别器用 CRNN、skeleton painter 用 PIL 画黑白字图、字体 ubuntu bold、**每张 LR 图生成上下两张字图（一张全大写一张全小写）**。

## 结论与局限

**结果**（TextZoom 21,740 对，训练 17,367，测试 easy/medium/hard = 1,619/1,411/1,343；4× 放大）

- 表 1（CRNN，未微调识别器）：Easy **65.2** / Medium **53.6** / Hard **39.8** / 平均 **53.7**；对比 TG 48.9、STT 48.1、TSRN 41.4、HR 上限 72.4、BICUBIC 26.8。MORAN 平均 60.5、ASTER 平均 64.1。
- 表 2（三档平均的保真度）：PSNR **21.51**（PCAN 21.49、TSRN 21.42、TG 21.40 次之）、SSIM **77.21**（×10⁻²，即 0.7721）。

**消融结论**（表 3–7，指标多为 CRNN 平均识别率）

- **单条线索的贡献排序**（表 5）：$h_{rec}$ 52.2 > $h_{vis}$ 51.7 > $h_{ling}$ 51.0（无线索 baseline 44.2）。**视觉线索在保真度上最强**（单用 PSNR 21.21 是三条里最高），语言线索最弱——"STISR 毕竟是视觉任务"。
- 两两组合推 **$h_{rec}+h_{ling}$ 识别率最好（53.7）**，$h_{rec}+h_{vis}$ 保真度最好（PSNR 21.38，高于任一单线索）。
- **对齐补偿不能省**（表 3）：不做 deformable 补偿 49.4 → 51.7，是抽提支里掉得最多的一项。
- 识别器微调（50.3→52.2）、多阶段训练（51.1→53.7）、DSTA（51.7→53.7）都有效，其中 MST 与 DSTA 各贡献 +2.6 / +2.0。
- $\alpha_4$ 网格搜索：0.2 最好（表 7），说明语言线索该**轻用**——权重给大了反而洗掉 LM 的预训练知识。

**局限 / 我观察到的点**

1. **保真度优势其实很薄**：PSNR 21.51 vs PCAN 21.49 只差 0.02 dB，而 **SSIM 77.21 反而低于 PCAN 的 77.53**。摘要说 "outperforms the SOTA methods in fidelity" 站不太住，真正站得住的是识别率（+4.8）。
2. **三线索的"增益"在识别率上是零**：表 5 里 $h_{rec}+h_{ling}$ 已是 53.7，加进 $h_{vis}$ 后**平均识别率一样是 53.7**，视觉线索只在 PSNR 上补了 21.28→21.51。所谓"三条线索都必需"（p.6）主要靠保真度支撑，不是识别率。
3. **识别线索单用会拉低保真度**（表 5：21.14 vs 无线索 21.38）——先验越强越往"识别友好"偏，这个权衡论文没讨论。
4. **视觉线索的生成方式很土**：解码→PIL 画固定字体的黑白字图→再拿可变形卷积去对位。一张图渲染两遍（大/小写）也意味着**每个 LR 输入要多算两路**，推理开销论文只在补充材料里比，本 PDF 看不到。
5. **两条线索都不是本文原创**：字形图/语言模型都来自前作，本文的贡献集中在"怎么抽、怎么拼成一套 $h_t$"（这点作者自己也说成 "clue generator"）。
6. **错误传播没有根治**：$h_{rec}$ 错了，$h_{vis}$ 会照着错字画、$h_{ling}$ 会照着错字纠，三条线索同源，只报告了 LM 能改对 "blrd→bird" 的例子，没有错误纠正率的分项统计。

## 我的疑问

- [ ] 视觉线索用固定字体（ubuntu bold）渲染，遇到艺术字、手写、中文会怎样？这套方法在中文场景文本上跑过吗？
- [ ] 既然 $h_{rec}+h_{ling}$ 就吃到 53.7，$h_{vis}$ 的 +0.23 dB 值不值它那两路渲染 + DCN 的开销？有没有算过 latency？
- [ ] 三条线索同源于一个识别器，能不能换成两个不同识别器/一次自纠错来真正打破错误传播？
- [ ] 表 5 的无线索 baseline 是 21.38，而三线索是 21.51；单纯把 $h_t$ 也喂给 STN 之前、或者加更多 block，是不是也能拿到同样的 PSNR？
- [ ] 后续 VTCTSRN（2025）在 ASTER 上做到 70.8 vs 本文 64.1，说明"原型/字形"这条路还有空间——那本文的 DCN 对位是不是不够？

## 关键摘录

> "Most recent approaches use the recognizer's feedback as clue to guide super-resolution. However, directly using recognition clue has two problems: 1) **Compatibility**. It is in the form of probability distribution, has an obvious modal gap with STISR — a pixel-level task; 2) **Inaccuracy**. it usually contains wrong information, thus will mislead the main task and degrade super-resolution performance."（p.1，摘要）

> "the recognition accuracy on LR/HR images is only 26.8%/72.4%, see Sec. 4.3"（p.2）

> "Here, we use Python Image Library (PIL) as $f_{sp}$ to draw black-white text images. ... First, the prediction confidence is lost during decoding, which may exacerbate the propagation of errors. Second, the text image is generated in horizontal direction with fixed font, while the recognition clue is interpolated to the pixel level, which may incur motion and shape misalignment."（p.3）

> "the recognition clue achieves the best recognition performance, and the visual clue outperforms the others in fidelity. The linguistical clue is inferior to the other two clues since STISR is a vision task."（p.6，表 5 讨论）

## 相关笔记

- [[VTCTSRN 视觉文本线索迁移的场景文本超分]] — 同一赛道（TextZoom / STISR）的**后继**：同样是"渲染字形图当先验"，但指出按同位融合会出伪影，改成图侧可变形采样 + 跨注意力；ASTER 70.8 vs 本文 64.1，可直接对读"同位 DCN 对位 vs 跨图采样"
- [[TPWGAN 小波域判别器引导的场景文本超分]] — 同一赛道的另一条路：完全不碰先验，只改监督（小波域损失 + 频域判别器），基线是 TATT；两篇正好代表"先验侧"与"监督侧"两种改进
- [[ASTER 注意力场景文本识别器]] — 本文三个评测识别器之一（未微调），也是"识别器输出 PD / attention map 当线索"这条线的上游
- [[MORAN 多目标矫正注意力场景文本识别]] — 同为评测识别器；其"预测偏移场做几何矫正"与本文用 deformable conv 对齐字形图是同一类思路
- [[像素重排：PixelUnshuffle 与 PixelShuffle]] — 本文骨架末端用 pixel shuffle 做 4× 上采样

---

#论文 #场景文本识别 #超分辨率 #可变形卷积
