---
created: 2026-11-07
source: E:\.dsh-paper-reader\data\小波变换\Liu 等 - 2025 - TPWGAN Wavelet-aware text prior guided super-resolution for scene text images.pdf（Image and Vision Computing 162 (2025) 105707）
type: 论文
tags: [论文, 小波变换, 超分辨率, 场景文本识别]
---

# TPWGAN：把小波监督同时塞进生成器和判别器的场景文本超分

> Shengkai Liu, Jun Miao, Yuanhua Qiao, Hainan Wang，*Image and Vision Computing* 162 (2025) 105707，2025-08-28 上线。页码均指本机 PDF 页码（8 页）。

## 一句话总结

不改主干、只在 TATT 骨架上动"监督放在哪"：生成器加**平稳小波变换（SWT）四子带的加权 $L_1$** 逼出笔画级高频，判别器**不再看 RGB 图**，改看"三支高频子带 ⊗ 二值文本区域掩码"的拼接，用空间注意力（CBAM 式）融合后做真假判别。TextZoom 上 PSNR 21.70 / SSIM 0.7965、ASTER 平均 64.4%，都超过基线 TATT（21.52 / 0.7930 / 63.6%）与当时 SOTA 的 DPMN（21.49 / 0.7925 / 63.9%）。

## 核心问题

STISR 的老矛盾（第 1 页）：像素级 $L_1/L_2$ 把字边缘磨平，对抗损失又把高频伪影带进来，两者打架，OCR 反而掉。作者把矛头指向**现有小波方法只把小波损失放在生成器、判别器仍留在 RGB 域**：

> "Although some recent approaches incorporate wavelet-domain losses into the generator, they typically retain RGB-domain losses during adversarial training, limiting their ability to distinguish authentic text details from artifacts."（p.1）

而 RGB 域判别器分不清"真的笔画高频"和"对抗生成出来的假高频"，因为它压根不知道高频在哪。于是本文的切口是：**把对抗博弈本身搬到小波高频域，并且只在高频域里比，同时告诉判别器"哪里是字"**。

## 关键方法

### 0. 整体结构（第 3 页，Fig. 2）

![[TPWGAN 架构图 Fig2.png]]

骨架沿用 TATT：LR 图 → **TPG（Text-Prior Generator）** 出识别概率（文本先验）→ **TP Interpreter** 把它和卷积抽出的 image feature 融合成 text map → **SR Block（Sequential-Recurrent Block）** 重建 → SR 图 + SR Mask（由 SR 图阈值化得到的文本区域二值掩码）。

怎么读这张图：注意三条 SWT 支路都在 **Y 通道**上做。HR 图和 SR 图各自 SWT 出四子带，**LL/LH/HL/HH 全部参与 L1 损失**（绿色框，式 7），但只有 **LH/HL/HH 三支（HF）进 SWT-Discriminator**，且每条都配一张对应的 mask（HR mask / SR mask）一起送进去。也就是说：低频负责"结构别跑偏"（靠 $L_1$），高频负责"笔画要真"（靠 $L_1$ + 对抗），两条监督线在 Loss Total 合并。

### 1. SWT 域重建损失（第 4 页，式 6–7）

生成器总损失 $L_G = L_{SR} + L_{SWT} + L_{adv}$（式 5），其中

$$L_{SR} = \alpha L_{TSSIM} + \beta L_1 + \gamma L_{TP}$$

与 TATT 相同（TSSIM + L1 + 文本感知感知损失，$\alpha=1,\beta=1,\gamma=0.1$）。新增的是

$$L_{SWT} = \mathbb{E}\sum_i \lambda_i \| SWT(G(x))_i - SWT(y)_i \|_1,\quad i \in \{LL, LH, HL, HH\}$$

权重取 $\lambda_{LL}=0.1$、$\lambda_{LH}=0.01$、$\lambda_{HL}=0.01$、$\lambda_{HH}=0.05$（p.5，经验调参）。**注意权重分布本身是有信息的结论**：LL 拿最大权重（0.1）保全局结构，HH（对角高频，通常在自然图里最不重要）反而给了 0.05 —— 比 LH/HL 高 5 倍，说明作者认为文字的对角笔画（斜撇捺、转折）需要额外照顾。这点论文没展开解释，属于我读出来的推断。

**为什么用 SWT 而不是 DWT**（p.4）：SWT 是**平移不变**的，不做下采样，所以高频子带和原图**逐像素对齐**——这对判别器把"高频证据"和"文本区域掩码"按同一坐标融合是硬性要求。DWT 下采样后高频图和 mask 尺寸对不上，得先上采样插值，等于又糊了一次。这是本文选 SWT 的真正理由，论文只写了一句 "SWT's shift-invariance ensures accurate alignment between HF components and the original image"。

### 2. SWT 判别器（第 3–4 页，Fig. 3，式 3–4）

![[TPWGAN 判别器 Fig3.png]]

三块：特征加权（空间注意力）→ 特征提取（4 个 DB）→ 分类（GAP + 2 FC，出**未归一化标量**）。

- **输入**：$F \in \mathbb{R}^{H\times W\times 4}$，即三支高频子带 + 文本掩码拼成 4 通道。
- **空间注意力**（式 2，抄 CBAM 的 SAM）：通道维上做 Avg/Max pool → 拼接 → $7\times7$ 卷积 → sigmoid → 与原特征逐元素乘：$F' = \sigma(f_{7\times7}([AvgPool(F); MaxPool(F)])) \otimes F$。
- **特征提取**：DB1 把通道 64→512（"enriching HF details"），后续 DB 是 stride-2 卷积下采样。
- **分类**：输出未归一化实数值，避免梯度消失（引 [42]）。
- **损失**：用 **Softplus** 替代 sigmoid-BCE（式 3–4）：

$$L_D = \mathbb{E}[\text{Softplus}(-D(F_{HR},M_{HR}))] + \mathbb{E}[\text{Softplus}(D(F_{SR},M_{SR}))]$$
$$L_{adv} = \mathbb{E}[\text{Softplus}(-D(F_{SR},M_{SR}))]$$

- **掩码怎么来**（p.4）：对灰度图取平均强度做阈值二值化，字为 1、背景为 0。作者称这让判别器"只看字，不把每个像素一视同仁"。

**这一步的动机很直接**：HF 子带里既含真实笔画边缘也含对抗伪影，单看频谱分不清；补一张"这里应该是字"的语义图，判别器才知道该在**文本区域内**的高频上较真。

### 3. 训练细节（第 5 页，§4.3）

Adam（$\beta_1=0.9,\beta_2=0.999$）、batch 64、500 epoch、生成器 lr $1\times10^{-3}$、判别器初值 $10^{-4}$、每 100 epoch 学习率减半；文本先验用**预训练 CRNN（CTC 损失）**算 $L_{TP}$；输入 16×64，输出 2×。前 10k 迭代判别器每 2 张图才算一次损失，稳对抗训练。

> 未找到官方实现（全文与参考文献均无 github 链接、无 "code available" 声明），本节的数值/结构全部来自论文正文，无可核对代码。

## 结论与局限

- **主结果**（第 5 页，Table 1/2，TextZoom）：PSNR/SSIM 平均 **21.70 / 0.7965**，比基线 TATT（21.52/0.7930）和 DPMN（21.49/0.7925）都高；识别率 ASTER **64.4%**（TATT 63.6、DPMN 63.9）、MORAN **61.3%**（59.5、60.6）、CRNN **53.1%**（52.6、53.2）。
  - ⚠️ 正文称相对 TATT 提升 "2.6%, 0.8%, 0.5%"，但按 Table 1 平均值算是 **+0.8 / +1.8 / +0.5 个百分点**，三个数对不上（2.6% 与 0.8% 疑似位置写串或口径不同）。**引用请以 Table 1 为准**，别抄正文那句话。
  - 注意 CRNN 这一列 TATT 是 52.6 而 TPWGAN 53.1，但 **DPMN 53.2 更高**——作者说 "surpassed DPMN across most metrics"，确实只是 most。
- **泛化**（第 6 页，Table 3，只在 TextZoom 训、直接测）：SVTP 75.7 / SVT 84.7 / IC13 89.1 / IC15 71.1 / IIIT5k 88.9，大多超过 TATT；**但 CUTE80 上 72.8 输给 TSRN 的 73.2**。
- **消融（Table 4，第 6–7 页）——这部分信息量最大**：
  - $L_{SR}\to L_{SR}+L_{SWT}$：Easy ACC 76.7→78.51、Medium 61.1→63.7、Hard 43.3→44.5，PSNR 三档全涨。小波重建损失是**便宜又稳**的一档。
  - 再加对抗（$L_{adv}$）：Easy PSNR 24.44→25.09、ACC 78.51→79.6；**Hard ACC 44.5→46.1（+1.6）最猛**——对抗主要在难样本上起作用，符合"难样本才有伪影可压"的直觉。
  - **掩码的贡献最小**：$L_{adv}(HF)\to L_{adv}(HF\&Mask)$ 只带来 Easy 25.09 vs 24.99、Hard 46.1 vs 45.6，Medium 的 PSNR 甚至**持平（19.17 vs 19.13 微降）**。
  - **反直觉的一条**：去掉空间注意力、直接 concat HF+mask（第 3 行）反而**比只喂 HF 子带更差**（Hard ACC 44.9 vs 45.6）。原因是 {0,1} 硬掩码让判别器过度依赖文本掩码、迅速拉低判别器损失，导致对真正的高频线索失敏。→ **要融合，必须先过注意力再加权，硬 concat 是有害的**。
- **作者承认的局限**（第 7 页 §5 + Fig. 5）：当**文字与背景边界本身看不清**时（低对比、复杂背景），SWT 编码的频域信息不再显著，小波监督就失效，生成器保不住字形、边缘与背景重叠还会生伪影。CUTE80 的弯曲艺术字输给 TSRN 大概率同源——字框/背景边界本来就模糊。改进方向：提升复杂背景下小波监督的鲁棒性、把频域监督下放到下游任务。

## 我的疑问

- [ ] 掩码只带来 +0.1% 级别的收益，但作者把它写进"三大贡献"之一，是否价值被高估？（消融里还出现 mask 反而拖累 Hard 的情况）
- [ ] 二值掩码是**对灰度取均值后阈值化**得到的——阈值到底取多少？论文没给，而这是复现的关键参数，也是掩码效果的一个隐变量。
- [ ] SWT 只作用在 Y 通道上（Fig. 2 标注 "SWT on Y Channel"），那色度通道（Cb/Cr）的高频（彩色字边缘）完全没被约束，这对彩色场景文字是否是缺口？
- [ ] 判别器吃的是 HF 子带 $\mathbb{R}^{H\times W\times 4}$，但掩码只有 1 通道、量级远大于小波系数（{0,1} vs 归一化小数），两者 concat 前是否做过归一化？没归一化可能正是"硬 concat 让判别器盯着 mask"的直接原因（对应消融第 3 行）。
- [ ] $\lambda_{HH}=0.05$ 是 $\lambda_{LH/HL}=0.01$ 的 5 倍，作者只说"empirical tuning"，文字任务里对角高频为何更关键值得单独验证。
- [ ] 全部实验都在 2×、16×64 的 TextZoom 设定下，4×/8× 放大率或其他语言（中文、古汉字）没测，泛化结论覆盖面有限。

## 关键摘录

> "Unlike prior work that applied wavelet-domain losses only to the generator, TPWGAN injects wavelet-aware supervision into both the generator and discriminator."（p.1）

> "Unlike previous approaches that only incorporate wavelet loss during training while retaining an RGB-based discriminator, our method fuses wavelet subbands with the binary mask of an SR text image as input, which contains a higher proportion of detail and text edge information."（p.3）

> "SWT's shift-invariance ensures accurate alignment between HF components and the original image."（p.4）

> "directly feeding a binary mask with values strictly in {0,1} makes the discriminator overly dependent on the text mask, thus reducing its sensitivity to more informative HF cues."（p.7）

> "In such scenarios, the frequency information encoded by SWT becomes less salient, diminishing the supervisory strength of the SWT-domain losses."（p.7）

## 相关笔记

- [[VTCTSRN 视觉文本线索迁移的场景文本超分]] — 同一赛道（TextZoom / STISR）的另一条路：那边用"渲染原型图"补字形、靠可变形卷积解决对齐；这篇不碰先验，只重排损失与判别器的频率域。两篇都引 TATT 当基线，可以直接对读"先验侧改进 vs 监督侧改进"。VTCTSRN 的 CRNN 63.0 vs 本文 53.1，说明先验侧空间还没吃完。
- [[WINE 小波引导的GAN反演与编辑]] — 同一组观测的另一种修法：WINE 证明通用 $L_1/L_2$ **在小波四子带上是等权的**、而 LL 误差是高频的 30 倍，于是只算高频、并给子带分权重。本文的 $\lambda_{LL}=0.1$ 反而是权重大头、给的是**反向**的权重直觉（保结构优先），两篇对照能看清"该往哪个子带加权"取决于任务是重建保真还是感知细节。
- [[LWTformer 可学习小波变换的古汉字图像修复]] — 把固定 SWT/DWT 换成**可学习**的小波核 + 子带方向加权（WACGA）；本文用的是固定 SWT、权重靠经验手调。若把本文的 $\lambda_i$ 换成可学习标量，就是一个很自然的后续实验点。
- [[DTWSR 扩散Transformer建模多级小波谱的单图超分]] — 同为"小波域监督"，但那篇把整条扩散链搬进谱域；本文只在损失侧用 SWT、主干仍是 TATT 那套循环重建块，属轻量改法（不改结构、只改监督）。
- [[AdaptIR 自适应频率增强的一体化图像复原]] — 同样是"频域里做判别/调制"的思路，可作为把本文做法搬到多退化一体化任务时的参照。
