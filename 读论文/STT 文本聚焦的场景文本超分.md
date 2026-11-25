---
created: 2026-11-25
source: E:\.dsh-paper-reader\data\SR\Chen 等 - 2021 - Scene Text Telescope Text-Focused Scene Image Super-Resolution.pdf（CVPR 2021；https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Scene_Text_Telescope_Text-Focused_Scene_Image_Super-Resolution_CVPR_2021_paper.pdf）
type: 论文
tags: [论文, 场景文本识别, 超分辨率, 损失函数]
---

# STT（Scene Text Telescope）：把监督从「每个像素」挪到「字符」上

> Jingye Chen, Bin Li\*, Xiangyang Xue（复旦大学 智能信息处理上海市重点实验室 / FudanOCR）。CVPR 2021。页码均指本机 PDF（正文 8 页 + 参考文献，PDF 共 10 页）；文中多次提到的补充材料不在本 PDF 内，附图表细节只能标 ⚠️ 未见。

## 一句话总结

第一个把「文本特有属性」拆成**布局**与**字符**两层写进超分框架的工作：主干把 TSRN 的双向 BLSTM 换成 **TBSRN**（自注意力 + 2-D 位置编码），以便处理倾斜/弯曲文本；监督侧加两个**只在训练期存在**的模块——**Position-Aware** 用冻结 Transformer 识别器在 HR 图上的逐字符注意力图当标签做 L1 蒸馏，**Content-Aware** 用 EMNIST 上训的 VAE 隐空间距离定义「字符混淆系数」，把普通交叉熵改成**加权交叉熵**（本文即 "text-focused loss" 的出处）。TextZoom 上 CRNN 平均识别率 41.4% → **48.1%**，ASTER 58.3% → 60.1%，MORAN 55.4% → 58.4%，且推理阶段零额外开销。

## 要点提炼（速览）

1. **成绩**（表 5a，p.7）：CRNN **48.1**（TSRN 41.4，+6.7）、ASTER **60.1**（58.3，+1.8）、MORAN **58.4**（55.4，+3.0）。三个识别器上全部领先，且是当时少数在**不做识别器微调**的前提下比较的工作。
2. **增益可完整拆解**（表 1–3，p.6）：41.4 →（换 TBSRN 主干）43.2 →（+POS）45.6 →（+普通 CE）47.1 →（CE 换成 WCE）48.1。**主干 +1.8、位置 +2.4、内容 +1.5、加权再 +1.0**。这是本文最值得记的一张账。
3. **出发点是「通用超分不适合文本图」**（p.1–2）：通用 SR 把文本图当普通自然图像；TSRN 用水平/垂直 BLSTM 建序列依赖，遇到**倾斜或弯曲**文本就失效（p.2）。
4. **理论包装是 Inattentional Blindness**（p.2）：人看文本图会自然忽略背景，所以「没必要提升整张图的质量」——这句直接推出了 POS + CON 两个模块的存在理由。
5. **TBSRN**（p.3）：CNN×2 → 展平 → 自注意力 → 位置前馈 → reshape → pixel shuffle（2×）。自注意力让**任意像素对**可相关，因此对任意方向鲁棒；代价是并行处理丧失了 RNN 的隐式位置感，必须显式拼 **2-D PE**（x/y 轴各 d_PE=32）。
6. **没有 2-D PE 会崩**（表 1，p.6）：平均 43.2 → **37.9（−5.3）**，比换主干带来的 +1.8 还大——位置线索在这套架构里是**前提**而非增益。
7. **POS 是注意力蒸馏**（p.4，式 2）：$L_{POS}=\Vert A_{HR}-A_{SR}\Vert_1$，$A_{HR}$ 来自 HR 图（字符间隙清晰，可当「字符位置标签」），长度等于标签长度 l；λ_POS=10 时 +2.4。
8. **CON 是混淆度加权交叉熵**（p.4–5，式 3–4）：VAE 在 EMNIST 上学字符的 2-D 隐表示，相近字符在隐空间聚在一起（Fig. 4 的 {C,c,e}、{2,z,Z}），取 $c_{ij}=1/d_{ij}$，把 softmax 分母里每个候选的分数乘以「它与真值的混淆系数」；作者用式 4 证明：$c_{ij}>1$ 时**梯度更大 = 惩罚更重**。
9. **两个模块只在训练期用**（p.2）：HR 图与识别器都只出现在监督分支，测试时整条支路消失，**没有额外时间开销**——这是它相对后续 clue-guided 方法（要在线跑识别器）的结构性优势。
10. **作为预处理器能迁移**（表 6，p.7–8）：人工高斯模糊越狠收益越大（radius=3 时 IC13 +7.8、IC15 +3.2、CT80 +7.3，三库平均 20.1 → 25.2）；但**真实**低分辨率 IC15（<16×64 的 352 张）NRTR +5.7 / SEED +4.8，**全集 1,811 张只 +1.1 / +0.9**——域偏移很明显。
11. **保真度是主动让掉的**（表 5b，p.7）：加上 POS/CON 后 PSNR 在 easy 上 24.13 → 23.82、hard 上 20.09 → 19.68，作者一句「not so important compared to accuracy」带过，但这个权衡后来一直是这条线（LEMMA 等）的核心争论。
12. **概念遗产**：[[文本聚焦损失]] 这个词的源头就在这里，后续 [[C3-STISR 三重线索引导的场景文本超分]]（KL 版）、[[MPGTSRN 多重视觉语义提示引导的场景文本超分]]、[[VTCTSRN 视觉文本线索迁移的场景文本超分]] 都沿用它当基线损失。

## 核心问题

现有两条路都不合用（p.1–2）：

- **插值**（bicubic/bilinear）：放大后仍然糊，识别器照样读错（Fig. 1）。
- **通用 SR**（SRCNN、Laplacian pyramid 等）：把文本图当普通图像，**完全不利用**文本级布局与字符级细节。
- **少量利用文本先验的**：PlugNet 把识别与超分做成多任务；TSRN 加水平 + 垂直 BLSTM 建序列依赖——但 BLSTM 只在**规则方向**上有意义，倾斜/弯曲文本失效（p.2）。

更深一层的问题是**监督的分配**：已有方法都用逐像素 L1/L2（p.4），而自然场景文本图的背景往往占大头，像素损失相当于把大部分梯度花在背景上，字符区反而欠拟合（p.2、p.4）。

由此得到三条贡献（p.2）：

1. **TBSRN**：用自注意力采序列信息，对任意方向鲁棒；
2. **Position-Aware + Content-Aware 模块**：分别突出字符的**位置**与**内容**，且测试期无开销；
3. **加权交叉熵**：缓解低分辨率下易混字符的问题。

## 关键方法

### 0. 总览（p.3，Fig. 2）

三个模块：**PSM**（绿，像素监督主路）、**POS**（红，位置感知）、**CON**（蓝，内容感知）。主路是 $I_{LR}$ → **STN**（治 TextZoom 人工裁剪带来的像素错位，沿用 [44]）→ 5× TBSRN → pixel shuffle → $I_{SR}$。红/蓝两支在图上都是**从 HR 图取标签、从 SR 图取预测**，共享同一个**冻结**的 Transformer 识别器；图注明确写 "Parameters in the pre-trained Transformer are frozen"。

> 读图时最容易滑过的一点：**HR 图只出现在监督支路里**。所以 POS/CON 不是网络结构的一部分，而是两个**损失构造器**——这也是「零测试开销」的由来。

### 1. TBSRN：自注意力 + 2-D 位置编码（p.3，Fig. 3）

单元结构：CNN×2 提特征 → flatten 成 1-D 序列 → Self-Attention → Position-Wise Feed-Forward → reshape 回特征图；堆 5 个后接 pixel shuffle 上采样（结构同 [[像素重排：PixelUnshuffle 与 PixelShuffle]]）。像素损失 $L_{PSM}=\Vert I_{HR}-I_{SR}\Vert_2^2$。

两个设计理由都写在正文里：

- **为什么自注意力**：它能把特征图上任意像素对关联起来，而 BLSTM 只能沿一条扫描方向建依赖，因此对倾斜/弯曲文本更鲁棒（p.3）；
- **为什么必须加 PE**：RNN 可以靠时间步隐式获得位置信息，自注意力是**并行**处理的，天然没有位置感（p.3），所以把 $P_x, P_y$ 与特征图 concat。式 (1) 就是标准正弦编码，按 grid $(m,n)$ 的**行 / 列**分别生成：

$$P_x[m,n,2i]=\sin(m/10000^{2i/d_{PE}}),\quad P_x[m,n,2i+1]=\cos(m/10000^{2i/d_{PE}})$$

$P_y$ 同理把 $m$ 换成 $n$；$d_{PE}=32$（p.5）。消融显示这一项值 5.3 个点（p.6 表 1），是全文最「便宜」的一处设计。

### 2. Position-Aware Module：拿 HR 的注意力图当字符位置标签（p.4）

先用 Syn90k + SynthText 预训练一个 Transformer 识别器，再取它在**每个时间步**的注意力图 $A_{HR}=(a_1,\dots,a_l)$，$l$ 为文本标签长度。作者的理由很直接：HR 图字符间隙清晰，所以它的注意力图可以当**字符区域标签**用。SR 图过同一个识别器得到 $A_{SR}$（长度与 $A_{HR}$ 相同），用 L1 拉近：

$$L_{POS}=\Vert A_{HR}-A_{SR}\Vert_1 \tag{2}$$

本质上这就是**把识别器的注意力当作监督做蒸馏**，逼 SR 图「让识别器看对地方」，进而把像素级监督集中到字符笔画上，而不是被背景稀释。

### 3. Content-Aware Module：混淆度加权交叉熵（p.4–5）

**先造一张混淆表**：用 EMNIST（数字 + 字母，共 62 类）训一个 VAE，取每个字符的 2-D 隐表示（Fig. 4），观察到易混字符在隐空间里挨得很近（{C,c,e}、{2,z,Z}）。设 $d_{ij}$ 为字符 $A_i,A_j$ 的欧氏距离，定义混淆系数

$$c_{ij}=\begin{cases}1/d_{ij}, & i\neq j\\ 1, & i=j\end{cases}$$

**再改写 softmax**：识别器在时间步 $t$ 输出向量 $o=\{o_1,\dots,o_{|A|}\}$，真值为 $A_j$ 时，把它在该步的加权激活写成

$$a_j=\frac{e^{o_j}}{\sum_{i=1}^{|A|}c_{ij}\,e^{o_i}} \tag{3}$$

内容损失 $L_{CON}=-\sum_t \ln a_{y_t}$。**当所有 $c_{ij}\equiv1$ 时它就退化成普通交叉熵**——这个「退化即基线」的写法让消融非常干净（表 3 里 CE 与 WCE 两两对照）。

**为什么它有用**（式 4，p.5）：把梯度推一遍，

$$\frac{\partial L_{CON}}{\partial w_{ki}}=\frac{c_{ij}\,e^{o_i}}{\sum_m c_{jm}e^{o_m}}\,h_k$$

若预测的 $A_i$ 与真值 $A_j$ 长得像（$c_{ij}$ 远大于 1），反向梯度数值随之放大——**易混字符被罚得更重**。这是个很小的改动：不新增参数、不新增前向，只改分母的加权。

关键在于分母把**每个候选 i 都按「与真值 j 的混淆系数」加权**：与真值越像的候选被压得越低，相当于把概率质量从 best-competitor 那里挤走。

### 4. 总损失与训练细节（p.5）

$$L=L_{PSM}+\lambda_{POS}L_{POS}+\lambda_{CON}L_{CON},\qquad \lambda_{POS}=10,\ \lambda_{CON}=0.0005$$

PyTorch；4× NVIDIA TITAN Xp (12GB)；Adam；batch 80；lr 1e-4；$d_{PE}=32$；识别器参数冻结；评测识别器用 CRNN / ASTER / MORAN 的官方 PyTorch 实现与预训练权重（p.5）。

### 5. 顺带提出的两个新指标：TR-PSNR / TR-SSIM（p.5）

因为「我们只关心文本区」，作者用 SynthText 的文本框标签训一个 U-Net 分割模型出文本区掩码（Fig. 7），只在掩码内算 PSNR/SSIM。这是本文的第二个小贡献，但**后续工作基本不用它**（各自仍报全图 PSNR/SSIM 或识别率），实际影响力有限。

## 结论与局限

### 结果

**主干消融**（表 1，p.6，指标 = CRNN 平均识别率）：BICUBIC 26.8 / SRCNN 29.2 / SRResNet 35.1 / **TSRN 41.4** / TBSRN-1 39.8 / TBSRN-3 42.1 / **TBSRN-5 43.2** / TBSRN-6 42.3 / **TBSRN-5 w/o PE 37.9**。块数到 6 反而掉 0.9（作者归因于过深）。

**POS 权重**（表 2，p.6）：λ=0 时 43.2；0.1 → 42.3；1 → 44.0；**10 → 45.6**；100 → 45.0。

**CON 权重**（表 3，p.6）：λ=0.0005 时最好；同权重下普通 CE 47.1、**WCE 48.1**；λ 取 0.001 / 0.01 都回落（46.2 / 46.1）。

**LPSM 不能去**（表 4，p.6）：去掉后平均识别率 48.1 → **36.6**，PSNR 23.82 → **4.97**、SSIM 0.8660 → 0.1968。作者还观察到缺了 LPSM 会生成**偏绿**的图，"not friendly to human eyes"。

**对比**（表 5a，p.7，三档 easy/medium/hard 平均）：CRNN 48.1（TSRN 41.4）/ MORAN 58.4（55.4）/ ASTER 60.1（58.3）；**加这两个模块到别的骨架上同样有效**，如 SRResNet + 双模块在 CRNN 上 35.1 → 41.1（+6.0），TSRN + 双模块 41.4 → 42.9。

**作为预处理器**（表 6，p.7–8）：见要点 10。

### 局限 / 我读出来的点

1. **保真度是明摆着让掉的**：表 5b 里 TBSRN 单独 PSNR 24.13/19.08/20.09，加 POS+CON 后 23.82/19.17/19.68——easy 与 hard 都降。作者用「识别率更重要」一句带过，但这正是后来这条线一直被质疑的软肋（识别友好 ≠ 看着更清）。
2. **加权交叉熵的增益位置有点反直觉**：表 3 里 WCE 相对普通 CE，medium 上 +2.4（44.7 → 47.1），而最需要「分辨易混字符」的 **hard 只 +0.3**（35.0 → 35.3）。硬样本上模型连字形轮廓都还没恢复，加权解决不了「笔画缺失」这类更底层的问题。
3. **λ_POS=0.1 比完全不用还差**（42.3 < 43.2）：小权重的注意力蒸馏更像噪声，说明这一项要么给足（10），要么别给。
4. **LPSM 去掉后 PSNR 只有 4.97**，说明 CON 是可以「骗」识别器的——单靠识别损失能把图推到识别器喜欢、但完全不像原图的形态，像素监督是唯一的锚。
5. **迁移到真实低分辨率几乎不涨**：IC15 全集 +1.1 / +0.9（p.8），而合成模糊能 +7.8——训练分布（TextZoom 的相机焦距退化）与真实压缩/小字退化之间的差距没有被处理，这也是后续 STIRER、TiGeSR 等要解的域问题。
6. **字符集被绑死在 EMNIST 62 类**：混淆系数依赖一个英数 VAE 隐空间，中文/日文等字符集要整套重建，论文没有任何跨字符集实验。
7. **训练管线偏重**：除超分主干外还要一个预训练 Transformer 识别器、一个 EMNIST VAE、一个 U-Net 文本区分割模型（为算指标）。推理虽零开销，但复现成本比 TSRN 高不少。
8. **只有 2× 设定**（16×64 → 32×128），与 TSRN 保持一致以便对比；更大倍率、更小输入都没测。
9. **失败案例自曝**（p.8，Fig. 10）：长文本且字号小、背景复杂/遮挡、艺术字与手写、训练集里没出现过的文本，都处理不好——尤其「未见标签」说明模型**记住了训练文本**而不是纯粹恢复了字形，这点值得警惕。

## 我的疑问

- [ ] 弯曲/透视文本的**注意力图本身**就可能对不准（HR 图也不总是清晰的），用错位的注意力当标签会不会反而教坏 SR？论文没给 POS 作用于弯曲样本的分项结果。
- [ ] 表 2 里 λ_POS = 0.1 掉到 42.3，是优化不稳定（梯度尺度随 $\lambda$ 线性变化）还是注意力图本身有噪声？跟 LR 与 LR-λ 一起搜一遍就能分清。
- [ ] WCE 在 hard 上只 +0.3，如果按「hard 子集单独调 λ_CON」会不会更高？作者只搜了全局 λ。
- [ ] TR-PSNR/TR-SSIM 依赖一个 U-Net 分割器，指标本身带误差，为什么后来没人用？是不是因为**换数据集就得重训分割器**？
- [ ] 与 [[C3-STISR 三重线索引导的场景文本超分]] 的 KL 写法（$KL(p_{SR},p_{HR})$）相比，WCE 与 KL 何者更稳？两篇都没有交叉实验，只在各自表格里互为基线。
- [ ] 训练期开销（识别器前向 + 冻结参数、VAE、分割器）究竟多少？论文只在补充材料里谈计算成本，本 PDF 看不到。

## 关键摘录

> "According to Inattentional Blindness [28, 29], when humans observe a text image, they will naturally pay more attention to text regions rather than backgrounds, i.e., there is no need to improve the quality of the whole image in the super-resolution task."（p.2，本文两个模块的立论基础）

> "Unlike those RNN-based models which can implicitly attain positional clues according to time steps, the Self-Attention Module is not aware of spatial positional information since the input is processed in parallel."（p.3，为什么必须加 2-D PE）

> "If $A_i$ and $A_j$ look similar, $c_{ij}$ will be a high value (i.e. much greater than 1), which results in a numerically higher gradient for backpropagation (i.e. get more punishment)."（p.5，加权交叉熵为什么有效）

> "the model tends to generate green-toned text images in the absence of $L_{PSM}$, which are not friendly to human eyes."（p.6，像素损失的锚定作用）

> "Although the performance does not reach the best, it is not so important compared to accuracy in this task."（p.7，作者对保真度让位的态度）

> "the model is hard to handle the LR images with long and small texts. ... Furthermore, artistic fonts and handwriting texts bring difficulties to the model. We also observe that our model is hard to tackle those images whose labels have not appeared in the training set."（p.8，失败案例）

## 相关笔记

- [[TextZoom 与 TSRN 真实场景文本图像超分]] — **直接前作与对比基线**：TextZoom 数据集、TSRN（5×SRB + 中心对齐 + 梯度轮廓损失）都被本文沿用；本文把它的 BLSTM 换成自注意力，CRNN 平均 41.4 → 48.1
- [[文本聚焦损失]] — 本文就是 "text-focused loss" 的命名出处；那篇把这个损失拆成 $L_{POS}$ + $L_{CON}$ 的两种写法（WCE vs KL）并排比较，可对照阅读
- [[C3-STISR 三重线索引导的场景文本超分]] — 同组后续（复旦 + 字节）：把 WCE 换成 **KL 蒸馏**，并补上字形图与语言两条线索；其表 1 里 "STT 48.1" 就是本文的成绩，可直接看 48.1 → 53.7 的演进
- [[MPGTSRN 多重视觉语义提示引导的场景文本超分]] — 沿 TFL 框架，把单一识别线索拆成边缘/分割/语义三路提示，代表「先验越做越细」的方向
- [[VTCTSRN 视觉文本线索迁移的场景文本超分]] — 批评「按同位融合」会出伪影，改成可变形采样 + 跨注意力；分数从本文 48.1 一路推到 63.0
- [[TPWGAN 小波域判别器引导的场景文本超分]] — 对照路线：**完全不碰识别器先验**，只在小波域改损失与判别器；两者构成「监督侧」与「先验侧」两种改进
- [[ASTER 注意力场景文本识别器]] — 本文三个评测识别器之一（60.1），也是 POS 模块「逐时间步注意力图」这套机制的原型
- [[MORAN 多目标矫正注意力场景文本识别]] — 同为评测识别器（58.4）；其偏移场矫正与本文 STN 治错位的思路同源
- [[感知损失 实时风格迁移与超分辨率]] — 概念对照：POS 用识别器注意力当监督，本质是**下游任务驱动的感知损失**，与 Johnson 用 VGG 特征算感知损失是同一种「换监督信号」的思路
- [[像素重排：PixelUnshuffle 与 PixelShuffle]] — 本文末端 2× 上采样用的 pixel shuffle

---

#论文 #场景文本识别 #超分辨率 #损失函数
