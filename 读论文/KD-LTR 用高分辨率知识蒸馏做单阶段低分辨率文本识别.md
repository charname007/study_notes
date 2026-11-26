---
created: 2026-11-25
updated: 2026-11-25
source: E:\.dsh-paper-reader\data\STR\Guo 等 - 2023 - One-stage Low-resolution Text Recognition with High-resolution Knowledge Transfer.pdf（arXiv:2308.02770v1；ACM MM '23, Ottawa；代码 https://github.com/csguoh/KD-LTR）
type: 论文
tags: [论文, 场景文本识别, 知识蒸馏, 低分辨率, 对比学习]
---

# KD-LTR：不做超分，用高分辨率知识蒸馏把识别器直接适配低分辨率输入

> 用户选中标题（第 1 页）："One-stage Low-resolution Text Recognition with High-resolution Knowledge Transfer"。作者 Hang Guo（清华深研院）、Tao Dai（深圳大学，通讯）、Mingyan Zhu、Guanghao Meng（清华深研院 / 鹏城实验室）、Bin Chen（哈工大深圳）、Zhi Wang、Shu-Tao Xia（p.1）。ACM MM '23（2023-10-29 ~ 11-03，Ottawa），DOI 10.1145/3581783.3611777，arXiv:2308.02770v1（p.1）。本机 PDF 共 10 页；以下页码均指 PDF 页码：p.1 摘要/引言/基金，p.2 Fig.1/贡献/相关工作，p.3 Fig.2/方法总览/基础识别器，p.4 视觉聚焦损失（式 1–3）与语义对比损失（式 4–5），p.5 soft logits（式 6–7）/总损失（式 8）/数据集/实现细节，p.6 四个消融表与温度分析，p.7 Table 5 主对比/Fig.4/4.5.1，p.8 Table 6/Table 7/鲁棒性/结论，p.9–10 参考文献。基金：国家重点研发 2022YFF1202104、NSFC 6230070671/62171248、深圳市科技计划、PCL2021A07（p.1）。

## 一句话总结

把「先超分再识别」的两阶段流水线**整个丢掉**：训练时挂一个**冻结的 HR 教师识别器**，用三层蒸馏损失把 HR 图像里的知识搬进**只吃 LR 图的学生识别器**；推理时丢掉教师，**不产生 SR 图、没有任何预处理模块**，一个识别器端到端出结果。三层知识分别是**视觉聚焦损失**（字符位置，分辨率无关）、**语义对比损失**（上下文语义）、**soft logits 损失**（词级 + 序列级软标签）。PARSeq-LTR 在 TextZoom 平均 **78.23%**，比「C3-STISR 超分 + PARSeq」高 **+9.4 点**，同时省 **3.13G FLOPs / 39.36M 参数**（p.7 Table 5）；作者称其为**首个单阶段 LTR 流水线**（p.2 贡献）。

## 核心问题

低分辨率场景文字笔画糊成一团，常规识别器掉点严重（p.1 引言）。主流做法是**两阶段**：第一阶段超分成识别友好图，第二阶段正常识别（TextSR/TSRN、STT、TATT、C3-STISR 一脉，p.2）。作者列了三条硬伤（p.1–2、p.2 Fig.1）：

1. **贵**：额外一个超分网络，训练与推理都要跑；当时的 SOTA STISR（C3-STISR）甚至比识别器 PARSeq 本身还大（p.1）。
2. **误差级联**：LR 信息丢失严重时超分也会重建错，而且错误被级联放大——Fig.1(a) 里 "Parrot" 被重建为 "Parret"，直接导致识别错（p.2）。
3. **像素像了、特征没像**：Fig.1(c) 可视化识别骨干的特征，SR 图虽然在像素空间模仿 HR，但**字符区域的特征仍有差异**（"Christie" 里的 "i"），所以超分目标与识别目标天然错配（p.2）。

作者还切了多任务学习这条替代路线（PlugNet、IFR）：它学 SR 与 STR 的共享视觉特征，但**仍局限在像素空间**，没有从语义空间、序列建模等视角给监督（p.8 §4.5.3）。而分辨率蒸馏此前只在分类/检测/人脸识别上做过，**没做过文本**——因为文本有序列性（p.3 相关工作）。核心追问于是变成两件事：**从 HR 搬什么知识？怎么搬？**（p.2）

## 关键方法

### 总览：双分支蒸馏，教师冻结（p.3 §3.1、Fig.2）

- **HR 教师分支**：随便一个现成识别器（ABINet/MATRN/PARSeq 都行），吃 HR 图，**训练全程冻结**，保证知识质量（p.3）。
- **LR 学生分支**：同架构识别器，吃 LR 图，目标是靠蒸馏补回丢失细节（p.3）。
- 两个分支共享同一套「 backbone → 序列建模 → 解码预测」结构（p.3 §3.2）：backbone（ResNet 或 ViT）出视觉特征 $Finmathbb{R}^{N	imes C	imes H	imes W}$；序列建模（LSTM/Transformer）把 2D 特征压成 1D 语义向量 $Hinmathbb{R}^{N	imes T	imes C}$；解码（投影 + softmax，可加语言模型）出字符概率。
- **唯一的结构性改动**：两分支输入分辨率不同，作者**改卷积 stride（CNN backbone）或 patch size（ViT backbone）**让深层视觉特征对齐（p.5 §4.2）。两者都用发布权重初始化，只微调学生（p.5）。
- 三层损失分别作用在 backbone 特征、语义向量、输出分布上，正好对应识别器的三个部位（p.3 Fig.2）。

### 第一层 · 视觉聚焦损失：搬「字符位置」，但分辨率无关（p.4 §3.3.1，式 1–3）

难点：两分支输入分辨率不同，特征**本来就不该数值相等**，直接上 L1/L2 对齐反而有害（p.4 原话见摘录）。两处设计：

1. **去分辨率域信息**：先做均值-方差归一化 $	ilde F=(F-mu)/sigma$，其中 $mu,sigmainmathbb{R}^{N	imes C}$ 是全局平均池化与标准差池化；再用**余弦相似度**只量方向一致性（式 1–2）。
2. **掩码蒸馏**：Fig.1(c) 说明字符特征比背景更难学、也更重要；于是拿**教师分支的注意力图当软掩码 $M$**，给像素重新加权，让学生聚焦字符区、屏蔽背景噪声（式 3）：
   $$L_{visual}=1-rac{1}{N}rac{1}{C}sum_{i=1}^{N}sum_{j=1}^{C}langle M	ilde F^{tea}_{i,j},,M	ilde F^{stu}_{i,j}
angle$$
   作者指出主流识别器（ABINet、MATRN、PIMNet 等）基本都是注意力式的，所以 $M$ 很容易拿到（p.4）。

### 第二层 · 语义对比损失：搬「上下文语义」，实例按字符序对齐（p.4 §3.3.2，式 4–5）

序列建模模块里每个字符位置对应一个语义向量：$H=mathrm{softmax}(QK^{	op}/sqrt{C})V$，$Q$ 是字符顺序的位置查询，$K,V$ 由视觉特征 $F$ 生成，于是 $h_i$ 就是第 $i$ 个字符的语义（式 4）。

然后在 $H^{stu}$ 与 $H^{tea}$ 之间做 InfoNCE 式对比（p.4 式 5）：以 $h_i$ 为 anchor，**正样本**是另一集合里同下标的 $h'_i$，**负样本**是 $H^{tea}cup H^{stu}$ 去掉这两个之后的所有元素；余弦相似度、温度 $	au$，$L$ 为 batch 内有效字符总长。

**关键差异**：SeqCLR 等方法用滑窗/图块**预定义**对比实例；这里实例已被注意力机制**按字符顺序对齐**，故可根据每张图的文字长度**数据依赖地**决定实例数——作者称这对任意朝向的文字更鲁棒（p.4，p.6 消融验证：比 SeqCLR 高 0.19 点）。

### 第三层 · Soft logits 损失：搬「软标签」，词级 + 序列级一起（p.4–5 §3.3.3，式 6–7）

Motivation：软标签能反映字符间相似度（易混字不该给高置信）。以往做法（WKD）只在每个时间步做 KL，是**词级**的、缺序列级监督；序列蒸馏（SKD）又把整条路径似然当原子目标，丢掉了「哪个字符confusable」的细粒度信息（p.4）。

做法：把教师分布改成两者加权（式 6）：
$$	ilde p^k_t=(1-alpha)p^k_t+alpharac{sum_{pi_t=k}prod_{t=1}^{T}p^{pi_t}_t}{sum_{pi}prod_{t=1}^{T}p^{pi_t}_t}$$
第二项就是「所有在第 $t$ 步输出字符 $k$ 的解码路径的边际概率」，等价于**所有路径对每个字符投票**，易混字自然拿到低分（p.4）。

工程近似：路径数指数级，实际用 **beam search 取 top-6**；并且**当最大路径似然低于阈值 $r=0.1$ 时，只用词级教师分布**当监督（p.5 §4.2）。损失函数为修改后教师分布 $	ilde p$ 与学生分布 $q$ 的 KL（式 7），同样用高蒸馏温度平滑分布（p.5）。

### 总损失与实现（p.5 式 8、§4.2）

$$L=lambda_1 L_{ce}+lambda_2 L_{visual}+lambda_3 L_{semantic}+lambda_4 L_{logits}$$
超参：$lambda_1=4,lambda_2=2,lambda_3=0.025,lambda_4=20$；$	au_{semantic}=0.1$、$	au_{logits}=4$（p.5）。4×TITAN X、batch 128、Adam、lr 5e-5、每 25 epoch ×0.1 衰减；作者特意说语义对比对 batch size 不敏感（p.5）。

## 结论与局限

**主结果（TextZoom，p.7 Table 5）**——三种骨干全部受益，且省算力：

| 骨干 | 方法 | 类型 | FLOPs(G) | Params(M) | Easy / Medium / Hard / **Avg** |
|---|---|---|---|---|---|
| ABINet | Bicubic | — | — | — | 77.52 / 56.98 / 42.81 / 60.23 |
| ABINet | C3-STISR | SR | 8.67 | 76.08 | 81.35 / 66.90 / 49.89 / 67.03 |
| ABINet | **Ours** | KD | **5.46** | **36.74** | 86.91 / 72.36 / 55.10 / **72.45** |
| MATRN | C3-STISR | SR | 13.07 | 83.50 | 81.90 / 68.04 / 51.08 / 67.96 |
| MATRN | **Ours** | KD | 9.86 | 44.16 | 86.91 / 73.14 / 56.96 / **73.27** |
| PARSeq | Bicubic | — | — | — | 90.36 / 75.34 / 57.11 / 75.30 |
| PARSeq | C3-STISR | SR | 6.06 | 63.17 | 84.25 / 68.25 / 50.86 / 68.83 |
| PARSeq | **Ours** | KD | **2.93** | **23.81** | 90.36 / 78.88 / 63.22 / **78.23** |

- **最反直觉的一行**：PARSeq + 双三次插值（75.30）**反超** C3-STISR 超分（68.83）。作者推测是超分模型「改动了原始 LR 图」反而伤了有强上下文建模的识别器（p.7 §4.5.1）——这是对「先超分」这条路线最狠的一击。
- **PARSeq-LTR 比 C3-STISR 高 9.4 点、省 3.13G FLOPs / 39.36M 参数**（p.7 正文），且**没有引入任何超分模块**。
- **泛化（Table 6，p.8）**：在 TextZoom 上训练的模型冻结后，直接搬到 5 个 STR 基准（图像人工降质），PARSeq-LTR 在 IC13/IC15/CUTE80/SVT/SVTP 上 **62.36/49.81/53.82/53.94/49.61** 五项全 SOTA，其中 IC15 比 C3-STISR +7.51 点；高斯模糊下掉点更慢、高斯噪声下也略优（Fig.5，p.8）。
- **其他 LTR 方案（Table 7，p.8）**：多任务学习的 PlugNet 68.60、IFR 69.04，仍低于本文 72.45，原因还是「只在像素空间学、缺多视角监督」。

**消融（均以 ABINet 为骨干，TextZoom，p.6）**：

| 表 | 对照 | Avg |
|---|---|---|
| Table 1 视觉聚焦 | 无 → 余弦 → +归一化 → +掩码 | 71.98 → 72.12(+0.14) → 72.22(+0.10) → **72.45(+0.23)** |
| Table 2 语义对比 | L2 72.19 / SeqCLR 72.26 / **Ours 72.45** | 比固定划分的 SeqCLR +0.19 |
| Table 3 soft logits | WKD 72.10 / SKD 71.62 / **Ours 72.45** | 单一层级都次优 |
| Table 4 联合 | 仅 CE 67.85 → +logits 71.87 → +semantic 71.99 → +visual 72.22 → 三者 72.45 | soft logits 一档独大 **+4.02** |

温度敏感性（p.6–7 Fig.3）：语义对比 $	au$ 小更好（放大语义向量差异，逼模型抠细微区别），过小则梯度不健康；soft logits 温度大更好（分布平滑、利于学字符相似性），过大则信息丢失——两者方向相反，最终取 0.1 与 4。

**诚实点 / 局限（作者承认 + 我观察到）**：

- **训练仍需成对 HR 数据**：推理虽不需超分，但蒸馏必须有 HR 教师 + LR-HR 配对（TextZoom 式数据），这不是「无 HR 监督」的方法（p.3、p.5）。
- **教师冻结 = 上限受教师约束**：教师若把 HR 图读错，蒸馏会把错误知识一并搬给学生；论文没有教师错误分析。
- **Table 1 的掩码并非全面受益**：avg +0.23 主要来自 Easy（86.20→86.91），而 Hard 反而从 **55.72 掉到 55.10**——「聚焦字符区」在最难样本上可能是负收益，正文没提。
- **Table 4 正文增量口径不严**：正文说视觉损失「further improved by 0.35%」，但 71.99+0.35≠72.45；0.35 实际是相对「仅 logits 的 71.87」（71.87→72.22），三者全开再 +0.23 才到 72.45（p.6）。
- **soft logits 的兜底逻辑可疑**：当最大路径似然 < 0.1（即教师最不确定的困难样本）时反而**退回只用词级监督**，恰好在最需要序列知识时把它关掉；$r$ 与 top-6 均无敏感性实验（p.5）。
- **消融增益普遍很小**：视觉/语义两档各只有 +0.1~0.35 点，且 Easy/Hard 时常反向摆动；真正的大头是 soft logits（+4.02），「三层知识」的贡献并不均衡。
- **评测面窄**：三个骨干都在 TextZoom 主战场，5 个 STR 集靠人工降质模拟 LR，全是英文词；无中文、无自然低分辨率真实分布、无实际推理时延（只报 FLOPs/Params）；Fig.1(c) 的「特征差异」动机只有定性可视化。
- **不产出 SR 图**：这是优点也是限制——无法像 [[STIRER 统一框架的低分辨率场景文本恢复与识别]] 那样给出保真指标（PSNR/SSIM），任务被严格限定为「只识别」。

## 我的疑问

- [ ] 教师冻结，那**教师读错 HR 时怎么办**？例如风格化字体、严重形变，HR 教师本身也会错（[[DEATRN v2 可变形集成注意力任意形状场景文本识别]] 指出常规集上 stylized 形近字仍是失败点），此时蒸馏是在「搬正确的位置和语义」还是在「搬 confident 的错误」？
- [ ] 阈值 $r=0.1$ 的兜底把困难样本的序列级知识关掉了——若反过来对低置信样本**更依赖**序列投票（因为词级分布本身不可信），会不会更好？这是本文设计里最值得改的一处。
- [ ] 为对齐分辨率改了学生的 **conv stride / patch size**（p.5），那 Table 5 里 FLOPs 对比（5.46G vs 8.67G）是否完全同架构口径？学生改动后是不是本身就与「原版 ABINet」不是同一个模型？
- [ ] 语义对比声称对任意朝向更鲁棒，但实验全是横向英文词；换到中文（字数量多、结构复杂）或竖排/弯曲文字，数据依赖的实例划分还成立吗？（对照 [[DEATRN v2 可变形集成注意力任意形状场景文本识别]] 的 Bezier 中心线约束）
- [ ] 与同在 MM'23 的 [[STIRER 统一框架的低分辨率场景文本恢复与识别]] 正面比会怎样？STIRER 靠「共享编码器 + 双解码器且 STR 不读 SR 图」切断误差传递，KD-LTR 靠「HR 教师蒸馏」；两者都不要串行 SR，但一篇走多任务、一篇走蒸馏，Table 5 里并没有互相对比。
- [ ] beam search 只取 top-6 路径，长单词（如 20+ 字符）的序列级投票是否足够？路径覆盖不全会不会系统性偏向高频短词？

## 关键摘录

> "Current solutions for low-resolution text recognition (LTR) typically rely on a two-stage pipeline that involves super-resolution as the first stage followed by the second-stage recognition. Although this pipeline is straightforward and intuitive, it has to use an additional super-resolution network, which causes inefficiencies during training and testing."（p.1 Abstract）

> "even though the super-resolution model attempts to imitate HR in the pixel space, there are still differences between both in the feature space."（p.2 §1）

> "aligning the student and teacher visual features with absolute numerical measures (e.g., L1 or L2) would be counterproductive."（p.4 §3.3.1）

> "Since the contrastive instances are already aligned in character order by the attention mechanism, we can obtain each contrastive instance in a data-dependent manner, which is more effective for images with arbitrary text orientations."（p.4 §3.3.2）

> "only sub-optimal results can be achieved with only one single-level loss. By contrast, by providing both local and global knowledge, the proposed method outperforms others."（p.6 §4.4.3）

> "Surprisingly, the recognition accuracy of bicubic even exceeds that of the SoTA STISR method [60] when applying a recognizer [3] with strong contextual modeling capabilities."（p.7 §4.5.1）

> "these methods are limited to pixel space learning as well as do not take into account supervision from different perspectives, such as semantic space and sequence modeling, which results in limited performance."（p.8 §4.5.3）

## 相关笔记

- [[STIRER 统一框架的低分辨率场景文本恢复与识别]] — **同为 MM'23、同主战场 TextZoom**：它走「共享编码器 + 双解码器、STR 不读 SR 图」，本文走「HR 教师蒸馏、单识别器」，两条不同的切断串行误差路线；STIRER 平均 74.8（ABINet 变体 68.6），本文 ABINet-LTR 72.45
- [[LRSTRN 恢复与识别迭代协同增强的低分辨率场景文本识别]] — ICME'26 后继工作：把 STIRER 的单向双解码升级为 N 阶段双向迭代注入、TextZoom 78.5；与本文「教师蒸馏」是两种增强低分辨率识别的思路
- [[C3-STISR 三重线索引导的场景文本超分]] — 本文 Table 5/6/7 里最强的 STISR 基线；PARSeq 下双三次都赢它 6.5 点，是本文论证「别做超分」的核心靶子
- [[TextZoom 与 TSRN 真实场景文本图像超分]] — 本文主数据集与 TSRN 基线的出处
- [[SVTR 场景文本识别]] — 同属「单一视觉模型做 STR」的路线，可与本文「直接用识别器吃 LR」对照（但 SVTR 靠 merging 压高度，本文不压）
- [[CPC 对比预测编码与 InfoNCE]] — 语义对比损失所用的 InfoNCE 原型；本文的变体是「实例按字符序由注意力对齐」，而非固定的时间步正样本
- [[MPGTSRN 多重视觉语义提示引导的场景文本超分]] — 同样在 TextZoom 上比较多种识别器口径（ABINet/MATRN/PARSeq），可对照不同论文的「识别器平均准确率」数值含义

#论文 #场景文本识别 #知识蒸馏 #低分辨率 #对比学习
