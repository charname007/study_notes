---
created: 2026-11-23
source: E:\.dsh-paper-reader\data\语义分割\Bell 等 - 2018 - Chromatin-associated RNA sequencing (ChAR-seq) maps genome-wide RNA-to-DNA contacts.pdf（eLife 2018;7:e27024，正文 p.1–12，方法 p.12–24；代码 https://github.com/straightlab/flypipe）
type: 论文
tags: [论文, 染色质相关RNA, 邻近连接, 剂量补偿]
---

# ChAR-seq：把 Hi-C 的思路搬到 RNA，一次画出全基因组 RNA-DNA 接触图

> Bell, Jukam, Teran, Risca, Smith, Johnson, Skotheim, Greenleaf, Straight（Stanford 生物化学/生物学/遗传学/应用物理），eLife。下文页码均指**本机 PDF 页码**（共 28 页；正文 p.1–12，方法 p.12–24）。
>
> ⚠️ 归档提醒：这篇在文献库里被放进「语义分割」专题，但内容是**染色质生物学的方法学**，与 CV 的语义分割无关（归档时错放）。本笔记放在 `读论文/`。

## 一句话总结

在**完整细胞核内**用一条特制"桥"接头，把**每条染色质相关 RNA 的 3′ 端**原位接到它邻近的**基因组 DNA** 上，测序这条嵌合分子就得到一张 **all-to-all 的 RNA→DNA 接触图**：一次实验同时回答"哪些 RNA 在染色质上"和"它们各自贴在哪里"。作者用果蝇细胞做概念验证，拿到 **2220 万条接触 / 约 16,800 条 RNA**，无偏地找回了新生转录本、X 染色体剂量补偿 ncRNA（roX1/roX2），并发现大量 snRNA/snoRNA 其实铺满基因组、还偏好 TAD 边界（p.1、p.2、p.6）。

![[ChAR-seq 方法总图 Fig1.png]]

> **怎么读这张图**（p.3–4 Fig. 1）：
> **A** 是全部实验流程——起始"十字形"是甲醛交联后的**核内**状态，蓝球是核小体，黑线是 DNA，红波浪是 RNA 的 3′ 端；随后五步：**接桥**（biotinylated & adenylated bridge + T4KQ RNA ligase，只有 5′App 端能接 RNA 3′-OH）→ **反转录**（Bst3.0 跨 DNA-RNA 交界延伸）→ **切基因组 DNA**（DpnII）→ **桥另一端接 DNA**（T4 DNA ligase）→ **二链合成、富集、测序**。注意"cut sites"红箭头在切前就画好了，说明 DpnII 位点是后续比对的锚点。
> **B** 是数据侧：read 跨过桥 → 按桥**极性**劈成 RNA 半边和 DNA 半边 → 分别比对转录组和基因组 → 合并成"这个 RNA 的结合位点"（红弧线，TSS 是转录起点）。
> **C** 是全基因组覆盖轨迹，三条分界线就是全文的**三类模式**：class I（Hsromega / chinmo / Ten-m）只在**自己所在的那一百 kb 内**有一根竖线，说明是**新生转录本**；class II（snRNA:U2 / 7SK）**每条染色体都有一层**信号；class III（roX1 / roX2）**只亮 chrX**。底部染色体小图里灰=主染色体骨架、深灰=异染色质、黑点=着丝粒。
> **D** 是 chr3L 上 850 kb 的放大，把 ChAR-seq 与 ATAC-seq、PRO-seq（±链）、RNA Pol II、H4K16ac 并排——看的是"接触位置和转录活性对不对得上"。
> **E** 是二维接触矩阵，**上**（RNA-to-DNA）与**下**（Hi-C DNA-to-DNA）形状完全不同：下面是典型的 TAD 三角，上面是交叉条纹 + 一条横带。这张对比图是全文的**护身符**——证明 ChAR-seq 信号不是 DNA-DNA 邻近的副产品。

## 核心问题

真核基因组大半被转录成非编码 RNA，其中一部分形成 RNP 复合物去结合并调控染色质：果蝇 roX1/roX2（MSL 复合物铺满雄性单条 X，乙酰化 H4K16 提升转录）、哺乳动物 Xist、HOTAIR/HOTTIP/enhancer RNA、TERRA、α-satellite（p.1–2）。

但现有手段都不够"无偏"（p.2）：

| 方法 | 能干什么 | 硬伤 |
|---|---|---|
| **ChIRP / CHART / RAP** | 用互补探针拉下**某一个已知 RNA**，再测它结合的 DNA/蛋白 | **必须先知道目标 RNA**，一次只做一个 |
| 计算预测 / 与已知因子关联 | 猜哪些 RNA 可能上染色质 | 只是猜，不是测 |
| 核分级 nuclear fractionation | 拿到"结合染色质的 RNA 池"再测序 | 只知道**是谁**，不知道**贴在哪** |

作者的立论就一句：需要**不需要先验知识**、能**同时**给出 RNA 身份与基因组坐标的方法。原文对此的表述是"the genomic targets of most chromatin-associated ncRNAs are unknown"（p.1）。

## 关键方法

### 1) 那条"桥"是全部巧思（p.2、Fig. 1—figure supplement 1）

桥是一段**双链寡核苷酸**，四个约束缺一不可：

1. **5′ 端腺苷化（5′App）** → 只有这个末端能被 **truncated T4 Rnl2tr R55K K227Q 突变体连接酶**接到 ssRNA 的 3′-OH；该突变体**不能自己腺苷化 5′ 端**，所以 RNA 只能接桥的**这一头**（单向性）。
2. **序列在酵母/果蝇/小鼠/人基因组里都不存在** → 测序时一眼认出桥；且桥**自带极性**，读穿它就能判定哪半段是 RNA、哪半段是 DNA（Fig. 1B）。
3. **另一端带 DpnII 兼容粘端** → 只接酶切后的 gDNA（单向性再锁一层）。
4. **生物素标记** → 链霉亲和素磁珠富集。

> 反过来看：这四条同时也是**四个失败模式**。少 5′App 就没有单向性（RNA 两头都能接）；少异源序列就分不清哪半是 RNA；少 DpnII 粘端就接不上 DNA；少生物素就从噪声里捞不出来。

### 2) 四天流程（p.2–3、Fig. 1A；protocol 明细 p.12–20）

1. **交联**：1% 甲醛 10 min，核内 RNA-DNA 邻近关系被"冻住"。
2. **打断 + 洗**：70 °C 4 min + Mg²⁺ 轻打断 RNA，洗掉**可溶的、没交联的** RNA——留下的才是"染色质相关"。
3. **原位接桥**：RNA 3′-OH → 桥 5′App（trT4KQ 连接酶，23 °C 过夜）。
4. **反转录固定**：Bst 3.0 能**穿过 DNA-RNA 交界**，把桥延伸成 cDNA，弱接触变共价嵌合。
5. **DpnII 切 gDNA**（果蝇中位间距 **~200 bp**）→ **桥另一端接 gDNA**（T4 DNA ligase）。
6. **二链合成**（RNase H + DNA Pol I）→ **解交联** → 超声到 ~200 bp → 链霉亲和素富集 → 建库 → **单端 152 bp** 测序。

### 3) 从 read 到接触图（p.2–3、Fig. 1B；pipeline p.20）

按桥极性劈成两半 → RNA 半边比对 FlyBase 转录组（**先强制 sense，失败再允许 antisense**）、DNA 半边比对 dm3 基因组 → 去 PCR 重复（Super Deduper）、去多重比对、去黑名单 → 同一 RNA 的所有接触汇总成它自己的全基因组图谱。**四个重复合并**后是 2220 万条高置信接触、约 16,800 条 RNA（p.2、p.20）。

### 4) 四道对照——方法学论文的价值全在这儿（p.2–4）

| 对照 | 做法 | 结果 | 排除了什么 |
|---|---|---|---|
| **RNase 对照** | 连接前加 RNase A + H | 唯一桥分子数**降 6 倍** | 桥乱贴 DNA 的非特异连接 |
| **spike-in** | 裂解后掺 0.1%/1%/10% 外源 RNA（MBP/HALO/GFP 体外转录 ~200 nt） | 假阳性随掺入量升，**10% 时仍 <0.5%** | 定量给出假接触率上限 |
| **重复性** | 重复间归一化接触数 | 高度一致 | 单次噪声 |
| **Hi-C / Mock-ChAR** | **不加桥**，改为给 DpnII 切口加生物素再连接（实质做 Hi-C） | **TAD 保留**、与已发表 CME-W1-cl8+ Hi-C 高相关；且 **RNA-DNA 谱 ≠ DNA-DNA 谱** | ① 流程没毁三维结构 ② 信号不是 DNA-DNA 邻近的影子 |

> 第二条推论值得单独记：**"DNA 挨得近"不等于"RNA 在那里"**。RNA 分布是独立信息，这正是 RNA-DNA 接触图能成为独立数据类型、而不是 Hi-C 附属品的原因（p.4、Fig. 1E）。

### 5) 顺带读出链方向（p.4）

只有 RNA 的 **3′-OH** 能接桥，所以 RNA 相对桥的朝向 = 相对转录方向的朝向 → 能判 **sense / antisense**：**85% 是 sense**，最大单一亚类是 **sense mRNA（32%）**（因为抓到新生转录本）；antisense 中 **96% 的 mRNA 是内含子来源**，其中 **64% 来自同一个 119 kb 的基因 CG42339**，提示该区域有未注释 ncRNA；非编码部分 **18% snoRNA、19% snRNA**。

### 6) 归一化与筛选：怎么从 16,812 条里挑出"真在染色质上"的（p.5–6）

1. 每条基因算 **CPKM**（每千碱基每百万 reads 的接触数）；
2. **88%（14,860/16,812）< 10 CPKM**，直接排除；
3. 剩下的 **1,952 条（12%）贡献了 83%（1,850 万）的接触**——极长尾；
4. 做 RNA-seq 拿表达量做分母：有些 RNA **接触数远超表达量**；
5. **双阈值**（>100 CPKM **且** 相对表达富集 >10 倍）圈出 **138 条**（Fig. 2E 红点），很多富集达 2–5 个数量级。

> 这一步在回答一个容易被忽略的问题：**接触多，是因为 RNA 丰度高，还是因为它真上染色质？** 不除表达量，任何"富集"都是把"多"当"准"。

## 主要结果

### 三类定位模式（p.5、Fig. 1C / Fig. 2A–C）

| 类别 | 接触图形态 | 例子 | 解释 |
|---|---|---|---|
| **I cis** | 贴在自己转录位点，形成**对角线** | Hsromega、chinmo、Ten-m、多数 mRNA | 新生转录本尚未离开模板 |
| **II trans 广泛** | 全基因组弥散 + 自身基因体一个峰 → **水平线** | snRNA:U2、snRNA:7SK | RNP 组分在全基因组工作（剪接、转录延伸） |
| **III 染色体特异** | 只铺某条染色体 | roX1、roX2 → chrX | 剂量补偿复合物靶向 |

### 用 roX1/roX2 做内部正负对照（p.6–8）

- **正对照**（雄性 CME-W1-cl8+）：roX1 富集 **7.6 倍**、roX2 **8.1 倍**（p < 10⁻¹⁰），从转录位点向外"喷"满 chrX（Fig. 3B–C）。
- **负对照**（雌性 Kc167，Sxl 抑制 msl2 翻译 → MSL2 组装不起来）：**检测不到 roX2 在 X 上的显著定位**；而其他 RNA（7SK、Hsromega）在两株细胞里高度一致（Fig. 3D）。
- **对 ChIRP-seq 的正面比较**（p.8、Fig. 3E–F）：roX1/roX2 在 ChAR-seq 里只占 0.1% / 0.36%（32,308 / 87,453 contacts），ChIRP 却各用了 ~2400 万 / ~2100 万 reads；**SNR（chrX 密集区 vs 常染色体随机）ChAR-seq 49.8 / 45.2，ChIRP 3.9 / 22.0**——reads 少两个数量级，信噪比反而全面更高。
- **分辨率**（p.8、Fig. 3G）：以 ChIRP 为参照算相关随分箱变化，出现**双相平台**——小平台 **~200 bp** 是 DpnII 位点分布偏倚（技术假象），主平台 **~25 kb** 才是真实分辨率，与 Hi-C 同量级。

### 生物学发现（p.8–11）

1. **剪接体是共转录的**：snRNA 占全部接触的 **23%**；主要剪接体 **U1/U2/U4/U5/U6 相互聚类，并与 7SK 聚在一起**——7SK 是 p-TEFb 的支架，负责解除 Pol II 启动子近端暂停，这条聚类等于把"剪接"和"转录延伸"直接连起来；snRNA 富集在**基因体/TSS 宽峰**而非增强子。（次要剪接体因丰度低没聚起来，作者如实交代。）
2. **snoRNA 是被忽视的大群体**：果蝇 200+ 种 snoRNA 在本数据里异常富集；**按富集标准筛出的染色质相关 RNA 中超过一半是 snoRNA**，多数与**异染色质**相关（推测与异染色质因子共定位于核仁），但**功能完全未知**（p.10–11）。
3. **给 RNA 贴染色质状态标签**：用"按 DpnII 位点数构造的均匀零模型 → 观测/期望 log2 → 全基因组 z-score"，再与 modENCODE 轨迹同法处理后算 Pearson 相关、层次聚类（Fig. 4D）。roX1/roX2 与 **MOF、H4K16ac、H3K36me3、JIL-1** 聚成一簇（方法自证）；其余分成**活跃染色质**（RNAPII、H4K8ac、H3K18ac）与**异染色质**（HP2、H3K9me3、HP1a）两个反相关阵营；**snRNA:U5 的变体分居两簇**（63BC vs 23D/38ABa/34A），与"U5:38ABa 在神经组织高表达、或参与可变剪接"的旧报道呼应（p.10–11）。
4. **偏好 TAD 边界**：7SK、U2 在 TAD 边界上富集 **~1.4 倍**，roX2 在 chrX 上 **~2.2 倍**（与"MSL2 高亲和位点偏好 TAD 边界"一致）；ATAC-seq 显示边界更开放 → 支持"果蝇 TAD 边界是转录活跃区"、也呼应"果蝇里活跃转录比 CTCF/cohesin 更能预测 TAD 划分"（p.10）。

## 结论与局限

**结论（作者口径）**

- 定位是"**大规模并行化的 de novo RNA 定位 assay**"，一次产出成百上千张 RNA 结合图，且不区分 RNA 是以新生转录本还是 RNP 组分身份存在（p.11）。
- 与 Hi-C **互补**：给出 RNA-DNA 相互作用矩阵，同时保留三维基因组结构（p.11）。
- 与同期 **GRID-seq** 的差别：GRID-seq 在桥两侧 19–23 bp 处酶切做尺寸选择（减少无效分子），但 19–23 bp 读长**更容易错配**；ChAR-seq 保留 20–100 bp 片段、用长单端读跨桥换映射置信度与极性。作者称两者在 roX2→chrX 检测上**能力相当**（p.11）。

**局限（作者承认 + 我的观察）**

1. **88% 的 RNA 因 <10 CPKM 被丢弃**——低丰度 ncRNA（很可能包含不少调控性 RNA）系统性缺席，"哪些 RNA 上染色质"的结论受覆盖度约束（p.6）。
2. **分辨率 ~25 kb**：能说"在哪个域"，说不出"在哪个调控元件"，比 ChIRP 的精细定位粗得多（p.8）。
3. **snoRNA 功能未知**：只证明了"在染色质上、多在异染色质"，机制是 open question（p.11）。
4. **两个细胞系、单一时点**，没有分化/刺激/肿瘤化的动态过程。
5. （我补）**"接触"≠"功能"**：邻近连接给的是空间邻近，不是因果调控。
6. （我补）**交联与捕获效率不等价**：甲醛对 RNA 二级结构、蛋白包裹敏感，不同 RNA 的捕获率不同，跨 RNA 的绝对比较要谨慎（只能做相对富集）。
7. （我补）**DpnII 偏倚**：所有接触锚定在 GATC，位点稀疏区（着丝粒、重复序列）覆盖受限——正是那个 200 bp 小平台的来源。
8. （我补）**SNR 指标只在染色体特异 RNA 上有意义**：对本来就全基因组铺开的 snRNA，"chrX vs 常染色体"的比值没有解释力。

## 我的疑问

- [ ] 化学打断的 RNA 断端未必都是**可连接的 3′-OH**（常见 2′,3′-环磷酸），这意味着实际可被捕获的分子只是断裂产物的一部分。这种"断端化学"的选择性会不会被误读成生物学差异？
- [ ] 不同长度/二级结构的 RNA 捕获效率差多少？作者用 MBP/HALO/GFP 做了假阳性标定（p.4），但**没有用不同序列/结构的 spike-in 标定捕获效率**——那 138 条 RNA 的富集倍数之间还能横向比吗？
- [ ] 雌性 Kc167 里"检测不到 roX2 定位"（p.8）——如果雌性细胞里 roX 表达量本身就极低，阴性结果究竟是"没有定位"还是"没抓到"？论文似乎没给这两株细胞里 roX 的 RNA-seq 表达量对照。
- [ ] 96% 的 antisense mRNA 来自内含子、其中 64% 挤在单个 119 kb 基因 CG42339（p.4）：这是"该基因座注释不全"，还是"反义转录普遍存在、只是这一处特别显眼"？单靠一个位点很难外推。
- [ ] TAD 边界富集只有 1.4 倍，而边界本身染色质更开放、DpnII 位点密度也可能不同。作者的随机箱对照有没有**匹配 DpnII 位点密度**？如果没有，这 1.4 倍里有多少是技术性的？
- [ ] 如果测序更深，让那 88% 的低 CPKM RNA 也进入分析，"染色质相关 RNA 的清单"会变多少？换句话说，现在的清单是"生物学边界"还是"测序深度边界"？

## 关键摘录

> "Here, we present a proximity ligation and sequencing method called Chromatin-Associated RNA sequencing (ChAR-seq) that maps all RNA-to-DNA contacts across the genome."（p.1，摘要）

> "Despite these well-studied examples, the genomic targets of most chromatin-associated ncRNAs are unknown, and the mechanisms by which these ncRNAs regulate the epigenetic and spatial organization of chromatin remain largely unexplored."（p.1，引言——全文动机就这一句）

> "ChAR-seq can be thought of as a massively parallelized de novo RNA mapping assay capable of generating hundreds to thousands of RNA-binding maps."（p.11，讨论——最准确的自我定位）

> "Despite their abundance and the their known role in RNA modification, we do not yet understand the functions of these modifications, or the implication of snoRNAs and scaRNAs chromatin association in cells (Cech and Steitz, 2014)."（p.11，作者原样承认"发现了但解释不了"，含原文笔误 "the their"）

> "We expect this approach will be particularly useful in organisms that use lncRNAs such as HOTAIR, HOTTIP and BRAVEHEART as scaffolds for ribonucleoproteins that regulate facultative heterochromatin."（p.11，展望）

## 与其他 RNA-DNA 接触方法的对照

| | **ChAR-seq**（本文） | **GRID-seq**（Li et al. 2017） | **ChIRP-seq**（Quinn et al. 2014） |
|---|---|---|---|
| 连接位置 | 核内原位 | 核内原位 | 先拉 RNA 再测 DNA |
| 读长策略 | 单端 152 bp 跨整条桥 | 酶切到 19–23 bp 做尺寸选择 | 探针富集，全部 reads 归目标 RNA |
| 一次能测几条 RNA | 全部（all-to-all） | 全部 | 一条 |
| 每条 RNA 的深度 | 低（几千到几万 contacts） | 低 | 极高（千万级 reads） |
| 主要优势 | 映射可信 + 链极性 | 无效分子少 | 单 RNA 精细定位 |
| 主要代价 | 88% 低丰度 RNA 被丢 | 短读错配风险 | 必须先知道目标 RNA |

## 相关笔记

- 本库目前以 CV 论文为主，**暂无同领域可双链的笔记**（全库 grep「染色质 / Hi-C / 邻近连接 / ChIRP / 测序」均无命中）。
- 若后续补读，最该跟着建的两篇：**GRID-seq**（Li et al. 2017，同期平行方法，p.11 有直接对比）、**in situ Hi-C**（Rao et al. 2014，方法学来源 + 本文对照实验的模板）。
- 方法论上可与 [[小波变换]] 之类"换一个变换域看同一份数据"的思路对照着看——本文换的不是频率，而是**把"分子身份"和"空间坐标"接在同一读段上**。
