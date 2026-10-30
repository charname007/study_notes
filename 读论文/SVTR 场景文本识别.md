---
created: 2026-10-30
source: Zotero 条目 KUSQBPWG / PDF 附件 7LJWRT33 — E:\Zotero\storage\7LJWRT33\Du 等 - 2022 - SVTR Scene Text Recognition with a Single Visual Model.pdf（arXiv:2205.00159v2, 2022-05-23）
zotero_key: KUSQBPWG
type: 论文
tags:
  - 论文
  - Transformer
  - Attention
  - 场景文本识别
---

# SVTR：用单一视觉模型做场景文本识别

> 本笔记依据 Zotero 条目 `KUSQBPWG`（PDF 附件 `7LJWRT33`）。同一库里还有一条标题完全相同的重复记录 `EJS4ZZXA`（webpage 类型，附件 `U6AK453T`），未采用。

## 一句话总结

把场景文本识别从「CNN 提特征 + 序列模型（RNN / 自回归 decoder / 语言模型）转写」的混合流水线，压成**只有一个视觉主干**：图像切成小 patch（character components，字符组件），过 3 个高度逐级减半的 stage 做**局部/全局混合 + 合并**，最后用**一次线性并行预测**出字符序列（靠 CTC 式去重收敛）。英文打平甚至超过带语言模型的 SOTA，中文领先很多，同时更快。

## 核心问题

场景文本识别（STR）本质是「图像 → 字符序列」的跨模态映射，主流做法固定由两块拼成：视觉模型提特征 + 序列模型做转写（第 1 页）。作者把已有范式归为三类（论文 Fig. 1，第 1 页）：

1. **CNN-RNN（Fig.1a，CRNN/CTC）**：效率高，但把 2D 特征 `reshape` 成 1D 序列对形变、遮挡敏感，限制效果（第 1 页）。
2. **Encoder-Decoder 自回归（Fig.1b，SAR/NRTR）**：引入上下文、精度上去了，但**逐字符解码**导致推理慢（第 1 页）。
3. **Vision-Language（Fig.1c，ABINet/VST）**：加语言知识 + 并行预测，但往往要靠大容量模型或复杂训练范式（比如字符级遮挡预训练）来保精度，效率被拖住（第 1–2 页）。

也曾有人直接用现成 CNN 或 ViT 当唯一视觉模型，但精度打不过 SOTA（第 2 页）。

作者的诊断是：**单视觉模型能不能成立，取决于能否提出有判别力的字符特征**，而这需要同时具备两种性质（第 2 页）：

- **intra-character 局部模式**（笔画级）——区分形近字的关键；
- **inter-character 长程依赖**——类似语言知识的互补信息。

而现有主干恰好一头缺一头：CNN 擅长局部相关、不擅长全局依赖；通用 ViT 又不给局部字符模式以优先权（第 2 页）。所以需要一个**为文本定制的视觉主干**，而不是拿现成 backbone 顶上去。

## 关键方法

整体是「三阶段、高度逐级减半」的网络（§2.1，第 3 页；论文 Fig. 2，第 2 页）：

```
输入 H×W×3
  → Progressive Overlapping Patch Embedding → CC₀: H/4 × W/4 × D₀
  → Stage 1: mixing blocks × L₁ → Merging → CC₁: H/8 × W/4 × D₁
  → Stage 2: mixing blocks × L₂ → Merging → CC₂: H/16 × W/4 × D₂
  → Stage 3: mixing blocks × L₃ → Combining → C: 1 × W/4 × D₃
  → 线性分类器（37 类英文 / 6625 类中文）→ 去重 → 字符串
```

宽度**始终保持 W/4 不变**，只在高度上做下采样——这是全篇最关键的设计取向（下面 merging 一节解释为什么）。

![SVTR 架构图 Fig2](SVTR%20架构图%20Fig2.png)

<small>裁自论文第 2 页 Figure 2，仅取图本体与图例，已去除该页正文与 "Figure 2." 图注。</small>

**怎么读这张图：**

- **主链（自上而下那条横带）**：最左是输入文本图（`qbhouse`），经蓝色 `Patch Embedding` 变成一堆小方块——图例里的「Character Component」，即**字符组件**。方块的颜色深浅表示不同 stage 的特征通道数（`D₀→D₃` 逐渐变深）。
- **三个橙色竖条 = `Mixing Blocks`**，被三个虚线大框括成 `Stage 1 / 2 / 3`；每个 stage 配的 block 数就是 Table 1 里的 `[L₁,L₂,L₃]`。
- **橙色竖条右边紧接的白色竖条 = `Merging`**（Stage 3 换成 `Combing`）。注意每次 merging 之后，**方块阵列的行数减半、列数不变**——这就是「高度逐级减半、宽度恒定」在图上最直观的体现。
- **⊙（圆里一个点）= Position Embedding**，只在 patch embedding 之后加一次；**⊕ = Element-wise Add**，每次 merging/混合块内部做残差时都会出现；**圆圈里一道斜线的符号 = Activation Function**。
- **下方四个放大框（虚线引出）**，从左到右分别是三处细节：
  - **左一**：`Patch Embedding` 内部 —— 输入连着两次 `Conv + BN + 激活`（对应「渐进重叠」：两层 3×3/stride 2 叠出来）。
  - **左二（`Mixing Block` 放大）**：`LN → Local/Global Mixing → ⊕ → LN → MLP → ⊕`，外面标 `× Lᵢ`。这是**残差 + 前置 LayerNorm**的 Transformer block 结构，只是把 self-attention 换成了「局部/全局混合」。
  - **左三（`Merging` 放大）**：`Conv → LN`，把 `h×w×dᵢ₋₁` 变成 `h/2×w×dᵢ`。
  - **右一（`C` 放大）**：`Pool → Linear → 激活 → Dropout`，把高度压到 1，得到 `1 × W/4 × D₃` 的特征序列，再交给 `FC` 输出字符（右侧黄色 `qbhouse`）。
- **一句话**：整张图就是在说「**同一个 stage 结构重复三次，唯一变化是高度减半、通道加宽，宽度永远不动**」。

### 1. 渐进重叠 patch embedding（§2.2，第 3 页；论文 Fig. 3）

ViT 的标准做法是**一次性**投影：4×4 不重叠线性投影，或 7×7 / stride 4 卷积。作者改成**两层连续的 3×3/strike 2 卷积 + BN**，通道逐级升到 `D₀`。

官方实现里这就是 `PatchEmbed` 的 `mode == "pope"`（pope = **p**rogressive **o**verlapping **p**atch **e**mbedding）分支：

```python
# PaddleOCR/ppocr/modeling/backbones/rec_svtrnet.py:302-306（第一级卷积；第二级 311-319 把通道升满 embed_dim）
ConvBNLayer(
    in_channels=in_channels,
    out_channels=embed_dim // 2,
    kernel_size=3,
    stride=2,
```

动机是「虽然多花一点算力，但**特征维是渐进加上去的**，利于融合」；消融（Table 2 左半）证明有效：比两种一步投影平均高 0.75% / 2.8%（第 5 页）。

💻 完整实现见 [[SVTR 的 PaddleOCR 实现]]

### 2. 混合块：Global Mixing + Local Mixing（§2.3，第 3–4 页；论文 Fig. 4）

这是全篇的核心创新，两个块都用 self-attention，只是**感受野不同**：

- **Global Mixing**：对**所有**字符组件做标准多头自注意力 → 建立跨字符的长程依赖，同时能削弱非文本组件、强化文本组件（第 3 页）。
- **Local Mixing**：每个组件只看**一个滑窗邻域**（类比卷积），模拟笔画级局部模式。窗口大小**经验设为 7×11**（第 3–4 页）。

![SVTR 全局局部混合块 Fig4](SVTR%20全局局部混合块%20Fig4.png)

<small>裁自论文第 3 页 Figure 4，仅取图本体与 (a)(b) 标号，已去除图注。</small>

**怎么读这张图：**

- **两种图都是「组件排成的 2D 网格」**，每个小方块 = 一个字符组件；**绿色块是当前 query**，红色块是它能看到（参与注意力）的组件，白色块是看不到的。
- **(a) Global Mixing**：除绿色 query 外**整片全红** —— 注意力范围是整张图，对应「跨字符 / 跨文本-非文本」的长程依赖。
- **(b) Local Mixing**：只有 query 周围 `7×11` 的窗口内是红的、其余全白。图上那条从绿块拉到右下角的**双向虚线箭头**表示窗口是**滑动的**（和卷积一样逐位置平移），不是固定分块。
- **对照着看**：(a) 负责「字与字之间」，(b) 负责「一个字内部」——作者的说法是一个管 inter-character dependence、一个管 intra-character 笔画特征，两者互补（第 3 页）。

官方实现里，两种混合共用同一个 `Attention` 类，靠 `mixer="Local"` + 一个 `-inf` 掩码实现局部窗口：

```python
# rec_svtrnet.py:204-205（softmax 之前把窗口外打成 -inf）
if self.mixer == "Local":
    attn += self.mask
```

**⚠️ 实现注意**：它是**稠密注意力 + 掩码**，不是真正的窗口化稀疏计算（`mask_inf = paddle.full([H * W, H * W], ...)`，`rec_svtrnet.py:190`）。N 只有 200（8×25）时这点开销无所谓，但「local mixing 省算力」这个直觉在官方代码里**并不成立**——它省的是「注意力该看哪」，不是「算多少」。

💻 完整实现见 [[SVTR 的 PaddleOCR 实现]]

### 3. Merging / Combining：只压高度，不动宽度（§2.4–2.5，第 4 页）

Merging 用 **3×3 卷积、高度 stride 2、宽度 stride 1** + LN，把 `h×w×dᵢ₋₁` 变成 `h/2×w×dᵢ`。最后一个 stage 换成 Combining（高度池化到 1 + FC + 激活 + dropout），因为「高度已经很小（比如 2）时再做卷积没有意义」（第 4 页）。

为什么只压高度？原文的理由很实在：**绝大多数文本是水平或接近水平的**，压高度能给每个字符建立多尺度表示，而**不动宽度就不会把相邻字符合并进同一个组件**（第 4 页）。这是「text-customized」最具体的一处体现。

```yaml
# PaddleOCR/configs/rec/rec_svtrnet.yml:52-54
out_char_num: 25 # W//4 or W//8 or W/12
out_channels: 192
patch_merging: 'Conv'
```

### 4. 预测：一次线性、并行、去重（§2.5，第 4 页）

最后一层是 `N` 个节点的线性分类器（英文 37、中文 6625），对 `W/4` 个位置**并行**输出，同一字符的多个组件会输出重复字符、非文本输出 blank，再**自动压缩**成最终串——也就是 CTC 的 collapse。

官方配置对应 `CTCHead` + `CTCLoss` + `CTCLabelDecode`：

```yaml
# PaddleOCR/configs/rec/rec_svtrnet.yml:66,69,72
name: CTCHead
name: CTCLoss
name: CTCLabelDecode
```

论文正文只写「parallel linear prediction with de-duplication」，**没提 CTC**——对照代码才知道去重就是 CTC 解码（见「我的疑问」）。

### 5. 四个规模变体（§2.6，第 4 页；Table 1，第 5 页）

| 变体 | [D₀,D₁,D₂] | [L₁,L₂,L₃] | Heads | D₃ | 混合块排布 | 参数量 | FLOPs |
|---|---|---|---|---|---|---|---|
| SVTR-T | [64,128,256] | [3,6,3] | [2,4,8] | 192 | [L]6[G]6 | 4.15M | 0.29G |
| SVTR-S | [96,192,256] | [3,6,6] | [3,6,8] | 192 | [L]8[G]7 | 8.45M | 0.63G |
| SVTR-B | [128,256,384] | [3,6,9] | [4,8,12] | 256 | [L]8[G]10 | 22.66M | 3.55G |
| SVTR-L | [192,256,512] | [3,9,9] | [6,8,16] | 384 | [L]10[G]11 | 38.81M | 6.07G |

（Table 1 明确说**不含 rectification 模块与线性分类器**，所以和 Table 4 的参数列对不上，见「我的疑问」。）

## 结论与局限

**实验设置**（§3.1–3.2，第 4 页）：英文用 MJ + ST（合成）训练，在 IC13(857 张，已过滤非字母数字)、SVT、IIIT5K、IC15(1811 张)、SVTP、CUTE80 六个基准测试；中文用 Chinese Scene Dataset（509,164 / 63,645 / 63,646）。输入先过 ASTER 的**矫正模块**（resize 到 32×64），AdamW（wd 0.05），英文 21 epoch、中文 100 epoch，4×V100，PaddlePaddle，指标是 word accuracy。

**关键数字（Table 4，第 6 页，压成锚点，不抄全表）**：

- **SVTR-L**：IC13 **97.2** / IIIT5K **96.3** / IC15 **86.6** / CUTE **95.1**，中文 **72.1**，40.8M 参数 / 18.0ms（1080Ti）。
- **SVTR-T**：IC13 **96.3** / IC15 **84.1**，中文 67.9，仅 **6.03M** 参数 / **4.5ms** —— 论文主打「小、快、还能打」的那一档。
- **领先谁、领先多少**：英文和 lan-aware 阵营（ABINet 97.4/96.2/86.0、VST 96.4/96.3/85.4）基本打平（IC13 还差 0.2），**真正的差距在中文**：对比表里最好的 SAR（62.5），SVTR 系列领先 **5.4%–9.6%**（第 5 页原话）。所以「单视觉模型能超过 VLM」这句话，**在英文上应读作「打平」，在中文上才成立**。

**消融（Table 2/3，第 5 页，这是信息密度最高的部分）**：

- **patch embedding**（Table 2 左）：Linear 投影 92.5/72.0 → 7×7 conv 一步投影 93.0/73.9 → **渐进重叠 93.5/74.8**。比两种一步投影平均高 0.75% / 2.8%，作者称尤其对不规则文本有效。
- **merging**（Table 2 右）：不加 merging（恒定分辨率）92.4/71.8、**1.10 GFLOPs**；加渐进 merging **93.5/74.8、0.29 GFLOPs** —— 精度和算力**同时变好**，这是全文最干净的一个「免费午餐」。
- **混合块排布**（Table 3）：不打乱顺序会显著掉点。`None` 91.6/68.2 → `[L]6[G]6`（先局部后全局）**93.5/74.8**（IC13 +1.9、IC15 +6.6）；反过来 `[G]6[L]6` 只有 92.2/71.9。作者的机理解释：**先局部**能引导全局块专注长程依赖，反过来会让全局块反复去关注局部特征、角色混乱（第 5 页）。
- **反直觉的一条**：全用局部 `[L]12` 只有 **91.3/70.2**，IC13 比「一块都不用」（91.6）**还低**；全用全局 `[G]12` 92.7/73.7，也明显不如混着用。即**两种块必须配对**，单独堆哪一种都不行。

**局限（作者没有单列 limitation 一节，结论只讲优点；以下是我从原文/代码里读出来的）**：

1. **「single visual model」不完全名副其实**：§3.2 明说「SVTR uses the rectification module [Shi et al., 2019]」，即前面还挂着一个 TPS 矫正网络（第 4 页）。严格说是「矫正网络 + 识别主干」，只是论文把矫正器当外部预处理不计数。
2. **英文不够有说服力**：IC13 97.2 < ABINet 97.4、SVTP 88.4 < ABINet 89.3、CUTE 95.1 = VST 95.1（Table 4）。所以「超过 VLM」主要靠中文和速度，不是英文精度。
3. **规模不单调**：SVTR-S 的 IC13（95.7）反而**低于** SVTR-T（96.3），SVTR-B 的 SVT（91.5）也低于 SVTR-T（91.6）。论文对此一字未提——容量上去未必更好，说明这四个变体的训练/调参没有完全对齐。
4. **消融是「阉割版」模型上做的**：§3.3 明说所有消融都用 SVTR-T **去掉矫正模块和数据增强**，所以消融结论能否等量迁移到完整模型，论文没有验证（第 4 页）。
5. **只评 word accuracy**：没有字符级指标，也没有对长文本/竖排/多语种混排的讨论；中文只用了 CSD 一个数据集。

## 我的疑问

- [ ] **Table 1 与 Table 4 的参数量对不上**：T 是 4.15M vs 6.03M，差 1.88M。Table 1 注明「不含 rectification 模块与线性分类器」，但 TPS 矫正头 + 37 类线性层真有 1.88M？如果差的是 TPS，那说明**矫正模块占了 SVTR-T 约 31% 的参数**——这与「极轻量」的定位关系很大，论文却没讨论。
- [ ] **`[L]6[G]6` 到底怎么数**：Table 3 的文字说「for each stage, six global mixing blocks ... then six local」，但 Table 1 里 SVTR-T 每 stage 只有 `[3,6,3]` 个 block。官方配置是一个 12 项的扁平 `mixer` 列表，按 stage 切开后实际是 **stage1 = 3×Local、stage2 = 3×Local+3×Global、stage3 = 3×Global**。所以 `[L]6[G]6` 更像「全网共 6 local + 6 global」的口径，论文的措辞与表/代码**对不上**。
- [ ] **为什么全局部比不用混合块还差**（`[L]12` 91.3 < `None` 91.6，IC13）？如果只是「感受野太小」，应该至少持平；这个掉点更像是**窗口 mask 让 attention 退化成可学习卷积、丢掉了全局 positional 信息**，作者只给了「角色混乱」这种定性解释。
- [ ] **去重到底是 CTC 还是别的**？论文只说「automatically condensed」，官方实现是 `CTCHead + CTCLoss + CTCLabelDecode`。如果就是 CTC，那「并行线性预测」相比 CRNN 的 CTC 到底赢在哪——赢在**特征质量**而不是解码方式？这个因果关系值得单独验证（比如把 SVTR 主干 + BiLSTM 对比）。
- [ ] **输入尺寸的口径**：论文说矫正后 resize 到 32×64（第 4 页），官方配置是 TPS 输入 `[32,64]`、**输出 `[32,100]`**，主干 `img_size: [32,100]`，`out_char_num: 25 = 100//4`。也就是说识别主干真正吃的是 **32×100**，论文那句 32×64 容易让人误判。
- [ ] **`prenorm` 这个参数名是反的**：`rec_svtrnet.py:272-277` 里 `if self.prenorm:` 走的其实是 `norm(x + mixer(x))`（**post-norm**），`else` 分支才是 `x + mixer(norm(x))`（**pre-norm**）；配置里写 `prenorm: False`，实际得到的是 pre-norm，才和 Fig. 2 画的「LN 在 mixing 之前」一致。读代码时极易读反。

## 关键摘录

> "Dominant scene text recognition models commonly contain two building blocks, a visual model for feature extraction and a sequence model for text transcription. This hybrid architecture, although accurate, is complex and less efficient."（第 1 页，Abstract）

> "We argue that the single visual model based scheme is effective only if discriminative character features could be extracted. Specifically, the model could successfully catch both intra-character local patterns and inter-character long-term dependence."（第 2 页）

> "More specifically, with the embedded components, we argue that text recognition requires two kinds of features. The first is local component patterns such as the stroke-like feature. It encodes the morphology feature and correlation between different parts of a character. The second is inter-character dependence such as the correlation between different characters or between text and non-text components."（第 3 页）

> "Typically, most image text appears horizontally or near horizontally. Compressing the height dimension can establish a multiscale representation for each character, while not affecting the patch layout in the width dimension. Therefore, it would not increase the chance of encoding adjacent characters into the same component across stages."（第 4 页，解释 merging 为什么只压高度）

> "By placing the local mixing block in front of the global one, it is beneficial to guide the global mixing block to focus on long-term dependence capturing. On the contrary, switching their permutation is likely to confuse the role of the global mixing block, which may repetitively attend to local characteristics."（第 5 页，混合块排布的机理）

> "Compared with SAR, the best one among the listed ones, accuracy gains ranging from 5.4% to 9.6% are observed, which are noticeable improvements."（第 5 页，中文结果）

> "The code is publicly available at https://github.com/PaddlePaddle/PaddleOCR."（第 1 页，Abstract 末尾）

> "The three lines together reveal that sub-character, character-level and cross-character clues are all captured by the recognizer, in accordance with the claim that SVTR perceives multi-grained character component features."（第 6 页，Fig. 6 注意力可视化分析）

## 相关笔记

- [[SVTR 的 PaddleOCR 实现]] — 本篇的代码对照：论文术语 ↔ PaddleOCR 类名、窗口 mask 的写法、`prenorm` 反命名陷阱
- [[PAGRN 渐进式注意力引导循环网络]] — 同一类思路的另一半：那篇在「特征怎么组合」上加注意力级联，这篇在「特征怎么提」上换主干；都靠消融表格讲清每个组件值多少
- [[Restormer 高效高分辨率图像修复 Transformer]] — 都是对通用 Transformer 做**任务定制**：Restormer 把注意力搬到通道维省算力，SVTR 把注意力拆成局部/全局两档
- [[各种Norm层的摆放]] — 读 SVTR 代码时 `prenorm` 参数名与实际行为相反，这篇是必备背景
- [[transformer的2道残差连接]] — SVTR block 同样是「注意力残差 + MLP 残差」两道
- [[Restormer 的 MDTA 与 GDFN 实现]] — 代码笔记的写法参照，也说明了「论文图与实现不完全一致」是常态

#论文 #Transformer #Attention #场景文本识别
