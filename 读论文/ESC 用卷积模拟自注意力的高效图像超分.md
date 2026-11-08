---
created: 2026-11-08
source: E:/.dsh-paper-reader/data/默认专题/Lee 等 - 2025 - Emulating Self-attention with Convolution for Efficient Image Super-Resolution.pdf（arXiv:2503.06671v2，2025-06-30；官方代码 github.com/dslisleedh/ESC，见 p.1）
type: 论文
tags: [论文, 超分辨率, Transformer, 卷积注意力]
---

# ESC：用卷积模拟自注意力的高效图像超分

## 一句话总结

发现自注意力在层间高度重复（相似度建模 89%、特征 87%，p.3），于是**每个 ESC Block 只保留第一层自注意力，其余层换成卷积化的 ConvAttn**（共享 13×13 大核 + 逐层动态 3×3 核）；省下的显存/延迟又让作者把窗口扩到 32×32 并接上 Flash Attention。结果：大部分自注意力被卷积替代后，PSNR 反而更高、延迟和显存大幅下降，且 Transformer 的长程建模与数据扩展性都保留下来了。

## 核心问题

Transformer 做轻量超分的瓶颈**不是 FLOPs 或参数量，而是访存**（p.1）：

- 自注意力要显式物化 score 矩阵 $S=QK^T$，还伴随张量 reshaping、window masking 等 memory-bound 操作；
- SR 架构没有 patchify stem / 下采样，特征图特别大，问题被放大。论文给的例子：SwinIR-light 在 ×2 重建 HD 图时，尽管 FLOPs 少 14.5×、参数少 17×，延迟仍是 CNN（RCAN）的 **4.7×**、显存 **2×**（p.1）。

作者的目标是"效率驱动的 Transformer"：既保留自注意力的长程建模 + 输入相关加权 + 数据扩展性，又把访存代价压下去，能跑在消费级 GPU 甚至手机上。

## 关键方法

### 0. 起点：自注意力的层间重复（p.2–p.3，Fig. 2）

对 SwinIR-light 在 Urban100×4 上测 CKA 与 cosine 相似度：注意力图 $A_{avg}$ 平均 **89%**、提取特征 $F$ 平均 **87%** 的层间相似度（p.3）。这与"跨层共享注意力图"的成功经验一致（[[PFT 渐进聚焦 Transformer 单图超分]] 也是利用跨层冗余，但做法是继承注意力图、每层仍需完整 score 矩阵）。本文更激进：**只在每个 block 的第一层留自注意力，其余层换卷积**。

### 1. 整体结构（p.3，Fig. 3 / Eq. 1）

$$F_0 = \mathrm{Conv}_{3\times3}(I_{LR}),\quad F = \mathcal{H}(F_0, LK) + F_0,\quad F_s = \mathcal{S}(I_{LR}),\quad I_{SR} = \mathcal{U}(F, F_s)$$

四个部件：3×3 浅层卷积 → 深层特征提取器 $\mathcal{H}$（N 个 ESCBlock + 一个 3×3 卷积）→ 与 $F_0$ 残差相加；并行的 image-wise skip $\mathcal{S}$；上采样器 $\mathcal{U}$（classic SR 用 Sub-Pixel Convolution，见 [[像素重排：PixelUnshuffle 与 PixelShuffle]]）。**关键**：一个端到端学出来、全网络共享的 $LK \in \mathbb{R}^{13\times13\times16\times16}$ 与 $F_0$ 一起喂进 $\mathcal{H}$，所有 ConvAttn 共用它（p.3）。

### 2. ESC Block（p.3–p.4，Fig. 4 / Eq. 3）

$$
F_i^{in} = \mathrm{ConvFFN}(\mathrm{LN}(F_{i-1})),\quad
F_{i,0} = F_i^{in} + \mathrm{SelfAttn}(\mathrm{LN}(F_i^{in}))
$$
$$
F_{i,j} = F_{i,j-1} + \mathrm{ConvAttn}_j(\mathrm{ConvFFN}_j(F_{i,j-1}), LK),\ j=1..M;\quad
F_i = F_{i-1} + \mathrm{Conv}_{3\times3}(\mathrm{LN}(F_{i,M}))
$$

- 第一层是 **32×32 窗口的朴素 window self-attention**，前置 ConvFFN 让它已带局部信息，因此省掉复杂的 QKV 投影模块（p.4）。
- 之后 M 次重复的 ConvFFN + ConvAttn 残差路径；**不用 shifted window**，靠这些卷积模块补跨窗口特征（p.4）。
- 配置：ESC 为 C,N,M = 64,5,5；ESC-light 为 64,3,5；ESC-FP 为 48,5,5（p.12 §7.1）。

### 3. ConvAttn：用两种卷积"扮"自注意力（p.4，Eq. 4）

先把通道切成两份，卷积只作用在**前 16 个通道**上（$F^{att} \in \mathbb{R}^{H\times W\times16}$），其余 $F^{idt}$ 原样旁路——这是为压访存量，与只算 FLOPs 的大核 CNN 不同（p.4）：

$$
F^{att}, F^{idt} = \mathrm{Split}_{16:C-16}(F^{CF}),\quad
DK_{i,j} = \mathrm{Conv}^{up}_{1\times1}\!\left(\phi(\mathrm{Conv}^{down}_{1\times1}(\mathrm{GAP}(F^{att})))\right)
$$
$$
F^{res} = (F^{att} \circledast DK_{i,j}) + (F^{att} \circledast LK),\quad
F^{fuse} = \mathrm{Conv}^{fuse}_{1\times1}(\mathrm{Concat}(F^{res}, F^{idt}))
$$

- **共享大核 $LK$（13×13）** 负责长程交互：基于"长程交互跨层变化不大"的假设，全网络共享一个核，避开逐层大核的参数爆炸与优化困难（p.4）。
- **逐层动态核 $DK$（3×3 depth-wise）** 由 GAP + 两层 1×1 Conv + GELU 生成，负责输入相关加权（[[PointWise和DepthWise]]）；核尺寸固定 3×3，因为放大它会让核估计器的隐维 h 成倍增长（p.4）。
- 作者明确对比：跨层共享注意力图的方案**每层仍要物化完整 attention map**，而这里只有一个共享 LK；大核 CNN 常用 pixel attention 注入输入依赖，这里用逐层 DK 更贴近自注意力（p.4）。
- ESC-FP 变体把 $LK$ 深度可分离分解成 $LK_c \in \mathbb{R}^{1\times1\times16\times16}$ 与 $LK_s \in \mathbb{R}^{13\times13\times1\times16}$，把 $DK$ 零填充到 13×13 后相加（Eq. 5，p.4），专门压 FLOPs/参数；作者说明**不把 DK 合并进 LK，因为会拖慢延迟**（p.4）。

> 官方实现地址见 p.1（github.com/dslisleedh/ESC）；本次未克隆/阅读该仓库源码，故不附代码片段 —— **未找到已核验的可用实现**，不臆造。

### 4. Flash Attention：窗口开到 32×32 的前提（p.4–p.5，p.12 §6）

窗口从 16 升到 32 会让 score 矩阵膨胀、峰值显存暴涨，所以引入 [[FlashAttention 精确注意力与IO感知]]（tiling + 重算，不物化 $S=QK^T$）。Table 1（p.4）在 $X\in\mathbb{R}^{288\times288\times64}$、WS 32×32 下：朴素 SA **44.7 ms / 2717 mb** → FA **2.8 ms / 223 mb**，即 **16.0× / 12.2×**；小输入 $64^2$ 上更是 54.6× / 14.3×。

**工程坑（补充材料 §6，p.12，很值得记）**：直接套 FA 会训练发散（Table 6：FA 无 RelPos → NaN/NaN），原因是融合 kernel 挡住了相对位置偏置 RelPos；换 RoPE 也只有 32.76/0.9343；最终用 **Flex Attention**（支持自定义 score 修改 + FA）保留 RelPos，达到 33.46/0.9395。作者称这是**首个在轻量 SR 中成功应用 Flash Attention** 的工作（p.2）。窗口不再往上加，因为训练 patch 是 64×64，>32 需要 padding 会导致训练不稳定（p.5）。

## 关键图

> 下图为论文 PDF 相应页面按文字 bbox 自动裁切（200 dpi）；可在阅读器核对整页：[p.3](http://127.0.0.1:43129/paper-reader/?topic=%E9%BB%98%E8%AE%A4%E4%B8%93%E9%A2%98&name=Lee%20%E7%AD%89%20-%202025%20-%20Emulating%20Self-attention%20with%20Convolution%20for%20Efficient%20Image%20Super-Resolution&page=3&q=Visualization%20of%20the%20overall%20architecture)、[p.4](http://127.0.0.1:43129/paper-reader/?topic=%E9%BB%98%E8%AE%A4%E4%B8%93%E9%A2%98&name=Lee%20%E7%AD%89%20-%202025%20-%20Emulating%20Self-attention%20with%20Convolution%20for%20Efficient%20Image%20Super-Resolution&page=4&q=Visualization%20of%20the%20proposed%20ESC%20block)。

![ESC 整体架构 Fig.3（p.3）](<ESC Fig3 整体架构.png>)

**怎么读这张图**：左端 LR 图经 3×3 Conv 出 $F_0$；主干是一串 ESC Block，每个块都额外接收一条**贯穿全网的 Shared Large Kernel**（图例中标 "Shared Large Kernel" 的箭头），卷积结果按元素相加；主干末端再与 $F_0$ 做残差相加。下方并行的 Skip 分支直接吃 $I_{LR}$，与主干特征一起送进 Upsampler 出 $I_{SR}$。要点是"共享大核"是全网络唯一一份参数，而不是每层一份。

![ESC Block 与 ConvAttn Fig.4（p.4）](<ESC Fig4 ESCBlock与ConvAttn.png>)

**怎么读这张图**：上半部是 ESC Block 主链——LN → ConvFFN →（第一层）Self-Attn → 之后 ×M 个 ConvFFN + ConvAttn 残差块 → LN → 3×3 Conv → 与块输入相加。下半部是 ConvAttn 放大图：特征先 Split 成 16 通道的 $F^{att}$ 和剩余的 $F^{idt}$；$F^{att}$ 一支经 GAP → 1×1 Conv → GELU → 1×1 Conv 生成 3×3 动态核 $DK$，另一支与共享的 13×13 $LK$ 卷积；两支结果相加后再与 $F^{idt}$ Concat、过 1×1 Conv 融合。注意"16 → h → 16·3²"的标注就是动态核估计器的通道与核参数量。

## 结论与局限

- **主结果（Table 2，p.5；摘要 p.1）**：ESC 在 Urban100×4 达 **27.07/0.8144**，比 HiT-SRF 高 **0.27 dB**，延迟/显存少 **3.7× / 6.2×**；比 ATD-light 高约 0.1 dB 且快 **8.7×**（21.9 ms vs 189.7 ms）。ESC-light 在 Urban100×2 比 ELAN-light 高 **0.29 dB**（33.05 vs 32.76）且延迟低 **22%**（73.2 vs 94.5 ms）；ESC-FP 相比 MambaIRV2-light 在 Manga109×4 更高，FLOPs/参数少 **20% / 32%**（p.1–p.2、Table 2 p.5）。
- **消融最有料（Table 3，p.6）**：只用自注意力 33.23、只用 ConvAttn 32.91，混合的 ESC **33.46**（Urban100×2，同延迟口径）→ 两者互补；LK 改 9×9 / 17×17 分别掉到 33.42 / 33.40 → **13×13 是最优**；不共享 LK 参数量从 947K 涨到 1985K 且掉点 → 缓解大参数优化难度比逐层长程建模更关键；只有 LK 共享而无 DK 掉到 33.36 → LK 与 DK 必须协同；WS 16/24 并加层补偿也只有 33.05/33.28 → **扩大窗口确实有效**。
- **长程能力没丢（p.7）**：Fig. 6 显示 $F^{idt}$ 抓局部纹理/边缘、$F^{att}$ 抓结构；LK 的梯度扩散比 DK 更远，而 DK 的输入相关多样性更高 —— 即"LK 管长程、DK 管实例相关"的分工成立。LAM 归因（Fig. 7）中 ESC 的扩散指数 **19.80 / 24.44**，高于 SRFormer-lt(11.48/13.67)、OmniSR(14.88/16.04)、HiT-SRF(17.89/21.76)，参考像素范围最宽。
- **数据扩展性保留（Table 4，p.7）**：DFLIP 大数据训练下 ×4 Urban100 达 **27.45/0.8229**，高于 ATD-lt§ 27.18、HiT-SRF§ 27.07、SRFormer-lt§ 26.90，且延迟 120.9 ms 远低于 ATD 的 733.5 ms。
- **泛化到别的任务（p.8）**：任意尺度 SR 把 $\mathcal{U}$ 换成 LTE，×2 比 RDN+LTE 高 **0.09 dB**（35.13 vs 35.04），未见尺度到 ×30 也不掉；真实世界 SR 的 ESC-Real 在 RealLQ250/RealSRSet/RealSR/DRealSR 四个集上 CLIPIQA 全最高（Table 8，p.13）。
- **真实设备（Table 9，p.13）**：MacBook M2 Air 上 ESC-lt 124.07 ms vs ELAN-lt 318.51 ms（**降 61%**），iPhone 12 上 25.57 ms vs 38.12 ms；OmniSR/ASID-D8 编译失败、HiT-SRF 88145.85 ms / OOM —— 轻量 Transformer 在端侧"能不能跑起来"本身就是贡献。
- **放大也成立（Table 10，p.14）**：ESC 放大到 12.5M 参数（窗口 48×48、N=8, M=5, C=192, $C_{ConvAttn}$=48, h=24，DF2K + ATD 训练策略），×4 Urban100 27.89/0.8351、Manga109 32.54/0.9295，与 SwinIR/EDT/CAT-A 等大模型同档。

**局限（作者未明说，我的观察）**：

1. **ConvAttn 只作用于 16 个通道**（C=64 时仅 1/4），且大核 $LK$ 输入无关。Fig. 6(c) 自己显示 $S_{LK}$ 的输入多样性低于 $S_{DK}$ —— 所谓"实例相关加权"几乎全靠 3×3 的 DK 承担；论文没有对 16 通道这个切分比例做消融（Table 3 只动了 LK 尺寸/共享/DK/窗口）。
2. **窗口 32 的上限是训练配置带来的**（64×64 patch + padding 不稳定，p.5），并非原理上限；放大版才用 48×48。FA 的收益与"能开大窗口"绑在一起，二者对最终 PSNR 的贡献没有完全解耦（消融里 WS16/24 加层补偿仍更差，说明窗口确实值钱，但没给"同窗口无 FA"的整体网络对照）。
3. **正文数字口径不一致**：p.2 说比 ATD-light 快 **8.9×**，p.6 §4.2.1 与 Table 2 算是 **8.7×**；p.2 说 FLOPs/参数少 20%/**32%**，p.6 写 20%/**31%**；p.6 称 ESC-FP 在 Urban100×4 超 MambaIRV2-light "0.18 dB"，而 Table 2 该格是 26.90 vs 26.82 = **0.08 dB**。引用时以 Table 2/Table 3 为准。
4. **真实世界 SR 是感知–失真权衡**：p.13 §9 只强调 CLIPIQA 全胜，但 Table 8 里 ESC-Real 的 PSNR/SSIM 明显低于 DASR（RealSR 24.52/0.7503 vs 25.86/0.7617；DRealSR 26.76/0.79534 vs 28.40/0.8302）。追求保真的场景不能直接套用。
5. **补充材料 §7.1（p.12）自相矛盾**："h 对 ESC 与 ESC-FP 设为 8，而 ESC-FP 用 4"，两句指向同一变体，疑为 ESC-light/ESC 与 ESC-FP 的笔误；复现时需以代码为准。
6. 效率数字是 **RTX4090 / FP32 / 100 次重建 1280×720** 的口径（p.5）；FA 依赖 Flex Attention 这类融合 kernel 支持，换硬件/推理栈的收益需重新测（Table 9 只给了端侧 ESC 的绝对延迟，没有同设备自注意力基线）。

## 我的疑问

- [ ] 89%/87% 的层间相似度是在 **SwinIR-light** 上测的（p.3），直接迁移到别的 Transformer  backbone 是否仍成立？如果某模型层间差异大，"只留第一层自注意力"会掉多少？
- [ ] 16 通道切分与 13×13 共享核是强先验：换成"按通道组学多个共享核"或"让 LK 也带轻量输入调制"，能否在同等访存下再涨点？
- [ ] DK 固定 3×3 是为了控核估计器参数，但它承载了几乎全部实例相关性；是否存在"3×3 DK + 稀疏大核动态化"的中间方案？
- [ ] 论文没报告 ConvAttn 相对自注意力的**单层**延迟/显存拆解（只有 Table 1 的 SA vs FA 和整体 Table 2），难以判断省下的访存里 ConvAttn 与 FA 各占多少。
- [ ] ESC-Real 的两阶段训练（1M 步 L1 + 400K 步 L1/对抗/感知，p.13）开销未与对比方法对齐，感知指标的领先有多少来自训练配方？

## 关键摘录

> "Motivated by the observations of self-attention's inter-layer repetition, we introduce a convolutionized self-attention module named Convolutional Attention (ConvAttn) that emulates self-attention's long-range modeling capability and instance-dependent weighting with a single shared large kernel and dynamic kernels."（p.1 摘要）

> "we establish a design strategy that retains self-attention only in the first layer of each block while replacing the remaining layers with our proposed efficient alternative, the Convolutional Attention (ConvAttn) module."（p.2）

> "our preliminary analysis demonstrates that the sophisticated similarity modeling of self-attention ($A_{avg}$) and the resulting extracted features ($F$) exhibit a high degree of inter-layer similarities, with averages of 89% and 87%."（p.3 §3.1）

> "We do not consider window sizes above 32, as it may cause training instability due to the necessity of padding, since the training patch size is 64×64."（p.5 §4.1）

> "We verified that this instability stems from FA's fused-kernel design, which blocks the use of relative positional bias (RelPos)."（p.12 §6，补充材料）

> "Our extensive experiments demonstrate that most self-attention layers can be effectively replaced with carefully designed convolutions to improve efficiency."（p.8 §5 结论）

## 相关笔记

- [[FlashAttention 精确注意力与IO感知]] — 本文 32×32 大窗口能成立的内存理论依据
- [[FlashAttention 的 online softmax 实现]] — FA 融合 kernel 为什么不能随手加 RelPos 的代码层面背景
- [[PFT 渐进聚焦 Transformer 单图超分]] — 同样利用跨层注意力冗余，但走"继承注意力图 + 稀疏计算"的另一条路，且同样用到 32×32 窗口
- [[PointWise和DepthWise]] — ConvAttn 的 $DK$ 是 depth-wise 动态卷积、ESC-FP 的大核分解是深度可分离卷积
- [[像素重排：PixelUnshuffle 与 PixelShuffle]] — classic SR 变体 $\mathcal{U}$ 用的 Sub-Pixel Convolution
- [[AdaptIR 自适应频率增强的一体化图像复原]] / [[StarIR 空间-频率融合的卷积图像复原]] — "用卷积替代/绕开注意力"的同期思路，可对比谁真正省了访存
