---
created: 2026-11-07
source: E:\.dsh-paper-reader\data\小波变换\Li 等 - 2026 - Lightweight Temporal-Frequency Perception Sparse State Space Models for Unified Image Restoration.pdf（IEEE TIP, vol. 35, 2026, pp. 7347–7362；DOI 10.1109/TIP.2026.3709505）；单位：太原理工大学 + 中科院沈阳自动化所
type: 论文
tags: [论文, 图像复原, 小波变换, 状态空间模型]
---

# Light-MambaIR：轻量级时频感知稀疏状态空间模型统一图像复原

> Li, Li, Dou, Cheng, Tian, Tang，*IEEE Transactions on Image Processing* vol. 35 (2026)，IEEE TIP 2026.3709505。下文页码均指本机 PDF 页码（p.1–16 与期刊页 7347–7362 对应）。

## 一句话总结

把 Mamba/VMamba 的四向扫描**用 top-k 动态稀疏**剪掉冗余（算力 −31.81%），再**并行挂一个 Haar 小波高频细节增强支路**补 SSM 缺的局部细节，两条支路按通道拆分 + Channel Shuffle 融合；最后在编码器末端加一个**可学习提示（prompt）式退化语义模块**做一体化复原——结果是五个退化任务上全面超过 31 个基线，同时参数量只有次优效率方法 AirNet 的 **12.5%**（p.8、p.12）。

## 核心问题

1. **一体化复原的两难**：all-in-one 方法（AirNet、DA-CLIP、NDR、MoCE-IR）靠复杂网络或多模态大模型换性能，参数与算力爆炸，无法上移动端/嵌入式；轻量路线又几乎没有针对"多退化"的设计（p.1–2）。
2. **SSM 本身的两个毛病**（p.1）：四向扫描机制带来**计算与特征冗余**；SSM 对**局部细节**的感知弱——而图像复原恰恰是像素级、靠细节吃饭的任务。
3. 作者的两个观察落点：局部分布式细节交给**频域（小波子带）**，冗余交给**稀疏**，两者都用 top-k 同一套机制实现，天然可共享实现（p.5–6）。

## 关键方法

整体是**三阶段 U 形**网络：3×3 卷积升维 → 编码器 3 级 CTFSSM（分辨率 H×W×C → H/2 × W/2 × 2C → H/4 → H/8 × 8C）→ 瓶颈处 DSPM 产出退化提示 → 解码器 3 级 CTFSSM（带跳连）→ 3×3 卷积出残差图（p.4）。

![Light-MambaIR 架构图](Light-MambaIR%20%E6%9E%B6%E6%9E%84%E5%9B%BE%20Fig2.png)

**怎么读这张图**：(a) 主干，黄块 CTFSSM 共 6 个（编码 3 + 解码 3），蓝色箭头升采样、红色箭头降采样，最下方绿色块 DSPM 只挂在编码器末端、把 H/8×W/8×8C 的全局退化语义送到瓶颈；(b) CTFSSM 内部三段：前端 DConv3×3+Norm → **Channel Split 后 DSVSM / SWDEM 并行**（各自带 scale 残差，缩放系数分别是 α 和 β）→ Channel Shuffle 合并 → 后端 1×1→3×3→Norm→ReLU 的 FFN 式精修；(c) DSPM：CTFSSM 提全局特征 → GAP → Linear → Softmax 得权重 → 加权可学习 prompt 分量 → scale 残差 → DConv 输出提示。图例：× 逐元素乘、⊕ 逐元素加。

### 1) CTFSSM：时域 + 频域并行，通道拆分再打乱（式 4，p.4）

$$F_f = \mathrm{LN}(\mathrm{DConv}_{3\times3}(X))$$
$$F_{tff} = f_{cs}\big[\mathrm{DSVSM}(f_{cp}(F_f)),\ \mathrm{SWDEM}(f_{cp}(F_f))\big] + sX$$
$$F_b = \mathrm{ReLU}\big(\mathrm{LN}(\mathrm{Conv}_{3\times3}(\mathrm{Conv}_{1\times1}(\mathrm{LN}(F_{tff}))))\big) + F_{tff}$$

关键在于 **按通道一分为二**：一半给 SSM 学全局依赖，一半给小波支路学高频细节，功能上分工、算力上各只处理一半通道；最后 Channel Shuffle 把两半的通道信息混起来（否则两条支路通道永远不相见）。消融显示**自适应拆分 + shuffle 只带来很小的增量、参数几乎不变**（p.12）——属于"免费"设计。

### 2) DSVSM + DSSSB：四向扫描之后再做 top-k（式 5–6，p.5）

DSVSM 与 VMamba 同构：一支 Linear→SiLU 保原信息，一支 Linear→DConv→SiLU 提空间特征，两支相乘后过 Linear 降维：

$$F_{DSVSM} = \mathrm{Line}\big(\mathrm{LN}(\mathrm{DSSSB}(\mathrm{SiLU}(\mathrm{DConv}(\mathrm{Line}(F)))))\big) \odot \big(\mathrm{SiLU}(\mathrm{Line}(F))\big)$$

DSSSB（Fig.3(b)）是四向扫描（1/2/3/4 四条路径把 patch 摊平成序列，各自用离散状态空间方程建模）→ **TKSB 剔除冗余** → 门控融合 → reshape 回 2D：

$$F_{DSSSB} = GF\big(TKSB(S_6(F_{seq}))\big)$$

![Light-MambaIR 的 DSVSM / DSSSB / TKSB](Light-MambaIR%20DSVSM%E4%B8%8ETKSB%20Fig3.png)

**怎么读这张图**：(a) DSVSM 两条并行支路在末端逐元素相乘，主干里 DSSSB 之后紧跟 LayerNorm；(b) DSSSB 的四条扫描路径用彩色小块示意（同一 patch 在不同遍历顺序里序号不同），四条序列各自过 S6 状态空间方程，再进 TKSB、门控融合；(c) TKSB 的掩码生成：**k 与 f_metric 两个分支**分别产出稀疏率和重要性度量 → Top-k 选索引 → Scatter 把未入选位置概率置 0 得掩码 → 与输入逐元素相乘后输出（所以稀疏发生在 SSM **之后**，掩码复用、可插拔）。

### 3) TKSB：稀疏率**从特征统计量里算出来**（式 7–8，p.5–6）

$$F_{TKSB} = f_{scatter}\big(g_{top\text{-}k}(f_k(F_s), f_{metric}(F_s))\big) \odot F_s$$
$$f_k = \rho \times \big(1 + 0.1 \times \sigma(f_{mean}(F_s))\big),\qquad f_{metric} = f_{std}(F_s)$$

- 稀疏率 $k$ **不是固定超参**，而是由整条序列的均值经 sigmoid 后把初始率 $\rho$ 上下浮动 **±10%**；
- 重要性度量用**标准差**——std 大的通道/位置被认为是"非冗余"的，保留；
- 低于阈值的元素用 scatter 把概率置 0，再与掩码相乘。

**⚠️ 机制读法**：因为掩码在 SSM 输出之后施加，它省的是**下游（BW/后续层）的计算与显存**，并不减少 SSM 递推本身；作者报的 FLOPs 降幅（−31.81%）对应的是被置零通道后续的卷积/线性开销（p.12）。

### 4) SWDEM：低频出系数、高频被调制（式 9，p.6）

![Light-MambaIR 的 SWDEM](Light-MambaIR%20SWDEM%20Fig4.png)

**怎么读这张图**：左侧 DWT 把特征拆成 LL（低频全局）与 LH/HL/HH（水平/垂直/对角细节）。上支路 **DACGB**：LL → 1×1 Conv → Pooling → 1×1 Conv → Sigmoid，产出**逐子带的动态调整系数**；中支路 **HCRB**：三路高频各走 DConv → GELU → 与对应系数逐元素相乘 → 1×1 Conv → 加回原始子带；随后高低频各自过 TKSB → Concat → IDWT 回空间域。注意**只有低频往下传系数，高频不回灌低频**，这是与"预测 LF 残差提升 HF"那类做法（见 [[预测LF残差提升HF，LF协同]]）不同的取向。

$$[cA, cH, cV, cD] = f_{DWT}(F_f),\quad F_{SL} = \mathrm{Sigmoid}(\mathrm{Conv}(\mathrm{GAP}(\mathrm{Conv}(cA))))$$
$$F_{SH} = \mathrm{Concat}\big(TKSB(\mathrm{Conv}(F_{SL} \odot (\mathrm{GELU}(\mathrm{DConv}(cH/cV/cD)))) + (cH/cV/cD))\big)$$
$$F_{SWDEM} = f_{IWT}\big(TKSB(\mathrm{Conv}(cA)),\ F_{SH}\big)$$

一句话：**低频当"调节旋钮"，高频被旋钮放大后再小波重构**。固定 Haar 小波（不做可学习），换取效率——与 [[LWTformer 可学习小波变换的古汉字图像修复]] 的可学习小波形成对照。

### 5) DSPM：可学习 prompt 而非文本/CLIP 提示（式 10，p.6）

$$\omega = \mathrm{Softmax}(\mathrm{Line}(\mathrm{GAP}(\mathrm{CTFSSM}(F))))$$
$$P = \mathrm{DConv}\Big(\sum_{c=1}^{N} \omega P_c + s \cdot \mathrm{CTFSSM}(F)\Big)$$

一组可学习参数 $P_c$ 当"退化原型"，GAP→Linear→Softmax 生成权重 $\omega$ 对它加权，再加 scale 残差。放在**编码器末端（分辨率已降 8×）**，理由写得直白：退化语义是**全局**属性，瓶颈处算最省（p.6）。t-SNE 可视化显示 DSPM 输出能按退化类型分簇（Fig.18，p.13）。

## 结论与局限

- **五任务一体化**（Rain100L / Snow100K / SOTS / SHIQ / LOL 混合训练，p.7–8）：全部指标最优，相比次优 MoCE-IR **平均 +1.83 dB PSNR、+1.2% SSIM**，各任务 PSNR 增益 1.18 / 1.68 / 1.63 / 2.5 / 1.71 dB（p.8）；LPIPS 同样最好（Fig.10）。
- **效率是最大卖点**（正文 p.10–11，Table VIII 排在 p.12）：参数仅 AirNet 的 **12.5%**；相对 PromptIR 参数 −96.85%、FLOPs −98.31%、运行时间 −48.97%、显存 −69.60%；相对 MoCE-IR 分别 −90.24% / −93.54% / −44.39% / −21.79%。Jetson Orin NX 上参数/FLOPs/显存与 3090 几乎一致，TensorRT 加速后平均 **0.15 s**（p.10–11）。
- **单退化设定**（各自单独训练，p.9–10）：去雨比 DRSformer **+0.02 dB / +0.1%**（几乎打平）、去雪比 LMQFormer +0.27 dB / +0.6%、去雾比 DehazeFormer +0.12 dB / +0.3%、高光去除比 DHAN +1.93 dB / 比 TSHRNet +1.5% SSIM、低光比 URetinex-Net +1.91 dB / +0.9%。⇒ **一体化与专修差距已经很小**，这是本文最有价值的信息之一。
- **消融（Table IX–XVI，p.12–14）**：
  - 结构：V1（标准 VSS + 无 DSPM + 无 Split&Shuffle）→ 全量模型 **+2.93 dB / +3.8% SSIM**，参数只多 **0.045 M**，FLOPs **−34.98%**——加了模块反而更省算力。
  - DSVSM：只用 SS2D 作基线，加 TKSB **FLOPs −31.81% 且 PSNR 还涨**；加门控融合几乎不增参/算力；两者同用 **+0.42 dB**。
  - SWDEM：TKSB 单独带来 **−7.5% FLOPs**；HCRB、DACGB 依次加，性能稳步涨、代价很小。
  - 稀疏策略：top-k 优于均匀稀疏与随机稀疏；**自适应 $k$ 优于固定稀疏率**（p.13）。
  - 分辨率：256/512/1024 三种输入下 FLOPs、时长、显存均优于 MoCE-IR（p.14）。
  - 轻量模块对照：CTFSSM 优于 EfficientVMamba 的 EVSS 与 MobileMamba Block（p.14）。
- **扩展退化**（p.10）：在雨滴、阴影上微调后仍能复原，但**原五任务的性能会随退化种类增多而轻微下降**——作者自己承认这是代价。
- **局限**（作者承认 + 我的判断）：
  1. 作者自承：**极低分辨率或严重退化**的图像上效果不理想，未来要补数据（p.14）。
  2. 一体化 vs 单退化的增益并不对称——去雨上只领先 0.02 dB（p.10），说明在"数据分布接近"的任务上该架构的天花板与专修模型相同，真正的领先来自跨任务统一部署而非绝对精度。
  3. DSPM 的 prompt 是**纯可学习参数**，没有文本/CLIP 语义监督（对照 DA-CLIP），所以它学到的是"隐式聚类中心"而非可解释的退化语义；$N$ 取多少、退化种类增加时是否线性膨胀，论文未报。
  4. 稀疏率 $f_k = \rho(1+0.1\sigma(\cdot))$ 只在 ±10% 内浮动，本质接近**固定稀疏率**，但消融里"自适应 k 优于固定 k"——两者的对比基线具体取了哪些固定值，只有表格没有正文说明（p.13）。
  5. 只报 PSNR/SSIM/LPIPS，"轻量"的验证只有参数/FLOPs/单卡耗时，缺**端侧真实吞吐**（Jetson 只说了与 3090 "几乎一致"和 0.15 s，未给 FPS 曲线，p.10–11），也没给 ARM CPU 等非 GPU 平台结果。

## 我的疑问

- [ ] TKSB 的掩码作用在 **SSM 之后**，那么"省计算"到底省在哪一层？如果把它前移到 SSM 输入侧（先稀疏再扫描），能否省掉递推本身、效果会不会掉？这决定它是"稀疏"还是"剪枝"。
- [ ] $f_{metric} = f_{std}(F_s)$ 用**标准差**当重要性度量，等于假设"波动大的通道信息量大"。对雨/雪这类**空间稀疏但幅度极大**的退化（局部雨线）是否会误杀？论文对五种退化统一用一个度量，没有分退化讨论。
- [ ] 稀疏率只在 $\rho$ 上下 ±10% 浮动，为何还要引入 $f_{mean}$ 这条依赖（多一次全局归约）？做成纯固定率、把省下的实现复杂度换成更细的 top-k 分组会不会更好？
- [ ] SWDEM 里 LL 只负责"出系数"、自己只过 TKSB+Conv，不做细节增强；反过来的方向（用高频预测低频残差）在 [[预测LF残差提升HF，LF协同]] 里有讨论，两种信息流向哪个更适合复原任务？可以做一个 swap 消融。
- [ ] DSPM 挂在编码器末端（H/8），对**空间不均匀**的退化（如局部阴影、局部高光）而言 8× 下采样的语义图够不够精细？与 AdaptIR 那类"逐像素频率调制"相比，粒度差一个量级。
- [ ] 论文声称 31 个基线、5 任务全面 SOTA，但每个任务只用**一个**测试集（Rain100L / Snow100K / SOTS / SHIQ / LOL，p.7），真实雨/雪泛化只给了 Fig.11 的少量样例，缺少 SPA-Data、Rain13K 等更强的检验。

## 关键摘录

> "However, their multi-directional scanning mechanism may introduce computational and feature redundancy, failing to satisfy lightweight deployment requirements."（p.1）

> "The frequency-domain sparse wavelet detail enhancement module utilizes wavelet transformation and convolution operations to extract and enhance details in different directions, and then uses a top-k mechanism to perform sparse processing."（p.1）

> "Only top-k values within the specified range are retained. For other elements with scores below the top-k threshold, we use the scatter function to replace their probability at the given index with 0."（p.5）

> "As the degradation type semantics belong to the global features of an image, we perform degradation semantic extraction at the end of the encoder, which also reduces the computational complexity of the network."（p.6）

> "Compared to the second-most-efficient method AirNet, our method requires only 12.5% of its parameters, which strongly demonstrates the high efficiency of our method."（p.8）

> "Compared with the current state-of-the-art efficient image restoration method MoCE-IR, our method also decreases the values of the four efficiency metrics by 90.24%, 93.54%, 44.39%, and 21.79%, respectively."（p.10）

> "its performance remains less than ideal for images with extremely low resolution or severe degradation."（p.14）

## 相关笔记

- [[AdaptIR 自适应频率增强的一体化图像复原]] —— 同是一体化五任务 + 频域 + 轻量；但那篇走**卷积 + 逐像素动态核频谱调制**、无 SSM，本篇走**SSM + 固定 Haar 小波 + top-k 稀疏**，可对照"频域到底该调制在哪一层"
- [[LWTformer 可学习小波变换的古汉字图像修复]] —— 把 DWT 做成可学习；本篇用固定 Haar 换效率，是同一权衡的两端
- [[StarIR 空间-频率融合的卷积图像复原]] —— 同样是空间-频率双支路融合（window FFT ⊗ 空间注意力），本篇用 Channel Split & Shuffle 做融合，可比较两种融合方式的开销
- [[Restormer 高效高分辨率图像修复 Transformer]] —— 本文后端 FFN 与主流复原骨干的写法来源，也是 Table I 中的一体化基线之一
- [[小波变换]] —— 概念索引页，本文 SWDEM 是"小波子带 + 动态系数"的一个实例
