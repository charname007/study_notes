---
created: 2026-11-09
source: E:\.dsh-paper-reader\data\默认专题\Wang 等 - 2020 - Scene Text Image Super-Resolution in the Wild.pdf（arXiv:2005.03341v3, 2020-08-02；代码 https://github.com/JasonBoy1/TextZoom）
type: 论文
tags: [论文, 场景文本识别, 超分辨率]
---

# TextZoom 与 TSRN：真实场景文本图像超分

> Wang, Xie, Liu, Wang, Liang, Shen, Bai（SenseTime Research + 港大 + 南大 + 阿德莱德 + 华科），ECCV 2020。页码均指本机 PDF 页码（正文 14 页 + 附录 + 参考文献，共 29 页）。这是一篇**数据集 + 方法**双贡献的开山之作：TextZoom 是第一个真实成对（real paired）场景文本超分数据集，TSRN 是配套的、面向文本的超分网络。

## 一句话总结

过去的文本超分只用 bicubic 合成低分辨率图训练，而真实模糊文字的退化复杂得多；本文用不同焦距相机拍出**真实 LR–HR 成对**数据的 TextZoom（p.1、p.5），并提出带序列残差块、中心对齐模块、梯度轮廓损失的 TSRN（p.1、p.7–9）。在 TextZoom 上，TSRN 把 CRNN 的识别率相对 BICUBIC 提升 **14.6%**、ASTER **11.1%**、MORAN **10.7%**（p.13 表 4），并明显超过 7 个 SISR 方法；用真实 LR 训练比合成 LR 训练在 CRNN 上高近 **14%**（p.15 表 5）。

## 核心问题

- **合成退化不真实**：SISR 方法都在 bicubic / 高斯模糊生成的合成 LR 上训练，学到的是"逆下采样"这种简单映射，泛化不到真实模糊文字；真实场景文字还受任意形状、光照、背景、模糊共同影响（p.1、p.3、p.5）。
- **缺真实成对数据**：此前既没有真实 LR–HR 配对的场景文本超分数据集，也没有专门面向文本的超分方法（p.1、p.3）。
- **文本是序列、且要求识别率而非视觉保真**：SISR 只追 PSNR/SSIM / 纹理重建，但文字有前后字符的语义关联，且下游评价指标是识别准确率；单个糊字只要其它字清楚，人仍能读出整词（p.4、p.8）。
- **配准误差**：LR/HR 由不同焦距拍摄、按同一矩形框裁剪，像素级错位不可避免；直接上 $L_1/L_2$ 会产生重影/伪影、甚至学错像素对应关系（p.7、p.8、p.19）。

## 关键方法

### 0. 总览：TSRN 架构（p.7，Fig. 3）

![TextZoom-TSRN 架构图 Fig3](TextZoom-TSRN%20架构图%20Fig3.png)

<small>裁自论文第 7 页 Fig. 3（按版面墨迹密度定位图形区，含上下留白）；⚠️ 未经人工目视复核，若裁切有偏移请手截替换。</small>

**怎么读这张图：** 基线是 SRResNet，两处改动（p.7）：① 网络最前面插入**中心对齐模块（Align）**；② 把原来的基本残差块全部换成 **Sequential Residual Block（SRB）**，堆叠 5 个。输入是 RGB 图与二值掩码拼成的 **RGBM 4 通道**（掩码由灰度均值简单生成，见 p.16 / A.3）；经对齐 → 浅层卷积 → 5×SRB + 残差短路 → PixelShuffle 上采样 + 卷积产出 SR 图。输出由 $L_2$（MSE）与本文提出的梯度轮廓损失 $L_{GP}$ 共同监督。

### 1. TextZoom 数据集（p.5–6、p.22–25）

- **来源**：从两个真实 SISR 数据集 **RealSR**（Canon 5D3 / Nikon D810，105/50/35/28mm）与 **SR-RAW**（SONY FE，24–240mm 共 7 档）中裁出文字区域（p.5、p.22–24）。只标注最长焦距图上的文字框，其余焦距按同一矩形裁剪，故错位不可避免；竖排/倒置图统一旋正（p.5、p.24）。
- **2× 成对设定**：HR 统一上采样到 **128×32**，LR 到 **64×16**（p.6、p.25）。四条取舍原则：不切 patch（要保字符完整）、按高度分档（>32px 收益饱和、<8px 识别率太低故丢弃）、8–32px 占多数、不做下采样以免把插值退化带进真实模糊图（p.5–6）。表 17 给了分高度识别率：16–32px 为 **79.5%**、8–16px **39.1%**、0–4px **0.3%**（p.25）。
- **难度划分**：RealSR 裁的 LR = easy；SR-RAW 且焦距 >50mm = medium；其余 = hard（p.6）。规模 17,367 训练 / easy 1,619 / medium 1,411 / hard 1,343；LR 识别率 35.7%→62.4%→42.7%→31.6%，HR 81.2%→94.2%→87.7%→76.2%（p.2 表 1）。
- **附带标注**：区分大小写的字符串（含标点）、框方向、原始焦距（p.5）；57.5% 为 ICDAR2015 9 万词表内的常见英文词（p.24）。

### 2. Sequential Residual Block（SRB，p.8、p.19）

SRB 是本文最核心的组件：在残差块里加入 **Bi-directional LSTM**，把特征图 permute/resize 成水平（及垂直）方向的序列，用 BLSTM 建"前后字符"的序列依赖后再还原回特征图，并做残差短路（p.8、p.19）。配置取 **5 个 SRB、每个 32 个 hidden units**（p.18–19 表 8/9：hidden 32 最优、block 5 饱和，加到 7 个反而掉点）。动机：文字是序列，模糊字可借上下文推断（p.4、p.8）。

### 3. Central Alignment Module（p.8、p.19–21）

用 **STN + TPS（薄板样条）** 预测控制点、对 LR 做柔性几何矫正，把文字区域拉向中心，使 $L_2$ 这类逐像素损失更合理、减轻重影（p.8、p.19）。附录 B 证明它**可即插即用**：SRResNet / LapSRN / 5×SRB 接上后平均识别率分别 51.7→53.0、53.0→53.7、56.2→57.8（p.21 表 13）；在人为放大错位的数据上增益更大（5×SRB 51.6→57.4，p.20 表 12），且优于 SR-RAW 的 CoBi Loss（p.21 表 14）。

### 4. Gradient Profile Loss（$L_{GP}$，p.9）

借用 Gradient Profile Prior（GPP）思想，用 **HR 图的梯度场当 GT**，在字符边界区间 $[x_0,x_1]$ 上约束 SR 与 HR 的梯度差（p.9 式 2）：

$$L_{GP}=E_x\|\nabla I_{hr}(x)-\nabla I_{sr}(x)\|_1,\quad x\in[x_0,x_1]$$

直觉：文字与背景对比强，锐化边界能让字形更明确；LR 的梯度曲线更宽、HR 更窄（p.9）。训练时 $L_2$ 权重 1、$L_{GP}$ 权重 $1e^{-4}$（p.10）。

> 官方实现见 https://github.com/JasonBoy1/TextZoom （p.1 声明将放出代码与模型）。本笔记未逐行核对仓库源码，**未附实现代码**，以免臆造。

### 5. 训练与评测设定（p.10）

Adam（momentum 0.9）、500 epochs、4× GTX 1080Ti；识别率用官方 released 模型 **ASTER / MORAN / CRNN** 评测，PSNR/SSIM 仅作参考（p.10、p.21）。

## 结论与局限

**结果（TextZoom，平均识别率）**

- 表 4（p.13）：TSRN ASTER **58.3%**、MORAN **54.8%**、CRNN **41.4%**，相对 BICUBIC 分别 +11.1 / +10.7 / +14.6；7 个 SISR 方法（SRCNN/VDSR/SRResNet/RRDB/EDSR/RDN/LapSRN）只带来 +2.3%~+5.8% 的增益，TSRN 则 +10.7%~+14.6%。
- 表 5（p.15，合成 LR vs 真实 LR）：TSRN 用真实 LR 训练 ASTER 58.3% vs 合成 49.7%、CRNN 41.4% vs 27.6%——**真实数据本身比换网络更值钱**（作者称近 9% / 近 14%）。
- 表 2（p.11，超分是否必要）：在 7 个通用测试集挑出的 436 张 CommonLR 上，"TSRN + 官方 ASTER" 达 80.3%，高于重新训练小图识别器 ReIm（79.3%）与在 TextZoom 上微调 ASTER（73.2%）；作者论证 SR 作为预处理比"训两个识别器"更省（p.11–12）。
- 表 6（p.16）：TSRN 仅 +0.72G FLOPs / +2.8M 参数，ASTER 推理 21.97→21.67 FPS 几乎不掉，CRNN 514.7→340.6 FPS 但精度涨 14.6%。
- 表 11（p.21）：TSRN PSNR/SSIM 在 easy 最优（25.07 / 0.8897），但 medium/hard 的 PSNR 并不突出——作者自认中心对齐会带来轻微像素位移，且"本任务里识别率比 PSNR/SSIM 重要"（p.21）。

**消融（表 3，p.12；均在 ASTER 平均识别率）**

- 5×SRB 替代 SRResNet 基本块：51.3 → **56.2（+4.9）**，是贡献最大的组件；
- 加中心对齐：56.2 → 57.8（+1.5）；
- 再加 $L_{GP}$：57.8 → **58.3（+0.5）**，幅度小但视觉更好（p.12 Fig. 5）。
- 附录 A.4（p.18）：hidden units 32 最优、4/5/6/7 个 SRB 中 5 个最好，堆更深反而降点。

**局限 / 我观察到的点**

1. **离 HR 上限还很远**：HR 图 ASTER 达 94.2%（easy）/87.7%（medium）/76.2%（hard）（p.2 表 1），而 TSRN 只有 75.1/56.3/40.1；作者结论明确写"low-resolution text recognition in the wild is far from being solved"（p.1、p.14）。
2. **语言与尺度单一**：几乎全为英文，作者把"加入中文、法语、德语"和"采集分布更合理的文字（避免极大/极小）"列为未来工作（p.14）；且只做了 2×（p.6 说其它尺度留给别的用途）。
3. **PSNR/SSIM 与识别率背离**：中心对齐改善识别却拉低 medium/hard 的 PSNR（p.21），说明逐像素保真不是好指标；但论文只把 PSNR/SSIM 放补充材料，未系统讨论 perception–distortion 权衡。
4. **对齐依赖"文字居中"这一人为假设**：HR 框是人工尽量居中的（p.25 C.3），所以中心对齐才有效；对任意版式/弯曲文字未必成立，数据集里弯曲文本本就很少（p.5）。
5. **掩码很土**：二值掩码仅由灰度均值生成（p.16 A.3），表 7 消融显示有掩码略优于无掩码，但增益有限。
6. **评测只靠 3 个 2017–2019 年的识别器**：结论建立在 ASTER/MORAN/CRNN 上，换用更强识别器（如后续 [[SVTR 场景文本识别]] 类视觉模型）时"SR 是否仍必要"未被验证。

## 我的疑问

- [ ] 用今天更强的识别器直接吃 LR，是否还需要显式 SR 预处理？表 2 的"SR 必要性"论证会怎样变化？
- [ ] 5×SRB 的 BLSTM 换成 Transformer / [[SVTR 场景文本识别]] 式全局混合，序列建模收益还能涨多少？
- [ ] 中心对齐依赖"HR 框居中"的人工先验；若换成可变形卷积对齐（见 [[C3-STISR 三重线索引导的场景文本超分]]）会不会更稳？
- [ ] 只做 2×、仅英文；多语言（尤其中文，笔画更密）与更大倍数下 $L_{GP}$ 还成立吗？

## 关键摘录

> "To this end, we propose a real scene text SR dataset, termed TextZoom. It contains paired real low-resolution and high-resolution images which are captured by cameras with different focal length in the wild. It is more authentic and challenging than synthetic data."（p.1，摘要）

> "Our results suggest that low-resolution text recognition in the wild is far from being solved, thus more research effort is needed."（p.1，摘要；p.14 结论重申）

> "Text images could not be cut into patches since the shape of the characters should maintain completed."（p.5–6，数据集构建四原则之一）

> "Different from [42], we build the in-network recurrence architecture not for detecting but for low-level reconstruction, so we only adapt the idea of building text line sequence dependence."（p.8，§4.2）

> "These methods could improve the average accuracy 2.3% ∼ 5.8%, while ours could improve 10.7% ∼ 14.6%."（p.13，§5.6）

## 相关笔记

- [[C3-STISR 三重线索引导的场景文本超分]] — 同一 TextZoom 赛道的后继（2022）：把识别线索扩成识别/视觉/语言三条，CRNN 平均 53.7% vs 本文 TSRN 41.4%，骨架仍是 5 个 TSRN block
- [[VTCTSRN 视觉文本线索迁移的场景文本超分]] — 同为 TextZoom/STISR 后继（2025），用渲染字形图 + 跨注意力，ASTER 70.8% vs 本文 58.3%
- [[TPWGAN 小波域判别器引导的场景文本超分]] — 同赛道但只改监督（小波域损失 + 频域判别器），代表"不碰先验"的另一条路
- [[TiGeSR 先文本后图像的两阶段场景文本超分]] — 两阶段"先修字形再增强整图"的更新路线（2026），仍以 TextZoom 之外的新数据集为主
- [[ASTER 注意力场景文本识别器]] — 本文三大评测识别器之一，也是"TPS/STN 矫正"与中心对齐模块的上游
- [[MORAN 多目标矫正注意力场景文本识别]] — 同为评测识别器，其偏移场矫正与本文 TPS 对齐可对照读
- [[SVTR 场景文本识别]] — 更强的新一代识别器，可用来重问"SR 预处理是否仍必要"

---

#论文 #场景文本识别 #超分辨率
