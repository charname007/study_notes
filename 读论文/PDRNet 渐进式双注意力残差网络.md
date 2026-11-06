---
created: 2026-11-06
source: E:\.dsh-paper-reader\data\渐进式注意力\Zhang 等 - 2022 - Progressive Dual-Attention Residual Network for Salient Object Detection.pdf（IEEE TCSVT 2022, vol.32, no.9；PDF 第 4 页页眉 5905、第 5 页 5906）
type: 论文
tags:
  - 论文
  - 显著性检测
  - Attention
  - 残差学习
---

# PDRNet：渐进式双注意力残差网络用于显著性目标检测

## 一句话总结

先用 **HFSM** 给出一个「定位基本对、但很糊」的粗显著图，再用 **5 个 DRM** 自深到浅**只学残差**——每一级只补「上一级还没预测对的部分」（漏检的物体碎片 + 多余的背景），逐级累加得到高分辨率结果。核心主张是"**残差式增量修正**优于"逐级特征融合"，因为后者会稀释深层的定位置信度（第 5 页）。

## 核心问题

显著性目标检测（SOD）要输出像素级显著图。论文指出当时方法存在两个互补的短板（第 1–2 页）：

1. **定位与细节不可兼得**：FCN 类方法靠深层语义定位、靠浅层纹理抠边界，但两者在融合中此消彼长。
2. **只用显著区域、浪费背景信息**：多数方法只在前景上做文章，忽略了 *"non-salient regions can provide auxiliary cues"*（第 2 页）——背景里恰恰藏着漏检的物体碎片和轮廓。
3. **逐级融合会稀释深层置信度**：作者对 top-down 逐级融合的批评是 *"these progressive fusion approaches will dilute the confident prediction of deep layers in the transmission process"*（第 5 页）。这句是全篇设计动机的支点，也是 [[浅层与深层]] 那篇概念笔记的直接延伸。

## 关键方法

整体框架（Fig.2，第 4 页）：ResNet-50 主干 + HED 式多 side-output 深监督（第 3 页）。

![PDRNet 架构图 Fig2](PDRNet%20架构图%20Fig2.png)

<small>裁自论文第 4 页 Figure 2，仅取图本体与图例，已去除图注与正文。</small>

**怎么读这张图：**

- **上排蓝色 = ResNet-50 的 Block1–5 + Pool6**，向下引出的箭头是六个初始 side-output `F₁…F₆`。
- **橙色 CEM = 上下文增强模块**，只作用在 `F₁…F₅`，输出 `F₁ᶜ…F₅ᶜ`（统一 64 通道）。
- **深蓝 HFSM** 只吃 `F₆`，产出 `F₆ʰ` → 1×1 卷积 → 粗显著图 `S₆ᵒ`（左下最右、最糊的那张）。
- **黄色 DA（= DRM 的核心）+ 绿框**：五个 DRM 串联，且**箭头方向自右向左**——这就是「渐进」的物理形态：从 Pool6 一路走向 Block1。
- **绿色箭头 = 上采样流**，⊕ = 逐元素相加：把上一级结果放大后**加**进本级残差，对应式 (13)。
- **粉色条 = 1×1 卷积**（出 `Sᵢᵒ`），**浅绿条 = 边缘预测层**。
- **最下方两行 = 六张深监督显著图 `S₁ᵒ…S₆ᵒ` + Ground truth**：可见 `S₆ᵒ` 只是模糊一团，`S₁ᵒ` 已是清晰剪影——这就是逐级细化的直观证据。
- **右下虚线框**是 DRM 的局部放大（Fig.4 的缩略），标注 `Fᵢᶜ、Fᵢⁱ₊₁、Fᵢᶠ、Fᵢᵇ、Fᵢʳᵉˢ` 的进出关系。
- **蓝虚线 = 边缘监督流，橙黄虚线 = 显著性监督流**，两者都只连前两级（第 12 页实验结论：边缘监督放前两级最合适）。

### 1. CEM（第 3 页）

3×3 卷积压缩通道降算力 → 两层卷积挖上下文，核大小 **k = {3, 3, 5, 5, 7}** 从浅到深递增 → 3×3 卷积统一到 **64 通道**。设计要点是**浅层不硬塞大感受野**。

### 2. HFSM（分层特征筛选模块，第 4–5 页，式 1–5）

解决"多个不同感受野的特征直接拼接、尺度间没有交互"的问题。

![PDRNet HFSM Fig3](PDRNet%20HFSM%20Fig3.png)

<small>裁自论文第 4 页 Figure 3，仅取图本体与图例，已去除图注。</small>

**怎么读这张图：**

- 输入 `F₆` 从左侧进入，**五条并行分支**自上而下分别是：1×1 卷积（`f₁ʰ`）、D=2、D=3、D=4 三条 3×3 空洞卷积、max-pooling（`f₅ʰ`）。
- **圆形 A = Attention Unit（通道注意力）**，只挂在 D=2/3/4 三条空洞分支上；对应式 (3) 的 `fₜᵃ = s ⊙ fₜ`。
- **浅蓝色粗箭头 = 跨尺度捷径连接**：D=2 的输出 `f₂ᵃ` 灌进 D=3 的注意力单元，D=3 的 `f₃ᵃ` 再灌进 D=4 的——**小感受野校准大感受野**（式 4 的 concat）。
- **圆形 C = 拼接**：右侧把 `f₁ʰ、f₂ᵃ、f₃ᵃ、f₄ᵃ、f₅ʰ` 五路拼起来 → 1×1 卷积 → 与 `conv(F₆)` 逐元素相加 → 输出 `F₆ʰ`（式 5）。
- 注意：**首尾两条分支（1×1 与 max-pool）不参与注意力交互**，作者明确说是为了保住原始语义（第 5 页）。

通道注意力本身（式 1–2）：GAP 得 `z ∈ ℝ^C` → 两层 FC（ReLU + sigmoid，压缩比 r = 2，C = 64）得权重 `s`。

### 3. DRM（双注意力残差模块，第 5–6 页，式 6–13）

![PDRNet DRM Fig4](PDRNet%20DRM%20Fig4.png)

<small>裁自论文第 5 页 Figure 4，仅取图本体与图例，已去除图注。</small>

**怎么读这张图（左半是"算注意力"，右半是"用注意力挖残差"）：**

- **左上**：`Fᵢᶜ` 与上一级 `Fᵢⁱ₊₁` 拼成 `Fᵢʳ`，分两路走 `k×1+1×k`（橙色）与 `1×k+k×1`（粉色）大核分解卷积 → ⊕ 相加 → 圆形 **σ** → 得到**前景注意图 `Aᵢᶠ`**（下方那张灰白掩码）。
- **圆形 **1** = 用 1 减去输入**：得到**反向注意图 `Aᵢᵇ = 1 − Aᵢᶠ`**（式 7）。
- **中部**：`Aᵢᵇ` 与 `Fᵢᶜ` 相乘后**取负**，再加回 `Fᵢᶜ` → 得 `Fᵢᵇ`（式 9）；`Aᵢᶠ` 与 `Fᵢᶜ` 相乘后加回 `Fᵢᶜ` → 得 `Fᵢᶠ`（式 8）。
- **右侧**：`Fᵢᶠ` 与 `Fᵢᵇ` 拼接（C）→ 3×3 卷积 → **SA（空间注意力）重加权** → 得残差特征 `Fᵢʳᵉˢ`（式 10–12）。
- **右上绿色 residual learning**：把上一级 `Fᵢⁱ₊₁` 上采样后与 `Fᵢʳᵉˢ` 相加 → `Fᵢᵈ`（式 13）。**这一步才是"progressive"的落点。**

四步文字版：

| 步骤 | 公式 | 作用 |
|---|---|---|
| ① 生成双注意力 | `Aᵢᶠ = σ(B₁+B₂)`，`Aᵢᵇ = 1 − Aᵢᶠ`（式 6–7，k=9） | 一张掩码认前景，一张认背景 |
| ② 双分支特征 | `Fᵢᶠ = Aᵢᶠ⊗Fᵢᶜ + Fᵢᶜ`；`Fᵢᵇ = −Aᵢᵇ⊗Fᵢᶜ + Fᵢᶜ`（式 8–9） | 背景支的负号把"藏在背景里的漏检"翻成可学的前景形式 |
| ③ 融合重加权 | `Fᵢᶠᵇ = conv(concat(Fᵢᶠ,Fᵢᵇ))`；`Fᵢʳᵉˢ = Aᵢᶠᵇ ⊗ Fᵢᶠᵇ`（式 10–12） | 空间注意力（avepool+maxpool 拼接后卷积）筛残差 |
| ④ 残差累加 | `Fᵢᵈ = Fᵢʳᵉˢ + up(Fᵢⁱ₊₁)`；`Sᵢᵒ = conv(Fᵢᵈ)`（式 13） | **只加不换**，深层置信度被旁路保住 |

## 深度理解："progressive" 到底指什么

不是课程学习、不是从易到难训练，而是**结构上的"级联逐级细化"**，三层含义绑在一起：

**① 尺度上的 coarse-to-fine：5 次串行 DRM**
`S₆ᵒ → S₅ᵒ → S₄ᵒ → … → S₁ᵒ`，每级换一层更浅、分辨率更高的 side-output，分辨率单调升高（第 2 页 "gradually reconstruct fine-resolution prediction"）。

**② 参数上的"增量修正"而非"重新预测"——这是 progressive 的实质**
每级**不重头预测**，只输出一个残差 `Fᵢʳᵉˢ`，加回上采样的旧结果。深层已确定的定位信息走旁路被保住，浅层只负责补细节。原文：*"the missing object parts and boundary details are well refined by simply adding them to the prior features"*（第 6 页）。所以标题里是 **residual** 不是 fusion。

**③ 监督也是逐级递进的**
6 个 side-output 全做深监督 + 前两级挂 6 条边缘监督（第 6 页、第 12 页）。

一句话：**progressive = 逐级、串联、只学残差**，不是简单的多尺度特征金字塔堆叠。

**与同专题上一篇 [[PAGRN 渐进式注意力引导循环网络]] 的区别**（同组同任务，2018 vs 2022）：

| | PAGRN (2018) | PDRNet (2022) |
|---|---|---|
| 注意力的角色 | **筛子**：SA+CA 串联筛选多级特征 | **挖掘器**：算前/背景权重去挖残差 |
| 层级传播 | 单向级联 + **多路循环反馈**（T=2 展开） | 单向级联 5 级，**无循环** |
| 重复利用旧预测的方式 | 顶层特征回流当可学习参数 | 旧预测**上采样后直接相加** |
| 速度 | Caffe/VGG19，未强调 | 48 FPS @ RTX2080Ti（第 2 页） |

## 损失与训练（第 6–7 页）

$$L=\sum_{i=1}^{6}\left(\mathcal{L}_B^{i}+\mathcal{L}_I^{i}\right)+\sum_{n=1}^{6}\mathcal{L}_b^{n}$$

- `𝓛_B`：像素级 BCE（式 14）
- `𝓛_I`：图像级 IoU 损失（式 15），让显著区域整体平滑高亮
- `𝓛_b`：类别平衡 BCE（式 17，α=|V⁻|/|V|、β=|V⁺|/|V|），**6 张边缘图**的监督，边缘图由 `Fˢ ∈ {F₁ᶠ,F₁ᵇ,F₁ᵈ,F₂ᶠ,F₂ᵇ,F₂ᵈ}` 经"两层卷积 + 残差"推出（式 16）

实现细节：PyTorch，单卡 RTX-2080Ti，DUTS-TR 训练，随机裁剪 + 水平翻转，多尺度 {224, 256, 288, 320, 352}，batch 8，SGD（lr 0.005 / momentum 0.9 / wd 0.0005），65 epochs 约 4 小时；测试统一 resize 到 **320×320**（第 7 页）。

## 结论与局限

- **精度**：在 5 个公开数据集上优于 **18 个** SOTA（Table I，第 8 页）；PARAMS/FLOPs 更少（Table II，第 8 页）。
- **速度**：单卡 RTX2080Ti 上 **48 FPS**（第 2 页）。
- **消融结论（第 11–12 页，最有价值的部分）**：
  - 去掉残差连接 → 加权 F-measure **掉约 1%**，作者据此断言 *"residual learning plays an important role in recovering missing details"*。**五个组件里残差连接贡献最直接。**
  - 去掉双分支融合、去掉空间重加权 → 都掉，证明"挖残差"和"筛残差"两步都要。
  - 只留前景支或只留背景支 → 都变差（Fig.10），说明互补性真实存在。
  - 注意力方法搭配不是随便选的：`A1+A1`、`A2+A2`、`A2+A1` 三种替代都劣于本文的 `A1`(算注意力)+`A2`(重加权) 组合。
  - 边缘监督：**放前两级最合适**，加更多层反而不好（"e_6fb+3d" 不如两级方案）。
- **作者承认的失败场景**（第 12 页，Fig.11）：① 多个大尺度且不连续的物体切不干净；② 小物体被大物体包住时，大的那个容易被误判；③ 标注本身带主观分歧 / 训练测试集偏置。作者给的解法是"更多样数据 + 更清晰的标注规则"——**没有在结构上给方案**。
- **未做**：只做 RGB 单图 SOD，不涉 RGB-D / 视频 / 协同显著；主干只用 ResNet-50。
- **开源情况**：本次 GitHub 检索未找到可核实的官方实现，故不附代码对照；复现只能依式 (1)–(18) 与 Fig.2–4 自行实现。

## 我的疑问

- [ ] **按式 (7)(8)(9) 代数推导，两条分支其实只差一项**：`Aᵇ = 1 − Aᶠ` 代入式 (9) 得 `Fᵇ = −(1−Aᶠ)⊗Fᶜ + Fᶜ = Aᶠ⊗Fᶜ`；而式 (8) 是 `Fᶠ = (1+Aᶠ)⊗Fᶜ`。也就是说所谓"dual attention"最终只是**同一张 `Aᶠ` 的两种缩放**（`(1+Aᶠ)` 与 `Aᶠ`），两张掩码并没有真正独立。那么消融里"w/o background"与"w/o foreground"的差异，究竟来自背景信息，还是来自 `+Fᶜ` 这个残差项的有无？论文没有做这个拆分。
- [ ] 式 (13) 的 `up(Fᵢⁱ₊₁)` 在 **i=5** 时上一级是 `F₆ʰ`（HFSM 输出）而不是 `F₆ᵈ`——Algorithm 1 用 if 分支处理了，但式 (13) 的文字表述把两者混为一谈，复现时容易错。
- [ ] 大核分解为什么取 **k=9**？与 HFSM 的 r={2,3,4} 是否做过联合搜索？
- [ ] 五级 DRM 全程 64 通道、无通道维扩张，浅层的表征容量是否是瓶颈？
- [ ] 边缘监督只在前两级有效，是因为浅层本身擅长边缘，还是因为深监督项太多削弱了主损失？
- [ ] 48 FPS 的测速是否含 5 次串行 DRM 的全部开销？推理时能否自适应地少走几级（简单图直接输出 `S₃ᵒ`）？

## 关键摘录

> "In this paper, a novel progressive dual-attention residual network (PDRNet) is proposed to exploit two complementary attention maps to guide residual learning, thus progressively refining prediction in a coarse-to-fine manner."（第 1 页，Abstract）

> "Besides, these progressive fusion approaches will dilute the confident prediction of deep layers in the transmission process."（第 5 页）

> "In this way, the missing object parts and boundary details are well refined by simply adding them to the prior features."（第 6 页）

> "In addition, we can see that the performance of 'w/o residual' is worse than our network, e.g., weighted F-measure decreases by about 1%, which proves that residual learning plays an important role in recovering missing details."（第 11 页）

> "Different from existing methods focusing on salient regions, we focus simultaneously on the foreground and background information of salient objects. We further take advantage of their complementarity to progressively refine the prediction through the residual learning structure in a coarse-to-fine manner."（第 12 页，Conclusion）

## 相关笔记

- [[PAGRN 渐进式注意力引导循环网络]] — 同专题、同任务的上一代工作：注意力当"筛子"+ 循环反馈；本篇把注意力改造成"残差挖掘器"+ 单向级联
- [[浅层与深层]] — 本文全部动机的前提：深层会稀释定位置信度，浅层才有边界细节
- [[渐进式注意力]] — 专题汇编页（本篇是其中第 3 条）
- [[transformer的2道残差连接]] — 思想同源：把"另一处算出来的信息"当增量注回去，而非替换
- [[PAN 渐进式注意力网络与视觉属性预测]] — 另一种"在多层反复施加注意力"的用法，可与本文的单向级联对照
