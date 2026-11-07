---
created: 2026-11-07
source: E:\.dsh-paper-reader\data\小波变换\Cui 等 - 2026 - Efficient All-in-One Image Restoration With Adaptive Frequency Enhancement.pdf（IEEE TIP, vol. 35, 2026, pp. 8080–8092；DOI 10.1109/TIP.2026.3715328）；官方代码 https://github.com/c-yn/AdaptIR（仓库仅有占位 README，无可复现代码）
type: 论文
tags: [论文, 图像复原, 小波变换, Transformer]
---

# AdaptIR：自适应频率增强的高效一体化图像复原

> Cui, Ren, Knoll，*IEEE Transactions on Image Processing* vol. 35 (2026)。下文页码均指本机 PDF 页码。

## 一句话总结

用「**动态卷积核 + 频域分组调制**」替代昂贵的退化先验分支，配一个不含注意力、只靠最大池化的卷积式 Transformer 主干和一个基于 Haar 小波的感受野扩张模块，拼出参数量少 48%、推理快 2 倍、却在五任务一体化复原上平均领先 AdaIR 0.55 dB 的 **AdaptIR**（p.1、p.6）。

## 核心问题

一体化复原（all-in-one）要求同一个模型处理去雾、去雨、去噪、去模糊、低光增强。已有路线（PromptIR 的可学习提示、Perceive-IR 借 DINO-v2、AdaIR 的二值频域旁路 + 交叉注意力）都要**显式提取退化先验**，参数与计算开销大（p.1）。

更关键的是频域这条路本身有先天缺陷：直接细化中间特征的傅里叶谱时，**由卷积定理，该操作是全局的**，导致频率学习很粗（p.2）。而退化在**同一张图内空间上不均匀**（p.2 Fig.3 的残差图），所以"全局 + 粗粒度"的频域调制根本不够用。

论文的两个观察：

1. 不同退化影响不同频带 —— 雾的差异集中在低频，去模糊延伸到高频（p.1，Fig.1 的 FFT 残差图）；
2. 退化在空间上不均 —— 需要**逐像素**的频率响应控制（p.2）。

## 关键方法

整体是 U 形三尺度结构：3×3 卷积提浅层特征 → 编码器 → **瓶颈处 RFEM → AFEM** → 解码器 + 细化阶段 → 3×3 卷积预测残差 R 与输入相加（p.3）。

### 1) AdaptIR Block：把注意力整个删掉（p.4）

$$\hat X = \mathrm{FFN}(\mathcal{N}(X')) + X',\qquad X' = \mathcal{P}\big(\mathrm{Conv}^D_{3\times3}(\mathrm{Conv}_{1\times1}(\mathcal{N}(X)))\big) + X \tag{1-2}$$

mixer 里只有一个 **3×3 最大池化**，没有任何注意力。取最大池化而非 MetaFormer 的平均池化，理由是"保留更多高频细节"（p.4 原文）。消融里这一项贡献 **+1.81 dB**（p.11）。

### 2) RFEM：用级联 Haar 小波扩感受野（p.4–p.5）

不靠堆深度/大核，而是把特征反复做 Haar 小波分解，**在每个子带上用 7×7 depthwise 卷积精修**，再逆变换回去（式 (3)–(6)）：

$$F^{i+1}_{LL},\ F^{i+1}_H = \mathcal{H}(F^i_{LL}),\qquad \hat F^{i+1}_{LL} = \mathrm{Conv}^D_{7\times7}(F^{i+1}_{LL}),\quad \hat F^{i+1}_H = \mathrm{Conv}^D_{7\times7}(F^{i+1}_H)$$

$$\hat F = \mathrm{Conv}^D_{7\times7}(F) + \mathcal{H}^{-1}(\hat F^1_{LL} + \hat F^2,\ \hat F^1_H)$$

Haar 用 stride-2 depthwise 卷积实现四个 2×2 滤波器组（D_LL 低通，D_LH / D_HL / D_HH 高通），逆变换用转置卷积（p.5）。关键洞察：**小核在缩小一半的系数图上卷积 ⇒ 等效感受野翻倍**，同时固定的小波分解还带来多频响应。深度取 3（第 4 级特征分辨率已小于 7×7 核，p.11）。

### 3) AFEM：给动态卷积核的频谱做分组调制（p.5，本文核心）

![AdaptIR 架构图 Fig4](AdaptIR%20架构图%20Fig4.png)

流程四步：

1. 一个 1×1 卷积为**每个像素**生成 K×K 卷积核 W(h,w) ∈ R^{K×K}（**跨通道共享**，只保留空间自适应 —— 这是效率的关键取舍，p.5）；
2. 对每个像素的核做 2D FFT 得谱 D(u,v)（式 (7)）；
3. 按"每个傅里叶系数到 DC 分量的 **L2 距离**"把谱**均匀切成 N 组** ${D_0, \dots, D_{N-1}}（p.5）；
4. 另一个 1×1 卷积生成逐像素注意力 A ∈ R^{H×W×N}，对每组谱逐元素调制，再 IFFT 回空间域与特征卷积：

$$A = \mathrm{Conv}_{1\times1}(Z),\qquad D'_n = A_{h,w,n}\otimes D_n \tag{8-9}$$

**为什么不是直接在空间域用动态核**：空间域的动态核只能"隐式"控制频率行为；显式在频域重标定，才能按退化类型**抑制噪声敏感的高频 / 保留光照相关的低中频**（p.5 原文）。N=2 时效果最好（p.12），整个模块只增加 **0.03 GFLOPs / 0.01M 参数**（p.11）。

**怎么读 Fig.4**：(a) 主流程，绿色块是 AdaptIR Block，瓶颈处橙为 RFEM、蓝为 AFEM，最后残差相加；(b) Block 内部 Norm→1×1→D3×3→**Pooling**→1×1→Norm，两条残差，无注意力；(c) RFEM 中蓝色漏斗是 WT/IWT，高频子带走旁路不再下分，低频继续级联，末级输出加回上一级低频；(d) AFEM 上路 1×1 生成 H×W×K×K 动态核 → FFT → 按频率距离分带（红虚线框为一组）→ 逐组 ⊗ 注意力 → IFFT；下路 1×1 生成 H×W×N 注意力。图例：⊗ 逐元素乘、⊕ 逐元素加、(·) 卷积。

![AdaptIR 动机图 Fig1](AdaptIR%20动机图%20Fig1.png)

**怎么读 Fig.1**：三列分别是退化图、GT、FFT(GT−Degraded) 的残差谱。上排是雾、下排是模糊：雾的残差能量集中在**十字中心（低频）**，模糊的残差向**高频扩散**成明显的"米字"瓣 —— 这正是"按退化类型分配频率预算"这一动机的直接证据（p.1）。

## 结论与局限

- **一体化五任务**（去雾/去雨/去噪/去模糊/低光，复合数据一次训练，p.6）：多数指标最优，相比 AdaIR **平均 +0.55 dB** 且**参数少 48%**；GoPro 去模糊 29.95 dB，**领先 AdaIR 1.83 dB**（p.6）。
- **单退化任务**（各自单独训练）：SOTS-Indoor 去雾比 MaIR **+1.71 dB** 且参数少 25%（p.7）；Snow100K 去雪 **34.27 dB**，比 MT-L V2 +0.26 dB（p.7）；真实去雨 SPA-Data 比 FourierMamba **+0.11 dB** 且参数少 **62%**（p.8）。
- **领域任务**：UHD 去雾比 ERR **+1.41 dB**（p.9）；医学 PET 合成/CT 去噪比 Restore-RWKV-light +0.24/+0.07 dB（p.9）；遥感 SateHaze1k 浓雾档比 FMambaIR +0.38 dB（p.9）。
- **效率**：去雪上比 MT-L V2 **快 11.1×**，一体化上推理速度是 AdaIR 的 **2 倍**（p.9）；直接泛化到 BSD68/Urban100/Kodak24 平均比 AdaIR +0.29 dB（p.10）。
- **消融（最有信息量，p.11–p.12）**：基线（去掉 max-pool/RFEM/AFEM）SOTS-Indoor 31.65 dB；+max-pool **+1.81 dB**、+RFEM **+2.99 dB**（贡献最大）、+AFEM **+1.51 dB**、三者合并 **+5.52 dB**。
  - **反直觉 1**：Max pooling **优于** MetaFormer 的 average pooling，+1.09 dB（33.46 dB），作者归因于保留细节（p.12）。
  - **反直觉 2**：RFEM 里把小波换成傅里叶，只有 33.21 dB，**低 3.24 dB** 反而多 0.24 GFLOPs —— 说明收益来自"小波保空间定位 + 逐级扩感受野"，不是"任何频域变换都行"（p.12）。
  - **反直觉 3**：把 7×7 depthwise 换成两个 1×1 点卷积"分别增强高低频"，参数 +0.42M、算力 +0.18 GFLOPs，PSNR **−1.42 dB** —— 再次证明增益来自**感受野**而非分支设计（p.11）。
  - AFEM 去掉频率调制是 33.63 dB，单组即到 34.53 dB，N=2 最佳（p.12）。
- **局限（作者未系统讨论，我的判断）**：
  1. "一体化"是有限度的 —— 五任务确为一次训练，但单退化与领域任务**每个数据集单独训一个模型**（p.6 原文），跨域泛化只验证了去噪三个集合（p.10）。
  2. 动态核**跨通道共享**（p.5 明说为效率而牺牲通道维），对通道差异大的退化可能欠表达，论文未对此做消融。
  3. RFEM/AFEM **只放在瓶颈**（p.3），此时分辨率已降 4×；UHD 上只给了 4090 吞吐，没有分辨率-延迟曲线。
  4. AFEM 的分组是**到 DC 的等距切分**（固定频率带，非可学习边界），这可能是最直接的改进点。
  5. 官方仓库 https://github.com/c-yn/AdaptIR 目前只有占位 README（commit dcbdc4d，2026-08-08），**无可复现代码**。

## 我的疑问

- [ ] AFEM 的"跨通道共享核"到底损失多少？可以做一个 channel-wise 动态核的对照消融，看 PSNR/参数量的曲线。
- [ ] 分组边界用"到 DC 的 L2 距离等距切分"是人为先验；若换成可学习的频率带边界（或按能量分位），能再涨多少？
- [ ] RFEM 用 Haar（最简小波）是效率选择；换成可学习小波（如 LWTformer 那类）在大模型档会不会更好？
- [ ] 瓶颈处串行 RFEM → AFEM 的顺序能否互换？两者都在频域操作，是否有冗余？
- [ ] 既然 max pooling 比 average pooling 好 1.09 dB，那"完全无注意力"这个结论在更大数据量/更大模型上还成立吗？

## 关键摘录

> "the frequency differences between hazy and clean images mainly lie in the low-frequency regions, whereas those in deblurring tasks extend to higher-frequency bands."（p.1）

> "due to the convolution theorem, this operation acts globally, leading to coarse frequency learning and suboptimal performance."（p.2）

> "we instead use max pooling to better capture high-frequency information that is essential for reconstructing fine details."（p.4）

> "This 1 × 1 convolution serves as a lightweight dynamic kernel generator."（p.5）

> "The spectrum D(u, v) is then evenly partitioned into N groups based on the L2 distance of each Fourier coefficient from the direct-current (DC) component."（p.5）

> "Without frequency modulation, the model achieves 33.63 dB (Table XIII(d)), whereas the frequency-based AFEM improves the performance to 34.53 dB when using a single group."（p.12）

> "this variant achieves only 33.21 dB PSNR, which is 3.24 dB lower than our approach."（p.12，指用傅里叶替代小波的 RFEM 变体）

## 相关笔记

- [[Adaptive Frequency]] —— 本论文 AFEM 的手写片段（1×1 卷积生成逐像素核 → FFT → 按 DC 距离分组 → 注意力 → IFFT），本篇是其完整展开
- [[关于联合小波变换的增强]] —— 本论文 RFEM 的手写片段（级联小波 → 增强 → 逆小波），并在那里留了"能否接交叉注意力"的疑问，本篇可对照回答
- [[预测LF残差提升HF，LF协同]] —— 同样是"低频继续分解、高频旁路"的小波思路，可与 RFEM 的级联结构对照
- [[Restormer 高效高分辨率图像修复 Transformer]] —— AdaptIR 的 FFN 与细化阶段沿用 Restormer 实现（p.4），且两者都以"高效"为核心卖点
- [[LWTformer 可学习小波变换的古汉字图像修复]] —— 反面取向：不满足于固定 Haar，把低通/高通核也做成可学习，用于古汉字修复；可与 RFEM 的「固定小波求效率」对照
