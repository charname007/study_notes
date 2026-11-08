---
created: 2026-11-08
source: CVPR 2026；E:\.dsh-paper-reader\data\默认专题\Zheng 等 - 2026 - DVAR Dynamic Visual Autoregressive Modeling for Image Super-Resolution.pdf；https://github.com/YuZheng9/DVAR
type: 论文
tags: [论文, 超分辨率, 图像复原, Transformer, 多尺度, 自回归]
---

# DVAR 动态视觉自回归单图超分

## 一句话总结
CVPR'26（南理工 PCA Lab + 南大）：把 VAR 的「按尺度预测」改造成**与输出尺寸解耦**的动态框架——用 LQ 图导出动态起始 token map 取代固定的 1×1 start token，再把一条**典范尺度序列按比例缩放**到任意目标尺寸（canonical scaling dynamic），并用动态采样调度器（DSS）缓解 teacher forcing 的 exposure bias；据作者称是第一个让 VAR 具备尺寸灵活性的框架，纯自回归路线在感知质量上超过现有 Real-ISR 方法（第 1 页）。

## 核心问题
真实世界超分的输入尺寸多变，而 vanilla VAR 有两个从图像生成任务继承来的静态假设（第 2、4 页）：

1. **尺寸依赖（size-dependency）**：生成过程必须从一个固定的 1×1 静态 start token 起步，但终点尺寸不同（256² vs 512²），于是模型被迫为每个分辨率背一套互不相容的绝对缩放序列（图 1(a) 举例：中间尺度可能是 8² 与 13²），**一套权重无法泛化到可变尺寸**。作者强调瓶颈不在 VQ-VAE tokenizer——全卷积编解码 + 最近邻量化本身天然尺寸自适应，而在自回归生成过程本身（第 2 页）。
2. **Exposure bias**：teacher forcing 用 GT 前缀训练，推理却要吃自己可能出错的预测，误差逐级累积（第 4 页）。

更根本的追问是：为什么一定要从 1×1 起步？作者认为这是通用图像生成（"from nothing"，从文本等抽象信号合成）的遗留约束；而图像复原是**refinement task**（"from something to something better"），LQ 输入本身就提供丰富的结构与内容先验，信息贫乏的 1×1 start token 不但多余而且次优（第 2 页）。

> 原文（第 2 页）："Image Restoration (IR), however, operates on an entirely different principle. IR is fundamentally a refinement task. ... The LQ input itself provides a wealth of structural and content priors, rendering the rigid, information-scarce 1×1 start token not only unnecessary but suboptimal."

## 关键方法

### 整体流程（第 3 页，图 2）
LQ 图 $x_{LQ}$ 先过**退化鲁棒编码器** $E_{dr}$ 得特征 $f_{LQ}$，一物两用：① 经卷积块生成 VAR Transformer 的条件前缀 token；② 经动态量化得多尺度 token 序列 $R^{LQ}=\{r^{LQ}_1,\dots,r^{LQ}_K\}$，其中 $r^{LQ}_1$ 就是**动态起始 token map**。$R^{LQ}$ 再经 VAE 解码器 $D$ 得到初始重建，由此导出文本 prompt。Transformer 在「前缀 + 文本 prompt + 目标尺寸嵌入」联合条件下，从 $r^{LQ}_1$ 自回归补完剩余尺度；训练时由 DSS 注入模拟误差。

![[DVAR 框架总览 Fig2.png]]

**怎么读这张图**（来源：论文图 2 / 官方仓库 assets/DVAR.png）：左侧 $x^{LQ}$ 进 Robust Encoder 出 $f^{LQ}$，分两路——上路经 Conv 变成 conditioning prefix tokens（青色），下路进 Dynamic Scale Quant 得到 $r_1^{LQ}, r_k^{LQ}$；$r_1^{LQ}$ 与初始重建（Initial Result）+ LLM 生成的 Text 一起喂给 VAR Transformer。中间那一串 $r_1 \to \hat r_1 \to r_2 \to \dots \to r_k$ 是**下一尺度预测**：每一级把上一级的 token map 插值放大后再预测下一级，逐级细化。EMA Book / Update / Sample & Random Choice 那条支路只在**训练期**生效（图中 "Only Training"），就是 DSS 的 count book；最终 $r_k$ 经 VQ-VAE Decoder 出 Result。图例里 Conditional tokens 与 Interploted tokens 区分「条件前缀」与「被插值上采样的前级预测」。

### 1. 典范缩放动态 canonical scaling dynamic（第 4–5 页，式 5、算法 1）
在基准分辨率 $(H_{base}, W_{base})$ 上定义**一条**典范尺度序列 $\{(h_k,w_k)\}_{k=1}^K$；对任意目标 $(H,W)$，按比例重缩放得到该尺寸专属序列（式 5，第 4 页）：

$$\tilde h_k = \mathrm{round}\Big(\frac{H}{H_{base}} h_k\Big),\quad \tilde w_k = \mathrm{round}\Big(\frac{W}{W_{base}} w_k\Big)$$

关键是**相邻步之间的相对缩放比在所有分辨率上保持一致**，于是单模型可处理可变甚至训练时没见过的尺寸与长宽比。论文脚注（第 4 页）：基准取 256×256，典范序列为 $(1,2,3,4,5,6,8,10,13,16)$。

配套两点设计洞见（第 5 页）：① 利用「生成 vs 复原」的差异，把 LQ 当作整个生成过程的**稳定结构锚点**；② 从「每个分辨率背一套固定 schedule」改成「学一条统一相对缩放规则」，早期尺度快速稳定全局结构、后期尺度专心细化纹理边缘（图 4 的多尺度可视化佐证）。

💻 官方实现见 [[代码细节笔记/DVAR 的动态尺度与采样调度器实现]]。核心一行（`models/quant.py:364-366`，仓库 `YuZheng9/DVAR@7dac046`）：

~~~python
self.v_patch_nums = [
    (round(H * i / 16.0), round(W * i / 16.0)) for i in self.v_patch_nums_copy
]
~~~

把配置里的 `patch_nums: [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]`（`config/inference.yaml`）当作**相对 16 的比例**，按当前特征图 H/W 动态生成各级 patch 尺寸——这正是式 5 的代码形态，也是「一条规则适配任意尺寸」的落点。

### 2. 动态采样调度器 DSS（第 5 页，算法 2）
经典 scheduled sampling 要额外跑一次完整前向拿模型预测，太贵。DSS 的洞察是：**VQ-VAE 的视觉 token 在码本空间里有几何邻近语义**（最近邻查找保证视觉相似的特征在码本中也相近），而 NLP 的离散 token 没有这种拓扑（第 5 页）。于是可以用 GT token 的 k-NN 来近似模型「可能犯的错」，一次近乎零成本的查表即可。

做法：为每个尺度 k 维护一个 **count book** $C_k \in \mathbb{R}^V$，记录模型历史上选中「GT token 第 v 近邻」的概率分布（`RankCount` 统计后归一化），每步用 **EMA** 更新；前向时按 $C_k$ 从 GT 的近邻集里采样一个 token，以概率 p 替换 GT 作为下一步输入——等于给训练注入逼真的「预测误差」，弥合训练–推理分布差（第 5 页）。

💻 代码印证（`models/dvar.py:725-736`、`models/quant.py:589-597`，同 commit）：

~~~python
rank_hit_V = pred_ranks.reshape(-1).bincount(minlength=self.vocab_size).float()
self.ema_rank_hit_SV[si].mul_(0.99).add_(rank_hit_V.mul(0.01))   # EMA 更新 count book
~~~

~~~python
sorted_idx = torch.argsort(d_no_grad, dim=1, descending=False)      # 码本距离排序 = k-NN 秩
~~~

注意论文称 "negligible overhead"，而 released 推理代码里还带 `scale_drop_rate: 0.3` 的随机尺度 dropout（`dvar.py:653-665`），属于训练期技巧，论文正文未展开。

### 3. 架构细节（第 6 页）
- **退化鲁棒编码器**：按 SUPIR 的编码器适配策略微调 VAR 的 VAE 编码器，但取**倒数第二层**特征作条件信号（最后一层过度压缩、丢内容与退化信息）；损失 $\|D(E_{dr}(x_{LQ}))-x_{HQ}\|_1 + \|f_{LQ}-\hat f_{LQ}\|_2 + \alpha L_{lpips} + \beta L_{GAN}$（式 6）。
- **VAR Transformer**：骨干选 Switti，用 **LoRA（r=16）** 做参数高效微调；动态缩放靠生成**尺寸专属注意力掩码**实现，另把目标尺寸嵌入注入 cross-attention 条件上下文（借鉴 SDXL），让模型 size-aware。
- **训练配置**：LSDIR + FFHQ 前 10,000 张，Real-ESRGAN 退化管线合成 LQ/HQ 对；随机裁剪 HQ 256–512² / LQ 64–128² 增强尺寸无关性；4× RTX 5880、batch 192、AdamW lr $5\times10^{-5}$、weight decay $5\times10^{-2}$（第 6–7 页）。

## 结论与局限

- **感知质量全面领先**：三个 benchmark（合成 DIV2K-Val 3,000 对、真实 RealSR / DRealSR）上，无参考 IQA 指标 CLIPIQA 与 MANIQA **全部 SOTA**，MUSIQ 稳定第二（表 1，第 6–7 页）。锚点数字：DIV2K-Val CLIPIQA 0.7405 / MANIQA 0.5641；RealSR 0.7050 / 0.5756；DRealSR 0.7351 / 0.5653。对比对象覆盖 GAN 系（BSRGAN、Real-ESRGAN、SwinIR）、扩散系（StableSR、SeeSR、SUPIR、OSEDiff、PiSA-SR）与 AR 系 VARSR。
- **保真度反而更低，作者自认**：PSNR/SSIM 明显偏低（如 DIV2K-Val 22.49 / 0.5689，DRealSR 26.62 / 0.7207），原因是 VQ 离散化不可避免引入 tokenization error，而扩散/AR–DM 混合的 VARSR 用连续 VAE 天然利于重建保真；但在 LPIPS、DISTS、FID 上 DVAR 对 VARSR 全面占优，符合 perception–distortion trade-off（第 7 页）。这和 [[感知损失 实时风格迁移与超分辨率]] 里「细节更好但 PSNR 更低」是同一类权衡。
- **消融一：动态缩放 + 尺寸条件的贡献（表 2，第 8 页，RealLR200）**：VARSR 0.7477 / 71.68 / 0.5758 → 只在 512² 训练、无尺寸条件 0.7531 / 71.89 / 0.5816 → 多分辨率训练、仍无尺寸条件 0.7621 / 72.59 / 0.5907 → 完整 DVAR **0.7767 / 73.17 / 0.6278**。最有料的结论是：**只见过 512² 的变体就已零样本超过 VARSR**，证明典范缩放动态学到的是尺寸不变规则；再叠显式尺寸条件又涨一档，说明「直接在原生分辨率更新特征 + 给尺寸线索」比 VARSR 的 resize / resize-back 更有效。图 4 显示 VARSR 早期尺度连干净结构都形不成，末端挂 diffusion MLP 也补不回高频；DVAR 早期就定好布局，后续尺度全花在纹理边缘上。
- **消融二：DSS 的贡献（表 3，第 8 页）**：带 DSS 在所有指标上一致优于不带（DIV2K-Val MANIQA 0.5444→0.5641、CLIPIQA 0.7152→0.7405、LPIPS 0.3295→0.3192；RealSR MANIQA 0.5417→0.5756、CLIPIQA 0.6756→0.7050），作者归因于训练期暴露于模拟误差，抑制推理期误差累积。
- **局限（作者承认 + 我观察到的）**：① VQ tokenization error 导致 PSNR/SSIM 偏低，是路线本身的代价（第 7 页）；② 官方仓库 README 的 Todo 明确 **训练代码与训练数据准备尚未发布**，只放了推理代码与预训练 adapter / VAE（https://github.com/YuZheng9/DVAR，commit 7dac046）；③ 消融只在 RealLR200 上做，且把长边 >512 的图先缩到 512，属于受控范围，未报告更大原生分辨率；④ 论文未系统报告推理耗时/显存与 DM 方法的同口径对比，「更高效」主要来自 VAR 范式本身的定性论述（第 2 页）；⑤ 文本 prompt 依赖外部 LLM/LLaVA 生成的 caption，推理管线不是纯视觉端到端。

## 我的疑问
- [ ] 式 5 的 round 会让任意长宽比下的相对缩放比产生量化偏差；极端宽高比（如 1:8）下这条典范序列还稳吗？论文只在 RealLR200（长边统一缩到 512）验证。
- [ ] DSS 的 count book 是**每个尺度共享、跨样本 EMA** 的全局统计，是否会对不同图像内容/退化类型欠拟合？与按样本动态估计相比差多少？
- [ ] 论文说「纯自回归」，但退化鲁棒编码器的训练用了 LPIPS + GAN 损失、VARSR 又挂了 diffusion MLP——「纯」的边界到底划在哪里？
- [ ] 训练代码未开源，Algorithm 2 里 RankCount / SampleNN / RandomChoice 的 p 取值、EMA 的 β 在正文没有给出具体数字，复现时只能靠 released 推理代码反推。

## 关键摘录
> 原文（第 1 页）："To our knowledge, DVAR is the first framework to grant VAR models size-flexibility, breaking their one-to-one dependency on a fixed resolution."

> 原文（第 2 页）："The root cause of the size-dependent bottleneck lies not in the VQ-VAE tokenizer, but in the autoregressive generation process itself."

> 原文（第 4 页）："This construction keeps the relative scaling ratios between consecutive steps consistent across resolutions, allowing a single model to operate on inputs of variable and even previously unseen sizes."

> 原文（第 5 页）："unlike discrete tokens in NLP, which lack an inherent semantic topology, visual tokens from a VQ-VAE are embedded in a feature space where geometric proximity is meaningful."

> 原文（第 7 页）："our VQ-based AR design inevitably introduces minor tokenization error and thus slightly lower PSNR/SSIM, yet DVAR still demonstrates a clear advantage over VARSR on LPIPS, DISTS, and FID across all datasets, consistent with the well-known perception–distortion trade-off"

> 原文（第 8 页）："even the variant trained only at 512×512 already outperforms VARSR, despite never seeing other resolutions or aspect ratios during training."

> 原文（第 8 页）："We introduced DVAR, a dynamic visual autoregressive framework that makes VAR models practical for image restoration."

## 相关笔记
- [[感知损失 实时风格迁移与超分辨率]] — 同为「感知质量涨、PSNR 降」的 perception–distortion 权衡
- [[ASTER 注意力场景文本识别器]] — 库内另一篇明确讨论 exposure bias / teacher forcing 的自回归模型
- [[SVTR 场景文本识别]] — 视觉序列建模的另一种形态（局部+全局混合块）
- [[DTWSR 扩散Transformer建模多级小波谱的单图超分]] — 同为生成式超分，走扩散 + 频域路线，可对比 VAR 与 DM 的先验注入方式
- [[PFT 渐进聚焦 Transformer 单图超分]] — 同属单图超分的 Transformer 效率改进，可对比「省算力」的切入点
