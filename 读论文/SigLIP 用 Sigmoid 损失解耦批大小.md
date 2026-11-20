---
created: 2026-11-20
updated: 2026-11-20
source: E:\.dsh-paper-reader\data\多模态对齐与对比学习\Zhai 等 - 2023 - Sigmoid Loss for Language Image Pre-Training.pdf（arXiv:2303.15343，ICCV 2023）
type: 论文
tags: [论文, 多模态, CLIP, 对比学习, 损失函数, SigLIP]
---

# SigLIP：把 softmax 换成 sigmoid，让批大小不再是约束

> Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer（Google DeepMind, Zürich）。页码指本机 PDF 页码。

## 一句话总结

标准对比损失要对**整个 batch 的相似度矩阵**做双向 softmax——这意味着必须有全局视野、必须大 batch、数值上还得靠减最大值来稳定。SigLIP 把它换成**逐对的 sigmoid 二分类**：每个图文对独立判断配或不配。于是 batch 与损失解耦，小 batch 下明显更好，还能推到 100 万而不过度受益——**32k 就够**。配合 Locked-image Tuning，仅 4 块 TPUv4 两天训出 **84.5%** ImageNet 零样本的 SigLiT。

## 要点提炼（速览）

1. **softmax 对比损失的三重负担**（p.1 原文）：需要对 batch 内所有图像方向、所有文本方向各归一化一次；需要全局视角；朴素实现数值不稳，通常要减最大值来稳住 softmax。
2. **sigmoid 损失**：不再做 batch 级归一化，而是对**每一个图文对**单独算 $\log\sigma(z)$ 或 $\log\sigma(-z)$——配对的对要拉近，不配对的拉远。损失因此只在「对这一对」的意义上成立，与 batch 里还有别的什么无关。
3. **小 batch 下显著更好**（Fig. 2 左）：SigLiT 训练 18B 样本时，sigmoid 损失在小 batch 区间**远超** softmax；大 batch 下两者趋同。
4. **大 batch 的收益很快饱和**（摘要与 Fig. 2）：把 batch 推到 100 万后作者发现收益迅速衰减，**32k 就足够**做图文预训练。这个结论在多语言（100+ 语言）的 SigLIP 上也成立（p.2）。
5. **批大小与损失解耦带来的副产品**：可以独立研究「样本数 vs 配对数」以及「负正比」的影响——这在 softmax 框架下是纠缠的。
6. **4 块 TPUv4 的 SigLiT**（摘要）：84.5% ImageNet 零样本，两天。这是「效率」这条线的标志性数字。
7. **Locked-image Tuning 的复用**：沿用 [[LiT 冻结图像塔只调文本塔]] 的做法，冻结图像塔只训文本侧，进一步压低成本。

## 核心问题

softmax 对比损失（CLIP / ALiGN 用的那个）有一个隐性约束：**为了让 softmax 有意义，batch 必须足够大**。因为分母是 batch 内所有负样本，batch 小则分母小、梯度信号弱。

但大 batch 意味着：显存墙、必须多卡同步、训练不稳定（CLIP 论文里就要对温度做截断）。于是问题变成：

**对比学习的「负样本要多」这个需求，到底来自任务本身，还是来自 softmax 这个具体形式？**

SigLIP 的答案是后者。

## 关键方法

### 1. 从 batch 级归一化到逐对独立

softmax 版本：$\mathcal{L}=-\log\frac{e^{z_{i,i}/\tau}}{\sum_j e^{z_{i,j}/\tau}}$——分母跨整个 batch。

sigmoid 版本：对每个 $(i,j)$ 独立算 $\log\sigma(z_{i,j})$（当 $i=j$）或 $\log\sigma(-z_{i,j})$（当 $i\neq j$），然后平均。**没有任何跨样本的归一化**。

这个改动看起来只是换了损失函数，但它把「batch 大小」和「负样本数量」这两个原本绑死的东西解开了。

### 2. 数值稳定性顺带解决

softmax 需要减最大值来避免溢出，sigmoid 天然有界，不需要这个技巧。这是一个附带但实际很重要的好处。

### 3. 用实验回答「多大 batch 才够」

作者把 batch 从很小一路推到 100 万，画出性能曲线（Fig. 2）。结论是收益在 32k 附近就基本饱和。**这直接给工程实践一个明确的指导值**，而不是「越大越好」的模糊说法。

## 结论与局限

**贡献**：把 CLIP 式对比学习从一个「必须多卡大 batch 才能训」的方法，变成「单机小 batch 也能训得好」的方法。这对学术界和资源受限的团队意义很大，也让 SigLIP 成为后来大量多模态模型（如 PaliGemma、多数开源 VLM）的图像塔首选。

**局限 / 值得注意的点**：

- **sigmoid 损失丢掉了 batch 内的竞争结构**。softmax 里每个样本要跟 batch 内所有负样本竞争（一个多分类问题），sigmoid 里每个对是独立的二分类。理论上后者对「难负样本」的利用更弱——它没有显式地让正样本在众多候选中胜出。实践中这个差距被数据和规模补上了，但在小数据 regime 下未必。
- **负正比成了一个需要调的超参**。softmax 下负样本数 = batch-1，sigmoid 下可以任意配比，多了一个自由度也多了调参负担（论文专门研究了这一点）。
- **「32k 就够」是经验结论**，依赖于其数据规模与模型规模。换到别的数据分布上这个数字未必成立。
- **本文的主战场仍是图文**，跨模态（音频、视频）上的 sigmoid 是否同样划算，论文没有覆盖。

## 相关

- [[CLIP 大规模图文对比预训练]] — softmax 版本的起点
- [[对比损失谱系 InfoNCE 到 Sigmoid]] — 概念笔记，三条损失的血缘与取舍
- [[LiT 冻结图像塔只调文本塔]] — SigLiT 直接复用的训练策略
- [[CPC 对比预测编码与 InfoNCE]] — 损失的最初形态
