---
created: 2026-10-30
source: Zotero 条目 5FUT8RQ4 / PDF 附件 Q2LCKMR8 — E:\Zotero\storage\Q2LCKMR8\Dao 等 - 2022 - FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness.pdf（arXiv:2205.14135v2, 2022-06-23）
zotero_key: 5FUT8RQ4
type: 论文
tags:
  - 论文
  - Attention
  - Transformer
  - IO感知
---

# FlashAttention：IO 感知的精确注意力

## 一句话总结

**把注意力的瓶颈从「算得快」重新定义为「少搬数据」**：标准注意力要在 GPU 显存（HBM，1.5 TB/s）里反复读写那个 `N×N` 的注意力矩阵，而 FlashAttention 用 **tiling（分块）+ recomputation（重算）** 让整张矩阵**从不落地到 HBM**，全程只在片上 SRAM（19 TB/s）里算完——结果**FLOPs 反而更多**，却快了 2–4×、显存从 `O(N²)` 降到 `O(N)`，而且 **attention 本身完全精确，不是近似**（第 2 页）。

## 核心问题

作者先给了一个前提判断：**GPU 算力早已超过显存带宽**，Transformer 里大多数算子（elementwise、softmax、norm、reduction）都是 **memory-bound**——时间几乎全花在搬数据上，而不是算数上（第 3 页）。但 PyTorch/TF 这类框架**不让你细粒度控制内存访问**（第 2 页），于是标准注意力的实现只能老老实实：

1. 读 Q、K 算出 `S = QKᵀ`（`N×N`），**写回 HBM**；
2. 再把 S **读回来**，算 `P = softmax(S)`，**再写回 HBM**；
3. 再读 P 和 V 算 `O = PV`，写回。

也就是说，那个 `N×N` 矩阵要**来回搬三趟**（第 4 页 Algorithm 2）。序列一长，`N²` 直接爆炸：既慢，又占显存，长度上限被卡死。

**为什么之前没人这么干？** 两个技术障碍（第 2 页）：
- softmax 的归一化需要**看到整行**才能算（而你只想一块一块地算）；
- 反向传播需要 `S`、`P` 这些 `N×N` 中间量算梯度（而作者不想存它们）。

## 关键方法

### 1. Tiling：分块算 + 在线 softmax

把 Q、K、V 切成块，从 HBM 搬进 SRAM，一块一块算。为了解决「softmax 要看整行」的问题，作者用了 **online softmax** 的递推：维护两个统计量 `m`（行最大值）和 `ℓ`（行指数和），每来一个新块就更新一次，最后乘上正确的归一化系数再合并（第 4–5 页，Algorithm 1 line 10–12）。

关键推导（第 5 页）：把向量 `x` 拆成 `[x⁽¹⁾ x⁽²⁾]`，则
- `m(x) = max(m(x⁽¹⁾), m(x⁽²⁾))`
- `ℓ(x) = e^{m(x⁽¹⁾)-m(x)}·ℓ(x⁽¹⁾) + e^{m(x⁽²⁾)-m(x)}·ℓ(x⁽²⁾)`

**只要记住 `(m, ℓ)`，softmax 就能一块一块地算**——这就是整篇论文的算法基石。

### 2. Recomputation：反向用「重算」代替「存矩阵」

反向传播本来需要 `S, P ∈ R^{N×N}`。作者改为：只存输出 `O` 和 softmax 统计量 `(m, ℓ)`，反向时**用 SRAM 里的 Q、K、V 块把 S 和 P 重新算一遍**（第 5 页）。

这本质是**选择性梯度检查点**（selective gradient checkpointing），但关键的差别是：常规 checkpointing 都是「拿速度换显存」，而这里**因为省掉了 HBM 往返，重算反而更快**（第 5 页 + Fig. 2）。这是全篇最反直觉、也最漂亮的一点。

### 3. IO 复杂度：给出了下界

- 标准注意力：`Ω(Nd + N²)` 次 HBM 访问
- FlashAttention：`O(N²d²M⁻¹)` 次（`d` 为 head dim，`M` 为 SRAM 大小）
- 实测**最多少 9×** HBM 访问（Fig. 2）

更进一步，作者证明了**下界**（Proposition 3）：对于 `d ≤ M ≤ Nd` 的所有 `M`，**不存在**能做到 `o(N²d²M⁻¹)` 次 HBM 访问的精确注意力算法（第 6 页）。也就是说这个方法在 IO 层面**已接近理论最优**。

### 4. Block-sparse 扩展

把同样的思路套到稀疏注意力上，得到 block-sparse FlashAttention：比 FlashAttention 本身还快 2–4×，支持到 64K 序列长度，且 IO 复杂度按稀疏率成比例改善（第 3 页）。

![FlashAttention 图1 IO层次与加速](FlashAttention%20图1%20IO层次与加速.png)

<small>裁自论文第 2 页 Figure 1，仅取图本体，已去除图注文字。</small>

**怎么读这张图（左中右三块）：**

- **左边（三角金字塔）= GPU 内存层次与带宽差异**，这是全篇的立论基础：
  - 塔尖橙色的 **SRAM**：19 TB/s，但只有 **20 MB**（片上、极快、极小）
  - 中间绿色的 **HBM**：1.5 TB/s，**40 GB**（显存，慢 12 倍，容量大）
  - 底层青色 **DRAM**（CPU 内存）：12.8 GB/s，>1 TB（更慢）
  - 带宽从下到上差三个数量级 → **数据放在哪一层，决定一切**。
- **中间（FlashAttention 数据流）= tiling 的实际执行方式**：
  - 绿色长条是 K、V、Q 分块；橙色小方块是当前正在算的块。
  - **红色箭头 Outer Loop**：外层循环负责搬运 K、V 块；
  - **蓝色箭头 Inner Loop**：内层循环再遍历 Q 的块，每块算完立刻写回 HBM；
  - 最关键的细节：中间那个 **虚线框（`N×N` 矩阵）没有变实**——图注明说 "prevents materialization of the large N×N attention matrix"，即**这个大矩阵从未被真正写进 HBM**，只在 SRAM 里以块的形式存在过。
- **右边（Attention on GPT-2 柱状图）= 收益的直接证据**：
  - 左柱 PyTorch：被拆成 **Matmul / Mask / Softmax / Dropout / Matmul** 五段，各自都要读写 HBM，总时长约 16 ms；
  - 右柱 FlashAttention：**整根柱子塌缩成一小块 "Fused Kernel"**（约 2 ms），时间轴上是 **7.6×** 加速；
  - 这就是 kernel fusion：把五步融成一个 CUDA kernel，一次搬运算完，中间量不落地。

## 实验与结论

**设置**：A100-SXM4-40GB（部分对比含 RTX 3090 / T4），BERT-large、GPT-2 small/medium，以及 Long Range Arena（LRA）五个长序列任务（第 6–7 页、附录 E）。

**关键结果（避免抄全表，只留锚点）**：
- **BERT-large 训练快 15%**：17.4 min vs Nvidia MLPerf 1.1 的 20.0 min（Table 1，第 7 页）
- **GPT-2 端到端最多 3×**（vs HuggingFace）、**1.7×**（vs Megatron-LM），且**困惑度完全一致**（Table 2，第 7 页）
- **长上下文**：context 从 1K 拉到 4K 后**仍比 Megatron 1K 快 30%**，且困惑度还低 0.7（Table 4，第 8 页）
- **纯 attention 算子层面**：GPT-2 上 **7.6×** 加速（Fig. 1 右）
- **显存**：从 `O(N²)` 线性化到 `O(N)`，能支持到 **64K** 序列长度（第 2 页）

**消融/对比的核心洞察**：
- 与 **Apex FMHA**（此前最快的注意力实现，专为 ≤512 短序列）比：短序列上 FlashAttention 略快，但它**能支持 64K**、支持更多 head dim（16/32/64/128）、支持更广的 GPU（Turing + Ampere）（Table 7，第 28 页）
- **FLOPs 更多但仍然更快**（Fig. 2 左）——直接验证了「HBM 访问次数才是耗时主因」这个论点
- **不同 GPU 上加速比不同**：RTX 3090 上更高（2.5–4.5×），因为其显存带宽更低（~900 GB/s vs A100 的 1.5 TB/s）；T4 上更低，因为 SRAM 更小、block 只能开更小——**与 IO 复杂度分析预测的一致**（第 29 页）
- LRA 上：调参后各注意力方法精度相近，但 FlashAttention 在 **wall-clock 上快得多**（附录 E.3）

**局限**（作者承认 + 我的判断）：
- 作者明确只做了**单 GPU 内**的 IO 优化，提到多 GPU、kernel regression、block-sparse matmul 是「未来工作」（第 3 页、§5）
- 需要**手写 CUDA**，与 PyTorch 生态割裂——这也是后来 v2/v3 与 `torch.nn.functional.scaled_dot_product_attention` 内建整合的动因
- 加速比**强依赖硬件**（SRAM 大小、带宽比），在 T4 这类小 SRAM 卡上收益就明显缩水
- IO 分析的下界只在「精确注意力」范围内成立，近似/稀疏方法可以绕过（论文自己也做了 block-sparse）

## 代码情况

**官方实现存在，论文第 3 页脚注直接给出**：`https://github.com/HazyResearch/flash-attention`。

⚠️ **注意仓库已迁移**：该链接现在返回 **301 → `https://github.com/Dao-AILab/flash-attention`**（Tri Dao 的 AILab 组织）。而且**当前仓库已包含 FlashAttention-2 与 Hopper 版本**，不再是 2022 年的原版形态——引用行号时必须绑定 commit，否则很快失效。

本次实读版本：

| 项 | 值 |
|---|---|
| 仓库 | `Dao-AILab/flash-attention`（由 `HazyResearch/flash-attention` 301 跳转而来） |
| commit | `819ef26aae39a1838d0597090880e044230d9261`（2026-10-29） |
| 关键文件 | `csrc/flash_attn/src/softmax.h`（189 行，online softmax 核心） |

**`softmax.h:129-167` 的 `Struct Softmax` 就是论文 Algorithm 1 的在线 softmax 落地**，核心是 `optimize_rescale_o`：

```cpp
// csrc/flash_attn/src/softmax.h:153-161
for (int mi = 0; mi < size(row_max); ++mi) {
    float scores_max_cur = !Check_inf ? row_max(mi)
                                      : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
    // 论文的 "m(x) = max(m(x1), m(x2))" —— 新块的最大值一更新，
    // 旧累加值就要按 exp(旧m - 新m) 整体缩放
    float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
    row_sum(mi) *= scores_scale;
    for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
}
```

这段代码把论文那两行公式（`m(x) = max(...)`、`ℓ(x) = e^{m(x1)-m(x)}·ℓ(x1) + ...`）写死了：**一旦发现更大的最大值，就把此前累积的 `row_sum` 和输出 `acc_o` 全部按 `exp2(Δm)` 缩放**——这就是「online」的全部含义。`exp2f` 而非 `exp` 是因为后续都在 log2 尺度上算，省一次乘法。

**配套对照**：`kernel_traits.h:51` 里的 `Flash_fwd_kernel_traits` 定义了 block 尺寸如何按 SRAM 容量推导（论文 `B_c = Θ(M/d)`、`B_r = Θ(min(M/d, M/B_c))` 的落地），`flash_fwd_kernel.h`（76 KB）则是主 kernel。**细读实现见 [[FlashAttention 的 online softmax 实现]]**。

## 我的疑问

- [ ] Proposition 3 的下界是在「精确注意力」的范围内，那么近似方法（如 block-sparse）在什么条件下能真正突破这个下界？
- [ ] 论文说 backward 的 IO 也是 `Θ(N²d²M⁻¹)`，但 backward 要多做一次重算，FLOPs 增加的代价在什么规模下会反超 IO 收益？
- [ ] 为什么 RTX 3090 加速比更高可归因于带宽更低，那 SRAM 大小的影响权重大概多少？（T4 的例子给了线索，但没给定量拆分）
- [ ] 当前仓库是 2026 年的版本（含 v2/v3 Hopper），2022 年的原始 kernel 长什么样？两者在 tiling 策略上的差异有多大？

## 关键摘录

> "Our main goal is to avoid reading and writing the attention matrix to and from HBM."（第 2 页）

> "We apply two well-established techniques to address these challenges. (i) We restructure the attention computation to split the input into blocks and make several passes over input blocks, thus incrementally performing the softmax reduction (also known as tiling). (ii) We store the softmax normalization factor from the forward pass to quickly recompute attention on-chip in the backward pass, which is faster than the standard approach of reading the intermediate attention matrix from HBM."（第 2 页）

> "Even with the increased FLOPs due to recomputation, our algorithm both runs faster (up to 7.6x on GPT-2, Figure 1 right) and uses less memory—linear in sequence length—than standard attention, thanks to the massively reduced amount of HBM access."（第 2 页）

> "FlashAttention is a new attention algorithm that computes exact attention with far fewer memory accesses."（第 2 页）

> "We prove a lower-bound: one cannot asymptotically improve on the number of HBM accesses for all values of M (the SRAM size) when computing exact attention."（第 6 页）

> "While gradient checkpointing has been suggested to reduce the maximum amount of memory required, all implementations (that we know off) have to trade speed for memory. In contrast, even with more FLOPs, our recomputation speeds up the backward pass due to reduced HBM accesses."（第 5 页）

## 相关笔记

- [[PAGRN 渐进式注意力引导循环网络]] — 同样在动「注意力该看什么范围」：PAGRN 用注意力筛特征、用循环传递语义；FlashAttention 则完全不改语义，只改**数据怎么搬**
- [[Restormer 高效高分辨率图像修复 Transformer]] — 把注意力从空间维搬到通道维以降低复杂度；FlashAttention 走的是另一条路：**复杂度不是重点，访问模式才是**
- [[SVTR 场景文本识别]] — 其「局部混合」用的是**稠密注意力 + -inf 掩码**，计算量没省；FlashAttention 的 tiling 才是真正减少 HBM 访问的做法
- [[transformer的2道残差连接]] — 残差与重算都属「把已有信息换个形式复用」，可对照理解 recomputation 的动机
- [[FlashAttention 的 online softmax 实现]] — 本文 `softmax.h` 的逐行解读

#论文 #Attention #Transformer #IO感知
