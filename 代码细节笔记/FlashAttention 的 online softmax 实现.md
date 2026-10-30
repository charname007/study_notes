---
created: 2026-10-30
source: Dao-AILab/flash-attention @ 819ef26aae39a1838d0597090880e044230d9261 — csrc/flash_attn/src/softmax.h
type: 代码
tags:
  - 代码
  - Attention
  - CUDA
  - online softmax
---

# FlashAttention 的 online softmax 实现

论文 Algorithm 1 里那个「记住 (m, ℓ) 就能分块算 softmax」的推导，在官方 CUDA 实现里落成了 `csrc/flash_attn/src/softmax.h` 的 `struct Softmax`。这篇笔记记录**代码到底怎么把公式写死成指令**。

## 场景

读 FlashAttention 论文时，最容易卡住的是 §3.1 那段 online softmax 的数学推导——看起来只是几个符号变换，但**为什么这样就「不需要看整行」了**，以及**代码里是怎么用一条循环表达的**，光看公式很难对上。这份笔记把两头连起来。

## 关键点

### 1. `Struct Softmax` 的状态只有两个张量

```cpp
// csrc/flash_attn/src/softmax.h:128-134
template <int kNRows>
struct Softmax {
    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max, row_sum;          // ← 就是论文里的 m 和 ℓ
    __forceinline__ __device__ Softmax() {};
```

`row_max` = 论文的 `m(x)`（该行当前见到过的最大分数），`row_sum` = 论文的 `ℓ(x)`（该行归一化因子）。**注意这两个都只是 `kNRows` 长度的一维数组**，不是 `N×N` 矩阵——这正是显存从 `O(N²)` 降到 `O(N)` 的直接原因。

### 2. `optimize_rescale_o` 分两条路径：首块 vs 后续块

```cpp
// softmax.h:136-167
template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1>
__forceinline__ __device__ void optimize_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
    Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
    if (Is_first) {
        // 第一块：直接初始化 m 和 ℓ
        reduce_max</*zero_init=*/true>(scores, row_max);
        scale_apply_exp2(scores, row_max, softmax_scale_log2);
        reduce_sum</*zero_init=*/true>(scores, row_sum);
    } else {
        // 后续块：先保存旧 m，再用新块更新 m
        Tensor scores_max_prev = make_fragment_like(row_max);
        cute::copy(row_max, scores_max_prev);              // ← 存旧 m(x1)
        reduce_max</*zero_init=*/false>(scores, row_max);   // ← m(x) = max(m(x1), m(x2))
        ...
```

`Is_first` 是个**编译期模板参数**，不是运行时分支——所以循环里没有 if 判断开销。

### 3. 核心那五行：一旦最大值变大，旧累加全部缩放

```cpp
// softmax.h:153-161
for (int mi = 0; mi < size(row_max); ++mi) {
    float scores_max_cur = !Check_inf ? row_max(mi)
                                      : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
    float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
    row_sum(mi) *= scores_scale;                                    // ℓ 缩放
    for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
        acc_o_rowcol(mi, ni) *= scores_scale;                       // 输出 O 缩放
    }
}
```

对照论文第 5 页的公式：

$$\ell(x) = e^{m(x^{(1)})-m(x)}\,\ell(x^{(1)}) + e^{m(x^{(2)})-m(x)}\,\ell(x^{(2)})$$

代码里的 `scores_scale` 就是那个 `e^{m(x_old) - m(x_new)}`。**因为新块可能带来更大的 max，旧的所有累积量都要乘这个衰减因子**——否则旧的 `ℓ` 和新块的 `ℓ` 不在同一个指数基准上，加法就错了。

同样，输出 `acc_o` 也要跟着缩放（公式 7 里那个 `diag(ℓ^{new})^{-1} diag(ℓ^{old})` 对角阵的效果，在这里用逐元素乘实现）。

### 4. 为什么是 `exp2f` 而不是 `expf`

因为整条流水线都在 **log2 尺度**上工作：`softmax_scale_log2` 把 `1/√d` 这个缩放因子也折进 log2。这样 `exp2f` 只需一条 `MUFU.EX2` 指令，而 `expf` 要拆成多条——在算子内部这种每块都要跑几万次的循环里，这个差别是实测级的。

### 5. 最后才做跨线程 reduce

```cpp
// softmax.h:169-172
template<bool Is_dropout=false, bool Split=false, typename Tensor0>
__forceinline__ __device__ TensorT normalize_softmax_lse(Tensor0 &acc_o, ...) {
    SumOp<float> sum_op;
    quad_allreduce_(row_sum, row_sum, sum_op);   // ← 只在最后归约一次
```

代码里有段注释很关键：

```cpp
// We don't do the reduce across threads here since we don't need to use the row_sum.
// We do that reduce at the end when we need to normalize the softmax.
```

即**中间每块累加时不做跨线程归约**（因为用不到精确的求和值，只有最后归一化才需要），把昂贵的 `quad_allreduce_` 推迟到最后一次。这是典型的「把同步开销从 O(块数) 降到 O(1)」。

## 与论文的映射表

| 论文符号 / 步骤 | 代码位置 | 说明 |
|---|---|---|
| `m(x)`（行最大值） | `row_max` | `TensorT`，长度 `kNRows` |
| `ℓ(x)`（行指数和） | `row_sum` | 同上 |
| `m(x) = max(m(x1), m(x2))` | `reduce_max<false>(scores, row_max)` | 就地更新最大值 |
| `ℓ` 的衰减因子 `e^{m_old - m_new}` | `scores_scale` | `exp2f` 一条指令 |
| 输出重缩放 | `acc_o_rowcol(mi, ni) *= scores_scale` | 逐元素乘 |
| Algorithm 1 line 10 | `optimize_rescale_o` 整体 | 每块核心 |
| Algorithm 1 line 12 | `normalize_softmax_lse` | 最终归一化 |

## 相关笔记

- [[FlashAttention 精确注意力与IO感知]] — 论文本身的解读（含 Fig.1 内存层次图）
- [[SVTR 的 PaddleOCR 实现]] — 同样是「论文说一套、代码是另一套」的对照案例（那边是掩码而非真窗口）
- [[Restormer 的 MDTA 与 GDFN 实现]] — 另一个把注意力公式落到代码的例子
- [[transformer的2道残差连接]] — 理解「重缩放 / 增量补充」这类操作的通用直觉

#代码 #Attention #CUDA #online softmax
