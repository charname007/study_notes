---
created: 2026-11-07
source: 官方仓库 https://github.com/c-yn/StarIR，commit fc5f3681f16e600a5777dcc284e2b7040fdb0caa（2026-04-09），文件 basicsr/models/archs/StarIR_arch.py
type: 代码
tags: [代码, 图像复原, 频域]
---

# StarIR 的 StarModule 与 DFFN 实现

对应笔记：[[StarIR 空间-频率融合的卷积图像复原]]

## 场景

读 StarIR 论文的 Star Module（FMB + SMB + star 相乘 + CMU）与 DFFN 时，公式（式 2–8）写得比较抽象，对照官方实现能看清**张量在哪个时刻被 split、哪个时刻做 FFT、滤波器到底多大**。

## 关键点

### 1) StarModule.forward：q / v 不是两条独立通路

论文写的是"$X_i = W^{dw}_{3\times3}(W_{1\times1}X)$ 给 FMB、$X_j = W^{dw}_{3\times3}(W_{1\times1}X)$ 给 SMB"，看起来是两条各自独立的支路。代码里其实是**一次投影出 2·dim 通道再 split**：

```python
self.to_hidden    = nn.Conv2d(dim, dim * 2, 1, bias=bias)
self.to_hidden_dw = nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2, bias=bias)
...
hidden = self.to_hidden(x)
q, v = self.to_hidden_dw(hidden).split([self.dim, self.dim], dim=1)
```

- **q** 走 FMB：切 8×8 窗口 → `torch.fft.rfft2` → 乘 `self.fft_FSAS` → `irfft2`；
- **v** 走 SMB：`SpatialOperation` = `dwconv3x3 + Sigmoid` 再自乘（式 (4) 的 σ）；
- 然后 `out = self.spatial(v) * out`，这一步就是式 (5) 的 star 相乘；
- 最后 `out = self.plain_channel(out) * out` 是 CMU（GAP → 1×1 → 乘），再 `project_out`（1×1）收尾。

```python
out = self.norm(out)
out = self.spatial(v) * out      # star operation (式 5)
out = self.plain_channel(out) * out   # CMU (式 6)
out = self.project_out(out)
```

⚠️ 注意 `self.norm` 是加在 **FMB 输出之后、star 之前**的，论文式 (2)–(5) 里没体现这一层 LayerNorm。

### 2) FFT 滤波器是"每通道一组固定尺寸的频响"

```python
self.patch_size = 8
self.fft_FSAS = nn.Parameter(torch.ones((dim, 1, 1, 8, 8 // 2 + 1)))
...
q_patch = rearrange(q, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=8, p2=8)
q_fft   = torch.fft.rfft2(q_patch.float())
out     = torch.fft.irfft2(q_fft * self.fft_FSAS, s=(8, 8))
```

- 尺寸 `(C, 1, 1, 8, 5)`：`rfft2` 只保留 8×(8/2+1)=8×5 的半谱，参数量 $8\times5\times C$，极小；
- 维度里那两维 `1, 1` 说明它**在所有空间位置上共享**（逐通道固定频响），不具备 AdaptIR 那种逐像素自适应；
- 输入转 `.float()`（fp32）再 FFT，混精度训练时要注意这部分是 fp32。

### 3) DFFN：GEGLU 在前，窗口 FFT 在后

论文式 (7)(8) 把 GEGLU 和频域调制并列写出，读起来像两条并行支路；代码里是**串行**的：

```python
x = self.project_in(x)                       # 1x1
x1, x2 = self.dwconv(x).chunk(2, dim=1)      # dw3x3 后对半切
x = F.gelu(x1) * x2                          # GEGLU
x = self.project_out(x)                      # 1x1
# 再对输出做 8x8 窗口 rfft -> 乘 self.fft -> irfft
```

DFFN 的 `self.fft` 形状是 `(dim, 1, 1, 8, 5)`，注意这里的通道数是 **dim**（输出维度），而 StarModule 里的也是 dim —— 两处滤波器规模相同。

### 4) CMU 的池化在去模糊任务上不是 GAP

```python
if basesize:
    gap = AvgPool2d(base_size=basesize)   # 自定义，大核自适应平均池化
else:
    gap = nn.AdaptiveAvgPool2d((1, 1))
```

文件开头的注释写明这部分源自 *Revisiting global information aggregation*，"**only used for deblurring tasks** (following previous works)"：生成大图时用大核池化替代 1×1 GAP，避免全局池化在超大分辨率上的行为差异。论文正文（式 6）只写了 GAP，没提这个实现细节。

### 5) 主干默认配置

```python
dim = 48, num_blocks = [7, 8, 15], num_refinement_blocks = 4, ffn_expansion_factor = 3
```

- 下采样 = `Upsample(0.5, bilinear)` + `Conv3x3` 升维（与论文 p.3 一致）；
- 上采样 = `Upsample(2, bilinear)` + `Conv3x3` 减半；
- `Fuse` = concat + 1×1 卷积（对应论文 p.3 的"拼接后 1×1 卷积减半"）；
- 单目任务输出 `self.output(...) + inp_img`（全局残差），双像素任务换成分支 `skip_conv`。

### 与论文的三处差异小结

| 论文 | 代码 |
|---|---|
| 式 (2)(4) 中 FMB / SMB 各有独立的 $W_{1\times1}\to W^{dw}_{3\times3}$ | 共用一个 `to_hidden` + `to_hidden_dw`，再 split 成 q/v |
| FMB 输出后直接 star 相乘 | 中间多一层 LayerNorm |
| 式 (7)(8) 读作并行双域 | 实际串行：GEGLU 提完特征再对结果做窗口 FFT 调制 |

## 相关笔记

- [[StarIR 空间-频率融合的卷积图像复原]] —— 论文解读（方法与消融）
- [[Restormer 的 MDTA 与 GDFN 实现]] —— 同类"逐行读官方代码找公式/实现差异"的笔记
- [[AdaptIR 自适应频率增强的一体化图像复原]] —— 同作者的另一条频域路线，对比逐像素动态核 vs 共享频响
- [[各种Norm层的摆放]] —— 代码里 LayerNorm 的位置与论文公式对不上，正是这篇文章讲的问题
