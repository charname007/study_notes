---
created: 2026-11-24
source: https://github.com/fundamentalvision/Deformable-DETR @ 11169a60c33333af00a4849f1808023eba96a931（2020-12-07）；配套 [[Deformable DETR 可变形注意力把端到端检测的收敛缩短 10 倍]]
type: 代码
tags: [代码, 目标检测, 可变形注意力]
---

# Deformable DETR 的 MSDeformAttn 实现

## 场景

读论文式 (2)(3) 时想确认三件事：①「一个 $3MK$ 通道的线性投影」在官方代码里到底是不是一个 Linear；② $\Delta p$ 号称「无约束实数」，可采样必须落在归一化坐标上，这中间的换算发生在哪一步；③ 论文附录 A.4 的初始化（$A_{mlqk}=1/(LK)$、8 方向网格偏移）具体怎么写。

仓库 `fundamentalvision/Deformable-DETR`（Apache 2.0）。本笔记对 commit **11169a6** 的 `models/ops/modules/ms_deform_attn.py` 逐行核对；**CUDA kernel 未读**，凡是那里的结论都标出来。

## 关键点

### 1. 论文的「一个 $3MK$ 通道投影」在代码里是两个 Linear

```python
# models/ops/modules/ms_deform_attn.py:55-56
self.sampling_offsets  = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
```

偏移与权重是两路独立投影，通道数合起来正好 $2MLK + MLK$，与论文 §4.1 的 $3MK$ 口径一致，只是实现上拆开了——好处是权重那路可以直接接 softmax（:100 行）。顺带一个符号差异：论文写 $MK$，代码写 `n_heads * n_levels * n_points`，即 $M\cdot L\cdot K$；$L=1$（单尺度）时两者才对得上。

### 2. 归一化：$\Delta p$ 的「无约束」是相对该层特征图说的

```python
# models/ops/modules/ms_deform_attn.py:100-110
attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
if reference_points.shape[-1] == 2:
    offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
    sampling_locations = reference_points[:, :, None, :, None, :] \
                         + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
elif reference_points.shape[-1] == 4:
    sampling_locations = reference_points[:, :, None, :, None, :2] \
                         + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
```

三件事只有读码才知道：

- **softmax 在最后一维**（`n_levels*n_points`）上做 → 权重是跨层跨点统一归一化的，对应论文 $\sum_l\sum_k A_{mlqk}=1$；
- **2 维参考点**时偏移要除以该层的 $(W_l,H_l)$（`offset_normalizer` 取 `input_spatial_shapes` 的 (W,H)，注意顺序是反的）→ 论文里「无约束」的 $\Delta p$ 实际是**以归一化特征图坐标为单位**的位移，不是像素位移；
- **4 维参考点（框）**时改成按框宽高缩放 `offsets / n_points * wh * 0.5` → 就是论文附录 A.4「采样偏移按上一层框的宽高调制」的实现。

### 3. 初始化：零权重 + 8 方向网格偏置

```python
# models/ops/modules/ms_deform_attn.py:62-73（_reset_parameters）
constant_(self.sampling_offsets.weight.data, 0.)
thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
for i in range(self.n_points):
    grid_init[:, :, i, :] *= i + 1
with torch.no_grad():
    self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
constant_(self.attention_weights.weight.data, 0.)
constant_(self.attention_weights.bias.data, 0.)
```

- 偏移：**权重全 0、bias 手动设成单位方向向量**（θ = 0, 2π/8, ... 的 cos/sin），再按第 $i$ 个采样点乘 $(i+1)$ 拉长 → 8 个头初始各朝一个方向、4 个点由近及远，等价于论文 A.4 的 $(-k,-k)\dots(k,k)$ 八方向；
- 权重：weight 与 bias 都置 0 → softmax 后每个点恰好 $1/(LK)$，与论文 A.4 一致；
- `value_proj` / `output_proj` 走 Xavier，bias 清零。

> 这解答了一个容易误读的说法：论文说「初始时注意力权重近似均匀」（第 3 页），在实现里**不是随机初始化恰好接近均匀，而是被显式设成均匀**——所以这里的均匀是设计出来的起点，不是巧合。

### 4. 采样本身不在 python 里

```python
# models/ops/modules/ms_deform_attn.py:112
output = MSDeformAttnFunction.apply(
    value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
```

双线性插值与加权求和在 CUDA 扩展里（`im2col_step = 64` 控制分块，见 :48）。⚠️ 本笔记没有读 `ms_deform_attn_cuda.cu`，所以「bilinear 具体怎么取、padding 怎么处理」这批细节未核对。

## 顺带纠正一个容易混的量

论文正文说「每个头采 $K$ 个点」，代码是「每个头、**每一层**采 $K$ 个点」，默认 $L=4,\ K=4$ → 每头实际 16 个采样点、总权重数为 $8\times4\times4=128$。读 Table 2 时要记住这里的 $K$ 是**每层每头**的点数，不是全部。

## 相关笔记

- [[Deformable DETR 可变形注意力把端到端检测的收敛缩短 10 倍]] —— 论文笔记（式 (2)(3)、复杂度、消融）
- [[DETR 用集合预测与二分匹配做端到端检测]] —— 前作；本文的改动就是把它的交叉注意力换掉
