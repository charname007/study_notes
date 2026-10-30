---
created: 2026-10-30
source: SVTR 论文 2.1–2.5 节；官方实现 https://github.com/PaddlePaddle/PaddleOCR （ppocr/modeling/backbones/rec_svtrnet.py）
type: 代码
tags:
  - 代码
  - Transformer
  - Attention
  - 场景文本识别
---

# SVTR 的 PaddleOCR 实现

## 场景

读 SVTR 论文时想知道三件事：① 论文的 mixing block / merging / combining 在代码里叫什么；② 局部混合的 `7×11` 窗口怎么落地；③ 论文说「parallel linear prediction with de-duplication」，代码里到底是不是这么干。

**仓库与版本（本次实读）**

| 项 | 值 |
|---|---|
| 仓库 | `PaddlePaddle/PaddleOCR`（论文 Abstract 第 1 页明写 code is publicly available 于此，属**官方实现**） |
| commit | `dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf`（2026-09-16） |
| 主干文件 | `ppocr/modeling/backbones/rec_svtrnet.py`（642 行） |
| 配置 | `configs/rec/rec_svtrnet.yml`（118 行） |
| 输出头 | `ppocr/modeling/heads/rec_ctc_head.py` |

论文作者本人**没有**单独仓库（GitHub 用户 `YongkunDu` 返回 404）；社区另有一份 PyTorch 移植 [`j-river/svtr-pytorch`](https://github.com/j-river/svtr-pytorch)（27 stars，最后提交 2022-05-24，含 `rec_svtrnet.py` / `svtr_model.py`），**未获官方背书**，本篇不采用。

## 关键点

### 1. 论文术语 ↔ 代码命名对照表

| 论文（§2.1–2.5） | 代码 | 位置 |
|---|---|---|
| progressive overlapping patch embedding | `PatchEmbed(mode="pope")` | `rec_svtrnet.py:281, 299-320` |
| 一步线性投影（对照基线） | `PatchEmbed(mode="linear")` | `rec_svtrnet.py:351-357` |
| SVTR block（mixing block 整体） | `Block` | `rec_svtrnet.py:215` |
| Global Mixing | `Attention(mixer="Global")` | `rec_svtrnet.py:151, 195-212` |
| Local Mixing | 同一个 `Attention` + `mixer="Local"` + 窗口掩码 | `rec_svtrnet.py:180-192, 204-205` |
| Merging | `SubSample(types="Conv", stride=[2,1])` | `rec_svtrnet.py:368, 388-396` |
| Combining（末级压高度） | `AdaptiveAvgPool2D([1,out_char_num])` + `last_conv` 1×1 | `rec_svtrnet.py:566-577` |
| 并行线性预测 + 去重 | `CTCHead`（`nn.Linear`）+ `CTCLoss` + `CTCLabelDecode` | `rec_ctc_head.py:73-75`；`rec_svtrnet.yml:66,69,72` |

一句话：论文里「Mixing Block」是**一个 stage 内 block 的统称**，代码里每个 block 是 `Block`，而「block 里的 mixer」才是 `Attention`/`ConvMixer`。

### 2. 渐进重叠 patch embedding

```python
# rec_svtrnet.py:299-320（mode="pope"，sub_num=2；只列第一级）
if mode == "pope":
    if sub_num == 2:
        self.proj = nn.Sequential(
            ConvBNLayer(
                in_channels=in_channels,
                out_channels=embed_dim // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                act=nn.GELU,
                bias_attr=None,
            ),
            ConvBNLayer(
                in_channels=embed_dim // 2,
                out_channels=embed_dim,
                ...
```

`ConvBNLayer` 就是论文图里的 `Conv → BN → 激活`：`forward` 是 `conv → norm → act` 三步（`rec_svtrnet.py:69-73`）。

注意 `mode` 还能取 `"linear"`（`rec_svtrnet.py:351-357`，用 `nn.Conv2D(1, embed_dim, kernel_size=patch_size, stride=patch_size)`）——这正是 Table 2 左半「Linear」那一行的对照配置，论文与代码的基线都留着。

### 3. 局部混合：论文说滑窗，代码用稠密掩码

论文 §2.3 说局部混合「similar to convolution, the mixing is carried out in a sliding window manner，窗口 7×11」（第 3–4 页）。代码的实现是：**正常算全图注意力，再把窗口外的分数打成 -inf**。

窗口掩码的构造（`rec_svtrnet.py:180-192`）：

```python
if mixer == "Local" and HW is not None:
    hk = local_k[0]
    wk = local_k[1]
    mask = paddle.ones([H * W, H + hk - 1, W + wk - 1], dtype="float32")
    for h in range(0, H):
        for w in range(0, W):
            mask[h * W + w, h : h + hk, w : w + wk] = 0.0
    mask_paddle = mask[:, hk // 2 : H + hk // 2, wk // 2 : W + wk // 2].flatten(
        1
    )
    mask_inf = paddle.full([H * W, H * W], "-inf", dtype="float32")
    mask = paddle.where(mask_paddle < 1, mask_paddle, mask_inf)
    self.mask = mask.unsqueeze([0, 1])
```

读法：先在 `(H+hk-1) × (W+wk-1)` 的 padded 画布上把每个 query 的 `hk×wk` 窗口内**置 0**（注意是「窗口内 = 0」，外面保持 1），再居中裁回 `H×W`，最后 `where(<1, 原值, -inf)` ——**窗口内保留 0、窗口外变 -inf**。

用的时候只在 softmax 前加一下（`rec_svtrnet.py:196-206`）：

```python
qkv = (
    self.qkv(x)
    .reshape((0, -1, 3, self.num_heads, self.head_dim))
    .transpose((2, 0, 3, 1, 4))
)
q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

attn = q.matmul(k.transpose((0, 1, 3, 2)))
if self.mixer == "Local":
    attn += self.mask
attn = nn.functional.softmax(attn, axis=-1)
```

**这意味着**：`attn` 本来就是 `[B, heads, N, N]` 的稠密矩阵（`N = H*W`），`mask` 也是 `[1,1,N,N]` 全尺寸。所以所谓「局部」只是**注意力权重被限制在看哪里**，计算量一点没省——作者在论文里强调的效率优势，在官方实现里体现为「N 很小所以稠密也不贵」，而不是「窗口化稀疏计算」。

### 4. Block：残差 + LayerNorm，但 `prenorm` 这个名字是反的

```python
def forward(self, x):
    if self.prenorm:
        x = self.norm1(x + self.drop_path(self.mixer(x)))
        x = self.norm2(x + self.drop_path(self.mlp(x)))
    else:
        x = x + self.drop_path(self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x
```

`rec_svtrnet.py:271-278`。**读代码时最容易读反的一处**：

- `prenorm=True` 走的分支是 `norm(x + mixer(x))` —— 这是 **post-norm**（LN 在残差之后）；
- `prenorm=False` 走的才是 `x + mixer(norm(x))` —— 这是 **pre-norm**（LN 在 mixer 之前）。

而配置里写的是 `prenorm: False`（`rec_svtrnet.yml:61`），所以**实际跑的是 pre-norm**，和论文 Fig. 2 画出的「LN → Local/Global Mixing → ⊕ → LN → MLP → ⊕」顺序一致。参数名与语义相反，纯属命名坑。

另外 `prenorm=False` 时主干末尾会补一道 `self.norm`（`rec_svtrnet.py:578-579, 620-621`），这也是 pre-norm 网络的标准收尾。

### 5. Merging / Combining / 输出头

Merging 走 `SubSample` 的 conv 分支（`rec_svtrnet.py:388-396`）：

```python
else:
    self.conv = nn.Conv2D(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        weight_attr=ParamAttr(initializer=KaimingNormal()),
    )
```

`stride=[2, 1]`（`rec_svtrnet.py:374` 默认值，调用处 `rec_svtrnet.py:499-505` 也传 `[2,1]`）——**高度减半、宽度不变**，与论文 §2.4 完全对应。另有 `types="Pool"` 分支（avg+max 池化后 1×1 线性，`rec_svtrnet.py:380-387`），配置默认用 `'Conv'`。

末级 combining（`rec_svtrnet.py:566-577`）：

```python
if last_stage:
    self.avg_pool = nn.AdaptiveAvgPool2D([1, out_char_num])
    self.last_conv = nn.Conv2D(
        in_channels=embed_dim[2],
        out_channels=self.out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        bias_attr=False,
    )
    self.hardswish = nn.Hardswish()
    self.dropout = nn.Dropout(p=last_drop, mode="downscale_in_infer")
```

论文 §2.5 说「pool 高度到 1，接**全连接层**、非线性激活、dropout」。代码用 `AdaptiveAvgPool2D([1,25])` + **1×1 卷积**代替 FC——数学上等价（1×1 conv 就是逐位置的 FC），但顺带把宽度也钉死成固定的 `out_char_num=25`。这解释了为什么输入宽度必须是 100：`100 // 4 = 25`。

### 6. 张量形状走一遍（用配置里的默认值）

配置出自 `rec_svtrnet.yml:51-61`：`img_size: [32,100]`、`embed_dim: [64,128,256]`、`depth: [3,6,3]`、`num_heads: [2,4,8]`、`out_channels: 192`。这套超参就是论文 Table 1 的 **SVTR-T** 那一行（`[64,128,256] / [3,6,3] / [2,4,8] / 192`）。

```
x                      [B, 3, 32, 100]
PatchEmbed(pope)       → [B, 64, 8, 25] → flatten/transpose → [B, 200, 64]
+ pos_embed[1,200,64]
blocks1 × 3 (Local)    [B, 200, 64]        HW = [8, 25]
sub_sample1 (Conv s2)  [B, 128, 4, 25]  → [B, 100, 128]
blocks2 × 6            [B, 100, 128]       HW = [4, 25]
sub_sample2 (Conv s2)  [B, 256, 2, 25]  → [B, 50, 256]
blocks3 × 3            [B, 50, 256]        HW = [2, 25]
avg_pool([1,25])       [B, 256, 1, 25]
last_conv 1×1 → 192    [B, 192, 1, 25]
SequenceEncoder(reshape) → [B, 25, 192]
CTCHead.fc (192→37)    [B, 25, 37]         ← 英文 37 类
CTCLoss / CTCLabelDecode → 去重成字符串
```

`HW` 由 `img_size // 2**sub_num` 算出（`rec_svtrnet.py:467`），`num_patches = (W//4)*(H//4) = 25*8 = 200`（`rec_svtrnet.py:294`）。高度一路 8→4→2，宽度始终 25。

### 7. 混合块排布：论文文字 vs 代码列表（**对不上**）

配置里 mixer 是一个 **12 项的扁平列表**（`rec_svtrnet.yml:58`）：

```yaml
mixer: ['Local','Local','Local','Local','Local','Local','Global','Global','Global','Global','Global','Global']
```

三个 stage 按 `depth=[3,6,3]` 把它切开（`rec_svtrnet.py:481, 515, 548`）：

- stage 1：`mixer[0:3]` → **Local × 3**
- stage 2：`mixer[3:9]` → **Local × 3 + Global × 3**
- stage 3：`mixer[9:12]` → **Global × 3**

即「全网共 6 个 Local + 6 个 Global，局部在前、全局在后」。

但论文 Table 3 的文字写的是「`[G]6[L]6` means for each stage, six global mixing blocks are carried out at first, and then six local mixing blocks」——按字面是**每个 stage 12 个 block**，而 Table 1 给 SVTR-T 的深度是 `[3,6,3]`（每 stage 只有 3 / 6 / 3 个）。两处口径不可能同时成立，代码支持的是「**全网 12 个**」这一读法。看消融结论时这点必须留意。

## 复现要点

1. **输入宽度被 `out_char_num` 钉死**：`AdaptiveAvgPool2D([1, out_char_num])` 会把宽度强制池化到 25，改 `img_size` 宽度必须同步改 `out_char_num`（注释自己写了 `# W//4 or W//8 or W/12`）。
2. **`img_size` 与实际喂进来的尺寸必须严格相等**，`PatchEmbed.forward` 里有 `assert H == self.img_size[0] and W == self.img_size[1]`（`rec_svtrnet.py:361-363`）。所以数据管线里 `SVTRRecResizeImg: [3,64,256]` 必须配 `tps_outputsize: [32,100]`，让 STN 把尺寸拉回 32×100，否则直接 assert 失败。
3. **`prenorm` 传 False 才是 pre-norm**（见上），改配置前先读 `Block.forward`。
4. **局部混合不省算力**：N 大时 `mask` 与 `attn` 都是 `N×N`，做长文本/高分辨率实验要自己追窗口化实现。
5. 论文的 `N=6625`（中文）在代码里由 `character_dict_path` + `character_type` 决定，不在 backbone 里。

## 相关笔记
- [[SVTR 场景文本识别]] — 论文整体内容
- [[Restormer 的 MDTA 与 GDFN 实现]] — 同类「论文公式 vs 官方代码」的对照写法
- [[各种Norm层的摆放]] — `prenorm` 反命名坑的背景知识
- [[transformer的2道残差连接]] — SVTR block 的两道残差

#代码 #Transformer #Attention #场景文本识别
