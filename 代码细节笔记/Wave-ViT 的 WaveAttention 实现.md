---
created: 2026-11-07
source: YehLi/ImageNetModel（官方实现，https://github.com/YehLi/ImageNetModel ，commit 0b4302e，2023-05-05）；对应论文 Wave-ViT (ECCV 2022)
type: 代码
tags: [代码, 小波变换, Transformer, Attention]
---

# Wave-ViT 的 WaveAttention 实现

## 场景

读 [[Wave-ViT 小波可逆下采样的多尺度ViT]] 时想确认三件事：论文里「DWT → 3×3 Conv → IDWT」这条链在代码里到底是什么顺序、张量形状怎么变、以及 GFLOPs 表里的「comparable cost」是不是真的。下面每一行都来自本次实际下载阅读的源码：

- 仓库：`YehLi/ImageNetModel`（论文摘要 p.1 直接给出，属官方实现）
- commit：`0b4302e0b10ff12013b03b799dda50393efa69a9`（2023-05-05）
- 文件：`classification/wavevit.py`、`classification/torch_wavelets.py`

## 关键点

### 1) 模块构成（wavevit.py:159-186）

```python
class WaveAttention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio):
        self.dwt  = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
        self.reduce = nn.Sequential(nn.Conv2d(dim, dim // 4, 1), nn.BatchNorm2d(dim // 4), nn.ReLU(inplace=True))
        self.filter = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True))
        self.kv_embed = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio) if sr_ratio > 1 else nn.Identity()
        self.q  = nn.Linear(dim, dim)
        self.kv = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2))
        self.proj = nn.Linear(dim + dim // 4, dim)   # 注意是 dim + dim//4
```

`proj` 的输入维度是 **dim + dim//4**，直接印证论文式(3)的 `Concat(head…, X_r)`——IDWT 支路是 D/4 通道、与 D 通道的注意力输出拼起来再投回 D。

### 2) 前向的真实顺序（wavevit.py:203-222）

```python
def forward(self, x, H, W):
    q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

    x = x.view(B, H, W, C).permute(0, 3, 1, 2)          # token → 2D 特征图
    x_dwt  = self.dwt(self.reduce(x))                   # 1×1 降维 → Haar DWT（四子带按通道拼成 D 通道）
    x_dwt  = self.filter(x_dwt)                         # 3×3 卷积做局部建模
    x_idwt = self.idwt(x_dwt)                           # 逆变换 → H×W×D/4
    kv     = self.kv_embed(x_dwt)                       # 可选再跨步 sr_ratio 降分辨率
    kv  = self.kv(kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    k, v = kv[0], kv[1]

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(torch.cat([x, x_idwt], dim=-1))        # 长程分支 + 局部重构分支
    return x
```

张量形状（以 stage 1、输入 H×W×64 为例）：

| 步骤 | 形状 |
|---|---|
| 输入 x | (B, N=H·W, 64) |
| reduce(x) | (B, 16, H, W) —— 通道降到 1/4 |
| DWT | (B, 64, H/2, W/2) —— 4 个 (B,16,H/2,W/2) 子带按通道拼接 |
| filter | (B, 64, H/2, W/2) |
| idwt | (B, 16, H, W) → 拉平成 (B, H·W, 16) |
| kv_embed(x_dwt) | 当 sr_ratio>1 再降一次；stage1 的 sr_ratio=4 |
| proj 输入 | (B, N, 64+16=80) → 输出 (B, N, 64) |

### 3) 与论文的三处出入（读码才发现）

1. **多了一次 `sr_ratio` 跨步卷积**。论文算法描述只讲 DWT 把 K/V 降到 H/2×W/2（p.7），但代码里 `kv_embed` 还会再按 `sr_ratios=[4,2,1,1]` 做一次跨步卷积——也就是 stage 1 的真实 K/V 分辨率是 **H/8 × W/8**，不是 H/2 × W/2。论文正文没有明确这一点。
2. **只有前两个 stage 用 Wavelets block**。`wavevit.py:423` 写死 `block_type='wave' if i < 2 else 'std_att'`，后两个 stage 退化成普通注意力。论文只说「四 stage 都用 Wavelets block 堆叠」（p.8），细节只能从代码读到。
3. **`reduce` 用的是 1×1 卷积 + BN + ReLU，`filter` 用的是普通 3×3 卷积（groups=1）**，不是 depthwise。论文只写「3×3 convolution to impose spatial locality」，没说是标准卷积，参数量差别不小。

### 4) DWT/IDWT 的实现（torch_wavelets.py）

- 不是 `nn.Conv2d`，而是自定义 `torch.autograd.Function`：`DWT_Function` / `IDWT_Function`（:11、:39），滤波器来自 `pywt.Wavelet('haar')` 的 `dec_lo/dec_hi` 与 `rec_lo/rec_hi`（:96-103、:73-80），注册成 buffer。
- **滤波器被转成 float16**（:110-113、:88），并要求输入也是 fp16，因此训练必须开 AMP；这也是论文里「negligible cost」的实现前提之一。
- `IDWT_Function.forward` 里 `x = x.view(B, 4, -1, H, W).transpose(1, 2)`（:46）——先把「按通道拼接的四个子带」还原成 4 组再上采，顺序必须与 DWT 的 `torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)`（:23）严格一致，写错顺序不会报错但精度会崩。

## 与库内其他小波实现的对照

- [[LWTformer 可学习小波变换的古汉字图像修复]]：那边把小波核做成可学习卷积，本篇是**固定 pywt 常量核 + 自定义 autograd**，两者在「是否学核」上正好相反。
- [[StarIR 的 StarModule 与 DFFN 实现]]：同属「论文一句带过、代码另有实现细节」的典型，可对照读法（先看 config，再看类定义）。
- [[AdaptIR 自适应频率增强的一体化图像复原]]：RFEM 也是 DWT→卷积→IDWT，但用 stride-2 depthwise 卷积实现 Haar、逆变换用转置卷积；本篇用 pywt 精确核 + 手写 Function，两种实现路径值得对比。

## 相关笔记

- [[Wave-ViT 小波可逆下采样的多尺度ViT]]
- [[小波变换]]
