---
created: 2026-10-30
source: Restormer 论文第 3–4 页 eq(1)(2)；官方仓库 https://github.com/swz30/Restormer （basicsr/models/archs/restormer_arch.py）
type: 代码
tags:
  - 代码
  - Transformer
  - Attention
  - 图像修复
---

# Restormer 的 MDTA 与 GDFN 实现

## 场景
读 Restormer 论文时想知道 eq(1)/(2) 落到代码里长什么样，以及**为什么代码和公式对不上**。以下实现按官方仓库 `swz30/Restormer` 的 `restormer_arch.py` 变量命名与层顺序整理。

## 关键点

### 1. MDTA —— 注意力搬到通道维（论文第 3–4 页，eq(1)）

公式：

$\hat{X}=W_p\,\text{Attn}(\hat{Q},\hat{K},\hat{V})+X,\qquad \text{Attn}=\hat{V}\cdot\text{Softmax}\!\left(\hat{K}\hat{Q}/\alpha\right)$

要点是把 token 序列维与通道维**对调**：`K·Qᵀ` 得到 `C×C` 而非 `HW×HW` 的注意力图。

```python
import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange

class Attention(nn.Module):
    """Multi-Dconv Head Transposed Attention (MDTA)"""
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        # per-head 可学习温度，对应公式里的 alpha
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # 一次卷积同时出 q/k/v，比三个独立分支省显存
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # 3x3 depthwise 覆盖 3*dim 个通道 -> 局部上下文注入
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1,
                                    padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # 关键一步：通道维当序列，空间维当特征
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)   # 论文公式没写，代码对 q/k 做 L2 归一化
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature   # -> b head c c
        attn = attn.softmax(dim=-1)

        out = attn @ v                                         # b head c hw
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        return self.project_out(out)
```

**公式与代码的四处差异（读官方实现时最容易懵的地方）：**

1. 代码对 `q`/`k` 沿通道维做了 **L2 归一化**，eq(1) 没体现。它把点积限制在 `[-1,1]`，让可学习温度 `α` 更稳。
2. `α` 实现为 `nn.Parameter(torch.ones(num_heads,1,1))`，**每个 head 一个**，乘在 matmul 之后、softmax 之前。
3. eq(1) 的 `+ X` 残差**不在 Attention 内**，而在 `TransformerBlock` 里（Pre-LN：`x = x + attn(norm1(x))`）。
4. `heads=[1, 2, 4, 8]` —— level-1 分头数是 **1**，最高分辨率那层其实是单头，分头只在更深的低分辨率层展开。

### 2. GDFN —— 门控 + 深度卷积的前馈网络（论文第 4 页，eq(2)）

公式：

$\hat{X}=W_p^0\,\text{Gating}(X)+X,\qquad \text{Gating}(X)=\varphi(W_d^1W_p^1(\mathrm{LN}(X)))\ \odot\ (W_d^2W_p^2(\mathrm{LN}(X)))$

```python
class FeedForward(nn.Module):
    """Gated-Dconv Feed-Forward Network (GDFN)"""
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)   # 默认 2.66，不是 4

        # 一次投影出两份（对应门控两条分支），省一次卷积
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2,
                                kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)   # 等价于两条 depthwise 分支
        x = F.gelu(x1) * x2                       # φ(...) ⊙ (...)，即门控
        return self.project_out(x)
```

同样注意：eq(2) 画了 `+ X`，但官方 `FeedForward` **内部没有残差**，残差在 block 里。公式把残差画进模块是为了表达完整路径，实现上提到 block 省一次重复加法。

### 3. 两个模块的组装（对应论文 Fig. 2）

```python
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn  = Attention(dim, num_heads, bias)                # MDTA
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn   = FeedForward(dim, ffn_expansion_factor, bias)   # GDFN

    def forward(self, x):
        x = x + self.attn(self.norm1(x))   # Pre-LN + 一道残差
        x = x + self.ffn(self.norm2(x))    # 再一道残差
        return x
```

主干关键超参（去雨/去模糊配置，论文第 3 页 + 官方默认值）：

```python
net = Restormer(
    inp_channels=3, out_channels=3,
    dim=48,                      # 顶层通道数
    num_blocks=[4, 6, 6, 8],     # 4 级编码器 block 数，由浅入深递增
    num_refinement_blocks=4,     # 高分辨率精修阶段 block 数
    heads=[1, 2, 4, 8],          # level-1 单头
    ffn_expansion_factor=2.66,   # 为控参把 gamma 从 4 调小
    bias=False, LayerNorm_type='WithBias',
)
```

该配置约 **26.1M 参数 / 141 GFLOPs @256×256**（论文第 8 页 Table 8 消融模型统计）。

前向路径：`encoder_level1 → down1_2 → enc2 → down2_3 → enc3 → down3_4 → latent → up4_3 → concat(enc3) → 1×1 降维 → dec3 → … → refinement → patch_unembed`，输出残差图 `R`，与输入相加得 `Î = I + R`（第 3 页）。

### 4. 复现要点

1. **输入必须是 8 的倍数**（4 级 ×2 下采样）；官方先 reflection pad、推理后裁回，否则 `PixelUnshuffle` 报错。
2. `dim=48` 时 `48//2=24`、`24*4=96` 恰好接上 `2C`；自改 `dim` 需保证通道能逐级对齐。
3. 训练 300K iter + 余弦退火 3e-4 → 1e-6（第 5 页），单卡复现要按比例缩减并接受掉点。
4. 消融（第 8 页 Table 7）：MDTA 比标准 SA 基线 +0.32 dB，门控单独 +0.12 dB，GDFN 完整比标准 FFN +0.26 dB，合计 +0.51 dB。

## 相关笔记
- [[Restormer 高效高分辨率图像修复 Transformer]] — 论文整体内容
- [[像素重排：PixelUnshuffle 与 PixelShuffle]] — 同架构里的降/升采样算子
- [[transformer的2道残差连接]] — 这里两道残差的作用
- [[PointWise和DepthWise]] — 1×1 与 3×3 depthwise 的分工

#代码 #Transformer #Attention #图像修复
