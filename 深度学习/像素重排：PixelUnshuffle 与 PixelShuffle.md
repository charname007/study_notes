---
created: 2026-10-30
source: Restormer 论文第 3 页；Shi et al. ESPCN (CVPR 2016)，即论文引文 [69]
type: 概念
tags:
  - 概念
  - 像素重排
  - 卷积
  - 图像修复
---

# 像素重排：PixelUnshuffle 与 PixelShuffle

## 是什么
两个**无可学习参数的纯张量重排**算子（reshape + transpose），互为逆运算：

| 操作 | 别名 | 形状变化 | 方向 |
|---|---|---|---|
| pixel-unshuffle | space-to-depth | `C×H×W → (C·r²)×(H/r)×(W/r)` | 分辨率↓，通道↑ |
| pixel-shuffle | depth-to-space | `(C·r²)×(H/r)×(W/r) → C×H×W` | 分辨率↑，通道↓ |

关键：没有插值、没有加权求和、没有信息损失，`pixel_shuffle(pixel_unshuffle(x)) == x` 严格成立。

## 为什么需要
Restormer 用它做编码器的降采样与解码器的上采样（论文第 3 页原话："For feature downsampling and upsampling, we apply pixel-unshuffle and pixel-shuffle operations [69], respectively."）。

相比 stride-2 卷积 / 池化 / 双线性插值的优势：
1. **零参数零信息损失** —— 池化丢一半像素，stride-2 卷积把 4 个像素压成固定加权和；unshuffle 一个都不丢，"怎么融合"交给紧随其后的卷积。
2. **严格可逆、几何对齐** —— 编解码重采样天然对称，不需要额外尺寸对齐逻辑。
3. **避开棋盘伪影** —— pixel-shuffle 源自 ESPCN 的亚像素卷积：先在小分辨率卷积出 r² 个通道再一次性铺开，比 ConvTranspose2d 的重叠相加干净。
4. **算力平移而非爆炸** —— 分辨率减半 token 数降到 1/4，通道翻倍 C² 涨 4 倍，两者抵消，每层算力近似恒定（配合 Restormer 的通道维 MDTA）。

## 怎么做 / 关键细节

r=2、单通道 4×4 的直观图：

```
输入 4×4：                 unshuffle 后：2×2 × 4 通道
a b c d                    ch0 (偶行偶列) = [a c ; i k]
e f g h                    ch1 (偶行奇列) = [b d ; j l]
i j k l                    ch2 (奇行偶列) = [e g ; m o]
m n o p                    ch3 (奇行奇列) = [f h ; n p]
```

### 纯原理版（手写，看清每一步）

```python
import torch

def pixel_unshuffle(x, r=2):
    """space-to-depth: (B,C,H,W) -> (B, C*r*r, H/r, W/r)"""
    B, C, H, W = x.shape
    assert H % r == 0 and W % r == 0, 'H、W 必须能被 r 整除'
    x = x.view(B, C, H // r, r, W // r, r)   # B,C,hr,r,wr,r
    x = x.permute(0, 1, 3, 5, 2, 4)          # B,C,r,r,hr,wr  把 r,r 提到前面
    return x.reshape(B, C * r * r, H // r, W // r)

def pixel_shuffle(x, r=2):
    """depth-to-space: (B, C*r*r, H, W) -> (B, C, H*r, W*r)"""
    B, Crr, H, W = x.shape
    C = Crr // (r * r)
    x = x.view(B, C, r, r, H, W)             # 逆向拆通道
    x = x.permute(0, 1, 4, 2, 5, 3)          # B,C,H,r,W,r
    return x.reshape(B, C, H * r, W * r)

x = torch.arange(4 * 4).float().view(1, 1, 4, 4)
y = pixel_unshuffle(x, 2)   # [1, 4, 2, 2]
z = pixel_shuffle(y, 2)     # [1, 1, 4, 4]
print(torch.equal(z, x))    # True
```

### Restormer 官方写法（卷积投影 + 重排绑定成模块）

```python
import torch.nn as nn

class Downsample(nn.Module):
    """H×W×C -> (H/2)×(W/2)×(2C)"""
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, padding=1, bias=False),
            nn.PixelUnshuffle(2),          # (n_feat//2)*4 == n_feat*2
        )
    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    """(H/2)×(W/2)×(2C) -> H×W×C"""
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),            # (n_feat*2)/4 == n_feat/2
        )
    def forward(self, x):
        return self.body(x)
```

读法：`n_feat // 2` 的卷积先把通道减半，unshuffle 的 ×4 再顶回 2 倍 —— 净效果就是论文图里的"通道翻倍"；上采样是它的镜像。裸重排本身不可学习，真正学"怎么融合"的是那个 3×3 卷积。

形状演示：
```python
x = torch.randn(1, 48, 128, 128)
Downsample(48)(x).shape    # [1, 96, 64, 64]
Upsample(96)(Downsample(48)(x)).shape   # [1, 48, 128, 128]
```

## 常见误区 / 注意事项
1. **整除性**：Restormer 是 4 级编码器（最深 1/8），输入 H/W 必须是 **8 的倍数**；官方实现先 reflection pad、推理后裁回，否则 `view` 直接报错。
2. **"可逆"只是几何层面的**：单次重排无损，但中间一旦接了卷积，信息就被改写，不要误认为整个模块可逆。
3. 别当成"降采样就等于丢信息"：它只是把空间信息搬进通道，通道容量不够时后续卷积会挤掉。

## 相关笔记
- [[Restormer 高效高分辨率图像修复 Transformer]] — 第 3 页的用法上下文
- [[Restormer 的 MDTA 与 GDFN 实现]] — 同属架构代码
- [[PointWise和DepthWise]] — 1×1 与 depthwise 的分工

#概念 #像素重排 #卷积 #图像修复
