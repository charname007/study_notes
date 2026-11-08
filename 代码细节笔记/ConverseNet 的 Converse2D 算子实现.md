---
created: 2026-11-08
source: https://github.com/cszn/ConverseNet @ 368aa39cebaa075b04926264c7494d969e8ee835 (2025-08-15)
type: 代码
tags: [代码, 卷积算子, 频域, 图像复原]
---

# ConverseNet 的 Converse2D 算子实现

## 场景

读 [[ConverseNet 反向卷积算子与图像复原]] 时对照官方实现，核验证文式(6)（FFT 闭式解）、式(9)（$\lambda$ 参数化）、Converse Block 与三网络骨架到底怎么落成代码，并记录**论文与 released 代码的不一致**。

仓库：`cszn/ConverseNet`（论文第 1 页给出的官方链接，README 自述 ICCV 2025）。本文所有代码均为本次真实读取 `main` 分支、HEAD commit `368aa39`（2025-08-15 "Update README.md"）所得，非默写。涉及文件：`models/util_converse.py`、`models/converse_dncnn.py`、`models/converse_usrnet.py`、`README.md`。

## 关键点

### 1. `Converse2D.forward` = 式(6) 的逐行翻译（util_converse.py）

~~~python
self.biaseps = torch.sigmoid(self.bias - 9.0) + self.eps        # 式(9) lambda: Sigmoid(b-9)+eps, b 逐通道可学
STy = self.upsample(x, scale=self.scale)                        # Y↑s：填零 s 倍上采样
if self.scale != 1:
    x = nn.functional.interpolate(x, scale_factor=self.scale, mode='nearest')   # X0 = Interp(Y,s)
FB  = self.p2o(self.weight, (h*self.scale, w*self.scale))       # FK = FFT(核)，p2o 用 roll 消除 PSF 偏心
FBC = torch.conj(FB)                                           # 共轭
F2B = torch.pow(torch.abs(FB), 2)                               # |FK|^2
FBFy = FBC * torch.fft.fftn(STy, dim=(-2, -1))                  # conj(FK)*F(Y↑s)
FR  = FBFy + torch.fft.fftn(self.biaseps * x, dim=(-2, -1))     # L = 上式 + lambda*F(X0)
x1  = FB.mul(FR)                                               # FK * L
FBR   = torch.mean(self.splits(x1, self.scale), dim=-1)         # (FK L)⇓s：s×s 块平均
invW  = torch.mean(self.splits(F2B, self.scale), dim=-1)        # |FK|^2⇓s
invWBR = FBR.div(invW + self.biaseps)                          # /( |FK|^2⇓s + lambda )
FCBinvWBR = FBC * invWBR.repeat(1, 1, self.scale, self.scale)   # conj(FK) ⊙s 上式并广播回 s×s
FX  = (FR - FCBinvWBR) / self.biaseps                          # 式(6) 括号内 = (L - ...)/lambda
out = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))             # F^{-1} → X*
~~~

对照论文式(6)：$L-\overline{F_K}\odot_s\frac{(F_KL)\Downarrow_s}{|F_K|^2\Downarrow_s+\lambda}$ 整体再 $/\lambda$——代码里 `FR` 就是 $L$，`invWBR` 是分式，`FCBinvWBR` 是共轭核乘回去，完全一致。`splits()` 用 `view + permute` 把 `[..., H, W]` 拆成 `[..., H/s, W/s, s²]`，`mean` 对应 $\Downarrow_s$（块平均）；`upsample()` 是 `z[..., st::scale, st::scale].copy_(x)` 的填零上采样，对应 $\uparrow_s$。

### 2. 四个实现要素在代码里的落点

- **核 Softmax 初始化**（式 8）：`self.weight.data = nn.functional.softmax(self.weight.data.view(1,C,-1), dim=-1).view(...)`，构造时即归一化（非负、和为 1）；`assert out_channels == in_channels` 强制 depthwise。
- **circular padding**：`F.pad(x, ..., mode=self.padding_mode)`，`Converse2D.__init__` 默认 `padding_mode='circular'`；forward 末尾把 padding 按 `*scale` 裁回去。与式(6) 的圆形边界假设自洽。
- **$\lambda$ 可学**（式 9）：`self.bias` 形状 `(1, C, 1, 1)`，`torch.sigmoid(self.bias-9.0)+eps`，初值全 0 → $\lambda\approx1.2\times10^{-4}$，即训练初期强数据保真。
- **$X_0=\mathrm{Interp}(Y,s)$**：`nearest` 插值（代码里还留了 `bilinear` 的注释备选和 `# x = torch.zeros_like(x)` 对应 $X_0=0$ 的消融开关）。

### 3. Converse Block：空间/通道解耦（util_converse.py）

`ConverseBlock` 两个残差子块，第一块插 Converse2D：

~~~python
self.conv1 = nn.Sequential(LayerNorm(in_channels, ..., data_format="channels_first"),
                           nn.Conv2d(in_channels, 2*out_channels, 1, 1, 0),   # 1×1 升维到 2C
                           nn.GELU(),
                           Converse2D(2*out_channels, 2*out_channels, kernel_size, scale=scale, ...),
                           nn.GELU(),
                           nn.Conv2d(2*out_channels, out_channels, 1, 1, 0))  # 1×1 压回 C
self.conv2 = nn.Sequential(LayerNorm(...), nn.Conv2d(...), nn.GELU(), nn.Conv2d(...))
def forward(self, x):
    x = self.conv1(x) + x
    x = self.conv2(x) + x
    return x
~~~

即论文 Fig 2 的 LayerNorm → 1×1 → GELU → Converse2D → GELU → 1×1（+残差），再叠一个纯 1×1 的 FFN 式子块。`ConverseBlockAlphaVariant` 另给两个可学 `alpha` 缩放残差（去模糊的 `ConverseUSRNet` 用的是这个变体），论文正文未提。`LayerNorm` 是自定义的 channels_first 版（对 C 维做均值/方差归一化），不是 `nn.LayerNorm` 默认行为。

### 4. 三网络骨架（converse_dncnn.py / converse_usrnet.py）

- `ConverseDnCNN`：`Conv2d(1→64,1×1)` → `20 × ConverseBlock` → `Conv2d(64→1,1×1)`，与 Fig 3(a) 一致。
- `ConverseUSRNet`：`conv1(3→64)` → 迭代 5 次：`ConvReverseDataNet`（数据模块，把 KernelNet 输出的 64 通道核当 K 直接喂进式(6)，`alpha` 学 $\lambda$）→ `MultiConverseBlockAlphaVariant`（7 块去噪先验）→ `conv2(64→3)`。`KernelNet` 用 `Linear(k²→64→64→16k²)` + GELU 把 7×7 模糊核嵌成 `16×7×7`，再由 `convs[i]` 的 1×1 投到 64 通道。这正是第 6 页"在特征域条件化模糊核"的实现。`ConverseNet`（盲去模糊）就是去掉 `KernelNet`/`d` 的纯 `MultiConverseBlock` 版本。

### 5. ⚠️ 论文 / 代码差异（复现时要注意）

| 项 | 论文 §5（第 6 页） | released 代码 |
|---|---|---|
| 去噪核尺寸 | "use a kernel size of **5 × 5**" | `converse_dncnn.py`: `kernel_size=7, padding=6` |
| padding 尺寸 | "circular padding of size **4**" | 同上 `padding=6`（注释建议 `kernel_size-1`） |
| Block 默认 | — | `ConverseBlock` 类默认 `kernel_size=3, padding=2, padding_mode='replicate'`（去噪显式传 `circular` 覆盖） |
| 上采样插值 | `Interp(Y,s)` | `nearest`（`bilinear` 被注释掉） |

去模糊 `converse_usrnet.py` 里 `MultiConverseBlock` 用的是 `kernel_size=3, padding=2`，与论文"5×5 默认"也不一致。README 的 Models 表给的 Converse-DnCNN 参数量 **734,913** 与论文表 4 一致，说明表 4 的实验大概率就是这份代码跑出来的——**那么论文 §5 的"5×5/4"文字与正式实验配置对不上**。复现实验请以 `main_test_converse_*.py` + 上述模型文件为准，别只照抄 §5。

另：仓库 `README.md` 把标题写成 "Reverse Convolution and Its **Application** to Image Restoration"（单数、无 s），与 PDF 标题 "...and Its **Applications**..."（复数）略有出入；citation 里 `booktitle={International Conference on Computer Vision}` 应为 ICCV 2025。

## 相关笔记

- [[ConverseNet 反向卷积算子与图像复原]] —— 论文六段解读与带页码摘录
- [[PointWise和DepthWise]] —— 1×1 Conv 补通道混合、Converse2D 管空间依赖的 DW/PW 分工
- [[StarIR 的 StarModule 与 DFFN 实现]] —— 同为"官方代码逐行对照 + 论文/代码差异"的读法
- [[Wave-ViT 的 WaveAttention 实现]] —— 另一种频域/可逆下采样算子的代码实现对照
