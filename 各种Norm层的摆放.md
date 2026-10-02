---
tags:
  - Norm
  - LayerNorm
  - BatchNorm
  - InsanceNorm
  - GroupNorm
---
在深度学习中，Normalization 层（BN、LN、GN、IN、RMSNorm 等）的摆放位置直接决定了**梯度的流动效率**、**深层网络的训练稳定性**以及**模型的收敛速度**。

Norm 层的摆放主要需要从两个维度来考虑：
1. **微观层面**：Norm 与线性层（Conv/Linear）和激活函数（ReLU/GELU）的先后顺序。
2. **宏观层面**：在残差连接（Residual Stream）中采用 **Pre-Norm** 还是 **Post-Norm**。

---

### 一、微观层面：与卷积/全连接层、激活函数的相对位置

#### 1. 经典默认：`Linear/Conv -> Norm -> Activation`（最常用）
* **代表模型**：标准 ResNet、VGG-BN、RegNet
* **逻辑**：线性变换（Conv/Linear）的输出分布往往会发生偏移（Internal Covariate Shift）。先用 Norm 将其强行拉回到零均值、单位方差的标准分布附近，再输入给激活函数。这样激活函数（如 ReLU、SiLU、GELU）可以工作在它最敏感、非线性最强的区间。
* **工程细节**：**当 Norm 紧跟在 Linear/Conv 后面时，前置的 Linear/Conv 必须设置 `bias=False`**。因为 Norm 自带可学习的偏移参数 $\beta$，前面的 bias 会在均值归一化步骤中被完全减去，纯属冗余参数。

#### 2. Pre-activation 模式：`Norm -> Activation -> Linear/Conv`
* **代表模型**：ResNet-v 2、DenseNet
* **逻辑**：在输入下一个权重层前先做归一化和非线性变换。这种设计通常出现在残差分支的内部，目的是让残差的主干（Skip Connection）保持纯粹的恒等映射（Identity Mapping）。

#### 3. 极少使用：`Linear/Conv -> Activation -> Norm`
* **原因**：如果先过激活函数（如 ReLU 将所有负数截断为 0），再做 Norm，输出的分布会被硬性打破单峰或对称性，容易破坏激活函数的非线性特征，通常收敛效果不如前两种。

---

### 二、宏观层面：残差结构中的 Pre-Norm vs Post-Norm

在现代深层网络（尤其是 Transformer、ViT、LLM）中，核心争论与演进是 **Pre-Norm** 与 **Post-Norm**：

```text
Post-Norm (早期设计):
输入 x ──┬─────────────────────► ( + ) ──► [ Norm ] ──► 输出
         │                       ▲
         └────► [ SubLayer ] ────┘
         (公式: x = Norm(x + SubLayer(x)))

Pre-Norm (现代主流):
输入 x ──┬─────────────────────────────► ( + ) ───────► 输出
         │                               ▲
         └────► [ Norm ] ──► [ SubLayer ] ─┘
         (公式: x = x + SubLayer(Norm(x)))
```

| 架构对比 | **Post-Norm**（如原始 Transformer, BERT） | **Pre-Norm**（如 GPT-2/3, LLaMA, ViT, Swin） |
| :--- | :--- | :--- |
| **残差路径** | 残差主干上每次都要经过 Norm | **残差主干是纯净的高速公路**，Norm 只在子层分支内部 |
| **训练稳定性** | 深层容易梯度爆炸/消失，极度依赖 Warmup 和精细的调参 | **极其稳定**，即使几百层也能直接训练，不需要复杂的 Warmup |
| **容量上限** | 理论上稍高（因为没有被残差分支无损稀释） | 极深层时残差流信号可能被放大，但容易训练且上限足够高 |
| **当前地位** | 逐渐被边缘化，仅在特定场景使用 DeepNorm 挽回 | **现代大语言模型和视觉 Transformer 的事实标准** |

> **关键提醒**：在采用 **Pre-Norm** 架构时，在网络的**最后一层（进入 LM Head 或分类头之前），必须额外补一个 Final Norm**，以约束由于不断累加而幅值变大的残差主干信号。

---

### 三、不同类型 Norm 在具体模型中的经典摆放

#### 1. BatchNorm (BN) —— 图像分类/CNN
* **标准位置**：`Conv -> BN -> ReLU`
* **特点**：依赖 Batch 维度计算。如果 Batch Size 太小（如显存吃紧的目标检测、分割任务），BN 的统计量会极不准确，此时应避免使用，改用 GN 或 SyncBN。

#### 2. LayerNorm / RMSNorm —— NLP、LLM 与新一代视觉模型
* **现代 LLM（LLaMA、Mistral 等）的标准范式**：
  * Attention 前：`RMSNorm -> Multi-Head Attention -> 残差相加`
  * FFN 前：`RMSNorm -> SwiGLU/MLP -> 残差相加`
  * 输出前：`Final RMSNorm -> Linear (Vocab Projection)`
* **现代纯卷积模型（ConvNeXt）**：
  * 抛弃了 BN，改用 LN，模仿 Transformer 的宏观结构：
    `Depthwise-Conv -> LayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv -> 残差相加`

#### 3. GroupNorm (GN) —— 目标检测、图像分割、扩散模型 (Diffusion)
* **标准位置**：`GN -> Activation -> Conv` 或 `Conv -> GN -> Activation`
* **应用场景**：Batch Size 极小（如 1 或 2）的视觉生成任务。
* **扩散模型（Stable Diffusion, DiT 等）中的变体**：
  * **AdaGN / AdaLN（自适应归一化）**：不仅是单纯的归一化，还用时间步（Timestep）和文本 Embedding 通过一个小 MLP 预测出 Norm 的 scale ($\gamma$) 和 shift ($\beta$)：
    $$\text{AdaLN}(x, c) = (1 + \gamma_c) \cdot \text{Norm}(x) + \beta_c$$
  * 摆放位置同样前置于每个 Attention 块和 ResBlock 的核心计算之前。

#### 4. InstanceNorm (IN) —— 风格迁移、GAN、医学图像
* **标准位置**：`Conv -> IN -> LeakyReLU`
* **逻辑**：仅针对单个样本的单个通道做归一化，剥离了整张图像的全局对比度和明暗风格，保留几何结构。

---

### 四、设计与排布的 3 条黄金法则

1. **“保护高速公路”原则（Identity Stream）**：
   在搭建带 Residual 的深层网络时，尽量保证残差主干（Skip connection）上不要插入任何阻碍梯度的层（包括激活和 Norm）。**让 Norm 退居到子模块（Attention、FFN、ConvBlock）的入口处（即 Pre-Norm 策略）**。
2. **算子融合省显存原则**：
   现代推理与训练框架（如 FlashAttention、Triton、TensorRT）通常会将 `Linear/Conv + Bias + Norm + Activation` 自动融合为单个算子（Fused Kernel）。保持 `Conv -> Norm -> Act` 这种紧凑顺序有利于引擎进行算子融合，降低访存开销（Memory Bandwidth）。
3. **参数剔除原则**：
   凡是紧接着 Norm 层的 Linear 或 Conv，都要显式声明 `bias=False`（除非使用的是没有可学习仿射参数的 Norm，如 `affine=False` 或 `elementwise_affine=False`）。