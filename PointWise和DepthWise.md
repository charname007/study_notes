---
tags:
  - Conv
  - 卷积
  - PointWise
  - DepthWise
---
在卷积神经网络（尤其是轻量化网络）中，**Pointwise 卷积（1×1 卷积）和 Depthwise 卷积（逐通道卷积）的位置关系并不是绝对唯一的**，而是取决于具体的网络架构设计。

最常见的位置关系有以下三种：

---

### 1. 先 Depthwise，后 Pointwise（最经典的“深度可分离卷积”）
* **结构顺序**：**DW (k×k) $\to$ PW (1×1)**
* **典型代表**：**MobileNetV 1**
* **设计逻辑**：
  * **第一步（DW）**：只在每个通道内独立做空间卷积（如 $3\times3$），负责**提取空间/空间局部特征（Spatial Filtering）**，但各通道之间互不通信。
  * **第二步（PW）**：用 $1\times1$ 卷积将所有通道的特征线性组合起来，负责**跨通道信息融合（Channel Mixing）**。
* **特点**：将传统卷积的“空间计算”和“通道计算”彻底解耦，计算量和参数量大幅下降。

---

### 2. Pointwise 包夹 Depthwise（“三明治”倒残差结构）
* **结构顺序**：**PW (升维) $\to$ DW (k×k) $\to$ PW (降维)**
* **典型代表**：**MobileNetV 2、MobileNetV 3、ConvNeXt 等**
* **设计逻辑**：
  * **第一步（PW 升维）**：用 $1\times1$ 卷积把通道数放大（例如扩大 6 倍），扩展特征维度。
  * **第二步（DW 空间特征提取）**：在高维空间中进行 Depthwise 卷积，因为通道数多，能提取更丰富、细致的空间特征。
  * **第三步（PW 降维）**：再用 $1\times1$ 卷积把通道数压缩回原来的低维，输出并配合残差连接（Linear Bottleneck）。
* **特点**：解决了 DW 卷积在通道数较少时表达能力不足的问题，是目前轻量化网络最主流的设计。

---

### 3. 先 Pointwise，后 Depthwise（Xception 的初始设计）
* **结构顺序**：**PW (1×1) $\to$ DW (k×k)**
* **典型代表**：**Xception**
* **设计逻辑**：
  * Xception 源自 Inception 的“极端情况”（Extreme Inception）假设，认为可以先在通道维度做映射（$1\times1$ 卷积），然后再对映射后的每个输出通道分别做局部的空间卷积。
  * *注：Xception 作者在论文中也指出，DW 在前还是 PW 在前，对最终分类精度的影响其实很小。*

---

### 总结
* 如果被问到标准的**深度可分离卷积（Depthwise Separable Conv）**：通常指 **先 Depthwise 后 Pointwise**（DW $\to$ PW）。
* 如果是现代更主流的**倒残差模块（Inverted Residual）**：则是 **PW (升维) $\to$ DW $\to$ PW (降维)**。
* 核心本质永远是：**DW 负责空间聚合（H×W），PW 负责通道融合（C）**。