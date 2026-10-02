---
tags:
  - Norm
  - LayerNorm
  - BatchNorm
  - InsanceNorm
  - GroupNorm
---
### 与卷积/全连接层、激活函数的相对位置
#### Linear/Conv -> Norm -> Activation（最常用）
- **逻辑**：线性变换（Conv/Linear）的输出分布往往会发生偏移（Internal Covariate Shift）。先用 Norm 将其强行拉回到零均值、单位方差的标准分布附近，再输入给激活函数。这样激活函数（如 ReLU、SiLU、GELU）可以工作在它最敏感、非线性最强的区间。
    
- **工程细节**：**当 Norm 紧跟在 Linear/Conv 后面时，前置的 Linear/Conv 必须设置 bias=False**。因为 Norm 自带可学习的偏移参数
    
    ```
    ββ
    ```
    
    ，前面的 bias 会在均值归一化步骤中被完全减去，纯属冗余参数。