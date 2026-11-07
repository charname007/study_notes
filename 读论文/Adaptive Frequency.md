![[Pasted image 20261009110457.png]]
Efficient All-in-One Image Restoration With  [[读论文/Adaptive Frequency|Adaptive Frequency]] Enhancement

利用 1\*1 卷积，为每一个像素生成 K\*K 卷积核，展开，傅里叶变换，在 u, v 频域，根据 DC component 距离划分为 N 组，另一个 1\*1 conv 生成对 N 组的分别的注意力，再逆傅里叶变换，卷积

→ 完整解读见 [[AdaptIR 自适应频率增强的一体化图像复原]]（AFEM 章节，论文 p.5）