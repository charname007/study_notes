Diffusion Transformer meets Multi-level Wavelet Spectrum  for Single Image Super-Resolution
![[Pasted image 20261009111143.png]]
认为 LF 中包含部分 HF（小波变换可以不断分解 LF）
LEDec 提取 LF 主体，再通过 HDDec 预测高频部分

→ 完整解读见 [[DTWSR 扩散Transformer建模多级小波谱的单图超分]]（§双解码器 + M_high 掩码，p.5 式 8–9、图 4；消融见表 6）