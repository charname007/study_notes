---
created: 2026-11-08
source: E:\.dsh-paper-reader\data\默认专题\Luo 等 - 2026 - Restore Text First, Enhance Image Later Two-Stage Scene Text Image Super-Resolution with Glyph Stru.pdf（CVPR 2026 Open Access；南开大学 PCA Lab + vivo BlueImage Lab）
type: 论文
tags: [论文, 超分辨率, 场景文本识别, 扩散模型]
---

# TiGeSR：先恢复文本、再增强图像的两阶段场景文本图像超分

> Minxing Luo, Linlong Fan, Qiushi Wang, Ge Wu, Yiyan Luo, Yuhang Yu, Jinwei Chen, Yaxing Wang, Qingnan Fan, Jian Yang（南开大学 + vivo，实习期工作）。页码均指本机 PDF 页码（11 页，正文至第 8 页 + 参考文献）。

## 一句话总结

把「字形结构恢复」与「整图增强」显式解耦：Stage 1 用扩散式文本修复器在文本区域恢复精细笔画，拼成一张文本掩码；Stage 2 把这张掩码当作 ControlNet 式条件注入 Stable Diffusion 3.5，引导全图超分——即 **"text-first, image-later"（先文本、后图像）**（p.1 摘要、p.3 第 3 节）。同时提出 **UZ-ST**（最大 ×14.29 变焦）真实场景中文文本超分数据集（p.1、p.5）。

## 核心问题

场景文本超分（STISR）长期存在「**可读性 vs 图像质量**」的权衡（p.1–2）：

- 通用/真实世界图像超分（含扩散类 HAT、SeeSR、DiffBIR、OSEDiff、DiT4SR 等）以整体画质为目标，常把文字区域扭曲成乱码；中文笔画复杂、轻微变形即改变语义，容忍度极低（p.1）。
- 只处理文本区域的方法（MARCONet、DiffTSR 等）虽提升可读性，但**缺少全局背景约束**，文字与背景间出现风格不一致和块伪影（p.2、p.3）。
- 已有数据集（TextZoom、CTR、Real-CE）退化轻微（≤×4），无法反映长焦真实场景（p.2、p.5）。

作者的关键观察：若把「文本」与「非文本」分开处理，二者并不必然互斥——先用专门机制重建文本结构，再用它引导整图复原，即可兼顾笔画保真与整体协调（p.2）。

## 关键方法

### 两阶段框架（p.3 Fig.2、第 3.1 节）

- **Stage 1 · 文本恢复**：OCR 检测器定位文本区域并给出识别文本作语义条件；每个区域经 VAE 编码后与噪声拼接，由 UNet $\epsilon_\theta$ 迭代去噪成「外观 RGB」与「结构 mask」两支，文本内容经 cross-attention 注入引导结构；T 步后把 mask 分支解码、拼回原位得到文本掩码 $\hat{x}_m$（p.3 第 3.2 节）。
- **两阶段训练策略**（p.4）：真实退化文本缺分割标注、合成数据又不够真实 → **Phase 1** 混用合成 + 真实数据学真实退化；**Phase 2** 冻结 RGB/mask 输出块、只用合成数据精修掩码质量。损失 $\mathcal{L}=\lambda_{td}\mathcal{L}_{td}+\lambda_{Seg}\mathcal{L}_{Seg}$，其中 $\mathcal{L}_{Seg}$ 用 MSE + Focal + Dice 在像素级监督掩码（式 1–3，p.4）。
- **Stage 2 · 图像增强**（p.4–5 第 3.3 节）：基于 **Stable Diffusion 3.5** 的 ControlNet 式网络 $\epsilon_\phi$，只在单一时间步 $t$ 用空文本嵌入 $c_{Null}$ 去噪：$\hat{z}_H = z_L - \sigma_t \epsilon_\phi(z_L, \hat{z}_m, t, c_{Null})$（式 4）。训练损失 = MSE + LPIPS 重建损失（式 5）+ 用 **Sobel 算子**抽文本边界的 $L_{edge}$ 边缘损失（式 6–7，p.5）。

### UZ-ST 数据集（p.5 第 4 节）

用 vivo X200 Ultra 的 4 段固定焦距（14/35/85/200 mm）采集，最大变焦 **×14.29**；**5,036 对** LR–HR 图像、**49,675 行**文本，含店招/海报/文档、白天/室内/夜间多场景；200 mm 为 GT，×14.29/×5.71/×2.35 三档分别 1,439/1,798/1,799 对。用**级联由粗到细对齐**（按焦距排序、逐级估计单应矩阵 warp 到更高焦邻居、再对 200 mm GT 精修 + 人工筛选）解决大变焦错位（p.4 Fig.3、p.5）。评价用 PSNR/SSIM/LPIPS/DISTS/FID + 文本裁剪区指标 + 基于 Levenshtein 比的 **OCR-A**（式 8，p.5）。

> 方法实现：本文为 CVPR 2026 新论文，未找到可用官方代码实现，故不附代码；可对照 [[C3-STISR 三重线索引导的场景文本超分]]、[[VTCTSRN 视觉文本线索迁移的场景文本超分]] 的「字形图/文本线索」思路理解 Stage 1 的动机差异。

## 结论与局限

- **主结果（Tab.2，p.6）**：TiGeSR 在 Real-CE 与 UZ-ST 上图像质量与文字准确率均最优——Real-CE PSNR 24.12 / SSIM 0.839 / LPIPS 0.164 / OCR-A **67.3%**；UZ-ST PSNR 25.48 / SSIM 0.830 / OCR-A **43.0%**（均高于最强对照 TADiSR 的 64.7% / 36.6%）。
- **文本区域（Tab.3，p.6）**：TiGeSR 是唯一让 OCR-A 相对 LR **不降反升**的方法（Real-CE +2.5%、UZ-ST +1.3%），其余方法普遍下降（如 SupIR −37.0%、OSEDiff −38.0%）。
- **消融 · 数据集有效（Tab.4，p.8）**：UZ-ST 微调让 OSEDiff/DiT4SR/Ours 的 OCR-A 分别 22.8→28.9%、19.3→23.7%、40.0→43.0%，说明即便无专门文本结构机制也能从数据获益，但仍不及本方法。
- **消融 · 掩码质量（Tab.6，p.8）**：以 Stage 2 固定为基线，空掩码 59.5%、标准字体 55.3%、SAM-TS 提取 57.9%、LDM 引导 60.1%，而本方法文本恢复管线 **67.3%** 最高——证明「真实退化下恢复出高质量字形掩码」是最大贡献点。
- **消融 · OCR 依赖（Tab.5，p.8）**：把 Stage 1 的 OCR 预测文本换成空/随机文本，OCR-A 仍有 40.4%/40.3%（对照 TADiSR 35.5%），用预测文本 44.6%——模型主要靠 LR 输入内禀结构，不过度依赖 OCR。
- **局限（作者承认/可观察）**：TADiSR 受 cross-attention 分辨率限制在小字/重度退化下失效（p.6–7）；Stage 1 依赖 OCR 检测与两阶段训练、Stage 2 基于 SD3.5 且用 tile 推理，计算开销文中未系统报告；实验仅在 Real-CE 与自建 UZ-ST 上做，未覆盖更多语言/退化类型（p.5–8）。

## 我的疑问

- [ ] Stage 1 与 Stage 2 是否端到端联合微调，还是完全级联训练？两阶段误差如何累积？
- [ ] 文本掩码在 Stage 2 以「单一时间步」注入，是否默认了扩散起点对齐？对更大/更小尺度是否稳健？
- [ ] 中文场景是重点，但 UZ-ST 是否含足够英文/多语言文本以验证泛化？

## 关键摘录

> "TiGeSR explicitly decouples glyph restoration from image enhancement: it first reconstructs precise text structures and uses them to guide full-image super-resolution."（p.1 Abstract）

> "We propose TiGeSR, the first two-stage scene text super-resolution framework with a 'text-first, image-later' paradigm that decouples glyph restoration from image enhancement."（p.2 Contributions）

> "Our method uses RGB output to jointly learn from real-world degradations and accurate synthetic masks, yielding high-quality masks and achieving the highest accuracy."（p.8 消融结论）

## 相关笔记

- [[C3-STISR 三重线索引导的场景文本超分]] — 同属 STISR 的「文本先验引导」路线，用渲染字形图 + 语言线索，对照本文的两阶段解耦
- [[VTCTSRN 视觉文本线索迁移的场景文本超分]] — 同样用字形原型图当线索，关注对齐问题
- [[TPWGAN 小波域判别器引导的场景文本超分]] — 同为场景文本超分，改「监督放在哪」（小波域）而非改框架
- [[DocWaveDiff 小波U-Net与扩散的文档图像预测精炼]] — 同用扩散 + 预测-精炼思路，但面向文档图像增强
- [[HDW-SR 小波高频引导扩散的单图超分]] — 扩散式超分中「高频/结构条件注入」的另一种做法

> ⚠️ 架构图待补：Fig. 2（两阶段框架，p.3）与 Fig. 3（UZ-ST 对齐，p.4）是信息密度最高的图；本机渲染工具可用，但当前模型无法读图核对裁剪，故暂不插图，待人工截取或切换可读图模型后补。
