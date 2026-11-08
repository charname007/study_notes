---
created: 2026-11-08
source: E:\.dsh-paper-reader\data\默认专题\Yu 等 - 2026 - RetFormer Multimodal Retrieval for Enhancing Image Recognition.pdf（CVPR 2026，CVF Open Access 版，pp. 2704–2714；本机 PDF 共 11 页）
type: 论文
tags: [论文, 检索增强, 多模态, CLIP]
---

# RetFormer：用图文检索增强长尾与含噪标签图像识别

> Tianrui Yu、Xiubo Liang*、Hongzhi Wang（浙江大学软件学院，宁波），CVPR 2026。下文页码 = 本机 PDF 页码（p.1 = 印刷页 2704）。

## 一句话总结

不把世界知识压进模型权重，而是外挂一个图文对知识库：查询图经冻结 CLIP 编码器得嵌入后 k-NN 检索出 top-k 图文邻居，再用「检索交叉融合模块」（跨模态注意力）算各邻居贡献，与原图 MLP 分支合并出 logits。这种半参数做法不显著增加计算开销，在长尾识别与含噪标签学习上取得 SOTA（p.1、p.2、p.3、p.4）。

## 核心问题

大模型把世界知识**隐式编码进海量参数**，由此带来灾难性遗忘、模型更新困难、可解释性差与扩展性问题；而真实数据又常同时具备长尾分布与噪声标签，真实分布难以估计（p.1）。以往方案多是「以图像为中心」的调权 / 噪声过滤，视觉识别里的检索增强仍欠探索，已有的 RAC、MAM 虽也做检索增强，却**只建模图像模态间的关系**，性能次优（p.2、p.6）。

作者的切入点是一个模态分工的观察（Fig. 1(c–e)，p.1–2）：同类检索结果里，图像模态给的是具体低层不变特征（形状、颜色、纹理），文本模态则携带高层抽象语义；不同类之间也存在可迁移的共享知识；文本还可能为噪声标签补上可选语义信息，且文本描述是可由专家总结的先验知识，在图像不足以学出通用类表征时尤其有用。

## 关键方法

### 1) 检索增强分类（p.3、p.4）

知识库 $\mathcal{D}=\{(I_i,T_i)\}_{i=1}^{L}$ 由图文对构成，**独立于训练集 $\mathcal{S}$**，且不假设含 $\mathcal{S}$ 的类标签（p.3）。两个冻结编码器 $\varepsilon_I,\\varepsilon_T$（CLIP，DataComp-1B 预训练）把库内图文转成嵌入 $E_i^I,E_i^T$；用查询图嵌入 $z_i$ 与各图像嵌入算余弦相似度取 top-k（式 2，p.4）。输出为两分支加权和（式 3，p.4）：

$$f(x_i)=\frac{\tau}{2}\left(L_I+L_R\right)=\frac{\tau}{2}\left(\mathrm{MLP}(z_i)+h\big(r(z_i,V_{NN}(z_i;\mathcal{V}_D))\big)\right)$$

主干分支是两层线性 + ReLU 的 MLP，检索分支由 $r(\cdot)$（交叉融合模块）产出，$\tau$ 平衡两者贡献（p.4）。

### 2) 检索交叉融合模块 —— 论文核心（p.4，式 4–6）

把 $z_i$ 与 top-k 邻居拼成图像嵌入 $E_{NN}^I$ 与文本嵌入 $E_{NN}^T$；**查询图自身对应的文本位置置 0 向量以防数据泄漏**。两者各自线性映射出 Q/K/V 三组矩阵（式 4）。关键在「交叉」二字（式 5）：

$$r(z_i,V_{NN})=\big[\mathrm{Att}(Q_T,K_I,V_I)+E_{NN}^I,\ \ \mathrm{Att}(Q_I,K_T,V_T)+E_{NN}^T\big]$$

即**用文本 Query 去查图像 K/V、用图像 Query 去查文本 K/V**，各加一次残差，$\mathrm{Att}(Q,K,V)=\sigma(QK^T/\sqrt{d})V$（缩放点积注意力，见 [[FlashAttention 精确注意力与IO感知]] 与 [[多头注意力+proj_out]]）。式 5 可堆 $L$ 层（式 6，默认 $L=4$，p.6），输入前按 ViT 设置加位置嵌入与 class token（p.4）；每层两处残差相加可对照 [[transformer的2道残差连接]]。

### 3) 知识库与复杂度（p.4、p.5）

- **三种建库来源**：下游数据集本身（保证每个查询至少有一个同类实例，但文本描述往往只有类名）；DataComp 的 xlarge 子集（约 1.4B 图文对）；All = 合并去重过滤后约 1.4B 图文对（p.4）。
- **向量化**：用 DataComp-1B 上预训练的 CLIP 参数初始化并**冻结**编码器，训练/验证前预先算好全部嵌入，使 1B 级内存数据集可用（p.4）。
- **检索**：Faiss 的 HNSW 近似 k-NN，亚线性 $O(\log N)$，10 亿级库查询毫秒级；因编码器冻结，k-NN 结果可预计算复用。代价是 ImageNet-LT 训练约需**额外 30GB 空间**（p.5）。

### 4) 梯度视角：为什么对尾类有效（p.5，式 7）

无检索模块时每个损失只对自己样本回传（one-to-one）；检索模块引入额外梯度项 $\partial L_i/\partial x_0$，等价于给每个标签注入「虚拟样本」，是一种**数据依赖的增强**。尾类缺样本，故受益最大（p.5）；这一解释与 p.8 的 k 消融、知识库规模消融相互印证。

![[RetFormer 架构图 Fig2.png]]

**怎么读这张图**（p.3 Fig. 2）：左侧 Query Image 先过 Image Encoder 得到查询嵌入，随后分**两路**——上路直接进 MLP 出 $L_I$；下路拿这个嵌入去 Knowledge Base 做 k-NN Search，检索出成对的 Image Embedding 与 Text Embedding（图中库内示例是 "A photo of indigo bunting." 一类图文描述）。检索到的两类嵌入进 **Retrieval Cross-fusion Module**：注意模块里 Dot Product 与 Element-wise Add 的交叉连线，正是式(5) 的「文本 Q 查图像 K/V、图像 Q 查文本 K/V」加上各自残差；输出经 FC 得 $L_R$。两路 logits 相加后由同一个损失训练。图中另一支虚线旁的 Text Encoder / Image Encoder 是**建库时**的冻结向量化器，不参与前向。

> 关键代码：⚠️ 论文未给出仓库链接，GitHub 与 Papers with Code 检索均未见官方实现（CVF Open Access 页面仅提供 PDF）。**未找到可用实现**，此处不挂代码。

## 结论与局限

- **任务与数据**：长尾识别用 CIFAR-100-LT（不平衡比 10/50/100）、ImageNet-LT、Places-LT、iNaturalist 2018；含噪标签用 WebVision（mini 设置取前 50 类）；按 Many(>100)/Med(20–100)/Few(<20) 三档报精度（p.2、p.5、p.6）。
- **实现**：ViT-B/16，用 CLIP(DataComp-1B) 初始化以**避免数据泄漏**（不用 ImageNet-21K）；25 epoch、lr 5e-4、batch 256、AdamW(wd 0.1)、label smoothing + Mixup、$L=4$、$k=32$（p.6）。
- **关键锚点**：ImageNet-LT overall **81.9%**（85.0/80.9/76.8），高于同样用 CLIP 初始化的 BALLAD 75.7、LIFT/PEL 78.3、MAM 78.3，也压过用额外数据的 RAC 76.7、VL-LTR 77.2（p.6 Table 1）；Places-LT overall **52.3%**、Few 51.9 为最高（p.6 Table 1）；iNaturalist 2018 overall **81.1%**（p.7 Table 2）；WebVision **81.7%** / ILSVRC12 **77.3%**（p.8 Table 4）。
- **消融洞察**（p.7–8，这部分比全表更值得留）：
  - **两模态缺一不可，且文本更值钱**（Table 5）：完整 81.9 → 去文本 76.0、去图像 78.1；冻结图像编码器 78.6；CLIP full FT 仅 74.6（Few 掉到 52.9）。
  - **k 不是越大越好**（Fig. 4 左）：精度升到 $k=32$ 后稳定；**Many-shot 反而随 k 下降**——头部类样本已够定决策边界，过大 k 引入的噪声标签反而有害；Few-shot 渐稳，与梯度视角一致。
  - **知识库越大越好，尾类涨幅最大**（Fig. 4 右）。
  - **L**：1/2 层低 0.4–0.7%（尾类尤甚），4→8 层仅 +0.11%，收益递减（Table 3）。
  - **HNSW 是必须的**（Table 6）：精确 k-NN 23.34 ms / 81.9，HNSW(recall 0.97) 7.69 ms / 81.8 —— 近似索引精度几乎不掉却约 3× 加速，否则检索会成为训练瓶颈。
- **作者承认的局限**：
  - Places-LT 表现「不及预期」，给了两条理由：场景分类 vs 大知识库文本多围绕**物体**，图文协同被削弱；检索可能**过度校正**不平衡，轻微伤及已充分覆盖的类（p.6）。还试了 $k=8,16$（p.6 末尾）。
  - WebVision 上 RCAL+ 因叠加半监督，在 ImageNet 验证集略高（p.7）；iNaturalist 2018 上 LiVT 的 Many-shot 更高（p.7）。
  - 存储开销约 30GB（p.5）；CIFAR-100-LT 结果只放在补充材料（p.6）。
- **我观察到的局限**：知识库的**文本质量与领域匹配是隐含前提**（Places-LT 正是反例）；k-NN 预计算依赖冻结编码器，检索侧表征无法端到端更新；全文未报训练时长/总算力开销（只报了单次查询时间与额外存储）。

## 我的疑问

- [ ] 查询图对应的文本嵌入置 0 防泄漏（p.4），那么检索分支是否完全无法利用「查询图自身」的文本描述？还是说检索到的邻居文本里可能含同类描述、已足够？
- [ ] $\tau$ 是固定超参还是可学？p.4 只称其为 balancing coefficient，正文未给取值。
- [ ] p.6 末 "We tested smaller k (8, 16) for Places-LT. Interestingly," 后被分页截断，需查补充材料看结论。
- [ ] 1.4B 知识库的检索质量与 DataComp 过滤偏差如何量化？会不会系统性偏向常见物体？

## 关键摘录

> "instead of statically compiling world knowledge into model weights, we construct an external image-text pair knowledge base for storing world knowledge and then utilize a retrieval module to identify and obtain relevant knowledge from a predefined knowledge base"（第 2 页）

> "We believe that the retrieval module can be considered as a data-dependent augmentation. The retrieval module implicitly extracts virtual samples and models multimodal inter-sample relationships by means of samples in their neighborhood distributions."（第 5 页）

> "Places-LT focuses on scene-oriented classification, but textual descriptions in large knowledge bases often center on objects. This mismatch can reduce the synergy between retrieved text and scene images."（第 6 页）

> "We acknowledge that LiVT surpasses us in Many-shot performance, but its Overall performance is far below RetFormer."（第 7 页）

## 相关笔记

- [[PRISM 语言引导原型学习的可解释图像识别]] — 同为 CVPR'26、同用 CLIP 文本支路增强图像识别；但 PRISM 把语言当**推理期之外的监督**（归因图），RetFormer 把外部图文库当**推理期输入**，两条「语言帮视觉」的路线正好对读
- [[FlashAttention 精确注意力与IO感知]] — 式(5) 缩放点积注意力的基础，回看 $\sigma(QK^T/\sqrt{d})V$ 与 IO 开销
- [[多头注意力+proj_out]] — 交叉融合里 Q/K/V 线性映射与输出投影的作用
- [[transformer的2道残差连接]] — 式(6) 每层两次残差相加（Att 输出 + $E_{NN}$），对照标准两道残差
