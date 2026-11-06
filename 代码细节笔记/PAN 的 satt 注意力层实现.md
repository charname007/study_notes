---
created: 2026-11-06
source: https://github.com/phseo/PAN @ b2a94f7 ；配套论文 [[PAN 渐进式注意力网络与视觉属性预测]]
type: 代码
tags: [代码, Attention, Theano, Lasagne]
---

# PAN 的 satt 注意力层实现

## 场景

读 PAN 论文时想确认三件事：论文第 7 页说 q 与特征 **concat 后过两层 FC**，代码是否一致？中间层 sigmoid 不归一化会不会压垮特征幅值？硬注意力到底怎么实现的？

仓库 https://github.com/phseo/PAN ，HEAD b2a94f7（2016-10-25），**Theano + Lasagne**（Python 2，print 语句），共 12 个文件。核心逻辑集中在 libs/theano_util.py 的 satt() 与 libs/spatialatt.py 的两个 MergeLayer。

## 关键点

### 1. satt()：一层注意力的完整构造（libs/theano_util.py）

```python
def satt(img, ref, img_shape, ref_size=10, num_int_ch=32, l_name=None, dropout=None,
         W_ini=lasagne.init.GlorotUniform(), nonlinearity=lasagne.nonlinearities.softmax,
         batch_norm=True, rtn_att_map=False, context_box=(1, 1)):
    # 局部上下文：用 context_box 大小的卷积一次性把邻域聚合进每个位置
    l_img = conv(img, num_filters=num_int_ch, filter_size=context_box,
                 name=l_img_name, W=W_ini, nonlinearity=None, pad='same')

    # 查询 q：one-hot(10) -> Embedding(32)，等价于 one-hot 直接乘一个矩阵
    l_q = lasagne.layers.EmbeddingLayer(ref, input_size=ref_size,
                                        output_size=num_int_ch, W=W_ini, name=l_q_name)

    # 融合：逐通道相加（不是 concat！）
    l_mix = SpatialElemwiseMergeLayer(l_img, l_q, T.add)
    l_nonlin = lasagne.layers.NonlinearityLayer(l_mix)
    l_bn = BatchNormLayer(l_nonlin) if batch_norm else l_nonlin

    # 打分：1x1 卷积把 32 通道压成 1 通道
    l_weight = conv(l_drp, num_filters=1, filter_size=(1, 1),
                    name=l_att_name, W=W_ini, nonlinearity=None)
    l_weight_flat = lasagne.layers.FlattenLayer(l_weight)

    # 归一化：softmax（全图）或 sigmoid（逐位置），由调用方传 nonlinearity 决定
    l_prob_flat = lasagne.layers.NonlinearityLayer(l_weight_flat, nonlinearity)
    l_prob = lasagne.layers.ReshapeLayer(l_prob_flat, (-1,) + img_shape)

    # 加权：alpha 乘回原特征图
    l_att_out = SpatialAttentionLayer(img, l_prob)

    return (l_att_out, l_prob) if rtn_att_map else l_att_out
```

**注意最后一步乘的是 img（注意力层的输入）而不是 l_img**：α 乘的是**上一层传下来的原始特征图**，不是经过 context 卷积后的中间表示。这点很关键，否则式(1) 就不再是"α 乘特征"。

### 2. 加权实现：那个 broadcast 决定了式(1) 对不对（libs/spatialatt.py）

```python
class SpatialAttentionLayer(MergeLayer):
    def __init__(self, img, att, **kwargs):
        super(SpatialAttentionLayer, self).__init__((img, att), **kwargs)

    def get_output_shape_for(self, input_shapes):
        # 断言 alpha 的空间尺寸与特征图一致：input[0] 是 (N,C,H,W)，input[1] 是 (N,H,W)
        if input_shapes[0][2] != input_shapes[1][1] or input_shapes[0][3] != input_shapes[1][2]:
            raise ValueError("Mismatch: not all input shapes are the same")
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        img, att = inputs[0], inputs[1]
        att = att.dimshuffle(0, 'x', 1, 2)   # (N,H,W) -> (N,1,H,W)
        return T.mul(img, att)               # 广播到所有通道
```

dimshuffle(0, 'x', 1, 2) 在通道维插一个 1 维，于是 (N,1,H,W) 广播乘 (N,C,H,W) —— **同一张 α 图作用于所有通道**，正是式(1) 的逐元素乘。若这里漏了 dimshuffle 会直接维度报错，不会静默算错，属于写得比较安全的实现。

### 3. train.py：注意力插在哪、哪一层用 softmax

```python
# 每层 pool 之后一个 satt，统一 sigmoid（逐位置独立门控）
l_satt0 = satt(l_pool0, l_q_in, (49,49), l_name='l_satt0',
               nonlinearity=lasagne.nonlinearities.sigmoid, context_box=(3,3))
l_bn0 = BatchNormLayer(l_satt0)          # BN 紧跟注意力，论文正文没提

l_conv1 = conv(l_bn0, num_filters=32, filter_size=(3,3), name='l_conv1_loc', W=W_ini)
# ... conv2/conv3 同构 ...

# 末层：不传 nonlinearity，用默认 softmax 全图归一化，并取回 alpha 图
l_satt3, l_att = satt(l_pool3, l_q_in, (4,4), l_name='l_satt3',
                      rtn_att_map=True, dropout=False)

# 软注意力分支：末层 f-hat 直接平均池化成一个向量再分类
l_avg_out = lasagne.layers.Pool2DLayer(l_satt3, (4,4), mode='average_exc_pad')
l_lin_out = lasagne.layers.DenseLayer(l_drp_out, num_units=num_classes,
                                      nonlinearity=lasagne.nonlinearities.softmax)
```

三个可核对的事实：

| 论文 | 代码 | 结论 |
|---|---|---|
| 中间层 sigmoid（第 4 页） | nonlinearity=sigmoid 显式传入 | ✅ 一致 |
| 末层 softmax 全图归一化（第 4 页） | l_satt3 不传 nonlinearity，取默认 softmax | ✅ 一致 |
| 每层之后接 BN | l_bn0 = BatchNormLayer(l_satt0) | ⚠️ **论文正文未提** |

**BN 的存在回答了"中间层 sigmoid 不归一化会不会把特征幅值系统性压小"**：会，所以每层注意力后补一道 BN 把分布拉回来。这也解释了为什么模型能深堆这么多层注意力而不崩。

### 4. 已发布代码 ≠ 论文完整模型

- train.py 只实现了**软注意力**分支（末层 average_exc_pad 平均池化 → Dense），对应 **PAN[S]**。
- 论文 §3.2 式(5)(6) 的**硬注意力边际化在这份代码里没有实现**，所以 Table 1 的 PAN[H] 行无法直接复现。
- 仓库只有 MNIST 系的数据生成与训练脚本，**没有 VG / VOC 版本的代码**（论文中的 VG 实验显然用了另一套未发布代码）。
- 文件头注释写的是 "Code for producing the resutls for the FFN-SPN network"（连拼写错误 "resutls" 都留着），说明这套 satt 是在 FFN-SPN 的代码上改的，与论文正文没有提及的 FFN 有谱系关系。

### 5. 与论文措辞不一致的两处（按重要程度排序）

**① q 的融合方式是 add 不是 concat（最重要）**

论文第 7 页："The function takes the **concatenation** of a query vector q and a feature vector f^l_{i,j}"；附录 A 第 13 页："This embedding is then **concatenated** with the one-hot query vector and fed to two fully connected layers"。

代码是 SpatialElemwiseMergeLayer(l_img, l_q, T.add) —— **逐通道相加**。二者的表达力不等价：

- **concat + FC**：FC 可以对 q 分量的每一种取值学独立的权重组合，表达力完整；
- **add + 后面只有 1×1 卷积**：等价于强制 q 的投影必须与特征**共享同一组通道基底**，是更强的结构性约束（也更省参数，只有 32 维）。

在 MNIST 上（10 个数字类、32 通道）这种强约束可能恰好是正则化，所以结果好；但**不能把"concat 后过两层 FC"这句话当成对代码的描述**。

**② q 的 one-hot 表述**

论文说 "q is a one-hot vector"，代码是 EmbeddingLayer(ref, input_size=10, output_size=32) —— 但这两者**等价**：one-hot 乘一个 10×32 矩阵 ≡ 查表，不是矛盾。真正的差异只有 ① 里的融合方式。

## 复现提示（若要移植到 PyTorch）

```python
# 一层 PAN attention 的等价写法（MNIST 配置：C=32, delta=1）
class PANAttention(nn.Module):
    def __init__(self, cin, cout, num_q, use_softmax=False, ctx=3):
        super().__init__()
        self.ctx = nn.Conv2d(cin, cout, ctx, padding=ctx // 2, bias=False)  # 局部上下文
        self.q   = nn.Embedding(num_q, cout)                                # one-hot -> 32
        self.mix = nn.ReLU()                                                # NonlinearityLayer
        self.bn  = nn.BatchNorm2d(cout)
        self.score = nn.Conv2d(cout, 1, 1)                                  # 出 alpha
        self.use_softmax = use_softmax

    def forward(self, x, q_idx):
        h = self.ctx(x)
        h = h + self.q(q_idx).view(-1, h.size(1), 1, 1)   # 逐通道 add (T.add)
        h = self.bn(self.mix(h))
        s = self.score(h)                                  # (N,1,H,W)
        if self.use_softmax:                               # 末层
            a = torch.softmax(s.reshape(s.size(0), -1), dim=-1).reshape_as(s)
        else:                                              # 中间层
            a = torch.sigmoid(s)
        return x * a, a
```

要点：a 是 (N,1,H,W)，乘 x 时自动广播到所有通道 —— 与 SpatialAttentionLayer 的 dimshuffle(0,'x',1,2) 完全对应。

## 相关笔记
- [[PAN 渐进式注意力网络与视觉属性预测]] — 本篇配套的论文笔记
- [[各种Norm层的摆放]] — 论文未提、代码却每层都插的 BN，其位置（注意力之后、下一层卷积之前）正好是这篇的讨论范围
- [[ASTER 注意力场景文本识别器]] — 同样是"query/attention + 逐位置打分"的经典实现，可对照其 concat 与 add 的选择

#代码 #Attention #Theano #Lasagne