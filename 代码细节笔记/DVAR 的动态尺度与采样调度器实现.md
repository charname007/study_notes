---
created: 2026-11-08
source: https://github.com/YuZheng9/DVAR @ 7dac046159c4476f5b1e4b8ad790eb11fd71a63b（2026-06-03）；配套 [[DVAR 动态视觉自回归单图超分]]
type: 代码
tags: [代码, 超分辨率, 自回归, 多尺度]
---

# DVAR 的动态尺度与采样调度器实现

## 场景
读 DVAR（CVPR'26）时想确认论文式(5)「典范缩放动态」和算法 2「DSS」在官方 released 代码里到底长什么样。仓库 `YuZheng9/DVAR`（Apache 2.0，CVPR 2026），README 的 Todo 显示**仅推理代码 + 预训练 adapter/VAE 已发布，训练代码与训练数据准备未发布**，所以这里以推理代码与 `models/dvar.py` 里残留的训练期逻辑为准。

## 关键点

### 1. 一条比例序列适配任意尺寸
`config/inference.yaml` 只写死一条**相对比例**序列，而不是各级绝对尺寸：

~~~yaml
use_adaptive_pn: true
patch_nums: [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
lora: true
lora_rank: 16
vqvae_vocab_size: 4096
~~~

`models/quant.py:356-372`（`_maybe_update_v_patch_nums`）把它按当前特征图 H/W 展开——分母 16 对应序列末项，也就是论文式(5)里基准分辨率的等价物：

~~~python
if self.use_adaptive_pn:
    self.v_patch_nums = [
        (round(H * i / 16.0), round(W * i / 16.0)) for i in self.v_patch_nums_copy
    ]
else:
    self.v_patch_nums = [
        (pn, pn) if isinstance(pn, int) else (pn[0], pn[1])
        for pn in self.v_patch_nums_copy
    ]
~~~

返回值是 `(h, w)` **元组**而非 VAR 原版的整数——这就是论文说的「目标专属 scale schedule」，也是支持非方形长宽比的关键。`models/dvar.py:745-762` 的 `update_patch_nums` 会同步重算 `first_l`、`L`、`last_level_pns` 和 RoPE 频率，所以一套权重能换序列。

### 2. DSS 的 count book 与 EMA
`models/dvar.py:525` 注册 `ema_rank_hit_SV`（形状 `(len(patch_nums), vocab_size)`），即论文里的 count book $C_k$。训练期 `dvar.py:702-741` 用模型 argmax 预测去查 GT 的 rank，bincount 后 EMA 更新：

~~~python
pred_ranks = gt_rank_BLV[batch_indices, cur+pos_indices, cur_pred]
rank_hit_V = pred_ranks.reshape(-1).bincount(minlength=self.vocab_size).float()
if self.record_rank_hit < 100:
    self.ema_rank_hit_SV[si].mul_(0.9).add_(rank_hit_V.mul(0.1))
else:
    self.ema_rank_hit_SV[si].mul_(0.99).add_(rank_hit_V.mul(0.01))
~~~

前 100 步用 0.9/0.1 快预热、之后 0.99/0.01 慢收敛——论文只写「EMA」，**没给 β，这个细节只有读码才知道**。`generate_pseudo_code_from_ranks`（`dvar.py:979+`）再把该分布归一化后 multinomial 采样出 rank，经 `gt_rank_BLV` 映射回真实 token index，就是算法 2 的 SampleNN。

### 3. k-NN 秩从哪来
`models/quant.py:587-598` 在 VQ 量化时直接对码本距离/相似度做全排序，一次拿到每个位置的完整近邻秩：

~~~python
sorted_idx = torch.argsort(d_no_grad, dim=1, descending=False)  # (B*h*w, vocab_size)
idx_N = sorted_idx[:, 0]
~~~

这解释了论文「k-NN 是零成本查表」的说法：量化过程本来就要算距离，排序顺手就带了秩表。

### 4. 顺带发现：训练期还有 scale dropout
`dvar.py:653-665` 按 `scale_drop_rate: 0.3`（见 `inference.yaml`）随机把某些尺度的输入替换成 `ema_var_input`：

~~~python
scale_mask[:, 1:] = rand_matrix[:, 1:] >= scale_drop_rate
...
x_BLC[b, condition_len + cur:condition_len + cur + scale_len] = ema_var_input[b, cur:cur + scale_len]
~~~

论文正文把它归入 DSS 的「模拟误差」，但没有单独消融，也没写进算法 2 的伪代码。

### 5. 坑与提醒
- README 的 clone 地址写的是 `Yuuuuuu9/DVAR`，而论文给的仓库是 `YuZheng9/DVAR`（同一仓库的别名/旧拼写），HF 权重路径也用 `Yuuuuuu9/DVAR`；按论文链接走 `YuZheng9/DVAR` 即可。
- `models/pipeline.py` 是**旧版 SwittiPipeline**，里面 `switti.patch_nums[-1]` 当整数用（`f_hat = sos.new_zeros(..., patch_nums[-1], patch_nums[-1])`），与 `dvar.py` 的 `(h,w)` 元组不兼容；实际推理入口是 `inference.py` → `DVAR.autoregressive_infer_cfg`，别照 pipeline.py 读。
- 训练代码未发布，Algorithm 1/2 的 RankCount、RandomChoice 的 p、EMA 的 β 只能从上述残留逻辑反推，不能当成官方定稿超参。

## 相关笔记
- [[DVAR 动态视觉自回归单图超分]] — 论文解读主笔记
- [[SVTR 的 PaddleOCR 实现]] — 另一处「论文表格与 config 不一致」的教训（读码要读 yml）
- [[FlashAttention 的 online softmax 实现]] — 同为「论文措辞容易误解，以代码为准」的案例
