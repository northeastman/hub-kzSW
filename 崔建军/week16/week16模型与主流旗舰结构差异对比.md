# Week16 模型与主流旗舰模型结构差异对比报告

> **数据来源**：week16 目录下 2 份 config.json + 源码（hy_v3、Qwen3.8-27B），与 `week16大模型结构演进/model_code/` 下 5 份 config.json + README（DeepSeek-V3/V4-Pro/V4-Flash、Kimi-K3、GLM-5.2），共 7 个模型。所有纯数字结构参数均逐字段核对 config.json；文字描述含源码 `modular_*.py` / `modeling_*.py` 及官方 README。
>
> **来源标注**：C = config.json 标准字段直接可查；S = 同名架构源码（modeling_*.py / modular_*.py）；R = 官方 README 公开信息。

---

## 一、概述

week16 目录下当前有两个模型：

| 模型 | 全称 | 类型 | 出品方 |
|------|------|------|--------|
| **hy_v3** | Tencent HunYuan 3 (Hy3-preview) | MoE 纯文本 | 腾讯混元 [S: `checkpoint="tencent/Hy3-preview"`] |
| **Qwen3.8-27B** | Qwen3.8-27B | Dense 多模态 | 阿里通义千问 |

本文将这两个模型与当前主流旗舰模型进行结构差异对比：

| 对比模型 | 类型 | 出品方 | 总参/激活 [R] |
|----------|------|--------|---------------|
| DeepSeek-V3 | MoE 纯文本 | 深度求索 | 671B/37B |
| DeepSeek-V4-Pro | MoE 纯文本 | 深度求索 | 1.6T/49B |
| DeepSeek-V4-Flash | MoE 纯文本 | 深度求索 | 284B/13B |
| Kimi-K3 | MoE 多模态 | 月之暗面 | 2.8T/104B |
| GLM-5.2 | MoE 纯文本 | 智谱AI | 744B/40B |

---

## 二、关键参数对比矩阵

### 表 1：基础结构参数（逐字段核对 config.json ✅）

| 模型 | model_type | layers | hidden | attn heads | KV heads | head_dim | vocab | 上下文 | rope_theta | dtype |
|------|-----------|--------|--------|-----------|----------|----------|-------|--------|-----------|-------|
| **hy_v3** | hy_v3 | 80 | 4096 | 64 | 8 | 128 | 120832 | 262144 | 1.12e7 | bf16 |
| **Qwen3.8-27B** | qwen3_5 | 64 | 5120 | 24 | 4 | 256 | 248320 | 262144 | 1e7 | bf16 |
| DeepSeek-V3 | deepseek_v3 | 61 | 7168 | 128 | 128 | 128+64¹ | 129280 | 163840 | 1e4 | bf16 |
| DeepSeek-V4-Pro | deepseek_v4 | 61 | 7168 | 128 | 1 | 512 | 129280 | 1048576 | 1e4 | bf16 |
| DeepSeek-V4-Flash | deepseek_v4 | 43 | 4096 | 64 | 1 | 512 | 129280 | 1048576 | 1e4 | bf16 |
| Kimi-K3 | kimi_k3 | 93 | 7168 | 96 | 96 | 128+64¹ | 163840 | 1048576 | NoPE² | bf16 |
| GLM-5.2 | glm_moe_dsa | 78 | 6144 | 64 | 64 | 192+64¹ | 154880 | 1048576 | 8e6 | bf16 |

¹ MLA 系模型无单一 head_dim，实际由 `qk_nope_head_dim + qk_rope_head_dim` 合成（V3: 128+64=192；Kimi: 128+64=192；GLM: 192+64=256）；hy_v3 与 Qwen3.8 使用标准 head_dim。
² Kimi-K3 的 24 层 Gated MLA 完全 NoPE（`mla_use_nope=true`），位置信息由 69 层 KDA 线性注意力提供。

### 表 2：注意力机制 / MoE / 多模态 / 稳定化（数字来自 config.json ✅；文字描述含源码 S / README R）

| 模型 | 注意力机制 [C/S] | MoE experts/topk/shared [C/R] | moe_intermediate [C] | 多模态 [C] | 量化 [C] | 特色稳定化 [C/S/R] |
|------|-----------------|-------------------------------|---------------------|-----------|---------|-------------------|
| **hy_v3** | 标准 GQA（64Q/8KV，head_dim=128 C:；qk_norm=true C:）[S: 继承 ApertusAttention] | 192 / 8 / shared=1 C:（num_shared_experts=1） | 1536 C: | 无 C:（ForCausalLM） | 无 | moe_router_use_sigmoid=true C:；moe_router_enable_expert_bias=true C:；route_norm=true C:；router_scaling_factor=2.826 C:；enable_lm_head_fp32=true C:；MTP=1 C:；first_k_dense_replace=1 C: |
| **Qwen3.8-27B** | 线性+全量 3:1 C:（full_attention_interval=4；layer_types=48 linear+16 full；linear_conv_kernel=4 C:） | 无（Dense，intermediate_size=17408 C:） | — | 视觉 C:（27层 ViT + M-RoPE mrope_section=[11,11,10]） | 无 | attn_output_gate=true C:；output_gate_type=swish C:；MTP=1 C:（mtp_num_hidden_layers=1） |
| DeepSeek-V3 | MLA 全量 C:（kv_lora_rank=512；q_lora_rank=1536） | 256 / 8 / shared=1 C: | 2048 C: | 无 C: | FP8 C:（block=[128,128]） | scoring_func=sigmoid C:；topk_method=noaux_tc C:；n_group=8 C:；first_k_dense_replace=3 C:；MTP=1 C: |
| DeepSeek-V4-Pro | 滑窗128 C: + CSA压缩 + DSA索引(index_topk=1024 C:) + HCA [S: compress_ratios 按层配置] | 384 / 6 / shared=1 C: | 3072 C: | 无 C: | FP4专家 C:（expert_dtype=fp4）+ FP8 C: | mHC C:（hc_mult=4, sinkhorn_iters=20）；swiglu_limit=10 C:；hash前3层 C:（num_hash_layers=3）；scoring_func=sqrtsoftplus C:；MTP=1 C: |
| DeepSeek-V4-Flash | 滑窗128 C: + CSA压缩 + DSA索引(index_topk=512 C:) [S: compress_ratios 含 HCA] | 256 / 6 / shared=1 C: | 2048 C: | 无 C: | FP4专家 C: + FP8 C: | 同 V4-Pro（mHC、swiglu_limit、hash前3层、sqrtsoftplus） |
| Kimi-K3 | 69层KDA线性 C: + 24层Gated MLA C:（mla_use_nope=true C:；linear head_dim=128 C:） | 896 / 16 / shared=2 [R: README 明确 2 个共享专家；config 无 n_shared_experts 字段] | 3072 C:（latent=3584 C:） | 视觉 C:（MoonViT-V2，image_placeholder [R]） | MXFP4 [R: README "MXFP4 weights / MXFP8 activations"] | hidden_act="situ" C:（β1=4, β2=25）；attn_res_block_size=12 C:（AttnRes）；mla_use_output_gate C:；LatentMoE C:（latent_moe_use_norm=true）；MTP=0 C: |
| GLM-5.2 | MLA+DSA稀疏 C:（index_topk=2048；indexer_types=[full,full,full,shared×3]循环 C:） | 256 / 8 / shared=1 C: | 2048 C: | 无 C: | 无 | IndexShare FSSS [S/R]（index_topk_freq=4 C:）；indexer_rope_interleave=true C:；first_k_dense_replace=3 C:；MTP=1 C:（index_share_for_mtp_iteration=true C:） |

---

## 三、分维度差异分析

### 3.1 注意力机制：hy_v3 / Qwen3.8 走了哪条路线？

当前旗舰模型的注意力机制分为四条路线，week16 两个模型各占一条，但都不是最激进的：

| 路线 | 代表模型 | 核心思想 | hy_v3 | Qwen3.8-27B |
|------|---------|---------|-------|-------------|
| **标准 GQA** | hy_v3、DeepSeek-V3（MLA 前身） | 分组查询，不做 KV 压缩 | ✅ 采用 | ✗ |
| **MLA（KV 压缩）** | DeepSeek-V3/V4、GLM-5.2、Kimi-K3 | 存 latent 而非全量 K/V | ✗ | ✗ |
| **线性注意力混合** | Qwen3.8-27B、Kimi-K3 | 线性递推 + 少量全量层兜底 | ✗ | ✅ 采用 |
| **DSA 稀疏 + 压缩** | DeepSeek-V4、GLM-5.2 | 廉价索引选 top-k token | ✗ | ✗ |

**hy_v3 的注意力**：采用标准 GQA（64 个查询头 / 8 个 KV 头 / head_dim=128），是 7 个模型中最"传统"的注意力设计。源码 `modular_hy_v3.py` 显示其注意力继承自 `ApertusAttention`（非 MLA），与 DeepSeek-V3 的 MLA、GLM-5.2 的 MLA+DSA、Kimi-K3 的 KDA+MLA 混合均不同。这意味着 hy_v3 的 KV cache 随序列长度线性增长，未做压缩——这在 256K 上下文下尚可接受，但与 DS-V4/GLM-5.2 的 1M 上下文路线有本质差距。

**Qwen3.8-27B 的注意力**：采用 Gated DeltaNet（GDN）线性注意力 + 每 4 层 1 层全量 GQA 的混合策略（`full_attention_interval=4`，64 层中 48 层线性 + 16 层全量）。线性层用短卷积（`linear_conv_kernel=4`）+ delta 规则做递推，全量层用 GQA（24Q/4KV，partial RoPE 25%）兜底精确检索。与 Kimi-K3 的 69 KDA + 24 MLA 思路相似（都是"线性为主 + 少量全量"），但 Kimi 用 MLA 做全量层且完全 NoPE，Qwen 用标准 GQA 做全量层且用 M-RoPE。

**关键差异**：
- hy_v3 **未采用**任何注意力压缩/稀疏/线性技术，是 7 个模型中唯一纯标准 GQA 的 MoE 模型
- Qwen3.8-27B 的线性混合与 Kimi-K3 同路线，但全量层用 GQA 而非 MLA，KV 压缩不如 Kimi
- DeepSeek-V4 的注意力最复杂：MLA + CSA 4:1 压缩 + DSA top-k + HCA 128:1 + 滑窗 128，1M 上下文 FLOPs 仅 V3 的 27%
- GLM-5.2 的 IndexShare（4 层共享一次索引）是 DSA 的工程优化，1M 上下文 FLOPs 再降 2.9×

### 3.2 MoE 架构：专家数、路由与共享专家

| 维度 | hy_v3 | DeepSeek-V4-Pro | Kimi-K3 | GLM-5.2 | DeepSeek-V3 |
|------|-------|----------------|---------|---------|-------------|
| 专家数 | 192 | 384 | 896 | 256 | 256 |
| 每token激活 | 8 | 6 | 16 | 8 | 8 |
| 稀疏比 | 24× | 64× | 56× | 32× | 32× |
| 共享专家 | 1 | 1 | 2 | 1 | 1 |
| 路由函数 | sigmoid | sqrtsoftplus | sigmoid | sigmoid | sigmoid |
| 负载均衡 | expert_bias [C] | noaux_tc [C] | noaux_tc [C] | noaux_tc [C] | noaux_tc [C] |
| 首层 Dense | 1 层 | 0 层（hash路由） | 1 层 | 3 层 | 3 层 |
| moe_intermediate | 1536 | 3072 | 3072 | 2048 | 2048 |
| 专家实现 | [S: Qwen3MoeExperts] | 原生 | [S: LatentMoE] | 原生 | 原生 |

**hy_v3 的 MoE 特色**：
- **专家数 192**：介于 DeepSeek-V3（256）和 MiniMax M2 之间，少于 GLM-5.2（256）、DS-V4-Pro（384）和 Kimi-K3（896）
- **moe_intermediate=1536**：7 个模型中最小，意味着每个专家的 FFN 中间层极窄，属于"极细粒度"专家设计（与 Qwen3.6-35B-A3B 的 512 同思路但更宽）
- **路由函数 sigmoid + expert_bias**：与 DeepSeek-V3 的 sigmoid 路由同族，但用 `moe_router_enable_expert_bias=true` 替代了 DS 的 `noaux_tc` 偏置方案——bias 直接加到路由 logit 上影响选人，但不影响权重归一化（`route_norm=true`）
- **router_scaling_factor=2.826**：独有的路由缩放因子，用于控制 top-k 权重的幅度，其他模型用 `routed_scaling_factor`（DS-V3: 2.5, GLM: 2.5, DS-V4-Flash: 1.5）
- **源码继承关系** [S]：`MiniMaxM2SparseMoeBlock`（MoE 块结构）+ `MixtralTopKRouter`（路由器）+ `Qwen3MoeExperts`（专家实现）——是一个"组装"型 MoE，取各家之长
- **first_k_dense_replace=1**：仅第 1 层为 Dense，比 DS-V3/GLM-5.2 的 3 层更激进，几乎所有层都走 MoE

**与旗舰模型的关键差异**：
- hy_v3 的专家数（192）和 moe_intermediate（1536）均小于旗舰模型（256~896 / 2048~3072），属于中等规模 MoE
- hy_v3 **未采用** LatentMoE（Kimi-K3 的潜空间压缩）、FP4 量化（DS-V4）、hash 路由（DS-V4 前 3 层）等激进技术
- hy_v3 的 sigmoid + expert_bias 路由是 DS-V3 路线的延续，未升级到 DS-V4 的 sqrtsoftplus + Muon 优化器
- Kimi-K3 是唯一使用 LatentMoE 的模型：把 7168 维 hidden 压缩到 3584 维潜空间，用同样预算买下 896 个专家

### 3.3 多模态能力

| 模型 | 多模态 | 视觉编码器 | 模态 |
|------|--------|-----------|------|
| **hy_v3** | ✗ | 无 | 纯文本 |
| **Qwen3.8-27B** | ✅ | 27 层 ViT（hidden=1152, depth=27, patch=16） | 文本+图像+视频 |
| DeepSeek-V3 | ✗ | 无 | 纯文本 |
| DeepSeek-V4 | ✗ | 无 | 纯文本 |
| Kimi-K3 | ✅ | MoonViT-V2（401M 参数）[R] | 文本+图像 |
| GLM-5.2 | ✗ | 无 | 纯文本 |

**Qwen3.8-27B 是 7 个模型中多模态能力最完整的**：配置了完整的 `vision_config`（27 层 ViT，patch_size=16，spatial_merge_size=2，out_hidden_size=5120 与文本 hidden 对齐），并支持视频（`video_token_id=248057`，`video_preprocessor_config.json` 存在）。位置编码使用 M-RoPE（多模态 RoPE），将时间、高度、宽度三维度分开编码（`mrope_section=[11, 11, 10]`）。

hy_v3 和 DeepSeek 系列、GLM-5.2 一样是纯文本模型，未配置视觉塔。Kimi-K3 虽是多模态（MoonViT-V2），但仅支持文本+图像，不支持视频。

### 3.4 上下文长度与位置编码

| 模型 | 上下文 | 位置编码 | rope_theta | 压缩/外推 |
|------|--------|---------|-----------|-----------|
| **hy_v3** | 256K | 标准 RoPE | 1.12e7 | 无外推 |
| **Qwen3.8-27B** | 256K | M-RoPE | 1e7 | 无外推 |
| DeepSeek-V3 | 160K | YaRN | 1e4 | yarn factor=40 |
| DeepSeek-V4 | 1M | YaRN + compress | 1e4 | yarn factor=16 + compress_rope_theta=160000 |
| Kimi-K3 | 1M | NoPE (MLA) + KDA | — | MLA 层零旋转 |
| GLM-5.2 | 1M | RoPE | 8e6 | interleave=true |

**关键差异**：
- hy_v3 和 Qwen3.8-27B 均为 **256K 上下文**，未达到旗舰模型的 1M 标准
- hy_v3 的 rope_theta=11158840（约 1.12e7）是标准 RoPE，无 YaRN 外推——说明 256K 是原生预训练长度
- Qwen3.8 的 M-RoPE 是多模态专用位置编码，partial_rotary_factor=0.25 意味着只旋转 25% 的维度
- DS-V4 的位置编码最复杂：YaRN 外推（factor=16, original_max=65536）+ 压缩专用 rope_theta（160000）+ 只旋转 64 维
- Kimi-K3 的 MLA 层完全 NoPE，位置信息全部由 69 层 KDA 线性注意力提供，是 7 个模型中最激进的位置编码方案

### 3.5 稳定化、量化与 MTP

| 维度 | hy_v3 | Qwen3.8-27B | DS-V4 | Kimi-K3 | GLM-5.2 |
|------|-------|-------------|-------|---------|---------|
| **激活函数** | SiLU C: | SwiGLU C:（output_gate=swish） | SiLU C: + swiglu_limit=10 C: | SiTU-GLU C:（β1=4, β2=25） | SiLU C: |
| **有界激活** | 无 | output_gate | 硬钳制 [-10,10] | 软钳制 \|f\|≤100 | 无 |
| **残差流** | 标准 | 标准 | mHC（4路混合 + Sinkhorn 20步） | AttnRes（跨12块读取） | 标准 |
| **QK-Norm** | ✅ C: | ✗（用 output_gate 替代） | ✅（隐含） | ✗ | ✅（隐含） |
| **量化** | 无 | 无 | FP4专家 + FP8 | MXFP4 QAT [R] | 无 |
| **MTP** | 1 层 C: | 1 层 C: | 1 层 C: | 0 层 C: | 1 层 C:（index_share） |

**hy_v3 的稳定化策略**：
- `qk_norm=true`：QK 归一化，防止注意力 logits 爆炸——与 DS-V4、GLM-5.2 一致
- `route_norm=true`：路由权重归一化，类似 DS/GLM 的 `norm_topk_prob=true`
- `enable_lm_head_fp32=true`：LM head 用 FP32 计算，防止输出层精度损失——这是 hy_v3 独有的配置
- `enable_attention_fp32_softmax=false`：注意力 softmax 不升级到 FP32（节省计算）
- `enable_moe_fp32_combine=false`：MoE 合并不用 FP32（节省显存）
- 未采用有界激活（swiglu_limit / SiTU）、残差流升级（mHC / AttnRes）、量化感知训练

**Qwen3.8-27B 的稳定化策略**：
- `attn_output_gate=true` + `output_gate_type=swish`：用 sigmoid 门控按通道调节注意力输出和 FFN 输出幅度
- 未采用 QK-Norm、有界激活、残差流升级、量化

**与旗舰模型的关键差异**：
- hy_v3 和 Qwen3.8-27B 均未采用量化感知训练（QAT），而 DS-V4（FP4+FP8）和 Kimi-K3（MXFP4）已将 QAT 作为标配
- hy_v3 和 Qwen3.8-27B 均未升级残差流（仍用标准加法残差），而 DS-V4 的 mHC（4 路超连接 + Sinkhorn 双随机投影）和 Kimi-K3 的 AttnRes（跨块快照读取）代表了残差流演进的前沿
- hy_v3 和 Qwen3.8-27B 均保留 MTP=1，与 DS/GLM 一致；Kimi-K3 是唯一不用 MTP 的模型

---

## 四、核心差异总结

### 4.1 hy_v3（腾讯混元3）的架构定位

hy_v3 是一个**务实型中等规模 MoE 模型**，其架构特点可归纳为：

| 维度 | hy_v3 选择 | 与旗舰的差异 |
|------|-----------|-------------|
| 注意力 | 标准 GQA + QK-Norm | 未压缩 KV（vs DS-V4 MLA+CSA、GLM DSA、Kimi KDA） |
| MoE | 192专家/8激活 + sigmoid+bias | 规模中等（vs Kimi 896、DS-V4 384）；路由保守（vs DS-V4 sqrtsoftplus） |
| 上下文 | 256K 原生 | 未冲 1M（vs DS-V4/GLM/Kimi 均 1M） |
| 量化 | 无 | 未做 QAT（vs DS-V4 FP4、Kimi MXFP4） |
| 残差 | 标准 | 未升级（vs DS-V4 mHC、Kimi AttnRes） |
| 多模态 | 纯文本 | 无视觉能力（vs Qwen3.8 视觉+视频、Kimi 视觉） |

源码继承关系表明 [S]，hy_v3 是一个"组装型"架构：注意力取自 Apertus、解码层取自 DeepSeek-V3、MoE 块取自 MiniMax-M2、专家取自 Qwen3-MoE、路由取自 Mixtral。这种设计哲学是**用成熟组件快速组装**，而非像 DS-V4/Kimi-K3 那样做系统级重构。

### 4.2 Qwen3.8-27B 的架构定位

Qwen3.8-27B 是一个**Dense 多模态模型**，其架构特点可归纳为：

| 维度 | Qwen3.8-27B 选择 | 与旗舰的差异 |
|------|-----------------|-------------|
| 注意力 | GDN 线性 + 全量 GQA 3:1 | 与 Kimi-K3 同路线（线性混合），但全量层用 GQA 非 MLA |
| FFN | Dense（无 MoE） | 无专家稀疏（vs 其他 5 个 MoE 模型均 256~896 专家） |
| 上下文 | 256K 原生 | 未冲 1M（vs DS-V4/GLM/Kimi 均 1M） |
| 量化 | 无 | 未做 QAT |
| 多模态 | 27层 ViT + M-RoPE + 视频 | 7 个模型中多模态最完整（vs Kimi 仅图像、其余纯文本） |
| 稳定化 | output_gate (swish) | 未用 QK-Norm / 有界激活 / 残差升级 |

Qwen3.8-27B 的核心价值在于**多模态能力**：是 7 个模型中唯一支持视频的模型（`video_token_id`、`video_preprocessor_config.json`），视觉塔配置完整（27 层 ViT，patch=16，spatial_merge=2，输出维度与文本 hidden_size=5120 对齐）。其线性注意力混合策略与 Kimi-K3 同属"线性为主 + 少量全量兜底"路线，但选择了更保守的 GQA 做全量层而非 MLA。

### 4.3 演进路线分野

将 7 个模型按"成熟配方下放 vs 激进系统重构"分类：

| 阵营 | 模型 | 特征 |
|------|------|------|
| **成熟配方组装** | hy_v3 | 标准GQA + sigmoid MoE + QK-Norm，用成熟组件快速组装 |
| **成熟配方下放** | Qwen3.8-27B | GDN线性3:1 + Dense FFN + M-RoPE多模态，路线稳健 |
| **压缩路线（激进）** | DeepSeek-V4 | MLA+CSA+HCA+滑窗，为 1M 上下文系统重构 |
| **稀疏索引路线** | GLM-5.2 | MLA+DSA+IndexShare，工程优化最务实 |
| **线性+潜空间路线（最激进）** | Kimi-K3 | KDA+MLA+LatentMoE+AttnRes+SiTU+MXFP4，全栈创新 |
| **MLA 原始路线** | DeepSeek-V3 | MLA 全量 + sigmoid MoE，开山之作 |

**结论**：hy_v3 和 Qwen3.8-27B 均属于"成熟配方"阵营——hy_v3 用标准 GQA + 组装型 MoE 做中等规模务实部署，Qwen3.8-27B 用线性混合 + Dense FFN + 完整多模态做多模态旗舰。两者均未采用 DS-V4 的压缩路线、GLM 的 DSA 稀疏索引、Kimi-K3 的 LatentMoE + AttnRes 等激进技术，代表了"工程成熟度优先于架构创新"的设计取向。

---

## 五、附录：文件路径与数据来源

### week16 目录模型

- **hy_v3**（腾讯混元3）：
  - [config.json](file:///D:/BaiduNetdiskDownload/AI/code_260709/hub-kzSW/崔建军/week16/hy_v3/config.json)
  - [modeling_hy_v3.py](file:///D:/BaiduNetdiskDownload/AI/code_260709/hub-kzSW/崔建军/week16/hy_v3/modeling_hy_v3.py)（自动生成源码）
  - [modular_hy_v3.py](file:///D:/BaiduNetdiskDownload/AI/code_260709/hub-kzSW/崔建军/week16/hy_v3/modular_hy_v3.py)（模块化源码模板，含继承关系）
  - [configuration_hy_v3.py](file:///D:/BaiduNetdiskDownload/AI/code_260709/hub-kzSW/崔建军/week16/hy_v3/configuration_hy_v3.py)

- **Qwen3.8-27B**：
  - [config.json](file:///D:/BaiduNetdiskDownload/AI/code_260709/hub-kzSW/崔建军/week16/Qwen3.8-27B/config.json)
  - [modeling_qwen3_5.py](file:///D:/BaiduNetdiskDownload/AI/code_260709/hub-kzSW/崔建军/week16/Qwen3.8-27B/transformers_source/modeling_qwen3_5.py)
  - [modular_qwen3_5.py](file:///D:/BaiduNetdiskDownload/AI/code_260709/hub-kzSW/崔建军/week16/Qwen3.8-27B/transformers_source/modular_qwen3_5.py)

### 对比模型（`week16大模型结构演进/model_code/`）

- DeepSeek-V3：[config](file:///D:/BaiduNetdiskDownload/AI/week16大模型结构演进/model_code/deepseek-ai_DeepSeek-V3_config.json)
- DeepSeek-V4-Pro：[config](file:///D:/BaiduNetdiskDownload/AI/week16大模型结构演进/model_code/deepseek-ai_DeepSeek-V4-Pro_config.json)
- DeepSeek-V4-Flash：[config](file:///D:/BaiduNetdiskDownload/AI/week16大模型结构演进/model_code/deepseek-ai_DeepSeek-V4-Flash_config.json) ｜ [readme](file:///D:/BaiduNetdiskDownload/AI/week16大模型结构演进/model_code/dsv4_readme.md)
- Kimi-K3：[config](file:///D:/BaiduNetdiskDownload/AI/week16大模型结构演进/model_code/moonshotai_Kimi-K3_config.json) ｜ [readme](file:///D:/BaiduNetdiskDownload/AI/week16大模型结构演进/model_code/kimi_k3_readme.md)
- GLM-5.2：[config](file:///D:/BaiduNetdiskDownload/AI/week16大模型结构演进/model_code/zai-org_GLM-5.2_config.json) ｜ [readme](file:///D:/BaiduNetdiskDownload/AI/week16大模型结构演进/model_code/glm52_readme.md)

### 参考资料目录

- 技术报告 PDF：[tech_reports/](file:///D:/BaiduNetdiskDownload/AI/week16大模型结构演进/tech_reports/)
- 课程 PPT：[大模型结构演进.pptx](file:///D:/BaiduNetdiskDownload/AI/week16大模型结构演进/大模型结构演进.pptx)

---

> **报告生成时间**：2026-08-19
> **数据核对方式**：所有 config.json 字段逐项核对；源码继承关系来自 `modular_hy_v3.py` 的 import 语句；README 数据来自官方 HuggingFace 页面。
