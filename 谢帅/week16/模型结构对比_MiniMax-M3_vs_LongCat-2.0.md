# MiniMax-M3 vs LongCat-2.0 结构分析与对比

> 基于本地文件分析：`config.json` / `README.md` / `configuration_*.py` / `generation_config.json`
> 分析日期：2026-08-20

---

## 〇、分析所需文件说明

当前提供的文件**已足够**完成架构分析：

| 文件 | 作用 | 是否已提供 |
|------|------|:---:|
| `config.json` | 架构总纲（层数/维度/MoE/注意力/稀疏配置） | ✅ 两个都有 |
| `README.md` | 官方架构与特性说明 | ✅ 两个都有 |
| `configuration_*.py` | config 字段定义 | ✅ MiniMax 有 |
| `generation_config.json` | 默认推理参数 | ✅ 两个都有 |
| `model.safetensors.index.json` | 权重分片索引（可逐层核对张量形状） | ⛔ 缺，**非必需**（锦上添花） |
| `modeling_*.py` | forward 实现源码 | ⛔ 缺，**非必需**（config 已能还原全貌） |

---

## 一、总体定位对比

| 维度 | **MiniMax-M3** | **LongCat-2.0 (FP8)** |
|------|----------------|------------------------|
| 出品方 | MiniMax | 美团 (Meituan) |
| 模态 | **原生多模态**（文本 + 图像 + 视频） | **纯文本** |
| 架构类名 | `MiniMaxM3SparseForConditionalGeneration` | `LongcatCausalLM` |
| 总参数 / 激活参数 | ~**428B** / ~**23B** | ~**1.6T** / ~**48B** |
| 上下文长度 | **1,048,576 (1M)** | **262,144 (256K)**，YaRN 从 8K 外推 |
| 词表大小 | 200,064 | 163,840 |
| 权重精度 | bfloat16 | **FP8 (e4m3, 128×128 分块量化)** |
| 定位 | 长上下文 + Agent + Coding + 多模态 | 长上下文 + Agent + Coding |

> **共同点**：都是「超大稀疏 MoE + 长上下文稀疏注意力 + MTP 多 Token 预测（投机解码）」这一代思路，只是在**注意力压缩方案**和**参数扩展方式**上走了不同技术路线。

---

## 二、MiniMax-M3 结构要点

**文本主干**：60 层，hidden_size 6144，64 注意力头，head_dim 128。

### 1. 注意力：GQA + MSA 稀疏注意力
- **GQA**：`num_attention_heads=64` / `num_key_value_heads=4` → 16:1 分组，大幅压缩 KV cache。
- **MiniMax Sparse Attention (MSA)**：`sparse_block_size=128`、`sparse_topk_blocks=16` —— 把序列切块，每个 query 只对 top-16 块做注意力。官方称 1M 上下文下相比 M2 **prefill 快 9×、decode 快 15×**，单 token 计算降到 1/20。
- **前 3 层用全注意力兜底**：`sparse_attention_freq` 前三位为 0，其余 57 层走 MSA。
- **QK-Norm**（`use_qk_norm=true`, `qk_norm_type=per_head`）。
- **部分 RoPE**：`partial_rotary_factor=0.5`（只有一半维度加旋转位置编码），`rope_theta=5,000,000`。

### 2. FFN：细粒度 MoE + 共享专家
- `num_local_experts=128`，每 token 激活 `num_experts_per_tok=4`。
- `n_shared_experts=1` **共享专家**（每 token 必过），`routed_scaling_factor=2.0`。
- 路由打分：**sigmoid**（`scoring_func=sigmoid`）+ routing bias（`use_routing_bias=true`）。
- **前 3 层 Dense，后 57 层 MoE**：`moe_layer_freq` 前三位为 0（dense_intermediate_size=12288）—— 底层 dense 稳住表示，高层 MoE 扩容。

### 3. 其他
- **MTP 投机解码**：`num_mtp_modules=7`，`num_nextn_predict_layers=1`。
- **激活函数**：`swigluoai`（带 `swiglu_alpha=1.702` / `swiglu_limit=7.0` 的 clamp 版 SwiGLU）。
- **归一化**：`use_gemma_norm=true`（Gemma 式 RMSNorm），`rms_norm_eps=1e-6`。

### 4. 视觉塔（vision_config）—— 体现"原生多模态"
- CLIP 式 ViT：hidden 1280、32 层、patch_size 14、支持到 2016×2016 动态分辨率。
- **3D RoPE**（`rope_mode=3d`，图像/视频通用）。
- **视觉 token 压缩**：`patch_merge` 做 2×2 空间合并 + 时间维 2 压缩。
- 投影到 6144 维接入文本主干（`projection_dim=6144`）。

---

## 三、LongCat-2.0 结构要点

**主干**：38 层，hidden_size 8192，64 头。层数更少但更"胖"（维度更大、专家更多）。

### 1. 注意力：MLA + LSA 稀疏注意力
- **MLA（Multi-head Latent Attention，DeepSeek 式低秩压缩）**：
  - `kv_lora_rank=512`、`q_lora_rank=1536`（低秩压缩 Q/KV）。
  - QK 拆分：`qk_nope_head_dim=128`（无位置）+ `qk_rope_head_dim=64`（带 RoPE）。
  - `v_head_dim=128`，`use_mla=1`，`attention_method=MLA`。
  - → 与 MiniMax 的 GQA 是**两条不同的省 KV cache 路线**。
- **LongCat Sparse Attention (LSA)** —— 带独立 **Lightning Indexer**：
  - `index_n_heads=32`、`index_head_dim=128`、`index_topk=2048`、`index_local_tokens=1024`、`index_init_tokens=16`。
  - 相比 DeepSeek DSA 的三个改进：
    - **Streaming-aware Indexing (SI)**：把碎片化内存访问变为连续顺序读，提升 HBM 带宽利用率。
    - **Cross-Layer Indexing (CLI)**：`cli_factor=2` —— 相邻 **2 层共享一次索引**，靠训练时跨层蒸馏摊薄索引成本。
    - **Hierarchical Indexing (HI)**：先块级粗召回，再块内细选 token，缩小索引候选空间。

### 2. FFN：超大规模 MoE + Zero-Expert
- `n_routed_experts=768`，每 token 激活 `moe_topk=12`。
- `expert_ffn_hidden_size=2048`（专家很细粒度），`ffn_hidden_size=12288`（dense 部分）。
- **Zero-Expert 机制**：`zero_expert_num=128`、`zero_expert_type=identity` —— 128 个"恒等/空"专家，token 可路由到"什么都不做"，实现动态算力分配。
- `moe_impl=mix`、`moe_switch_token_num=1024`（前若干 token 用 dense，之后切 MoE）。
- 路由 `routed_scaling_factor=9`。

### 3. N-gram Embedding（LongCat 独有）
- 继承自 LongCat-Flash-Lite，在 **MoE 正交的稀疏维度**扩展参数。
- 模型内含 **135B N-gram Embedding 参数**（`oe_vocab_size_ratio=100.567`、`oe_neighbor_num=5`、`oe_split_num=4`），提升参数利用效率。

### 4. 其他
- **MTP 投机解码**：`mtp_num_layers=3`（3 步草稿），3 个 MTP 草稿步共享一次索引（配合 CLI）。
- **长上下文外推**：`rope_scaling` 用 **deepseek_yarn**，`factor=120`，从 `original_max_position_embeddings=8192` 外推到 256K，`rope_theta=1,000,000`。
- **FP8 量化**：`quant_method=fp8`、`fmt=e4m3`、`weight_block_size=[128,128]`；LayerNorm / router / indexer / embedding 等敏感层保留高精度（见 `ignored_layers`）。

---

## 四、核心差异对照表

| 结构维度 | **MiniMax-M3** | **LongCat-2.0** |
|----------|----------------|-----------------|
| 层数 / hidden | 60 层 / 6144 | 38 层 / 8192 |
| **KV 压缩路线** | **GQA**（64Q:4KV） | **MLA**（低秩 latent 压缩） |
| **稀疏注意力** | MSA（块级 top-k，无独立索引器） | LSA（独立 Lightning Indexer + SI/CLI/HI） |
| 跨层共享索引 | 无 | **CLI，每 2 层共享** |
| 位置编码 | 部分 RoPE（0.5），θ=5e6 | RoPE 拆分（nope+rope），YaRN 外推，θ=1e6 |
| **MoE 专家数** | 128（选 4）+ 1 共享 | **768（选 12）** + 128 Zero-Expert |
| 专家特殊机制 | 共享专家 | **Zero-Expert（恒等专家）** |
| 路由打分 | sigmoid + bias | classifier + e_score_correction_bias |
| 额外参数扩展 | 无 | **N-gram Embedding（135B）** |
| **MTP 步数** | 7 | 3 |
| 归一化 | Gemma RMSNorm + QK-Norm | RMSNorm（双 layernorm.0/.1） |
| 精度 | bf16 | **FP8 (e4m3)** |
| 模态 | 文本 + 图像 + 视频（3D RoPE ViT） | 纯文本 |
| 上下文 | 1M | 256K（YaRN 外推） |

---

## 五、一句话总结

- **MiniMax-M3**：以「**GQA + 块级 MSA 稀疏注意力**」压注意力开销，用「共享专家 + 前 3 层 Dense」的经典 MoE 布局，最大亮点是 **1M 原生多模态**（文本/图像/视频统一 3D RoPE）。走**多模态 + 极致长上下文**路线。

- **LongCat-2.0**：以「**MLA 低秩 + LSA（Lightning Indexer + 跨层共享 CLI）**」压注意力，用「768 专家 + Zero-Expert + N-gram Embedding」把参数堆到 **1.6T**，并以 **FP8** 落地部署。走**极致参数规模 + 硬件高效（含 ASIC/NPU）**路线。

> 两者代表了 2026 年前沿开源大模型的两种主流工程取向：
> **MiniMax 押注"多模态 + 长上下文统一"，LongCat 押注"稀疏规模 + 推理效率极限"。**
