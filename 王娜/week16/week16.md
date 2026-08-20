# 大模型结构对比：小红书 dots3 与 DeepSeek-V4、Kimi K3、GLM-5.2

> 基于`dots3_readme.md`、`dots3_config.json` 以及 DeepSeek-V4、Kimi K3、GLM-5.2 的 README / 配置文件整理分析。

---

## 一、各模型总体定位

| 模型 | 出品方 | 定位 | 总参数量 | 激活参数量 | 上下文长度 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **dots3-note preview** | 小红书（dots studio） | 多模态 MoE 预览版，dots3 家族最轻量成员 | 280B | 16B | 512K |
| **DeepSeek-V4-Flash** | DeepSeek | 高效百万 token 长文本 MoE 模型 | 284B | 13B | 1M |
| **DeepSeek-V4-Pro** | DeepSeek | 旗舰级百万 token 长文本 MoE 模型 | 1.6T | 49B | 1M |
| **Kimi K3** | Moonshot AI | 原生多模态 Agentic MoE 模型，首个开源 3T 级模型 | 2.8T | 104B | 1M |
| **GLM-5.2** | 智谱 AI（z.ai） | 长程任务旗舰 MoE 模型 | 未公开 | 未公开 | 1M |

---

## 二、核心结构参数对比

| 维度 | dots3-note | DeepSeek-V4-Flash | DeepSeek-V4-Pro | Kimi K3 | GLM-5.2 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **架构类型** | Multimodal MoE | MoE（纯文本） | MoE（纯文本） | MoE（原生多模态） | MoE（纯文本） |
| **总层数** | 46（1 dense + 45 MoE） | 43 | 61 | 93（1 dense + 92 MoE） | 78 |
| **Hidden Size** | 5120 | 4096 | 7168 | 7168 | 6144 |
| **Attention Heads** | 128 | 64 | 128 | 96 | 64 |
| **Dense FFN / 中间层** | 13824 | — | — | 33792 | 12288 |
| **每专家隐藏维** | 1536 | 2048 | 3072 | 3072 | 2048 |
| **路由专家数** | 256 | 256 | 384 | 896 | 256 |
| **共享专家数** | 1 | 1 | 1 | 2 | 1 |
| **每 token 激活专家数** | 8 | 6 | 6 | 16 | 8 |
| **词表大小** | 152064 | 129280 | 129280 | 163840 | 154880 |
| **位置编码基频** | 80M | 10K（Yarn 扩 16×） | 10K（Yarn 扩 16×） | 未明确 | 8M |
| **精度支持** | BF16 / FP8 | FP8 Mixed（专家 FP4） | FP8 Mixed（专家 FP4） | MXFP4 权重 / MXFP8 激活 | BF16 |
| **MTP 层** | 1 层（1.13B） | 1 层 | 1 层 | 0 层 | 1 层 |

---

## 三、注意力机制对比

| 模型 | 注意力机制 | 关键特点 |
| :--- | :--- | :--- |
| **dots3-note** | 13 × 全注意（DSA/Full） + 33 × 滑动窗口注意（SWA），约 1:3 | 全注意层的稀疏索引 top-2048；滑动窗口 size=513；混合 MLA 结构（kv_lora_rank=512、q_lora_rank=1024） |
| **DeepSeek-V4** | 混合注意：CSA（Compressed Sparse Attention）+ HCA（Heavily Compressed Attention） | 1M 上下文下单 token 推理 FLOPs 仅为 V3.2 的 27%、KV Cache 仅 10%；通过 `compress_ratios` 分层压缩 |
| **Kimi K3** | 69 × KDA（Kimi Delta Attention）+ 24 × Gated MLA | KDA 为线性注意变体，搭配 Attention Residuals（AttnRes）；MLA 采用 nope 投影与输出门控 |
| **GLM-5.2** | DSA + IndexShare | 每 4 个稀疏注意层共享同一个 indexer，1M 上下文下 per-token FLOPs 降低 2.9×；`index_topk_freq=4` |

### 关键发现
- **dots3** 采用“少数全注意 + 多数滑动窗口”的低成本长上下文策略，与 GLM-5.2 的 IndexShare、DeepSeek-V4 的 CSA/HCA 目标一致：在超长上下文下减少 KV 与计算开销。
- **DeepSeek-V4** 和 **GLM-5.2** 明确给出 1M 长文本的效率优化指标，dots3 目前 README 中仅强调 512K 支持，未给出对应对比数据。
- **Kimi K3** 的 KDA 属于线性/次二次注意路线，与前三者的“稀疏/压缩全注意”路线不同，兼顾超长上下文与训练稳定性。

---

## 四、MoE 设计对比

| 维度 | dots3-note | DeepSeek-V4-Flash | DeepSeek-V4-Pro | Kimi K3 | GLM-5.2 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **专家规模** | 256 路由 + 1 共享 | 256 路由 + 1 共享 | 384 路由 + 1 共享 | 896 路由 + 2 共享 | 256 路由 + 1 共享 |
| **Top-k** | 8 | 6 | 6 | 16 | 8 |
| **激活占比（专家）** | 16B / 280B ≈ 5.7% | 13B / 284B ≈ 4.6% | 49B / 1.6T ≈ 3.1% | 104B / 2.8T ≈ 3.7% | 未公开 |
| **路由函数** | sigmoid（noaux_tc） | sqrtsoftplus（noaux_tc） | sqrtsoftplus（noaux_tc） | sigmoid（noaux_tc） | sigmoid（noaux_tc） |
| **专家并行/量化** | BF16/FP8 | FP4 专家 + FP8 其他 | FP4 专家 + FP8 其他 | MXFP4 权重 + MXFP8 激活 | BF16 |
| **共享专家设计** | 1 个共享专家 | 1 个共享专家 | 1 个共享专家 | 2 个共享专家 | 1 个共享专家 |

### 关键发现
- **Kimi K3** 专家数最多（896）、Top-k 最大（16），是“极稀疏 + 高选择性”的代表；其激活参数高达 104B，计算强度最大。
- **dots3** 在 280B 总参数量下激活 16B，稀疏比（5.7%）高于 DeepSeek-V4-Flash（4.6%），说明它用更多总参数换取更低的单 token 成本。
- **DeepSeek-V4-Pro** 虽然总参数量达 1.6T，但专家用 FP4，存储与传输成本被大幅压缩。

---

## 五、多模态能力对比

| 模型 | 输入模态 | 视觉编码器 | 音频编码器 | 多模态特点 |
| :--- | :--- | :--- | :--- | :--- |
| **dots3-note** | 文本、图像、视频、音频 | MoE ViT（7B 总参 / 1.2B 激活） | Dense（800M） | 原生四模态；视频输入自动包含音轨；配置中 vision_config 含金字塔路由 `pyramid_num_routed` |
| **DeepSeek-V4** | 文本 | 无 | 无 | 纯语言模型 |
| **Kimi K3** | 文本、图像、视频 | MoonViT-V2（401M） | 无 | 原生多模态 Agentic 模型；vision_config 使用 patch merge + SD2 时序池化 |
| **GLM-5.2** | 文本 | 无 | 无 | 纯语言模型（GLM-5 系列可能另有 VL 变体，但本配置未体现） |

### 关键发现
- **dots3-note** 是五款中唯一同时支持图像、视频、音频输入的模型，且视觉编码器本身也是 MoE（7B 总参数、1.2B 激活），音频使用 800M Dense 编码器。
- **Kimi K3** 支持文本与图像/视频，视觉编码器相对轻量（401M），强调“原生多模态 Agent”。
- DeepSeek-V4 与 GLM-5.2 当前版本为纯文本模型，多模态不在本次文件讨论范围。

---

## 六、训练/推理优化特性对比

| 特性 | dots3-note | DeepSeek-V4 | Kimi K3 | GLM-5.2 |
| :--- | :--- | :--- | :--- | :--- |
| **长上下文优化** | 滑动窗口 + 稀疏全注意（512K） | CSA + HCA（1M） | KDA + Gated MLA（1M） | IndexShare（1M） |
| **残差/连接优化** | 未明确 | Manifold-Constrained Hyper-Connections（mHC） | Attention Residuals（AttnRes） | 未明确 |
| **优化器** | 未明确 | Muon | 未明确 | 未明确 |
| **量化训练** | 支持 FP8 推理 | FP8/FP4 混合量化 | MXFP4/MXFP8 量化感知训练 | BF16 |
| **投机解码** | 支持 MTP | 支持 MTP | 未明确 | 改进版 MTP，接受长度提升 20% |
| **工具/Agent** | 支持 tool calling（OpenAI 兼容） | 支持 reasoning / tool use | 原生 Agentic，支持工具调用 | 支持工具调用 |

---

## 七、dots3-note 结构特点总结

基于 `dots3_readme.md` 与 `dots3_config.json`：

1. **轻量多模态 MoE**：280B 总参数中仅激活 16B，是 dots3 家族最小成员，强调“能力-延迟-成本”平衡。
2. **混合注意力**：13 层全注意 + 33 层滑动窗口注意（约 1:3），全注意层使用 top-2048 的稀疏索引，滑动窗口 size=513，兼顾长上下文与效率。
3. **原生四模态**：文本、图像、视频、音频统一输入；视觉编码器为 MoE ViT，音频为独立 Dense 编码器。
4. **MLA 变体**：配置中同时出现 `kv_lora_rank`、`q_lora_rank`、`qk_nope_head_dim`、`qk_rope_head_dim`，说明采用多层注意力键值压缩（MLA）以降低 KV Cache。
5. **专家路由**：256 路由专家 + 1 共享专家，top-8，sigmoid 评分，noaux_tc 路由。
6. **MTP 投机解码**：1 层共享 MTP（1.13B），与 DeepSeek-V4、GLM-5.2 类似。

---

## 八、综合对比结论

| 维度 | 领先/特色 |
| :--- | :--- |
| **总参数量最大** | Kimi K3（2.8T） |
| **激活参数量最大** | Kimi K3（104B） |
| **上下文最长（公开）** | DeepSeek-V4 / Kimi K3 / GLM-5.2 均达 1M；dots3-note 为 512K |
| **多模态最完整** | dots3-note（文本/图像/视频/音频） |
| **长文本效率优化最明确** | DeepSeek-V4（CSA/HCA，1M 下 FLOPs/KV 大幅减少）、GLM-5.2（IndexShare，2.9× FLOPs 降低） |
| **专家规模最大** | Kimi K3（896 专家，top-16） |
| **轻量化部署** | DeepSeek-V4-Flash（284B/13B）与 dots3-note（280B/16B）均面向高效推理；Flash 用 FP4/FP8 进一步压缩 |
| **开源协议** | dots3-note（Apache 2.0）、DeepSeek-V4（MIT）、GLM-5.2（MIT）；Kimi K3 使用自有 Kimi K3 License |

---

## 参考文件

- `dots3_readme.md`
- `dots3_config.json`
- `dsv4_readme.md`
- `deepseek-ai_DeepSeek-V4-Flash_config.json`
- `deepseek-ai_DeepSeek-V4-Pro_config.json`
- `deepseek-ai_DeepSeek-V3_config.json`
- `kimi_k3_readme.md`
- `moonshotai_Kimi-K3_config.json`
- `glm52_readme.md`
- `zai-org_GLM-5.2_config.json`
