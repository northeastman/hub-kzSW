# 课外开源模型结构对照

课件已覆盖 DeepSeek-V4 / Kimi K3 / GLM-5.2 / Qwen3.6，下表为另外 5 个开源（或开源权重）模型。

| 维度 | MiniMax-M3 | Llama 4 Scout | Llama 4 Maverick | Gemma 4-26B-A4B | Nemotron 3 Super | GPT-OSS-120B |
|------|------------|---------------|------------------|-----------------|------------------|--------------|
| **机构** | MiniMax | Meta | Meta | Google | NVIDIA | OpenAI |
| **总参 / 激活** | 428B / ~23B | 109B / 17B | 400B / 17B | 25.2B / 3.8B | 120.6B / 12.7B | 116.8B / 5.1B |
| **注意力** | GQA + MSA 块稀疏 | iRoPE（RoPE 分块 + NoPE 全局） | 同 Scout，1M 上下文 | 5:1 滑窗/全局 + p-RoPE | 少量 GQA 锚点，主体 Mamba-2 | 1:1 滑窗 128 / 全注意力 |
| **MoE** | 稀疏 MoE | 16 routed + 1 shared，每 token 1 个 routed | 128 routed + 1 shared，每 token 1 个 routed | 8/128 + 1 shared | LatentMoE：512 expert，top-22，潜空间 1024 | 4/128 |
| **上下文** | 1M | 10M（训练 256K） | 1M | 256K | 1M | 131K（YaRN） |
| **多模态** | 原生图/视频 | 早期融合图文 | 早期融合图文 | 图（部分规格含音频） | 文本为主 | 纯文本 |
| **位置编码** | RoPE（部分维） | iRoPE + 推理温度缩放 | 同左 | 局部标准 RoPE，全局 p-RoPE | **无位置编码** | RoPE + YaRN |
| **核心创新** | 按 GQA 组独立选 KV 块 | 10M 上下文 + 极稀 MoE | 同激活、更多专家 | 端侧多模态 + K=V | SSM+Transformer+LatentMoE | Attention Sink + MXFP4 |
| **协议** | MiniMax Community | Llama Community | Llama Community | Gemma ToU / Apache 2.0 系 | 权重+数据+配方开源 | Apache 2.0 |
