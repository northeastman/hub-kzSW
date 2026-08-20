# 第十六周作业：调研课堂未讲的开源模型结构特点

课件 `model_code/` 已包含 DeepSeek-V4、Kimi K3、GLM-5.2、Qwen3.6，本报告改为调研 **课堂上未出现** 的开源模型。

## 目录

| 文件 | 说明 |
|------|------|
| [report.md](./report.md) | 主报告：5 个课外开源模型的结构特点 |
| [comparison_table.md](./comparison_table.md) | 结构参数对照表 |

## 调研对象（均不在本周课件中）

1. **MiniMax-M3** — MiniMax Sparse Attention（块级稀疏 GQA）
2. **Llama 4 Scout / Maverick** — iRoPE + 交替 dense/MoE + 早期融合
3. **Gemma 4** — 局部/全局混合注意力 + p-RoPE + K=V 共享
4. **Nemotron 3 Super** — Mamba-2 / Attention 混合 + LatentMoE
5. **GPT-OSS** — 128-token 滑窗与全注意力 1:1 交替 + Attention Sink
