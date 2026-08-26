"""
手动实现 Transformer 模型
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadAttention(nn.Module):
    """
    Attention(Q,K,V) = softmax( QKᵀ/√dₖ ) · V
    """

    def __init__(self, hidden, n_head):
        super().__init__()
        assert hidden % n_head == 0  # 特征维度必须是头数的整数倍
        self.n_head = n_head
        self.d_k = hidden // n_head
        self.qkv = nn.Linear(hidden, hidden * 3)  # 一次性算 Q K V
        self.out = nn.Linear(hidden, hidden)

    def forward(self, x, mask=None):
        B, T, H = x.shape
        # [B, T, H] -> 3 个 [B, T, H]，chunk在最后一个维度上切分
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # view [B, T, H] → [B, T, n_head, d_k]
        # transpose [B, T, n_head, d_k] → [B, n_head, T, d_k]，矩阵乘法、softmax、mask 都不要求内存连续
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)

        # 交换最后两维[B, n_head, T, d_k] -> [B, n_head, d_k, T]
        k_t = k.transpose(-2, -1)
        # 矩阵乘法[B, n_head, T, d_k] @ [B, n_head, d_k, T] -> [B, n_head, T, T]
        qk = q @ k_t
        # 除以√d_k缩放防止点积过大，避免 softmax 梯度消失
        scores = qk / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # Weights[i] = softmax( Q[i] · Kᵀ / √d_k )
        # 在最后一维做归一化
        attn = F.softmax(scores, dim=-1)

        # 矩阵乘法[B, n_head, T, T] @ [B, n_head, T, d_k] -> [B, n_head, T, d_k]
        out = attn @ v
        # 交换中间两维，还原原始形状[B, n_head, T, d_k] -> [B, T, n_head, d_k]
        # contiguous  强制拷贝张量，在内存中重新排布成连续存储，保证后续 view 合法可用
        out = out.transpose(1, 2).contiguous().view(B, T, H)

        return self.out(out)


class EncoderLayer(nn.Module):
    """
    EncoderLayer = MultiHeadAttention + LN1 + FFN + LN2
    """

    def __init__(self, hidden, n_head, ff):
        super().__init__()
        self.attn = MultiHeadAttention(hidden, n_head)
        self.ln1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ff),
            nn.GELU(),
            nn.Linear(ff, hidden),
        )
        self.ln2 = nn.LayerNorm(hidden)

    def forward(self, x, mask=None):
        x = self.ln1(x + self.attn(x, mask))  # 残差 + LN
        x = self.ln2(x + self.ffn(x))
        return x


class TransformerEncoder(nn.Module):
    """
    多层EncoderLayer堆叠
    TransformerEncoder = EncoderLayer * n_layer
    """

    def __init__(self, hidden=768, n_head=12, ff=3072, n_layer=12):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(hidden, n_head, ff) for _ in range(n_layer)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


def transformer_torch(x, d_model=768, nhead=12, dim_feedforward=3072, num_layers=12):
    layer = nn.TransformerEncoderLayer(d_model=d_model,  # 特征维度
                                       nhead=nhead,  # 多头头数
                                       dim_feedforward=dim_feedforward,  # FFN 中间维度
                                       batch_first=True,
                                       dropout=0.0,
                                       activation="gelu",  # 激活函数
                                       norm_first=False)  # LN 放在残差外侧
    encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
    return encoder(x), encoder


# 权重从官方复制到手写模型
def copy_official_weights_to_custom(official_model, custom_model):
    for off_layer, cus_layer in zip(official_model.layers, custom_model.layers):
        # 拆分官方合并的QKV权重
        qkv_w = off_layer.self_attn.in_proj_weight
        qkv_b = off_layer.self_attn.in_proj_bias
        qw, kw, vw = qkv_w.chunk(3, dim=0)
        qb, kb, vb = qkv_b.chunk(3, dim=0)

        cus_layer.attn.qkv.weight.data.copy_(torch.cat([qw, kw, vw], dim=0))
        cus_layer.attn.qkv.bias.data.copy_(torch.cat([qb, kb, vb], dim=0))

        # 注意力输出层
        cus_layer.attn.out.weight.data.copy_(off_layer.self_attn.out_proj.weight)
        cus_layer.attn.out.bias.data.copy_(off_layer.self_attn.out_proj.bias)

        # FFN
        cus_layer.ffn[0].weight.data.copy_(off_layer.linear1.weight)
        cus_layer.ffn[0].bias.data.copy_(off_layer.linear1.bias)
        cus_layer.ffn[2].weight.data.copy_(off_layer.linear2.weight)
        cus_layer.ffn[2].bias.data.copy_(off_layer.linear2.bias)

        # 归一化层
        cus_layer.ln1.weight.data.copy_(off_layer.norm1.weight)
        cus_layer.ln1.bias.data.copy_(off_layer.norm1.bias)
        cus_layer.ln2.weight.data.copy_(off_layer.norm2.weight)
        cus_layer.ln2.bias.data.copy_(off_layer.norm2.bias)


def main():
    torch.manual_seed(42)
    hidden = 512
    n_layer = 1
    n_head = 8
    ff = 1024

    x = torch.randn(2, 16, 512)
    model = TransformerEncoder(hidden, n_head, ff, n_layer)

    with torch.no_grad():
        out_torch, torch_enc = transformer_torch(x, hidden, n_head, ff, n_layer)
        # 把官方权重赋值给手写模型，保证参数完全一致
        copy_official_weights_to_custom(torch_enc, model)
        out = model(x)

    print(f"自定义out形状: {out.shape}")
    print(f"官方out_torch形状: {out_torch.shape}")

    # ===================== 多维度量化对比 =====================
    diff = out - out_torch
    abs_diff = torch.abs(diff)

    # 1. 最大单点误差（最关键）
    max_err = abs_diff.max().item()
    # 2. 平均误差
    mean_err = abs_diff.mean().item()
    # 3. SSE 总平方误差和
    sse = torch.sum(torch.square(diff)).item()
    # 4. 是否所有元素近似相等
    all_close = torch.allclose(out, out_torch, atol=1e-5)

    print("\n========= 误差指标汇总 =========")
    print(f"最大绝对误差: {max_err:.2e}")
    print(f"平均绝对误差: {mean_err:.2e}")
    print(f"SSE总平方误差: {sse:.2e}")
    print(f"全部元素近似相等（容忍1e-5）: {all_close}")

    # 判定规则
    if max_err < 1e-5:
        print("\n✅ 手写代码和PyTorch官方计算逻辑完全一致，微小误差为浮点数正常精度损耗")
    else:
        print("\n❌ 代码逻辑存在错误，计算结果不一致")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", message="enable_nested_tensor is True")
    main()
