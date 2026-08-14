#用pytorch实现transformer层
import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    """
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        assert self.d_k * num_heads == d_model, "d_model must be divisible by num_heads"

        # 线性变换矩阵
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力
        Args:
            Q: (batch_size, num_heads, seq_len, d_k)
            K: (batch_size, num_heads, seq_len, d_k)
            V: (batch_size, num_heads, seq_len, d_k)
            mask: (batch_size, 1, 1, seq_len) or (batch_size, 1, seq_len, seq_len)
        Returns:
            attention output and attention weights
        """
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用mask（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # softmax归一化
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        output = torch.matmul(attn_weights, V)

        return output, attn_weights

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: (batch_size, seq_len, d_model)
            mask: (batch_size, 1, 1, seq_len) for decoder, None for encoder
        Returns:
            output: (batch_size, seq_len, d_model)
            attn_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)

        # 线性变换并分割成多头
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # 合并多头输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 最终线性变换
        output = self.W_o(attn_output)

        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    """
    位置前馈网络
    """
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.dropout(self.linear2(self.relu(self.linear1(x))))


class PositionalEncoding(nn.Module):
    """
    位置编码
    """
    def __init__(self, d_model=512, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层
    """
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, 1, 1, seq_len)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # 多头自注意力 + 残差连接 + 层归一化
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


class TransformerDecoderLayer(nn.Module):
    """
    Transformer解码器层
    """
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: (batch_size, tgt_seq_len, d_model)
            encoder_output: (batch_size, src_seq_len, d_model)
            src_mask: (batch_size, 1, 1, src_seq_len)
            tgt_mask: (batch_size, 1, tgt_seq_len, tgt_seq_len)
        Returns:
            output: (batch_size, tgt_seq_len, d_model)
        """
        # 掩码多头自注意力
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # 交叉注意力
        cross_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_output))

        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x


class TransformerEncoder(nn.Module):
    """
    Transformer编码器（多层）
    """
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048,
                 max_len=5000, dropout=0.1, pad_idx=0):
        super(TransformerEncoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, src, mask=None):
        """
        Args:
            src: (batch_size, src_seq_len)
            mask: (batch_size, 1, 1, src_seq_len)
        Returns:
            output: (batch_size, src_seq_len, d_model)
        """
        # 词嵌入 + 位置编码
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # 通过多个编码器层
        for layer in self.layers:
            x = layer(x, mask)

        return x


class TransformerDecoder(nn.Module):
    """
    Transformer解码器（多层）
    """
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048,
                 max_len=5000, dropout=0.1, pad_idx=0):
        super(TransformerDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            tgt: (batch_size, tgt_seq_len)
            encoder_output: (batch_size, src_seq_len, d_model)
            src_mask: (batch_size, 1, 1, src_seq_len)
            tgt_mask: (batch_size, 1, tgt_seq_len, tgt_seq_len)
        Returns:
            output: (batch_size, tgt_seq_len, d_model)
        """
        # 词嵌入 + 位置编码
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # 通过多个解码器层
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return x


class Transformer(nn.Module):
    """
    完整的Transformer模型
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048,
                 max_len=5000, dropout=0.1, pad_idx=0):
        super(Transformer, self).__init__()

        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, num_heads, num_encoder_layers,
            d_ff, max_len, dropout, pad_idx
        )

        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, num_heads, num_decoder_layers,
            d_ff, max_len, dropout, pad_idx
        )

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model

    def generate_masks(self, src, tgt, pad_idx=0):
        """
        生成源序列mask和目标序列mask
        """
        # src_mask: 忽略padding位置
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

        # tgt_mask: 忽略padding位置 + 因果mask（防止看到未来信息）
        tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_seq_len = tgt.size(1)
        tgt_causal_mask = torch.tril(torch.ones((tgt_seq_len, tgt_seq_len), device=tgt.device)).bool()
        tgt_mask = tgt_pad_mask & tgt_causal_mask

        return src_mask, tgt_mask

    def forward(self, src, tgt):
        """
        Args:
            src: (batch_size, src_seq_len)
            tgt: (batch_size, tgt_seq_len)
        Returns:
            output: (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        src_mask, tgt_mask = self.generate_masks(src, tgt)

        # 编码
        encoder_output = self.encoder(src, src_mask)

        # 解码
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)

        # 输出层
        output = self.fc_out(decoder_output)

        return output


# ─── 使用示例 ────────────────────────────────────────────────
if __name__ == '__main__':
    # 参数设置
    SRC_VOCAB_SIZE = 10000
    TGT_VOCAB_SIZE = 10000
    D_MODEL = 512
    NUM_HEADS = 8
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    D_FF = 2048
    MAX_LEN = 5000
    DROPOUT = 0.1
    PAD_IDX = 0

    # 创建模型
    model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        d_ff=D_FF,
        max_len=MAX_LEN,
        dropout=DROPOUT,
        pad_idx=PAD_IDX
    )

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试数据
    batch_size = 32
    src_seq_len = 20
    tgt_seq_len = 18

    src = torch.randint(1, SRC_VOCAB_SIZE, (batch_size, src_seq_len))
    tgt = torch.randint(1, TGT_VOCAB_SIZE, (batch_size, tgt_seq_len))

    # 前向传播
    output = model(src, tgt)
    print(f"输入形状: src={src.shape}, tgt={tgt.shape}")
    print(f"输出形状: {output.shape}")  # (batch_size, tgt_seq_len, tgt_vocab_size)

    # 单独测试编码器
    print("\n--- 测试单独编码器 ---")
    encoder = TransformerEncoder(SRC_VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_ENCODER_LAYERS, D_FF)
    encoder_output = encoder(src)
    print(f"编码器输出形状: {encoder_output.shape}")

    # 单独测试多头注意力
    print("\n--- 测试多头注意力 ---")
    attn = MultiHeadAttention(D_MODEL, NUM_HEADS, DROPOUT)
    x = torch.randn(batch_size, src_seq_len, D_MODEL)
    attn_output, attn_weights = attn(x, x, x)
    print(f"注意力输出形状: {attn_output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
