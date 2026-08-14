"""
字符级语言模型训练脚本
支持RNN/LSTM切换，基于PyTorch实现
自动保存验证集最优模型，使用困惑度(PPL)评估
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import argparse
import math
import time


# ==================== 数据处理模块 ====================

class CharDataset(Dataset):
    """
    自定义字符数据集类
    使用滑动窗口切分样本，y为x整体右移一位
    """
    def __init__(self, text, char2idx, seq_length):
        self.text = text
        self.char2idx = char2idx
        self.seq_length = seq_length

        # 计算总样本数
        self.n_samples = len(text) - seq_length

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 获取输入序列 x
        x_str = self.text[idx:idx + self.seq_length]
        # 获取目标序列 y（x整体右移一位）
        y_str = self.text[idx + 1:idx + self.seq_length + 1]

        # 转换为索引
        x = [self.char2idx[char] for char in x_str]
        y = [self.char2idx[char] for char in y_str]

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def load_corpus(corpus_path):
    """
    读取目录下所有.txt语料文件并拼接
    """
    txt_files = glob.glob(os.path.join(corpus_path, "*.txt"))

    if not txt_files:
        raise FileNotFoundError(f"在 {corpus_path} 目录下未找到.txt文件")

    print(f"找到 {len(txt_files)} 个语料文件:")
    for f in txt_files:
        print(f"  - {os.path.basename(f)}")

    all_text = ""
    for file_path in txt_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_text += f.read()

    print(f"\n语料总长度: {len(all_text)} 字符")
    return all_text


def build_vocab(text):
    """
    基于语料构建字符词表
    生成char2idx和idx2char双向映射
    """
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    char2idx = {char: idx for idx, char in enumerate(chars)}
    idx2char = {idx: char for char, idx in char2idx.items()}

    print(f"\n词表大小: {vocab_size}")
    print(f"字符示例: {''.join(chars[:50])}...")

    return char2idx, idx2char, vocab_size


def split_dataset(dataset, val_ratio=0.1):
    """
    随机划分训练集和验证集
    """
    total_size = len(dataset)
    indices = np.random.permutation(total_size)

    val_size = int(total_size * val_ratio)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    print(f"\n训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    return train_dataset, val_dataset


# ==================== 模型定义模块 ====================

class CharLanguageModel(nn.Module):
    """
    字符级语言模型
    结构: Embedding -> RNN/LSTM -> Dropout -> Linear
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 dropout=0.5, model_type='lstm'):
        super(CharLanguageModel, self).__init__()

        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding层
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 循环层 (RNN或LSTM)
        if model_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif model_type == 'rnn':
            self.rnn = nn.RNN(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}, 请使用 'rnn' 或 'lstm'")

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 全连接输出层
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        前向传播
        x: (batch_size, seq_length)
        """
        # Embedding: (batch_size, seq_length) -> (batch_size, seq_length, embed_dim)
        embed = self.embedding(x)

        # RNN/LSTM
        if self.model_type == 'lstm':
            output, hidden = self.rnn(embed, hidden)
        else:
            output, hidden = self.rnn(embed, hidden)

        # Dropout
        output = self.dropout(output)

        # 全连接层: (batch_size, seq_length, hidden_dim) -> (batch_size, seq_length, vocab_size)
        output = self.fc(output)

        return output, hidden

    def init_hidden(self, batch_size, device):
        """
        初始化隐藏状态
        """
        if self.model_type == 'lstm':
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            return (h0, c0)
        else:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            return h0


# ==================== 训练与评估模块 ====================

def calculate_perplexity(loss):
    """
    计算困惑度 PPL = exp(loss)
    """
    return math.exp(loss)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    训练一个epoch
    返回: 平均损失, 平均PPL
    """
    model.train()
    total_loss = 0
    n_batches = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        # 前向传播
        output, _ = model(batch_x)

        # 计算损失
        # output: (batch_size, seq_length, vocab_size)
        # batch_y: (batch_size, seq_length)
        loss = criterion(output.view(-1, output.size(-1)), batch_y.view(-1))

        # 反向传播
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    avg_ppl = calculate_perplexity(avg_loss)

    return avg_loss, avg_ppl


def evaluate(model, dataloader, criterion, device):
    """
    验证模型
    返回: 平均损失, 平均PPL
    """
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # 前向传播
            output, _ = model(batch_x)

            # 计算损失
            loss = criterion(output.view(-1, output.size(-1)), batch_y.view(-1))

            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches
    avg_ppl = calculate_perplexity(avg_loss)

    return avg_loss, avg_ppl


def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs, save_path, device):
    """
    完整训练流程
    自动保存验证集PPL最低的最优模型
    """
    best_val_ppl = float('inf')
    best_epoch = 0

    print("\n" + "="*80)
    print("开始训练")
    print("="*80)
    print(f"{'Epoch':^6} | {'Train Loss':^12} | {'Train PPL':^12} | "
          f"{'Val Loss':^12} | {'Val PPL':^12} | {'Best':^6}")
    print("-"*80)

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # 训练
        train_loss, train_ppl = train_epoch(model, train_loader,
                                           criterion, optimizer, device)

        # 验证
        val_loss, val_ppl = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        # 判断是否为最优模型
        is_best = val_ppl < best_val_ppl
        if is_best:
            best_val_ppl = val_ppl
            best_epoch = epoch
            # 保存最优模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_ppl': val_ppl,
            }, save_path)

        # 打印日志
        best_mark = "★" if is_best else ""
        print(f"{epoch:^6} | {train_loss:^12.4f} | {train_ppl:^12.2f} | "
              f"{val_loss:^12.4f} | {val_ppl:^12.2f} | {best_mark:^6}")

    print("-"*80)
    print(f"\n训练完成！最优模型保存在第 {best_epoch} 轮，验证集PPL: {best_val_ppl:.2f}")
    print(f"模型已保存至: {save_path}")
    print("="*80)

    return best_val_ppl, best_epoch


# ==================== 主函数 ====================

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='字符级语言模型训练')

    # 模型相关参数
    parser.add_argument('--model_type', type=str, default='lstm',
                       choices=['rnn', 'lstm'], help='模型类型: rnn或lstm')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='训练轮数')
    parser.add_argument('--seq_length', type=int, default=50,
                       help='序列长度')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--embed_dim', type=int, default=128,
                       help='嵌入维度')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='循环层层数')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout概率')

    # 训练相关参数
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='验证集比例')

    # 路径参数
    parser.add_argument('--corpus_path', type=str, default='./data',
                       help='语料库目录路径')
    parser.add_argument('--save_path', type=str, default='./best_model.pth',
                       help='模型保存路径')

    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")

    # 加载语料
    print(f"\n正在加载语料库: {args.corpus_path}")
    text = load_corpus(args.corpus_path)

    # 构建词表
    char2idx, idx2char, vocab_size = build_vocab(text)

    # 创建数据集
    dataset = CharDataset(text, char2idx, args.seq_length)

    # 划分训练集和验证集
    train_dataset, val_dataset = split_dataset(dataset, args.val_ratio)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, drop_last=True)

    # 创建模型
    print(f"\n创建模型: {args.model_type.upper()}")
    print(f"  - 词表大小: {vocab_size}")
    print(f"  - 嵌入维度: {args.embed_dim}")
    print(f"  - 隐藏层维度: {args.hidden_dim}")
    print(f"  - 层数: {args.num_layers}")
    print(f"  - Dropout: {args.dropout}")

    model = CharLanguageModel(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        model_type=args.model_type
    ).to(device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - 可训练参数量: {total_params:,}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 打印训练配置
    print(f"\n训练配置:")
    print(f"  - 训练轮数: {args.num_epochs}")
    print(f"  - 批次大小: {args.batch_size}")
    print(f"  - 序列长度: {args.seq_length}")
    print(f"  - 学习率: {args.learning_rate}")
    print(f"  - 验证集比例: {args.val_ratio}")

    # 开始训练
    best_val_ppl, best_epoch = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        save_path=args.save_path,
        device=device
    )

    # 测试最优模型
    print(f"\n加载最优模型进行测试...")
    checkpoint = torch.load(args.save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_ppl = evaluate(model, val_loader, criterion, device)
    print(f"最优模型验证集表现:")
    print(f"  - 损失: {test_loss:.4f}")
    print(f"  - PPL: {test_ppl:.2f}")

    # 文本生成示例
    print(f"\n生成示例文本:")
    model.eval()
    with torch.no_grad():
        # 随机选择起始字符
        start_idx = np.random.randint(0, len(text) - args.seq_length)
        seed_text = text[start_idx:start_idx + args.seq_length]

        generated = seed_text
        current_seq = seed_text

        for _ in range(50):
            # 准备输入
            x = torch.tensor([[char2idx[c] for c in current_seq]],
                           dtype=torch.long).to(device)

            # 预测下一个字符
            output, _ = model(x)
            next_char_idx = torch.argmax(output[0, -1]).item()
            next_char = idx2char[next_char_idx]

            generated += next_char
            current_seq = current_seq[1:] + next_char

        print(f"  {generated}")


if __name__ == "__main__":
    main()
