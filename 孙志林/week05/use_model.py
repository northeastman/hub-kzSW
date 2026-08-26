"""
使用生成式语言模型生成文本
加载 best_model.pt，用户输入文本后自动续写
"""
import torch
import torch.nn.functional as F

from train_language_model import LM

def load_model(checkpoint_path="best_model.pt", device="cpu"):
    """
    加载训练好的模型及词表
    返回: model, char2idx, idx2char, args
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    char2idx = ckpt["char2idx"]
    idx2char = ckpt["idx2char"]
    args = ckpt["args"]
    vocab_size = len(char2idx)

    model = LM(
        vocab_size=vocab_size,
        embed_dim=args["embed_dim"],
        d_model=args["d_model"],
        num_layers=args["num_layers"],
        dropout=args["dropout"],
        nhead=args["nhead"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, char2idx, idx2char, args

def generate(model, char2idx, idx2char, prompt, max_new_tokens=100,
             temperature=0.8, top_k=30, top_p=0.9, device="cpu"):
    """
    自回归生成文本

    参数:
        prompt:         用户输入的起始文本
        max_new_tokens: 最大生成字符数
        temperature:    温度系数，<1 更保守，>1 更随机
        top_k:          top-k 采样，只从概率最高的 k 个字符中选
        top_p:          top-p (nucleus) 采样，累积概率阈值
        device:         运行设备

    返回:
        生成的完整文本（prompt + 续写内容）
    """
    model.eval()

    ids = [char2idx.get(c) for c in prompt if c in char2idx]
    if not ids:
        return prompt

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    generated = list(ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            logits = logits[0, -1, :]

            logits = logits / max(temperature, 1e-8)

            if top_k > 0:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                threshold = topk_vals[-1]
                logits[logits < threshold] = -float("inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[1:] = sorted_mask[:-1].clone()
                sorted_mask[0] = False
                indices_to_remove = sorted_indices[sorted_mask]
                logits[indices_to_remove] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_id)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]], dtype=torch.long, device=device)],
                dim=1
            )

    result = "".join(idx2char.get(i, "") for i in generated)
    return result

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    print("正在加载模型...")
    model, char2idx, idx2char, args = load_model("best_model.pt", device)
    print(f"模型加载完成! 词表大小: {len(char2idx)}, 训练窗口: {args['seq_len']}")

    print("\n" + "=" * 50)
    print("  文本续写工具（输入 quit 退出）")
    print("=" * 50)

    while True:
        try:
            prompt = input("\n请输入文本: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not prompt:
            continue
        if prompt.lower() == "quit":
            print("再见!")
            break

        result = generate(
            model, char2idx, idx2char,
            prompt=prompt,
            max_new_tokens=80,
            temperature=0.8,
            top_k=30,
            top_p=0.9,
            device=device,
        )

        print("-" * 40)
        print(f"续写结果:\n{result}")
        print("-" * 40)

if __name__ == "__main__":
    main()