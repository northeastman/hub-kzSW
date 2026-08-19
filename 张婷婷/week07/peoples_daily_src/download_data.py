"""
下载人民日报 NER 数据集并保存为本地 JSON 文件

与原 cluener 项目的区别：
  - 只下载人民日报 NER（去掉 cluener2020 下载部分）
  - 人民日报数据为 CoNLL 格式（每行"字符 BIO标签"，空行分隔句子）
  - 3类实体：PER（人名）/ ORG（组织）/ LOC（地点），共 7 个 BIO 标签

使用方式：
  python download_data.py
"""

import json
import argparse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"

PEOPLES_DAILY_URLS = {
    "train":      "https://raw.githubusercontent.com/OYE93/Chinese-NLP-Corpus/master/NER/People%27s%20Daily/example.train",
    "validation": "https://raw.githubusercontent.com/OYE93/Chinese-NLP-Corpus/master/NER/People%27s%20Daily/example.dev",
    "test":       "https://raw.githubusercontent.com/OYE93/Chinese-NLP-Corpus/master/NER/People%27s%20Daily/example.test",
}

PEOPLES_DAILY_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def _parse_conll(raw_text: str) -> list[dict]:
    """解析 CoNLL 格式：每行 '字符 BIO标签'，空行分隔句子。"""
    records = []
    tokens, tags = [], []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            if tokens:
                records.append({"tokens": tokens, "ner_tags": tags})
                tokens, tags = [], []
        else:
            parts = line.split()
            if len(parts) >= 2:
                tokens.append(parts[0])
                tags.append(parts[-1])
    if tokens:
        records.append({"tokens": tokens, "ner_tags": tags})
    return records


def download_peoples_daily(save_dir: Path):
    """从 GitHub 下载人民日报 NER 数据集（CoNLL 格式，3类实体：PER/ORG/LOC）。"""
    print("=" * 60)
    print("正在下载人民日报 NER 数据集...")
    print("  数据集：3类实体（PER人名 / ORG机构 / LOC地名）")
    print("  规模：训练集 ~20864 句，验证集 ~2318 句，测试集 ~4636 句")
    print("  来源：GitHub OYE93/Chinese-NLP-Corpus（无需账号）")
    print("=" * 60)

    save_dir.mkdir(parents=True, exist_ok=True)

    for split_name, url in PEOPLES_DAILY_URLS.items():
        print(f"  下载 {split_name} ← {url}")
        with urllib.request.urlopen(url, timeout=60) as resp:
            raw_text = resp.read().decode("utf-8")

        records = _parse_conll(raw_text)
        out_path = save_dir / f"{split_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"  [{split_name}] {len(records)} 条 → {out_path}")

    label_path = save_dir / "label_names.json"
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump(PEOPLES_DAILY_LABELS, f, ensure_ascii=False, indent=2)
    print(f"\n人民日报 NER 标签体系（共{len(PEOPLES_DAILY_LABELS)}个）：{PEOPLES_DAILY_LABELS}")
    print()


def main():
    parse_args()
    download_peoples_daily(DATA_DIR)
    print("=" * 60)
    print("人民日报 NER 数据下载完成！")
    print(f"  数据目录: {DATA_DIR}")
    print()
    print("下一步：python explore_data.py")


def parse_args():
    parser = argparse.ArgumentParser(description="下载人民日报 NER 数据集")
    return parser.parse_args()


if __name__ == "__main__":
    main()
