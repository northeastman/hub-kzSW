#!/usr/bin/env python3
"""统计 v1 / v2 两个 SKILL.md 的 token 消耗，评估上下文占用差异。

关键点：skill 触发时，SKILL.md 的正文会被加载进模型上下文。v2 把冗长的格式细节
挪到了 reference.md，采用"渐进式披露"——只有真正需要核对格式时才加载 reference.md，
因此每次触发 skill 的常态 token 成本大幅下降。
"""
import os
import tiktoken

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V1_SKILL = os.path.join(BASE, "skill-v1-未优化", "log-stats", "SKILL.md")
V2_SKILL = os.path.join(BASE, "skill-v2-优化后", "log-stats", "SKILL.md")
V2_REF = os.path.join(BASE, "skill-v2-优化后", "log-stats", "reference.md")

enc = tiktoken.get_encoding("cl100k_base")


def toks(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return len(enc.encode(text)), len(text)


def main():
    v1_t, v1_c = toks(V1_SKILL)
    v2_t, v2_c = toks(V2_SKILL)
    ref_t, ref_c = toks(V2_REF)

    print("=== SKILL.md Token 消耗对比（cl100k_base 编码）===\n")
    print("%-28s %-10s %-10s" % ("文件", "tokens", "字符数"))
    print("%-28s %-10d %-10d" % ("v1 SKILL.md", v1_t, v1_c))
    print("%-28s %-10d %-10d" % ("v2 SKILL.md（常态加载）", v2_t, v2_c))
    print("%-28s %-10d %-10d" % ("v2 reference.md（按需）", ref_t, ref_c))
    print()
    print("每次触发 skill 的常态上下文 token：")
    print("  v1 = %d tokens" % v1_t)
    print("  v2 = %d tokens" % v2_t)
    print("  降幅 = %.1f%%（省下 %d tokens）" % (
        (v1_t - v2_t) / v1_t * 100, v1_t - v2_t))
    print()
    print("即使 v2 偶尔需要加载 reference.md，总量 = %d tokens，仍比 v1 少 %.1f%%。" % (
        v2_t + ref_t, (v1_t - (v2_t + ref_t)) / v1_t * 100))


if __name__ == "__main__":
    main()
