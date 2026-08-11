"""
基线评估缓存工具（优化版，方案 6）。

优化点：
  - 基线 Skills 不变时（temperature=0），基线评估答案也可复用
  - 缓存 key = 初始 Skills 的版本指纹 + 评估集 mtime
  - 命中缓存直接返回，跳过 60 次 LLM 调用
  - teaching_mode=True 时禁用缓存（原版行为）

注意：
  - DeepSeek 即便 temperature=0 也有微小波动，缓存是"近似复现"
  - 缓存文件放在 outputs/baseline_cache.json，重置实验时会被清空
  - 教学场景下建议禁用缓存，保证学生看到实时结果
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime


def _compute_skill_fingerprint(skills: dict[str, str]) -> str:
    """对当前 Skills 内容做 hash，作为缓存 key 的一部分。"""
    h = hashlib.sha256()
    for name in sorted(skills.keys()):
        h.update(name.encode("utf-8"))
        h.update(skills[name].encode("utf-8"))
    return h.hexdigest()[:16]


def _compute_eval_set_fingerprint(eval_set_path: Path) -> str:
    """对 eval_set.json 的 mtime + size 做 hash，避免内容变了缓存还命中。"""
    stat = eval_set_path.stat()
    return f"{int(stat.st_mtime)}_{stat.st_size}"


def try_load_baseline_cache(
    cache_path: Path,
    skills: dict[str, str],
    eval_set_path: Path,
) -> dict | None:
    """
    尝试加载基线缓存。
    返回 dict（命中）或 None（未命中）。

    缓存命中条件：
      1. 缓存文件存在
      2. Skills 指纹匹配（初始 Skills 内容未变）
      3. eval_set 指纹匹配（评估集未修改）
    """
    if not cache_path.exists():
        return None

    skill_fp = _compute_skill_fingerprint(skills)
    eval_fp = _compute_eval_set_fingerprint(eval_set_path)

    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    if cached.get("skill_fingerprint") != skill_fp:
        return None
    if cached.get("eval_set_fingerprint") != eval_fp:
        return None

    print(f"  [BaselineCache] ✓ 命中缓存，跳过 {cached.get('total', 0)} 次 LLM 调用")
    return cached


def save_baseline_cache(
    cache_path: Path,
    result: dict,
    skills: dict[str, str],
    eval_set_path: Path,
):
    """保存基线评估结果到缓存。"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_data = {
        **result,
        "skill_fingerprint": _compute_skill_fingerprint(skills),
        "eval_set_fingerprint": _compute_eval_set_fingerprint(eval_set_path),
        "cached_at": datetime.now().isoformat(),
    }
    cache_path.write_text(
        json.dumps(cache_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  [BaselineCache] ✓ 已保存基线缓存至 {cache_path.name}")


def clear_baseline_cache(cache_path: Path):
    """清除基线缓存（重置实验时调用）。"""
    if cache_path.exists():
        cache_path.unlink()
        print(f"  [BaselineCache] 已清除 {cache_path.name}")
