from __future__ import annotations

from typing import List

import jieba


def tokenize_text(text: str) -> List[str]:
    return jieba.lcut(text.lower())

