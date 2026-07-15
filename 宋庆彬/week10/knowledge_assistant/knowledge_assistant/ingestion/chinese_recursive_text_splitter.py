from __future__ import annotations

import re
from typing import Any, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter


def _split_text_with_regex_from_end(text: str, separator: str, keep_separator: bool) -> List[str]:
    if separator:
        if keep_separator:
            splits = re.split(f"({separator})", text)
            merged = ["".join(item) for item in zip(splits[0::2], splits[1::2])]
            if len(splits) % 2 == 1:
                merged += splits[-1:]
            result = merged
        else:
            result = re.split(separator, text)
    else:
        result = list(text)
    return [item for item in result if item]


class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(
        self,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or [
            "\n\n",
            "\n",
            "。|！|？",
            r"\.\s|\!\s|\?\s",
            r"；|;\s",
            r"，|,\s",
        ]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        separator = separators[-1]
        new_separators = []
        for index, candidate in enumerate(separators):
            escaped = candidate if self._is_separator_regex else re.escape(candidate)
            if candidate == "":
                separator = candidate
                break
            if re.search(escaped, text):
                separator = candidate
                new_separators = separators[index + 1 :]
                break

        escaped_separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex_from_end(text, escaped_separator, self._keep_separator)

        good_splits = []
        merge_separator = "" if self._keep_separator else separator
        for split in splits:
            if self._length_function(split) < self._chunk_size:
                good_splits.append(split)
            else:
                if good_splits:
                    final_chunks.extend(self._merge_splits(good_splits, merge_separator))
                    good_splits = []
                if not new_separators:
                    final_chunks.append(split)
                else:
                    final_chunks.extend(self._split_text(split, new_separators))

        if good_splits:
            final_chunks.extend(self._merge_splits(good_splits, merge_separator))
        return [re.sub(r"\n{2,}", "\n", chunk.strip()) for chunk in final_chunks if chunk.strip()]

