import re


class TextCleaner:

    def clean(self, text: str) -> str:

        if not text:
            return ""

        # unicode 空格
        text = text.replace("\u3000", "")
        text = text.replace("\xa0", " ")

        # 去掉页码
        text = re.sub(
            r"\n\s*\d+\s*\n",
            "\n",
            text
        )

        # 去除 OCR 单独残留字符
        text = re.sub(
            r"\n[^\u4e00-\u9fa5]{1,5}\n",
            "\n",
            text
        )

        # 合并空白
        text = re.sub(
            r"[ \t]+",
            " ",
            text
        )

        # 合并多余换行
        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text
        )

        return text.strip()


    def is_valid(self,text):

        if len(text)<30:
            return False

        chinese_chars=len(
            re.findall(
                r"[\u4e00-\u9fa5]",
                text
            )
        )

        # 中文比例过低
        if chinese_chars / len(text)<0.3:
            return False

        return True
