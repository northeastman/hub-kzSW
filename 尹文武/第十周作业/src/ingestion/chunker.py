import re
from typing import List, Dict

from src.ingestion.text_cleaner import TextCleaner


class BookChunker:


    def __init__(
        self,
        chunk_size=800,
        overlap=150
    ):

        self.chunk_size = chunk_size
        self.overlap = overlap

        self.cleaner = TextCleaner()



    def detect_title(self,text):

        """
        判断章节标题
        """

        patterns=[
            r"^第[一二三四五六七八九十百]+章",
            r"^[一二三四五六七八九十]+、",
            r"^[一二三四五六七八九十]+．",
        ]


        for p in patterns:

            if re.match(p,text.strip()):
                return True


        return False



    def merge_pages(
        self,
        pages
    ):

        """
        页级合并
        """

        sections=[]

        current={
            "text":"",
            "start_page":None,
            "end_page":None,
            "title":None
        }


        for page in pages:


            text=self.cleaner.clean(
                page["text"]
            )


            if not self.cleaner.is_valid(text):
                continue


            page_num=page["metadata"]["page"]


            paragraphs=text.split("\n")


            for para in paragraphs:


                para=para.strip()

                if not para:
                    continue


                if self.detect_title(para):

                    if current["text"]:

                        sections.append(current)


                    current={
                        "text":para+"\n",
                        "title":para,
                        "start_page":page_num,
                        "end_page":page_num
                    }


                else:

                    if current["start_page"] is None:
                        current["start_page"]=page_num


                    current["end_page"]=page_num

                    current["text"]+=para+"\n"



        if current["text"]:
            sections.append(current)


        return sections

    def split_section(self, section):
        text = section["text"].strip()
        chunks = []

        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            # 尽量在句号处分割
            if end < text_len:
                cut = text.rfind("。", start, end)
                if cut == -1 or cut <= start:
                    cut = end
                else:
                    cut = cut + 1
            else:
                cut = end

            chunk = text[start:cut].strip()

            if chunk:
                chunks.append(chunk)

            # 防死循环：下一轮必须前进
            next_start = cut - self.overlap

            if next_start <= start:
                next_start = cut

            start = max(next_start, 0)

            if start >= text_len:
                break

        return chunks

    def create_chunks(
        self,
        pages,
        book_name
    ):


        results=[]


        sections=self.merge_pages(
            pages
        )


        for section in sections:


            chunks=self.split_section(
                section
            )


            for i,c in enumerate(chunks):

                results.append(
                    {

                    "content":c,

                    "metadata":{

                        "book":book_name,

                        "chapter":
                            section["title"],

                        "start_page":
                            section["start_page"],

                        "end_page":
                            section["end_page"],

                        "chunk_index":i
                    }

                    }
                )


        return results
