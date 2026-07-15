import json
from pathlib import Path
import sys


# =========================
# 添加项目根目录
# =========================

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.append(
    str(ROOT_DIR)
)


from src.ingestion.chunker import BookChunker



# =========================
# 路径配置
# =========================

OCR_DIR = ROOT_DIR / "data" / "ocr_output"

CHUNK_DIR = ROOT_DIR / "data" / "chunks"



# =========================
# 加载OCR JSON
# =========================

def load_ocr_json(file_path: Path):

    """
    加载百度 Unlimited-OCR JSON

    返回:
        pages列表
    """

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)


    pages = data.get(
        "pages",
        []
    )


    result = []


    for page in pages:


        text = page.get(
            "text",
            ""
        )


        page_num = page.get(
            "page_num"
        )


        result.append(
            {
                "text": text,

                "metadata":{

                    "source":
                        file_path.name,

                    "page":
                        page_num

                }
            }
        )


    return result



# =========================
# 保存chunks
# =========================

def save_chunks(
    book_name,
    chunks
):

    output_dir = (
        CHUNK_DIR /
        book_name
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    output_file = (
        output_dir /
        "chunks.json"
    )


    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2
        )


    return output_file



# =========================
# 单本书处理
# =========================

def process_book(
    file_path: Path,
    chunker: BookChunker
):


    book_name = file_path.stem


    print(
        "\n=============================="
    )

    print(
        f"[BOOK] {book_name}"
    )


    # OCR加载

    pages = load_ocr_json(
        file_path
    )


    print(
        f"[PAGES] {len(pages)}"
    )


    # chunk

    chunks = chunker.create_chunks(
        pages,
        book_name
    )


    print(
        f"[CHUNKS] {len(chunks)}"
    )


    # 保存

    output = save_chunks(
        book_name,
        chunks
    )


    print(
        f"[SAVE] {output}"
    )


    return len(chunks)



# =========================
# 主程序
# =========================

def main():


    CHUNK_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    chunker = BookChunker(
        chunk_size=800,
        overlap=150
    )


    json_files = list(
        OCR_DIR.glob(
            "*.json"
        )
    )


    if not json_files:

        print(
            "[ERROR] 没有找到OCR JSON文件"
        )

        return



    print(
        f"[FOUND OCR FILES] {len(json_files)}"
    )


    total_chunks = 0


    success = 0


    for file in json_files:
        print(f"[START] {file.name}", flush=True)

        try:

            count = process_book(
                file,
                chunker
            )


            total_chunks += count
            success += 1

            print(f"[OK] {file.name}, chunks={count}", flush=True)

        except Exception as e:


            print(
                f"[FAILED] {file.name}"
            )

            print(
                e
            )



    print(
        "\n=============================="
    )

    print(
        "[DONE]"
    )

    print(
        f"成功处理: {success}/{len(json_files)}"
    )

    print(
        f"总chunks: {total_chunks}"
    )



if __name__ == "__main__":

    main()
