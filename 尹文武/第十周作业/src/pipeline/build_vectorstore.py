import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parent.parent.parent

sys.path.append(
    str(ROOT_DIR)
)


from src.embedding.embedder import BGEEmbedder
from src.vectorstore.faiss_store import FAISSStore



# =========================

CHUNK_DIR = (
    ROOT_DIR
    /
    "data"
    /
    "chunks"
)


MODEL_PATH = "C://Users//sean//Downloads//LLM_modules//bge-large-zh-v1.5"


FAISS_DIR = (
    ROOT_DIR
    /
    "data"
    /
    "faiss_index"
)



# =========================


def load_chunks():


    all_chunks=[]


    for book_dir in CHUNK_DIR.iterdir():


        if not book_dir.is_dir():
            continue


        file=(
            book_dir
            /
            "chunks.json"
        )


        if not file.exists():
            continue


        print(
            f"[LOAD] {file}"
        )


        with open(
            file,
            encoding="utf-8"
        ) as f:

            chunks=json.load(f)


        all_chunks.extend(
            chunks
        )


    return all_chunks



def main():


    chunks=load_chunks()


    print(
        f"[TOTAL CHUNKS] {len(chunks)}"
    )



    texts=[
        c["content"]
        for c in chunks
    ]


    metadata=[
        {
            **c["metadata"],
            "content": c["content"]
        }
        for c in chunks
    ]



    # embedding

    embedder=BGEEmbedder(
        MODEL_PATH
    )


    vectors=embedder.encode(
        texts,
        batch_size=32
    )


    print(
        vectors.shape
    )



    # FAISS

    store=FAISSStore(
        dimension=vectors.shape[1]
    )


    store.add(
        vectors,
        metadata
    )


    store.save(
        FAISS_DIR
    )



if __name__=="__main__":

    main()
