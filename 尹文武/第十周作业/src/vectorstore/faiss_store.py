import faiss
import json
from pathlib import Path
import numpy as np



class FAISSStore:


    def __init__(
        self,
        dimension
    ):

        self.dimension = dimension


        self.index = faiss.IndexFlatIP(
            dimension
        )


        self.metadata=[]



    def add(
        self,
        embeddings,
        metadata
    ):


        embeddings=np.asarray(
            embeddings,
            dtype="float32"
        )


        self.index.add(
            embeddings
        )


        self.metadata.extend(
            metadata
        )


        print(
            f"[FAISS ADD] {len(metadata)} vectors"
        )



    def save(
        self,
        path
    ):

        path=Path(path)

        path.mkdir(
            parents=True,
            exist_ok=True
        )


        faiss.write_index(
            self.index,
            str(path/"index.faiss")
        )


        with open(
            path/"metadata.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                self.metadata,
                f,
                ensure_ascii=False,
                indent=2
            )


        print(
            "[FAISS SAVED]"
        )



    def load(
        self,
        path
    ):


        path=Path(path)


        self.index=faiss.read_index(
            str(path/"index.faiss")
        )


        with open(
            path/"metadata.json",
            encoding="utf-8"
        ) as f:

            self.metadata=json.load(f)



    def search(
        self,
        query_embedding,
        top_k=5
    ):


        scores,ids=self.index.search(
            query_embedding,
            top_k
        )


        results=[]


        for score,idx in zip(
            scores[0],
            ids[0]
        ):


            if idx==-1:
                continue


            item=self.metadata[idx].copy()

            item["score"]=float(score)


            results.append(
                item
            )


        return results
