from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

class BGEEmbedder:


    def __init__(
        self,
        model_path
    ):
        self.model_path = str(Path(model_path))
        print(
            f"[LOAD EMBEDDING] {self.model_path}"
        )

        self.model = SentenceTransformer(
            self.model_path,
            device=device
        )


    def encode(
        self,
        texts,
        batch_size=64
    ):

        """
        文本转向量

        bge-large-zh-v1.5:
        dim=1024
        """

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )


        return np.asarray(
            embeddings,
            dtype="float32"
        )
