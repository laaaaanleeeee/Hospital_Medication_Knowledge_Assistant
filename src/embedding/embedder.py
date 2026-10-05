from sentence_transformers import SentenceTransformer


class BGEEmbedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cuda"
    ):
        self.model = SentenceTransformer(
            model_name,
            device=device
        )

    def encode_documents(
        self,
        texts: list[str],
        batch_size: int = 4
    ):
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True
        )

    def encode_query(self, query: str):
        return self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )