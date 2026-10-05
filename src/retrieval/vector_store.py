import chromadb


class ChromaVectorStore:
    def __init__(
        self,
        persist_directory: str = "data/vector_db/chroma",
        collection_name: str = "drug_chunks"
    ):
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "Vietnamese drug information chunks"
            }
        )

    def count(self) -> int:
        return self.collection.count()

    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        embeddings,
        batch_size: int = 5000
    ):
        embeddings = embeddings.tolist()

        for start in range(0, len(ids), batch_size):
            end = start + batch_size

            print(
                f"Adding batch: "
                f"{start} → {min(end, len(ids))}"
            )

            self.collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                embeddings=embeddings[start:end]
            )

    def query(
        self,
        query_embedding,
        n_results: int = 5
    ):
        return self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )