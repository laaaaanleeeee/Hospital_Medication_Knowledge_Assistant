from src.embedding.embedder import BGEEmbedder
from src.retrieval.vector_store import ChromaVectorStore
from src.retrieval.bm25_retriever import BM25Retriever


class HybridRetriever:

    def __init__(
        self,
        embedder: BGEEmbedder,
        vector_store: ChromaVectorStore,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        candidate_k: int = 20
    ):
        # --------------------------------------------------
        # 1. Vector Retrieval
        # --------------------------------------------------

        query_embedding = self.embedder.encode_query(query)

        vector_results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=candidate_k
        )

        vector_documents = vector_results["documents"][0]
        vector_metadatas = vector_results["metadatas"][0]

        # --------------------------------------------------
        # 2. BM25 Retrieval
        # --------------------------------------------------

        bm25_results = self.bm25_retriever.retrieve(
            query=query,
            n_results=candidate_k
        )

        # --------------------------------------------------
        # 3. RRF Fusion
        # --------------------------------------------------

        fused_results = {}

        # Vector ranking
        for rank, (
            document,
            metadata
        ) in enumerate(
            zip(
                vector_documents,
                vector_metadatas
            ),
            start=1
        ):

            chunk_id = self._make_chunk_id(metadata)

            if chunk_id not in fused_results:
                fused_results[chunk_id] = {
                    "text": document,
                    "metadata": metadata,
                    "rrf_score": 0.0
                }

            fused_results[chunk_id]["rrf_score"] += (
                1 / (self.rrf_k + rank)
            )

        # BM25 ranking
        for rank, result in enumerate(
            bm25_results,
            start=1
        ):

            metadata = result["metadata"]
            document = result["text"]

            chunk_id = self._make_chunk_id(metadata)

            if chunk_id not in fused_results:
                fused_results[chunk_id] = {
                    "text": document,
                    "metadata": metadata,
                    "rrf_score": 0.0
                }

            fused_results[chunk_id]["rrf_score"] += (
                1 / (self.rrf_k + rank)
            )

        # --------------------------------------------------
        # 4. Sort by RRF score
        # --------------------------------------------------

        ranked_results = sorted(
            fused_results.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        return ranked_results[:n_results]

    def _make_chunk_id(self, metadata):
        return (
            f"{metadata['drug_id']}_"
            f"{metadata['chunk_index']}"
        )