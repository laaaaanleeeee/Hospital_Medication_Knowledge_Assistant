import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.embedding.embedder import BGEEmbedder
from src.retrieval.vector_store import ChromaVectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.reranker import BGEReranker


def main():

    query = "Voltaren có những chống chỉ định nào?"

    print("=" * 70)
    print("RERANKER TEST")
    print("=" * 70)

    # 1. Hybrid retrieval
    embedder = BGEEmbedder(
        model_name="BAAI/bge-m3",
        device="cuda"
    )

    vector_store = ChromaVectorStore(
        persist_directory="data/vector_db/chroma",
        collection_name="drug_chunks"
    )

    bm25_retriever = BM25Retriever()

    hybrid_retriever = HybridRetriever(
        embedder=embedder,
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
        rrf_k=60
    )

    candidates = hybrid_retriever.retrieve(
        query=query,
        n_results=20,
        candidate_k=20
    )

    print(f"\nHybrid candidates: {len(candidates)}")

    # 2. Reranking
    reranker = BGEReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        use_fp16=True
    )

    results = reranker.rerank(
        query=query,
        candidates=candidates,
        top_k=5
    )

    # 3. Display
    print("\n" + "=" * 70)
    print("RERANKED TOP 5")
    print("=" * 70)

    for i, result in enumerate(
        results,
        start=1
    ):
        metadata = result["metadata"]

        print(
            f"\n#{i}"
            f" {metadata['drug_name']}"
            f" | {metadata['section']}"
        )

        print(
            f"Score: "
            f"{result['reranker_score']:.4f}"
        )

        print(
            f"Text: "
            f"{result['text'][:200]}..."
        )


if __name__ == "__main__":
    main()