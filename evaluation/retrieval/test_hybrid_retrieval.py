import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.embedding.embedder import BGEEmbedder
from src.retrieval.vector_store import ChromaVectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever


DEVICE = "cuda"


def main():

    print("=" * 70)
    print("TEST HYBRID RETRIEVAL + RRF")
    print("=" * 70)

    # --------------------------------------------------
    # Initialize components
    # --------------------------------------------------

    embedder = BGEEmbedder(
        model_name="BAAI/bge-m3",
        device=DEVICE
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

    # --------------------------------------------------
    # Query
    # --------------------------------------------------

    query = "Voltaren có những chống chỉ định nào?"

    print(f"\nQuery: {query}")

    results = hybrid_retriever.retrieve(
        query=query,
        n_results=5,
        candidate_k=20
    )

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("HYBRID TOP 5 RESULTS")
    print("=" * 70)

    for rank, result in enumerate(
        results,
        start=1
    ):

        metadata = result["metadata"]

        print(f"\n--- Result {rank} ---")
        print(
            f"RRF Score : "
            f"{result['rrf_score']:.6f}"
        )
        print(
            f"Drug      : "
            f"{metadata['drug_name']}"
        )
        print(
            f"Section   : "
            f"{metadata['section']}"
        )
        print(
            f"Chunk     : "
            f"{metadata['drug_id']}_"
            f"{metadata['chunk_index']}"
        )
        print(
            f"Source    : "
            f"{metadata['source_url']}"
        )

        print("\nText:")
        print(result["text"][:500])


if __name__ == "__main__":
    main()