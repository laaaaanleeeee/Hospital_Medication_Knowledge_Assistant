import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.embedding.embedder import BGEEmbedder
from src.retrieval.vector_store import ChromaVectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.reranker import BGEReranker
from src.rag.context_builder import ContextBuilder
from src.rag.prompt_builder import PromptBuilder
from src.rag.llm import GeminiLLM
from src.rag.pipeline import RAGPipeline


def main():

    print("=" * 70)
    print("RAG PIPELINE TEST")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Embedding
    # --------------------------------------------------

    embedder = BGEEmbedder(
        model_name="BAAI/bge-m3",
        device="cuda"
    )

    # --------------------------------------------------
    # 2. Vector store
    # --------------------------------------------------

    vector_store = ChromaVectorStore(
        persist_directory="data/vector_db/chroma",
        collection_name="drug_chunks"
    )

    # --------------------------------------------------
    # 3. BM25
    # --------------------------------------------------

    bm25_retriever = BM25Retriever()

    # --------------------------------------------------
    # 4. Hybrid retriever
    # --------------------------------------------------

    hybrid_retriever = HybridRetriever(
        embedder=embedder,
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
        rrf_k=60
    )

    # --------------------------------------------------
    # 5. Reranker
    # --------------------------------------------------

    reranker = BGEReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        use_fp16=True
    )

    # --------------------------------------------------
    # 6. Context builder
    # --------------------------------------------------

    context_builder = ContextBuilder()

    # --------------------------------------------------
    # 7. Prompt builder
    # --------------------------------------------------

    prompt_builder = PromptBuilder()

    # --------------------------------------------------
    # 8. LLM
    # --------------------------------------------------

    llm = GeminiLLM()

    # --------------------------------------------------
    # 9. RAG pipeline
    # --------------------------------------------------

    pipeline = RAGPipeline(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm=llm,
        candidate_k=20,
        top_k=5
    )

    # --------------------------------------------------
    # 10. Query
    # --------------------------------------------------

    query = "Voltaren có những chống chỉ định nào?"

    result = pipeline.run(query)

    print("\n" + "=" * 70)
    print("CONTEXT SENT TO GEMINI")
    print("=" * 70)
    print(result["context"])

    # --------------------------------------------------
    # 11. Output
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("QUERY")
    print("=" * 70)

    print(result["query"])

    print("\n" + "=" * 70)
    print("TOP RERANKED RESULTS")
    print("=" * 70)

    for index, item in enumerate(
        result["results"],
        start=1
    ):
        metadata = item["metadata"]

        print(
            f"\n#{index}"
            f" {metadata['drug_name']}"
            f" | {metadata['section']}"
        )

        print(
            f"Reranker score: "
            f"{item['reranker_score']:.4f}"
        )

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(result["answer"])

    assert result["answer"].strip()
    assert len(result["results"]) == 5
    assert result["context"].strip()

    print("\nSTATUS: PASSED")


if __name__ == "__main__":
    main()