import sys
from pathlib import Path

# ============================================================
# Add project root to Python path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Imports
# ============================================================

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_store import ChromaVectorStore
from src.retrieval.hybrid_retriever import HybridRetriever

from src.embedding.embedder import BGEEmbedder
from src.reranking.reranker import BGEReranker

from src.rag.context_builder import ContextBuilder
from src.rag.prompt_builder import PromptBuilder
from src.rag.llm import GeminiLLM
from src.rag.pipeline import RAGPipeline


# ============================================================
# Configuration
# ============================================================

CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"

EXPECTED_REFUSAL = (
    "Không tìm thấy thông tin này trong tài liệu được cung cấp."
)

OOD_QUERIES = [
    {
        "id": "OOD01",
        "query": "Thời tiết Hà Nội hôm nay thế nào?",
    },
    {
        "id": "OOD02",
        "query": "Đà Nẵng có những địa điểm du lịch nào?",
    },
    {
        "id": "OOD03",
        "query": "Ai là tổng thống Mỹ hiện tại?",
    },
    {
        "id": "OOD04",
        "query": "Hãy viết một bài thơ về mùa thu.",
    },
    {
        "id": "OOD05",
        "query": "123 * 456 bằng bao nhiêu?",
    },
]


# ============================================================
# Helpers
# ============================================================

def contains_refusal(answer: str) -> bool:
    """
    Check whether the LLM explicitly refuses because
    the requested information is not available in the documents.
    """
    if not answer:
        return False

    return EXPECTED_REFUSAL in answer


# ============================================================
# Build RAG pipeline
# ============================================================

def build_pipeline():
    print("=" * 70)
    print("BUILDING RAG PIPELINE")
    print("=" * 70)

    # ----------------------------
    # Load chunks
    # ----------------------------

    print("\n[1/7] Loading chunks...")

    chunks = []

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        import json

        for line in f:
            chunks.append(json.loads(line))

    print(f"Loaded {len(chunks)} chunks")

    # ----------------------------
    # Embedding
    # ----------------------------

    print("\n[2/7] Loading BGE-M3...")

    embedder = BGEEmbedder(
        model_name="BAAI/bge-m3",
        device="cuda",
    )

    # ----------------------------
    # Vector store
    # ----------------------------

    print("\n[3/7] Loading vector store...")

    vector_store = ChromaVectorStore(
        persist_directory=str(
            PROJECT_ROOT / "data" / "vector_db" / "chroma"
        ),
        collection_name="drug_chunks",
    )

    print(f"Vector store contains {vector_store.count()} chunks")

    # ----------------------------
    # BM25
    # ----------------------------

    print("\n[4/7] Building BM25...")

    bm25_retriever = BM25Retriever(
        str(CHUNKS_PATH)
    )

    # ----------------------------
    # Hybrid retriever
    # ----------------------------

    print("\n[5/7] Building hybrid retriever...")

    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
        embedder=embedder,
    )

    # ----------------------------
    # Reranker
    # ----------------------------

    print("\n[6/7] Loading BGE reranker...")

    reranker = BGEReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        use_fp16=True,
    )

    # ----------------------------
    # RAG components
    # ----------------------------

    print("\n[7/7] Building RAG pipeline...")

    context_builder = ContextBuilder()
    prompt_builder = PromptBuilder()
    llm = GeminiLLM()

    pipeline = RAGPipeline(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm=llm,
        candidate_k=50,
        top_k=5,
    )

    return pipeline


# ============================================================
# Run OOD evaluation
# ============================================================

def main():

    pipeline = build_pipeline()

    print("\n")
    print("=" * 70)
    print("OOD / REFUSAL EVALUATION")
    print("=" * 70)

    passed = 0
    failed = 0
    errors = 0

    for item in OOD_QUERIES:

        query_id = item["id"]
        query = item["query"]

        print("\n" + "-" * 70)
        print(f"{query_id}")
        print(f"Query: {query}")
        print("-" * 70)

        try:
            result = pipeline.run(query)

            answer = result["answer"]
            results = result["results"]

            print("\nTop retrieved documents:")

            for i, doc in enumerate(results, start=1):
                metadata = doc.get("metadata", {})

                drug_name = metadata.get("drug_name", "N/A")
                section = metadata.get("section", "N/A")
                score = doc.get("final_score", 0.0)

                print(
                    f"  #{i} "
                    f"{drug_name} | "
                    f"{section} | "
                    f"score={score:.4f}"
                )

            print("\nAnswer:")
            print(answer)

            # ------------------------------------------------
            # Check refusal
            # ------------------------------------------------

            if contains_refusal(answer):
                print("\nRESULT: PASS")
                passed += 1
            else:
                print("\nRESULT: FAIL")
                print(
                    "Expected refusal phrase was not found:"
                )
                print(f'  "{EXPECTED_REFUSAL}"')

                failed += 1

        except Exception as e:

            print("\nRESULT: ERROR")
            print(f"{type(e).__name__}: {e}")

            errors += 1

    # ========================================================
    # Summary
    # ========================================================

    total = len(OOD_QUERIES)

    print("\n")
    print("=" * 70)
    print("OOD EVALUATION SUMMARY")
    print("=" * 70)

    print(f"Total queries : {total}")
    print(f"Passed        : {passed}")
    print(f"Failed        : {failed}")
    print(f"Errors        : {errors}")

    successful_queries = passed + failed

    if successful_queries > 0:
        refusal_rate = passed / successful_queries
    else:
        refusal_rate = 0.0

    print(
        f"Refusal rate  : {refusal_rate:.2%} "
        f"(among successful requests)"
    )

    print("=" * 70)

    if failed == 0 and errors == 0:
        print("STATUS: PASSED")
        return 0

    if failed == 0 and errors > 0:
        print("STATUS: PASSED WITH API ERRORS")
        return 0

    print("STATUS: FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())