import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.embedding.embedder import BGEEmbedder
from src.retrieval.vector_store import ChromaVectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever


DEVICE = "cuda"

BENCHMARK_PATH = (
    ROOT_DIR
    / "evaluation"
    / "embedding"
    / "benchmark_queries.json"
)

TOP_K = 5
CANDIDATE_K = 20


def load_benchmark():
    with open(
        BENCHMARK_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def section_hit(section, expected_sections):
    section = section.lower()

    return any(
        expected.lower() in section
        for expected in expected_sections
    )


def evaluate():

    print("=" * 70)
    print("HYBRID RETRIEVAL + RRF EVALUATION")
    print("=" * 70)

    benchmark = load_benchmark()

    print(f"\nBenchmark queries: {len(benchmark)}")

    # --------------------------------------------------
    # Initialize retrievers
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
    # Metrics
    # --------------------------------------------------

    total_queries = len(benchmark)

    drug_hits = {
        1: 0,
        3: 0,
        5: 0
    }

    section_hits = {
        1: 0,
        3: 0,
        5: 0
    }

    # --------------------------------------------------
    # Evaluate each query
    # --------------------------------------------------

    for i, item in enumerate(
        benchmark,
        start=1
    ):

        query = item["query"]
        expected_drug = item["expected_drug"]
        expected_sections = item["expected_sections"]

        results = hybrid_retriever.retrieve(
            query=query,
            n_results=TOP_K,
            candidate_k=CANDIDATE_K
        )

        print("\n" + "-" * 70)
        print(f"[{i}/{total_queries}] {query}")
        print(f"Expected drug: {expected_drug}")
        print(
            f"Expected sections: "
            f"{expected_sections}"
        )

        # --------------------------------------------------
        # Display Top 5
        # --------------------------------------------------

        for rank, result in enumerate(
            results,
            start=1
        ):

            metadata = result["metadata"]

            drug_name = metadata["drug_name"]
            section = metadata["section"]
            score = result["rrf_score"]

            print(
                f"{rank}. "
                f"{drug_name} | "
                f"{section} | "
                f"RRF {score:.6f}"
            )

        # --------------------------------------------------
        # Drug Hit@K
        # --------------------------------------------------

        for k in [1, 3, 5]:

            top_k_results = results[:k]

            drug_hit = any(
                expected_drug.lower()
                in result["metadata"]["drug_name"].lower()
                for result in top_k_results
            )

            if drug_hit:
                drug_hits[k] += 1

        # --------------------------------------------------
        # Section Hit@K
        # --------------------------------------------------

        for k in [1, 3, 5]:

            top_k_results = results[:k]

            section_found = any(
                section_hit(
                    result["metadata"]["section"],
                    expected_sections
                )
                for result in top_k_results
            )

            if section_found:
                section_hits[k] += 1

    # ------------------------------------------------------
    # Final Results
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"\nTotal queries: {total_queries}")

    print("\nDrug Retrieval:")

    for k in [1, 3, 5]:

        percentage = (
            drug_hits[k]
            / total_queries
            * 100
        )

        print(
            f"Drug Hit@{k}: "
            f"{percentage:.2f}% "
            f"({drug_hits[k]}/{total_queries})"
        )

    print("\nSection Retrieval:")

    for k in [1, 3, 5]:

        percentage = (
            section_hits[k]
            / total_queries
            * 100
        )

        print(
            f"Section Hit@{k}: "
            f"{percentage:.2f}% "
            f"({section_hits[k]}/{total_queries})"
        )


if __name__ == "__main__":
    evaluate()