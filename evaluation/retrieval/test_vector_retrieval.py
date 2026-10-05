import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.embedding.embedder import BGEEmbedder
from src.retrieval.vector_store import ChromaVectorStore


DEVICE = "cuda"

BENCHMARK_PATH = (
    ROOT_DIR
    / "evaluation"
    / "embedding"
    / "benchmark_queries.json"
)


def load_benchmark():
    with open(
        BENCHMARK_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def check_drug_hit(results, expected_drug, k):
    metadatas = results["metadatas"][0][:k]

    return any(
        metadata["drug_name"] == expected_drug
        for metadata in metadatas
    )


def check_section_hit(results, expected_sections, k):
    metadatas = results["metadatas"][0][:k]

    return any(
        metadata["section"] in expected_sections
        for metadata in metadatas
    )


def main():

    print("=" * 70)
    print("VECTOR RETRIEVAL EVALUATION")
    print("=" * 70)

    benchmark = load_benchmark()

    print(f"\nBenchmark queries: {len(benchmark)}")

    # Load embedding model

    print("\nLoading BGE-M3...")

    embedder = BGEEmbedder(
        model_name="BAAI/bge-m3",
        device=DEVICE
    )

    # Load vector DB

    vector_store = ChromaVectorStore(
        persist_directory="data/vector_db/chroma",
        collection_name="drug_chunks"
    )

    print(
        f"Documents in DB: "
        f"{vector_store.count()}"
    )

    # Evaluation

    ks = [1, 3, 5]

    drug_hits = {
        k: 0
        for k in ks
    }

    section_hits = {
        k: 0
        for k in ks
    }

    print("\n" + "=" * 70)
    print("RUNNING EVALUATION")
    print("=" * 70)

    for index, item in enumerate(benchmark, start=1):

        query = item["query"]
        expected_drug = item["expected_drug"]
        expected_sections = item["expected_sections"]

        print(
            f"\n[{index}/{len(benchmark)}] "
            f"{query}"
        )

        query_embedding = embedder.encode_query(
            query
        )

        results = vector_store.query(
            query_embedding=query_embedding,
            n_results=5
        )

        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # Show Top 5
        for rank, (
            metadata,
            distance
        ) in enumerate(
            zip(metadatas, distances),
            start=1
        ):
            print(
                f"  {rank}. "
                f"{metadata['drug_name']} "
                f"| {metadata['section']} "
                f"| distance={distance:.4f}"
            )

        # Calculate metrics
        for k in ks:

            if check_drug_hit(
                results,
                expected_drug,
                k
            ):
                drug_hits[k] += 1

            if check_section_hit(
                results,
                expected_sections,
                k
            ):
                section_hits[k] += 1

    # Final results

    total = len(benchmark)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"\nTotal queries: {total}")

    print("\nDrug Retrieval:")

    for k in ks:

        score = (
            drug_hits[k]
            / total
            * 100
        )

        print(
            f"  Drug Hit@{k}: "
            f"{score:.1f}% "
            f"({drug_hits[k]}/{total})"
        )

    print("\nSection Retrieval:")

    for k in ks:

        score = (
            section_hits[k]
            / total
            * 100
        )

        print(
            f"  Section Hit@{k}: "
            f"{score:.1f}% "
            f"({section_hits[k]}/{total})"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()