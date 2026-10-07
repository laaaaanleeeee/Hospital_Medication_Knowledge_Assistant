
import csv
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

# Add project root to Python import path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.embedding.embedder import BGEEmbedder
from src.retrieval.vector_store import ChromaVectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.reranker import BGEReranker

QUERY_FILE = ROOT_DIR / "evaluation" / "retrieval" / "evaluation_queries.json"
OUTPUT_DIR = ROOT_DIR / "evaluation" / "retrieval" / "results"

TOP_K_VALUES = (1, 3, 5)
CANDIDATE_K = 50


def normalize_text(text: str) -> str:
    """Normalize text for robust metadata comparison."""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def normalize_section(section: str) -> str:
    """Normalize common naming variations in section labels."""
    section = normalize_text(section)

    aliases = {
        "tac dung phu": "tac dung ngoai y muon",
        "lieu luong cach dung": "lieu luong cach dung",
    }
    return aliases.get(section, section)


def is_expected_drug(actual: str, expected: str) -> bool:
    return normalize_text(actual) == normalize_text(expected)


def is_expected_section(actual: str, expected_sections: list[str]) -> bool:
    actual_normalized = normalize_section(actual)

    return any(
        actual_normalized == normalize_section(section)
        for section in expected_sections
    )


def evaluate_query(
    item: dict,
    hybrid_retriever,
    reranker,
) -> dict:
    query = item["query"]
    answerable = item["answerable"]
    expected_drug = item.get("expected_drug")
    expected_sections = item.get("expected_sections", [])

    start = time.perf_counter()

    candidates = hybrid_retriever.retrieve(
        query=query,
        n_results=CANDIDATE_K,
        candidate_k=CANDIDATE_K,
    )

    
    debug_ids = {"Q09", "Q16", "Q20"}

    if item["id"] in debug_ids:
        print(f"\n{'=' * 70}")
        print(f"DEBUG {item['id']}: {query}")
        print(f"Candidates before reranking: {len(candidates)}")

        for i, candidate in enumerate(candidates, start=1):
            metadata = candidate.get("metadata", {})
            print(
                f"{i:02d}. "
                f"Drug={metadata.get('drug_name')} | "
                f"Section={metadata.get('section')} | "
                f"Chunk={metadata.get('chunk_index')} | "
                f"RRF Score={candidate.get('rrf_score')}"
            )


    reranked = reranker.rerank(
        query=query,
        candidates=candidates,
        top_k=max(TOP_K_VALUES),
    )

    
    if item["id"] == "Q09":
        print("\n--- Q09: RESULTS AFTER RERANKING ---")

        for rank, result in enumerate(reranked, start=1):
            metadata = result.get("metadata", {})

            print(
                f"{rank}. "
                f"Drug={metadata.get('drug_name')} | "
                f"Section={metadata.get('section')} | "
                f"Chunk={metadata.get('chunk_index')} | "
                f"BGE={result.get('reranker_score', 0):.4f} | "
                f"Boost={result.get('section_boost', 0):.1f} | "
                f"Final={result.get('final_score', 0):.4f}"
            )


    elapsed = time.perf_counter() - start

    # Rank of the first result matching both drug and section.
    relevant_ranks = []

    for rank, result in enumerate(reranked, start=1):
        metadata = result.get("metadata", {})

        drug_match = (
            expected_drug is not None
            and is_expected_drug(
                metadata.get("drug_name", ""),
                expected_drug,
            )
        )

        section_match = is_expected_section(
            metadata.get("section", ""),
            expected_sections,
        )

        if answerable and drug_match and section_match:
            relevant_ranks.append(rank)

    first_relevant_rank = (
        relevant_ranks[0] if relevant_ranks else None
    )

    row = {
        "id": item["id"],
        "query": query,
        "answerable": answerable,
        "expected_drug": expected_drug or "",
        "expected_sections": " | ".join(expected_sections),
        "num_candidates": len(candidates),
        "num_reranked": len(reranked),
        "first_relevant_rank": first_relevant_rank or "",
        "reciprocal_rank": (
            1.0 / first_relevant_rank
            if first_relevant_rank
            else 0.0
        ),
        "elapsed_seconds": round(elapsed, 3),
    }

    for k in TOP_K_VALUES:
        top_results = reranked[:k]

        drug_hit = any(
            expected_drug is not None
            and is_expected_drug(
                result.get("metadata", {}).get("drug_name", ""),
                expected_drug,
            )
            for result in top_results
        )

        section_hit = any(
            is_expected_section(
                result.get("metadata", {}).get("section", ""),
                expected_sections,
            )
            for result in top_results
        )

        joint_hit = any(
            expected_drug is not None
            and is_expected_drug(
                result.get("metadata", {}).get("drug_name", ""),
                expected_drug,
            )
            and is_expected_section(
                result.get("metadata", {}).get("section", ""),
                expected_sections,
            )
            for result in top_results
        )

        row[f"drug_hit_at_{k}"] = int(drug_hit) if answerable else ""
        row[f"section_hit_at_{k}"] = int(section_hit) if answerable else ""
        row[f"joint_hit_at_{k}"] = int(joint_hit) if answerable else ""

    # Include result details for manual inspection.
    row["top_results"] = json.dumps(
        [
            {
                "rank": rank,
                "drug_name": result.get("metadata", {}).get("drug_name"),
                "section": result.get("metadata", {}).get("section"),
                "reranker_score": result.get("reranker_score"),
                "source_url": result.get("metadata", {}).get("source_url"),
            }
            for rank, result in enumerate(reranked, start=1)
        ],
        ensure_ascii=False,
    )

    return row


def print_summary(rows: list[dict]) -> None:
    positive_rows = [row for row in rows if row["answerable"]]
    negative_rows = [row for row in rows if not row["answerable"]]

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total queries:       {len(rows)}")
    print(f"Answerable queries:  {len(positive_rows)}")
    print(f"Out-of-domain queries: {len(negative_rows)}")

    if positive_rows:
        for k in TOP_K_VALUES:
            for metric in ("drug_hit", "section_hit", "joint_hit"):
                key = f"{metric}_at_{k}"
                hits = sum(row[key] for row in positive_rows)
                total = len(positive_rows)
                print(
                    f"{key}: {hits}/{total} "
                    f"({hits / total:.2%})"
                )

        mrr = sum(
            row["reciprocal_rank"] for row in positive_rows
        ) / len(positive_rows)

        print(f"MRR@{max(TOP_K_VALUES)}: {mrr:.4f}")

    if negative_rows:
        empty_results = sum(
            row["num_reranked"] == 0 for row in negative_rows
        )
        print(
            "Out-of-domain queries returning zero results: "
            f"{empty_results}/{len(negative_rows)}"
        )
        print(
            "Note: non-empty retrieval results do not necessarily "
            "mean they are relevant."
        )


def build_evaluation_components():
    """Initialize hybrid retrieval and reranking components."""

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

    reranker = BGEReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        use_fp16=True
    )

    return hybrid_retriever, reranker


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(QUERY_FILE, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    queries = benchmark["queries"]

    # TODO: Initialize these using the same setup as
    # evaluation/retrieval/test_rag_pipeline.py
    #
    # hybrid_retriever = ...
    # reranker = BGEReranker(...)
    #
    # Do not create a new vector database here.

    hybrid_retriever, reranker = build_evaluation_components()

    rows = []

    for index, item in enumerate(queries, start=1):
        print(f"[{index}/{len(queries)}] {item['id']}: {item['query']}")

        try:
            row = evaluate_query(item, hybrid_retriever, reranker)
            rows.append(row)

            print(
                f"  Relevant rank: {row['first_relevant_rank'] or 'MISS'}"
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            raise

    json_path = OUTPUT_DIR / "retrieval_results.json"
    csv_path = OUTPUT_DIR / "retrieval_results.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    if rows:
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print_summary(rows)
    print(f"\nJSON results: {json_path}")
    print(f"CSV results:  {csv_path}")


if __name__ == "__main__":
    main()
