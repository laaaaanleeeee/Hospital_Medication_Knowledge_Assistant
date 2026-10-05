import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# CONFIG

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "chunks"
    / "chunks.jsonl"
)

QUERIES_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "embedding"
    / "benchmark_queries.json"
)

MODELS = {
    "BGE-M3": "BAAI/bge-m3",
    "Multilingual-E5": "intfloat/multilingual-e5-large",
}

TOP_K = 5


# DEVICE

print(f"Device: {DEVICE}")

if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# LOAD DATA

def load_chunks():
    print("\nLoading chunks from:")
    print(CHUNKS_FILE)

    chunks = []

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            line = line.strip()

            if line:
                chunks.append(
                    json.loads(line)
                )

    print(f"Loaded chunks: {len(chunks)}")

    return chunks


def load_queries():
    print("\nLoading benchmark queries from:")
    print(QUERIES_FILE)

    with open(
        QUERIES_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        queries = json.load(f)

    print(f"Loaded queries: {len(queries)}")

    return queries


# EMBEDDING

def encode_documents(
    model,
    texts,
    model_name
):
    """
    Encode all document chunks.

    E5 requires:
        passage: <text>
    """

    if model_name == "Multilingual-E5":

        texts = [
            f"passage: {text}"
            for text in texts
        ]

    embeddings = model.encode(
        texts,
        batch_size=4,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    return embeddings


def encode_queries(
    model,
    queries,
    model_name
):
    """
    Encode queries.

    E5 requires:
        query: <text>
    """

    if model_name == "Multilingual-E5":

        queries = [
            f"query: {query}"
            for query in queries
        ]

    embeddings = model.encode(
        queries,
        batch_size=4,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True
    )

    return embeddings


# RETRIEVAL

def retrieve(
    query_embedding,
    document_embeddings,
    top_k=5
):
    """
    Cosine similarity because vectors
    are normalized.

    cosine similarity =
        dot(query, document)
    """

    scores = np.dot(
        document_embeddings,
        query_embedding
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    return [
        (
            int(index),
            float(scores[index])
        )
        for index in top_indices
    ]


# RELEVANCE

def normalize_text(text):
    return text.strip().lower()


def is_drug_relevant(
    chunk,
    expected_drug
):
    metadata = chunk["metadata"]

    drug_name = normalize_text(
        metadata.get(
            "drug_name",
            ""
        )
    )

    expected_drug = normalize_text(
        expected_drug
    )

    return drug_name == expected_drug


def is_relevant(
    chunk,
    expected_drug,
    expected_sections
):

    metadata = chunk["metadata"]

    drug_name = normalize_text(
        metadata.get(
            "drug_name",
            ""
        )
    )

    section = normalize_text(
        metadata.get(
            "section",
            ""
        )
    )

    expected_drug = normalize_text(
        expected_drug
    )

    expected_sections = [
        normalize_text(section_name)
        for section_name in expected_sections
    ]

    return (
        drug_name == expected_drug
        and section in expected_sections
    )


# HIT CALCULATION

def calculate_hit(
    chunks,
    ranked_indices,
    expected_drug,
    expected_sections,
    k
):
    """
    Section-level relevance.
    """

    top_indices = ranked_indices[:k]

    for index in top_indices:

        chunk = chunks[index]

        if is_relevant(
            chunk,
            expected_drug,
            expected_sections
        ):
            return 1

    return 0


def calculate_drug_hit(
    chunks,
    ranked_indices,
    expected_drug,
    k
):

    top_indices = ranked_indices[:k]

    for index in top_indices:

        chunk = chunks[index]

        if is_drug_relevant(
            chunk,
            expected_drug
        ):
            return 1

    return 0


# PRINT QUERY RESULT

def print_query_result(
    chunks,
    query_data,
    ranked_results
):

    query = query_data["query"]

    expected_drug = query_data[
        "expected_drug"
    ]

    expected_sections = query_data[
        "expected_sections"
    ]

    print("\n" + "=" * 90)

    print(f"QUERY: {query}")

    print(
        f"EXPECTED DRUG: "
        f"{expected_drug}"
    )

    print(
        f"EXPECTED SECTIONS: "
        f"{expected_sections}"
    )

    print("=" * 90)

    for rank, (
        index,
        score
    ) in enumerate(
        ranked_results,
        start=1
    ):

        chunk = chunks[index]

        metadata = chunk["metadata"]

        drug_name = metadata.get(
            "drug_name",
            ""
        )

        section = metadata.get(
            "section",
            ""
        )

        relevant = is_relevant(
            chunk,
            expected_drug,
            expected_sections
        )

        marker = "✓" if relevant else " "

        print(
            f"[{rank}] {marker} "
            f"score={score:.4f}"
        )

        print(
            f"    Drug   : {drug_name}"
        )

        print(
            f"    Section: {section}"
        )


# BENCHMARK MODEL

def benchmark_model(
    model_name,
    model_path,
    chunks,
    queries
):

    print("\n")
    print("#" * 90)

    print(
        f"MODEL: {model_name}"
    )

    print(
        f"PATH : {model_path}"
    )

    print("#" * 90)

    # Load model

    start_time = time.time()

    model = SentenceTransformer(
        model_path,
        device=DEVICE
    )

    load_time = (
        time.time()
        - start_time
    )

    print(
        f"\nModel loaded in "
        f"{load_time:.2f}s"
    )

    # Prepare documents

    document_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # Encode documents

    print(
        f"\nEncoding "
        f"{len(document_texts)} chunks..."
    )

    start_time = time.time()

    document_embeddings = encode_documents(
        model,
        document_texts,
        model_name
    )

    embedding_time = (
        time.time()
        - start_time
    )

    print(
        f"Embedding time: "
        f"{embedding_time:.2f}s"
    )

    print(
        f"Vector shape: "
        f"{document_embeddings.shape}"
    )

    print(
        f"Vector dimension: "
        f"{document_embeddings.shape[1]}"
    )

    # Encode queries

    query_texts = [
        query["query"]
        for query in queries
    ]

    query_embeddings = encode_queries(
        model,
        query_texts,
        model_name
    )

    # Metrics

    section_hits = {
        1: 0,
        3: 0,
        5: 0
    }

    drug_hits = {
        1: 0,
        3: 0,
        5: 0
    }

    # Query benchmark

    for query_index, query_data in enumerate(
        queries
    ):

        query_embedding = (
            query_embeddings[query_index]
        )

        ranked_results = retrieve(
            query_embedding,
            document_embeddings,
            TOP_K
        )

        ranked_indices = [
            index
            for index, score
            in ranked_results
        ]

        print_query_result(
            chunks,
            query_data,
            ranked_results
        )

        # Section-level Hit@K

        for k in [1, 3, 5]:

            section_hits[k] += calculate_hit(
                chunks,
                ranked_indices,
                query_data[
                    "expected_drug"
                ],
                query_data[
                    "expected_sections"
                ],
                k
            )

        # Drug-level Hit@K

        for k in [1, 3, 5]:

            drug_hits[k] += calculate_drug_hit(
                chunks,
                ranked_indices,
                query_data[
                    "expected_drug"
                ],
                k
            )

    # Calculate percentages

    total_queries = len(queries)

    section_results = {
        k: (
            section_hits[k]
            / total_queries
            * 100
        )
        for k in [1, 3, 5]
    }

    drug_results = {
        k: (
            drug_hits[k]
            / total_queries
            * 100
        )
        for k in [1, 3, 5]
    }

    # Print results

    print("\n")
    print("-" * 90)

    print(
        f"RESULTS - {model_name}"
    )

    print("-" * 90)

    print(
        "Section Retrieval"
    )

    print(
        f"Hit@1: "
        f"{section_results[1]:.2f}%"
    )

    print(
        f"Hit@3: "
        f"{section_results[3]:.2f}%"
    )

    print(
        f"Hit@5: "
        f"{section_results[5]:.2f}%"
    )

    print()

    print(
        "Drug Retrieval"
    )

    print(
        f"Drug Hit@1: "
        f"{drug_results[1]:.2f}%"
    )

    print(
        f"Drug Hit@3: "
        f"{drug_results[3]:.2f}%"
    )

    print(
        f"Drug Hit@5: "
        f"{drug_results[5]:.2f}%"
    )

    print("-" * 90)

    return {
        "section": section_results,
        "drug": drug_results
    }


# MAIN

def main():

    print("=" * 90)

    print(
        "EMBEDDING MODEL BENCHMARK"
    )

    print("=" * 90)

    # Load dataset

    chunks = load_chunks()

    queries = load_queries()

    # Run benchmark

    all_results = {}

    for model_name, model_path in MODELS.items():

        results = benchmark_model(
            model_name=model_name,
            model_path=model_path,
            chunks=chunks,
            queries=queries
        )

        all_results[
            model_name
        ] = results

    # Final comparison

    print("\n")
    print("=" * 90)

    print(
        "FINAL COMPARISON"
    )

    print("=" * 90)

    print(
        f"{'Model':<30}"
        f"{'Hit@1':>10}"
        f"{'Hit@3':>10}"
        f"{'Hit@5':>10}"
        f"{'Drug@1':>10}"
        f"{'Drug@3':>10}"
        f"{'Drug@5':>10}"
    )

    print("-" * 90)

    for model_name, results in all_results.items():

        section = results["section"]
        drug = results["drug"]

        print(
            f"{model_name:<30}"
            f"{section[1]:>9.2f}%"
            f"{section[3]:>9.2f}%"
            f"{section[5]:>9.2f}%"
            f"{drug[1]:>9.2f}%"
            f"{drug[3]:>9.2f}%"
            f"{drug[5]:>9.2f}%"
        )

    print("=" * 90)


if __name__ == "__main__":
    main()