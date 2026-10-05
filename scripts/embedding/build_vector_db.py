import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import torch

from src.embedding.embedder import BGEEmbedder
from src.retrieval.vector_store import ChromaVectorStore


# CONFIG

CHUNKS_PATH = Path("data/chunks/chunks.jsonl")

BATCH_SIZE = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# LOAD CHUNKS

def load_chunks(path: Path):
    chunks = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


# MAIN

def main():

    print("=" * 70)
    print("BUILD VECTOR DATABASE")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Chunks file: {CHUNKS_PATH}")

    # Load chunks

    chunks = load_chunks(CHUNKS_PATH)

    print(f"\nLoaded chunks: {len(chunks)}")

    if not chunks:
        raise ValueError("No chunks found.")

    # Initialize embedder

    print("\nLoading BGE-M3...")

    embedder = BGEEmbedder(
        model_name="BAAI/bge-m3",
        device=DEVICE
    )

    print("BGE-M3 loaded.")

    # Prepare data

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [
        chunk["metadata"]
        for chunk in chunks
    ]

    # Unique ID for each chunk
    ids = [
        f"chunk_{i:08d}"
        for i in range(len(chunks))
    ]

    # Generate embeddings

    print("\nEncoding chunks...")

    embeddings = embedder.encode_documents(
        texts,
        batch_size=BATCH_SIZE
    )

    print("\nEmbedding completed.")
    print(f"Embedding shape: {embeddings.shape}")

    # Initialize ChromaDB

    print("\nInitializing ChromaDB...")

    vector_store = ChromaVectorStore(
        persist_directory="data/vector_db/chroma",
        collection_name="drug_chunks"
    )

    print(
        f"Existing documents: "
        f"{vector_store.count()}"
    )

    # Insert vectors

    print("\nAdding documents to ChromaDB...")

    vector_store.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings
    )

    # Verify

    count = vector_store.count()

    print("\n" + "=" * 70)
    print("VECTOR DATABASE BUILD COMPLETED")
    print("=" * 70)

    print(f"Expected documents : {len(chunks)}")
    print(f"Actual documents   : {count}")

    if count == len(chunks):
        print("\nSTATUS: SUCCESS")
    else:
        print("\nSTATUS: FAILED")


if __name__ == "__main__":
    main()