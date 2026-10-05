import numpy as np
import torch
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.embedding.embedder import BGEEmbedder


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("=" * 70)
    print("TEST BGE-M3 EMBEDDER")
    print("=" * 70)

    print(f"Device: {DEVICE}")

    embedder = BGEEmbedder(
        model_name="BAAI/bge-m3",
        device=DEVICE
    )

    print("\nModel loaded successfully.")

    # Test document embedding
    document = """
    Thuốc Voltaren 75mg/3ml được chỉ định trong điều trị
    các bệnh lý viêm và đau theo hướng dẫn sử dụng.
    """

    print("\nEncoding document...")

    document_embedding = embedder.encode_documents(
        [document]
    )

    print(f"Document embedding shape: {document_embedding.shape}")
    print(f"Embedding dimension: {document_embedding.shape[1]}")

    # Test query embedding
    query = "Voltaren được dùng để điều trị những bệnh gì?"

    print("\nEncoding query...")

    query_embedding = embedder.encode_query(query)

    print(f"Query embedding shape: {query_embedding.shape}")

    # Calculate similarity
    similarity = np.dot(
        document_embedding[0],
        query_embedding
    )

    print(f"\nCosine similarity: {similarity:.4f}")

    print("\n" + "=" * 70)
    print("TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()