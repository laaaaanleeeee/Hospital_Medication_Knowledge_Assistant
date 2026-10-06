import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(
        self,
        chunks_path: str = "data/chunks/chunks.jsonl"
    ):
        self.chunks_path = Path(chunks_path)

        self.documents = []
        self.metadatas = []

        self._load_chunks()
        self._build_index()

    def _load_chunks(self):

        with open(
            self.chunks_path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                item = json.loads(line)

                self.documents.append(
                    item["text"]
                )

                self.metadatas.append(
                    item["metadata"]
                )

        print(
            f"Loaded chunks: "
            f"{len(self.documents)}"
        )

    def _tokenize(self, text: str):

        text = text.lower()

        # Keep Vietnamese characters,
        # numbers and word characters.
        tokens = re.findall(
            r"\w+",
            text,
            flags=re.UNICODE
        )

        return tokens

    def _build_index(self):

        tokenized_documents = [
            self._tokenize(document)
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

        print("BM25 index built successfully.")

    def retrieve(
        self,
        query: str,
        n_results: int = 5
    ):

        query_tokens = self._tokenize(query)

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = scores.argsort()[::-1]

        top_indices = ranked_indices[
            :n_results
        ]

        results = []

        for index in top_indices:

            results.append(
                {
                    "text": self.documents[index],
                    "metadata": self.metadatas[index],
                    "score": float(scores[index])
                }
            )

        return results