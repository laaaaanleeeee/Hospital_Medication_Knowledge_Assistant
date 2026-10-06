from FlagEmbedding import FlagReranker


class BGEReranker:

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True
    ):
        self.model = FlagReranker(
            model_name,
            use_fp16=use_fp16
        )

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5
    ):
        pairs = [
            [query, candidate["text"]]
            for candidate in candidates
        ]

        scores = self.model.compute_score(
            pairs,
            normalize=False
        )

        if not isinstance(scores, list):
            scores = [scores]

        ranked_results = []

        for candidate, score in zip(
            candidates,
            scores
        ):
            result = candidate.copy()
            result["reranker_score"] = float(score)

            ranked_results.append(result)

        ranked_results.sort(
            key=lambda x: x["reranker_score"],
            reverse=True
        )

        return ranked_results[:top_k]