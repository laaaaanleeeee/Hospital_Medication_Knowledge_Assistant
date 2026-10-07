from FlagEmbedding import FlagReranker


class BGEReranker:

    SECTION_INTENTS = {
        "dosage": [
            "liều",
            "liều dùng",
            "liều lượng",
            "cách dùng",
            "uống",
            "dùng thuốc",
        ],
        "contraindication": [
            "chống chỉ định",
            "không được dùng",
            "không nên dùng",
        ],
        "side_effect": [
            "tác dụng phụ",
            "tác dụng ngoài ý muốn",
        ],
        "interaction": [
            "tương tác thuốc",
            "tương tác",
            "dùng cùng",
            "phối hợp với",
        ],
        "active_ingredient": [
            "hoạt chất",
            "thành phần hoạt chất",
            "thành phần",
        ],
    }

    SECTION_KEYWORDS = {
        "dosage": [
            "liều",
            "cách dùng",
            "liều lượng",
        ],
        "contraindication": [
            "chống chỉ định",
        ],
        "side_effect": [
            "tác dụng phụ",
            "tác dụng ngoài ý muốn",
        ],
        "interaction": [
            "tương tác",
        ],
        "active_ingredient": [
            "thành phần hoạt chất",
            "hoạt chất",
        ],
    }

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
    ) -> list[dict]:

        if not query.strip():
            raise ValueError("Query must not be empty.")

        if top_k <= 0:
            return []

        if not candidates:
            return []

        # --------------------------------------------------
        # 1. BGE semantic reranking
        # --------------------------------------------------

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

        if len(scores) != len(candidates):
            raise ValueError(
                "Number of reranker scores does not match "
                "number of candidates."
            )

        # --------------------------------------------------
        # 2. Detect query intent
        # --------------------------------------------------

        intent = self._detect_intent(query)

        ranked_results = []

        for candidate, score in zip(candidates, scores):

            result = candidate.copy()

            reranker_score = float(score)

            result["reranker_score"] = reranker_score

            # --------------------------------------------------
            # 3. Section-aware adjustment
            # --------------------------------------------------

            section = candidate.get(
                "metadata", {}
            ).get(
                "section",
                ""
            )

            section_boost = self._section_boost(
                section=section,
                intent=intent
            )

            result["section_boost"] = section_boost

            result["final_score"] = (
                reranker_score + section_boost
            )

            ranked_results.append(result)

        # --------------------------------------------------
        # 4. Final ranking
        # --------------------------------------------------

        ranked_results.sort(
            key=lambda item: item["final_score"],
            reverse=True
        )

        return ranked_results[:top_k]

    def _detect_intent(self, query: str) -> str | None:

        query_lower = query.lower()

        best_intent = None
        best_score = 0

        for intent, keywords in self.SECTION_INTENTS.items():

            score = sum(
                1
                for keyword in keywords
                if keyword in query_lower
            )

            if score > best_score:
                best_score = score
                best_intent = intent

        return best_intent

    def _section_boost(
        self,
        section: str,
        intent: str | None
    ) -> float:

        if intent is None:
            return 0.0

        section_lower = section.lower()

        keywords = self.SECTION_KEYWORDS.get(
            intent,
            []
        )

        if any(
            keyword in section_lower
            for keyword in keywords
        ):
            return 1.0

        return 0.0