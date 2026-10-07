class RAGPipeline:

    def __init__(
        self,
        hybrid_retriever,
        reranker,
        context_builder,
        prompt_builder,
        llm,
        candidate_k: int = 50,
        top_k: int = 5
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.llm = llm

        self.candidate_k = candidate_k
        self.top_k = top_k

    def run(
        self,
        query: str
    ) -> dict:

        # 1. Hybrid retrieval
        candidates = self.hybrid_retriever.retrieve(
            query=query,
            n_results=self.candidate_k,
            candidate_k=self.candidate_k
        )

        # 2. Reranking
        reranked_results = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=self.top_k
        )

        # 3. Build context
        context = self.context_builder.build(
            reranked_results
        )

        # 4. Build prompt
        prompt = self.prompt_builder.build(
            context=context,
            question=query
        )

        # 5. Generate answer
        answer = self.llm.generate(
            prompt
        )

        return {
            "query": query,
            "answer": answer,
            "results": reranked_results,
            "context": context
        }