from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        if not results:
            return "I could not find relevant information in the knowledge base."

        context_parts = []
        for index, result in enumerate(results, start=1):
            source = result.get("metadata", {}).get("source", result.get("id", "unknown"))
            context_parts.append(f"[{index}] Source: {source}\n{result['content']}")
        context = "\n\n".join(context_parts)
        prompt = (
            "Answer the question using only the provided context. "
            "If the context does not contain the answer, say so clearly. "
            "Cite relevant context numbers such as [1].\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        return self.llm_fn(prompt)
