from __future__ import annotations

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv

from src import (
    EMBEDDING_PROVIDER_ENV,
    GEMINI_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    Document,
    EmbeddingStore,
    GeminiEmbedder,
    LocalEmbedder,
    MockEmbedder,
    OpenAIEmbedder,
    SentenceChunker,
)


DATA_DIR = Path("data/thu-vien-vnuhcm")

# Keep this as the only strategy choice so team members can swap one line fairly.
CHUNKER = SentenceChunker(max_sentences_per_chunk=3)

QUERIES = [
    ("Sinh viên chính quy được mượn bao nhiêu tài liệu và trong bao nhiêu ngày?", {"audience": "student"}),
    ("Giảng viên và nghiên cứu sinh được mượn tài liệu trong bao nhiêu ngày?", {"audience": "faculty"}),
    ("Độc giả ngoài ĐHQG-HCM được mượn bao nhiêu tài liệu và có được gia hạn không?", None),
    ("Những tài liệu nào không được mượn về nhà?", None),
    ("Dịch vụ cung cấp thông tin theo yêu cầu có những loại phí nào?", None),
]


class CachedEmbedder:
    """Cache embeddings by content hash to avoid duplicate API calls."""

    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self._cache: dict[str, list[float]] = {}
        self._backend_name = getattr(embedder, "_backend_name", type(embedder).__name__)

    def __call__(self, text: str) -> list[float]:
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if content_hash not in self._cache:
            self._cache[content_hash] = self.embedder(text)
        return self._cache[content_hash]


def create_embedder():
    """Select the configured real backend, with an explicit mock fallback."""
    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "").strip().lower()
    supported_providers = {"gemini", "openai", "local", "mock"}
    if provider not in supported_providers:
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            provider = "gemini"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        else:
            provider = "mock"

    try:
        if provider == "gemini":
            embedder = GeminiEmbedder(
                model_name=os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL)
            )
        elif provider == "openai":
            embedder = OpenAIEmbedder(
                model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL)
            )
        elif provider == "local":
            embedder = LocalEmbedder(
                model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL)
            )
        elif provider == "mock":
            embedder = MockEmbedder()
        else:
            raise ValueError("Unsupported embedding provider")
    except Exception as error:
        print(f"Embedding backend '{provider}' unavailable: {type(error).__name__}")
        print("Falling back to mock embeddings.")
        embedder = MockEmbedder()
    return CachedEmbedder(embedder)


def parse_document(path: Path) -> tuple[dict[str, str], str]:
    """Read YAML-like front matter and the cleaned Markdown body."""
    parts = path.read_text(encoding="utf-8").split("---", 2)
    if len(parts) != 3:
        raise ValueError(f"Missing front matter: {path}")

    metadata: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"')
    return metadata, parts[2].strip()


def load_chunked_documents() -> list[Document]:
    documents: list[Document] = []
    for path in sorted(DATA_DIR.glob("*.md")):
        metadata, content = parse_document(path)
        chunks = CHUNKER.chunk(content)
        for index, chunk in enumerate(chunks):
            documents.append(
                Document(
                    id=f"{path.stem}#{index}",
                    content=chunk,
                    metadata={
                        **metadata,
                        "doc_id": path.stem,
                        "chunk_index": str(index),
                        "source_file": str(path),
                    },
                )
            )
    return documents


def main() -> int:
    documents = load_chunked_documents()
    embedder = create_embedder()
    store = EmbeddingStore(collection_name="hust-library-benchmark", embedding_fn=embedder)
    store.add_documents(documents)

    print(f"Chunker: {CHUNKER.__class__.__name__}")
    print(f"Embedding backend: {embedder._backend_name}")
    print(f"Loaded documents: {len(documents)} chunks")
    for question, metadata_filter in QUERIES:
        results = store.search_with_filter(
            question,
            top_k=3,
            metadata_filter=metadata_filter,
        )
        filter_label = metadata_filter or "none"
        print(f"\nQuery: {question}")
        print(f"Filter: {filter_label}")
        for rank, result in enumerate(results, start=1):
            print(
                f"  {rank}. score={result['score']:.4f} "
                f"doc_id={result['metadata']['doc_id']} "
                f"chunk={result['metadata']['chunk_index']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
