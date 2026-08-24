"""HR policy RAG pipeline using Docling, HuggingFace embeddings, Qdrant, and Groq.

The script loads a PDF from a URL, splits it into chunks, embeds those chunks,
stores them in an in-memory Qdrant collection, retrieves the most similar
chunks for a question, and asks Groq to answer using only that context.
"""

from __future__ import annotations

from typing import Any

from dotenv import load_dotenv
from groq import Groq
from langchain_docling import DoclingLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-20b"
"""Fallback Groq chat model when ``GROQ_MODEL`` is not set."""

SYSTEM_PROMPT = """You are a helpful HR assistant.
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information, say so — do not make things up.
Always cite the section name when referencing specific information."""
"""System instructions sent with every Groq chat completion."""


class GroqSettings(BaseSettings):
    """Groq API credentials and model name loaded from the environment.

    Values are read from a ``.env`` file or process environment variables.
    ``GROQ_API_KEY`` is required; ``GROQ_MODEL`` defaults to ``DEFAULT_MODEL``.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: SecretStr = Field(..., description="Groq API key used to authenticate chat requests.")
    groq_model: str = Field(
        default=DEFAULT_MODEL,
        description="Groq chat model identifier, for example openai/gpt-oss-20b.",
    )


class IngestConfig(BaseModel):
    """Configuration for document loading, chunking, embedding, and storage."""

    source_url: HttpUrl = Field(..., description="HTTPS URL of the PDF to ingest.")
    chunk_size: int = Field(default=1000, ge=1, description="Maximum characters per text chunk.")
    chunk_overlap: int = Field(default=200, ge=0, description="Overlap in characters between consecutive chunks.")
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformer model name used by HuggingFaceEmbeddings.",
    )
    qdrant_location: str = Field(default=":memory:", description="Qdrant location, e.g. :memory: or a server URL.")
    collection_name: str = Field(default="atliqai_hr_policies", description="Qdrant collection that stores the vectors.")


class RetrievedChunk(BaseModel):
    """A single document chunk returned from similarity search."""

    score: float = Field(..., description="Similarity score from the vector store; higher is more similar.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Loader metadata attached to the chunk.")
    page_content: str = Field(..., description="Text content of the retrieved chunk.")


class RAGQuery(BaseModel):
    """User question and generation settings for the RAG step."""

    query: str = Field(..., min_length=1, description="Natural-language question to answer from the document.")
    top_k: int = Field(default=3, ge=1, le=20, description="Number of nearest chunks to retrieve.")
    system_prompt: str = Field(default=SYSTEM_PROMPT, description="System prompt given to the chat model.")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="Sampling temperature for Groq generation.")


class RAGResult(BaseModel):
    """Answer produced by the RAG pipeline plus the context that grounded it."""

    answer: str = Field(..., description="Model-generated answer, or a fallback message if nothing was retrieved.")
    context: str = Field(default="", description="Formatted source block sent to the model.")
    chunks: list[RetrievedChunk] = Field(default_factory=list, description="Raw retrieved chunks used as context.")


class VectorStorePreview(BaseModel):
    """Summary of a Qdrant collection used when printing store diagnostics."""

    points_count: int = Field(..., description="Number of vectors currently stored in the collection.")
    point_id: str | int | None = Field(default=None, description="ID of the first scrolled point, if any.")
    payload: dict[str, Any] | None = Field(default=None, description="Payload of the first point.")
    vector_preview: list[float] = Field(default_factory=list, description="First few embedding values of the first point.")
    vector_length: int = Field(default=0, description="Full embedding dimensionality of the first point.")


def load_document(url: str | HttpUrl) -> list:
    """Load a PDF from a remote URL using Docling.

    Args:
        url: Location of the PDF. Accepts a string or a Pydantic ``HttpUrl``.

    Returns:
        A list of LangChain ``Document`` objects produced by ``DoclingLoader``.
    """
    loader = DoclingLoader(str(url))
    return loader.load()


def split_documents(docs: list, config: IngestConfig) -> list:
    """Split loaded documents into overlapping character chunks.

    Args:
        docs: Documents returned by :func:`load_document`.
        config: Ingest settings that supply ``chunk_size`` and ``chunk_overlap``.

    Returns:
        Chunked LangChain ``Document`` objects ready for embedding.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    return text_splitter.split_documents(docs)


def build_embeddings(config: IngestConfig) -> HuggingFaceEmbeddings:
    """Create a HuggingFace embedding model from ingest configuration.

    Args:
        config: Ingest settings that supply ``embedding_model``.

    Returns:
        An embeddings instance used both to index chunks and to embed queries.
    """
    return HuggingFaceEmbeddings(model_name=config.embedding_model)


def store_document(
    chunks: list,
    embeddings: HuggingFaceEmbeddings,
    config: IngestConfig,
) -> QdrantVectorStore:
    """Index document chunks into a Qdrant vector store.

    Args:
        chunks: Split documents to embed and persist.
        embeddings: Embedding model used to encode each chunk.
        config: Ingest settings that supply Qdrant location and collection name.

    Returns:
        A ``QdrantVectorStore`` populated with the given chunks.
    """
    return QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        location=config.qdrant_location,
        collection_name=config.collection_name,
    )


def query_document(
    vector_store: QdrantVectorStore,
    query: str,
    top_k: int = 3,
) -> list[RetrievedChunk]:
    """Retrieve the top-k most similar chunks for a query.

    Args:
        vector_store: Qdrant-backed store containing the ingested document.
        query: Natural-language search string.
        top_k: Maximum number of chunks to return.

    Returns:
        A list of :class:`RetrievedChunk` models ordered by the vector store.
    """
    results_with_scores = vector_store.similarity_search_with_score(query, k=top_k)
    return [
        RetrievedChunk(
            score=score,
            metadata=doc.metadata,
            page_content=doc.page_content,
        )
        for doc, score in results_with_scores
    ]


def build_context(retrieved_chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a single context string for the LLM.

    Args:
        retrieved_chunks: Chunks returned by :func:`query_document`.

    Returns:
        A string with numbered source blocks separated by horizontal rules.
        Returns an empty string when no chunks are provided.
    """
    parts = [f"[Source {i}]\n{chunk.page_content}" for i, chunk in enumerate(retrieved_chunks, 1)]
    return "\n\n---\n\n".join(parts)


def resolve_credentials(
    api_key: str | None = None,
    model_name: str | None = None,
) -> GroqSettings:
    """Resolve Groq credentials from explicit arguments or the environment.

    Args:
        api_key: Optional API key that overrides ``GROQ_API_KEY``.
        model_name: Optional model name that overrides ``GROQ_MODEL``.

    Returns:
        A :class:`GroqSettings` instance with the effective key and model.

    Raises:
        ValidationError: If no API key is provided and ``GROQ_API_KEY`` is unset.
    """
    overrides: dict[str, str] = {}
    if api_key:
        overrides["groq_api_key"] = api_key
    if model_name:
        overrides["groq_model"] = model_name
    return GroqSettings(**overrides)


def rag(
    vector_store: QdrantVectorStore,
    rag_query: RAGQuery,
    settings: GroqSettings | None = None,
) -> RAGResult:
    """Run retrieve-then-generate: search Qdrant, then answer with Groq.

    Args:
        vector_store: Store containing the ingested document embeddings.
        rag_query: Question, retrieval depth, system prompt, and temperature.
        settings: Optional Groq credentials. Loaded from the environment when omitted.

    Returns:
        A :class:`RAGResult` with the model answer, formatted context, and raw chunks.
        If retrieval returns nothing, ``answer`` explains that no content was found
        and generation is skipped.
    """
    chunks = query_document(vector_store, rag_query.query, top_k=rag_query.top_k)
    if not chunks:
        return RAGResult(answer="No relevant content found in the document.")

    context = build_context(chunks)
    user_message = f"Context:\n{context}\n\nQuestion: {rag_query.query}"

    groq_settings = settings or resolve_credentials()
    groq_client = Groq(api_key=groq_settings.groq_api_key.get_secret_value())

    response = groq_client.chat.completions.create(
        model=groq_settings.groq_model,
        messages=[
            {"role": "system", "content": rag_query.system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=rag_query.temperature,
    )
    answer = response.choices[0].message.content or ""
    return RAGResult(answer=answer, context=context, chunks=chunks)


def _vector_preview(vector: Any, preview_size: int = 5) -> tuple[list[float], int]:
    """Extract a short preview and length from a Qdrant vector payload.

    Args:
        vector: Dense list, or a named-vector mapping from Qdrant.
        preview_size: Number of leading values to keep in the preview.

    Returns:
        A pair of ``(preview_values, full_length)``. Empty when the vector is missing.
    """
    if vector is None:
        return [], 0
    if isinstance(vector, dict):
        vector = next(iter(vector.values()), [])
    values = list(vector)
    return values[:preview_size], len(values)


def get_vector_store_preview(
    vector_store: QdrantVectorStore,
    collection_name: str,
) -> VectorStorePreview:
    """Read collection size and the first stored point for diagnostics.

    Args:
        vector_store: Qdrant store whose client will be queried.
        collection_name: Name of the collection to inspect.

    Returns:
        A :class:`VectorStorePreview` with point count and an optional first-point sample.
    """
    collection_info = vector_store.client.get_collection(collection_name=collection_name)
    records, _next_page_offset = vector_store.client.scroll(
        collection_name=collection_name,
        limit=1,
        with_payload=True,
        with_vectors=True,
    )
    if not records:
        return VectorStorePreview(points_count=collection_info.points_count)

    first_point = records[0]
    preview, length = _vector_preview(first_point.vector)
    return VectorStorePreview(
        points_count=collection_info.points_count,
        point_id=first_point.id,
        payload=first_point.payload,
        vector_preview=preview,
        vector_length=length,
    )


def print_vector_store_details(vector_store: QdrantVectorStore, collection_name: str) -> None:
    """Print collection size and a truncated view of the first stored vector.

    Args:
        vector_store: Qdrant store to inspect.
        collection_name: Name of the collection to inspect.
    """
    preview = get_vector_store_preview(vector_store, collection_name)
    print(f"Number of vectors in the vector store: {preview.points_count}")
    if preview.point_id is None:
        return
    print("--- FIRST POINT DETAILS ---")
    print(f"Point ID: {preview.point_id}")
    print(f"Metadata Payload (Text): {preview.payload}")
    print(
        f"Raw Embedding Vector (Preview): {preview.vector_preview}... "
        f"[Total Length: {preview.vector_length}]"
    )


def ingest(config: IngestConfig) -> QdrantVectorStore:
    """Load, split, embed, and store a document according to ``config``.

    Args:
        config: Source URL, chunking, embedding, and Qdrant settings.

    Returns:
        A populated :class:`QdrantVectorStore` ready for retrieval.
    """
    docs = load_document(config.source_url)
    chunks = split_documents(docs, config)
    embeddings = build_embeddings(config)
    return store_document(chunks, embeddings, config)


def main() -> None:
    """Ingest the sample HR policy PDF and answer a demo question.

    Environment variables ``GROQ_API_KEY`` and optional ``GROQ_MODEL`` must be
    available (typically via a ``.env`` file). The vector store is created in
    memory and discarded when the process exits.
    """
    config = IngestConfig(
        source_url="https://raw.githubusercontent.com/tnahddisttud/sample-doc/refs/heads/main/AtliqAI_HR_Policies.pdf",
    )
    vector_store = ingest(config)

    rag_query = RAGQuery(query="What is the company's policy on remote work?", top_k=3)
    result = rag(vector_store, rag_query)

    print(f"Answer: {result.answer}")
    print(f"{'=' * 250}")
    print(f"\n\nSOURCES:\n {result.context}")


if __name__ == "__main__":
    main()
