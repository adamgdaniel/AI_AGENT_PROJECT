import io
import os
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import numpy as np
from google import genai
from google.genai import types
from google.cloud import storage
from pypdf import PdfReader


@dataclass
class ChunkRecord:
    doc_name: str
    chunk_id: int
    text: str
    page_start: int | None = None
    page_end: int | None = None
    score: float | None = None


@dataclass
class RagIndex:
    loaded: bool = False
    bucket_name: str | None = None
    prefix: str | None = None
    docs: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[ChunkRecord] = field(default_factory=list)
    vectors: np.ndarray | None = None
    last_error: str | None = None


_INDEX = RagIndex()
_LOCK = Lock()


def _get_genai_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Falta GEMINI_API_KEY.")
    return genai.Client(api_key=api_key)


def _get_storage_client() -> storage.Client:
    return storage.Client()


def _extract_pdf_pages(pdf_bytes: bytes) -> list[dict[str, Any]]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": idx, "text": text})
    return pages


def _chunk_pages(
    pages: list[dict[str, Any]],
    doc_name: str,
    chunk_size: int,
    overlap: int,
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    if not pages:
        return chunks

    joined = []
    page_map = []
    cursor = 0
    for item in pages:
        page_header = f"\n[PAGE {item['page']}]\n"
        piece = page_header + item["text"]
        joined.append(piece)
        start = cursor
        cursor += len(piece)
        end = cursor
        page_map.append((start, end, item["page"]))
    full_text = "".join(joined)

    start = 0
    chunk_id = 1
    step = max(1, chunk_size - overlap)
    while start < len(full_text):
        end = min(len(full_text), start + chunk_size)
        text = full_text[start:end].strip()
        if text:
            covered_pages = [p for s, e, p in page_map if not (e <= start or s >= end)]
            page_start = min(covered_pages) if covered_pages else None
            page_end = max(covered_pages) if covered_pages else None
            chunks.append(
                ChunkRecord(
                    doc_name=doc_name,
                    chunk_id=chunk_id,
                    text=text,
                    page_start=page_start,
                    page_end=page_end,
                )
            )
            chunk_id += 1
        if end >= len(full_text):
            break
        start += step
    return chunks


def _embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    client = _get_genai_client()
    model_name = os.getenv("RAG_EMBEDDING_MODEL", "gemini-embedding-001")
    output_dim = int(os.getenv("RAG_EMBEDDING_DIM", "768"))
    batch_size = int(os.getenv("RAG_EMBED_BATCH_SIZE", "16"))

    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = client.models.embed_content(
            model=model_name,
            contents=batch,
            config=types.EmbedContentConfig(output_dimensionality=output_dim),
        )
        vectors.extend([emb.values for emb in result.embeddings])

    array = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def build_rag_index_from_gcs() -> dict[str, Any]:
    bucket_name = os.getenv("GCS_BUCKET_NAME", "").strip()
    prefix = os.getenv("GCS_PREFIX", "").strip()
    chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "1400"))
    overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "180"))
    max_pdfs = int(os.getenv("RAG_MAX_PDFS", "5"))

    if not bucket_name:
        raise RuntimeError("Falta GCS_BUCKET_NAME.")

    storage_client = _get_storage_client()
    bucket = storage_client.bucket(bucket_name)
    blobs = list(storage_client.list_blobs(bucket, prefix=prefix))
    pdf_blobs = [b for b in blobs if b.name.lower().endswith(".pdf")][:max_pdfs]

    if not pdf_blobs:
        raise RuntimeError("No se han encontrado PDFs en el bucket/prefijo configurado.")

    docs: list[dict[str, Any]] = []
    all_chunks: list[ChunkRecord] = []

    for blob in pdf_blobs:
        pdf_bytes = blob.download_as_bytes()
        pages = _extract_pdf_pages(pdf_bytes)
        chunks = _chunk_pages(
            pages=pages,
            doc_name=blob.name,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        docs.append(
            {
                "name": blob.name,
                "pages_with_text": len(pages),
                "chunks": len(chunks),
                "size_bytes": blob.size,
            }
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        raise RuntimeError("Se han encontrado PDFs, pero no se ha podido extraer texto útil.")

    vectors = _embed_texts([chunk.text for chunk in all_chunks])

    with _LOCK:
        _INDEX.loaded = True
        _INDEX.bucket_name = bucket_name
        _INDEX.prefix = prefix
        _INDEX.docs = docs
        _INDEX.chunks = all_chunks
        _INDEX.vectors = vectors
        _INDEX.last_error = None

    return get_rag_status()


def get_rag_status() -> dict[str, Any]:
    with _LOCK:
        return {
            "loaded": _INDEX.loaded,
            "bucket_name": _INDEX.bucket_name,
            "prefix": _INDEX.prefix,
            "docs": list(_INDEX.docs),
            "num_docs": len(_INDEX.docs),
            "num_chunks": len(_INDEX.chunks),
            "last_error": _INDEX.last_error,
        }


def mark_rag_error(message: str) -> None:
    with _LOCK:
        _INDEX.last_error = message


def retrieve_relevant_chunks(question: str, top_k: int | None = None) -> list[ChunkRecord]:
    with _LOCK:
        if not _INDEX.loaded or _INDEX.vectors is None or len(_INDEX.chunks) == 0:
            return []
        chunks = list(_INDEX.chunks)
        vectors = _INDEX.vectors.copy()

    client = _get_genai_client()
    model_name = os.getenv("RAG_EMBEDDING_MODEL", "gemini-embedding-001")
    output_dim = int(os.getenv("RAG_EMBEDDING_DIM", "768"))
    top_k = top_k or int(os.getenv("RAG_TOP_K", "4"))

    result = client.models.embed_content(
        model=model_name,
        contents=question,
        config=types.EmbedContentConfig(output_dimensionality=output_dim),
    )
    query_vec = np.array(result.embeddings[0].values, dtype=np.float32)
    norm = np.linalg.norm(query_vec)
    if norm == 0:
        norm = 1.0
    query_vec = query_vec / norm

    scores = vectors @ query_vec
    best_idx = np.argsort(scores)[::-1][:top_k]

    output: list[ChunkRecord] = []
    for idx in best_idx:
        record = chunks[int(idx)]
        output.append(
            ChunkRecord(
                doc_name=record.doc_name,
                chunk_id=record.chunk_id,
                text=record.text,
                page_start=record.page_start,
                page_end=record.page_end,
                score=float(scores[int(idx)]),
            )
        )
    return output


def build_context(chunks: list[ChunkRecord]) -> str:
    sections = []
    for item in chunks:
        page_info = ""
        if item.page_start is not None:
            if item.page_start == item.page_end:
                page_info = f" (página {item.page_start})"
            else:
                page_info = f" (páginas {item.page_start}-{item.page_end})"
        sections.append(
            f"Fuente: {item.doc_name}{page_info}\n"
            f"Chunk {item.chunk_id}\n"
            f"{item.text}"
        )
    return "\n\n---\n\n".join(sections)
