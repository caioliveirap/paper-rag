import json
import logging
import os

import openai
import requests
from rank_bm25 import BM25Okapi

from .env import load_env
from .ingest import get_chroma_collection
from .models import RetrievedChunk

load_env()

logger = logging.getLogger(__name__)

_openai_client = None
_bm25_index: BM25Okapi | None = None
_bm25_chunks: list[RetrievedChunk] = []
_bm25_collection_count: int = -1


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _openai_client


def embed_query(query: str) -> list[float]:
    client = _get_openai_client()
    model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    response = client.embeddings.create(model=model, input=[query])
    return response.data[0].embedding


def _load_bm25_index() -> tuple[BM25Okapi | None, list[RetrievedChunk]]:
    global _bm25_index, _bm25_chunks, _bm25_collection_count

    collection = get_chroma_collection()
    count = collection.count()
    logger.info("BM25 index check: collection count = %d, cached count = %d", count, _bm25_collection_count)

    if count == 0:
        return None, []

    if count == _bm25_collection_count:
        return _bm25_index, _bm25_chunks

    logger.info("Rebuilding BM25 index for %d documents.", count)
    result = collection.get(include=["documents", "metadatas"])
    docs = result["documents"]
    metadatas = result["metadatas"]
    ids = result["ids"]

    chunks = []
    for doc, meta, chunk_id in zip(docs, metadatas, ids):
        page_numbers = json.loads(meta.get("page_numbers", "[]"))
        chunk = RetrievedChunk(
            chunk_id=chunk_id,
            paper_id=meta["paper_id"],
            text=doc,
            section=meta["section"],
            page_numbers=page_numbers,
            token_count=meta["token_count"],
            chunk_index=meta["chunk_index"],
            relevance_score=0.0,
        )
        chunks.append(chunk)

    tokenized = [doc.lower().split() for doc in docs]
    index = BM25Okapi(tokenized)

    _bm25_index = index
    _bm25_chunks = chunks
    _bm25_collection_count = count

    return _bm25_index, _bm25_chunks


def vector_search(
    query_embedding: list[float],
    top_k: int,
    paper_filter: str | None = None,
) -> list[RetrievedChunk]:
    collection = get_chroma_collection()

    if collection.count() == 0:
        return []

    where = {"paper_id": paper_filter} if paper_filter else None
    kwargs = {"query_embeddings": [query_embedding], "n_results": top_k, "include": ["documents", "metadatas", "distances"]}
    if where:
        kwargs["where"] = where

    result = collection.query(**kwargs)

    chunks = []
    for doc, meta, distance, chunk_id in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
        result["ids"][0],
    ):
        page_numbers = json.loads(meta.get("page_numbers", "[]"))
        # Convert L2 distance to similarity score
        score = 1.0 / (1.0 + distance)
        chunk = RetrievedChunk(
            chunk_id=chunk_id,
            paper_id=meta["paper_id"],
            text=doc,
            section=meta["section"],
            page_numbers=page_numbers,
            token_count=meta["token_count"],
            chunk_index=meta["chunk_index"],
            relevance_score=score,
        )
        chunks.append(chunk)

    logger.info("Vector search returned %d results.", len(chunks))
    return chunks


def bm25_search(
    query: str,
    top_k: int,
    paper_filter: str | None = None,
) -> list[RetrievedChunk]:
    index, chunks = _load_bm25_index()

    if index is None:
        return []

    tokenized_query = query.lower().split()
    scores = index.get_scores(tokenized_query)

    scored_chunks = list(zip(scores, chunks))

    if paper_filter:
        scored_chunks = [(s, c) for s, c in scored_chunks if c.paper_id == paper_filter]

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    top = scored_chunks[:top_k]

    results = [
        c.model_copy(update={"relevance_score": float(s)}) for s, c in top
    ]
    logger.info("BM25 search returned %d results.", len(results))
    return results


def reciprocal_rank_fusion(
    vector_results: list[RetrievedChunk],
    bm25_results: list[RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(vector_results):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank + 1)
        chunk_map[chunk.chunk_id] = chunk

    for rank, chunk in enumerate(bm25_results):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank + 1)
        chunk_map[chunk.chunk_id] = chunk

    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
    return [
        chunk_map[cid].model_copy(update={"relevance_score": rrf_scores[cid]})
        for cid in sorted_ids
    ]


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    jina_api_key = os.environ.get("JINA_API_KEY", "")
    logger.info("Calling Jina reranker for %d candidates, top_k=%d.", len(chunks), top_k)

    payload = {
        "model": "jina-reranker-v2-base-multilingual",
        "query": query,
        "documents": [c.text for c in chunks],
        "top_n": top_k,
    }
    headers = {
        "Authorization": f"Bearer {jina_api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post("https://api.jina.ai/v1/rerank", json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data["results"]:
        original_chunk = chunks[item["index"]]
        reranked = original_chunk.model_copy(update={"relevance_score": item["relevance_score"]})
        results.append(reranked)

    results.sort(key=lambda c: c.relevance_score, reverse=True)
    return results


def retrieve(
    query: str,
    top_k: int = 5,
    paper_filter: str | None = None,
) -> list[RetrievedChunk]:
    logger.info("Retrieving for query: %r, top_k=%d, paper_filter=%r", query, top_k, paper_filter)

    top_k_candidates = int(os.environ.get("TOP_K_CANDIDATES", "20"))

    query_embedding = embed_query(query)
    vector_results = vector_search(query_embedding, top_k_candidates, paper_filter)
    bm25_results = bm25_search(query, top_k_candidates, paper_filter)

    if not vector_results and not bm25_results:
        return []

    fused = reciprocal_rank_fusion(vector_results, bm25_results)
    return rerank(query, fused, top_k)


def format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for chunk in chunks:
        header = f"[Source: {chunk.paper_id} | Section: {chunk.section} | Score: {chunk.relevance_score:.3f}]"
        parts.append(f"{header}\n{chunk.text}")
    return "\n\n---\n\n".join(parts)
