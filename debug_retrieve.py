import logging
logging.disable(logging.CRITICAL)

from app.ingest import get_chroma_collection
from app.retrieve import vector_search, bm25_search, reciprocal_rank_fusion, rerank, embed_query

QUERY = "What is the optimal infection time?"
TOP_K = 5

collection = get_chroma_collection()
total_count = collection.count()

result = collection.get(include=["metadatas"])
figure_count = sum(
    1 for meta in result["metadatas"] if "Figure" in meta.get("section", "")
)

print(f"Total chunks in collection: {total_count}")
print(f"Figure chunks: {figure_count}")
print()

query_embedding = embed_query(QUERY)

print("=== Vector Search (top 5) ===")
vector_results = vector_search(query_embedding, top_k=TOP_K)
for chunk in vector_results[:TOP_K]:
    print(f"  paper_id={chunk.paper_id} | section={chunk.section} | score={chunk.relevance_score:.4f}")
    print(f"  {chunk.text[:200]!r}")
    print()

print("=== BM25 Search (top 5) ===")
bm25_results = bm25_search(QUERY, top_k=TOP_K)
for chunk in bm25_results[:TOP_K]:
    print(f"  paper_id={chunk.paper_id} | section={chunk.section} | score={chunk.relevance_score:.4f}")
    print(f"  {chunk.text[:200]!r}")
    print()

print("=== RRF Fusion (top 5) ===")
fused = reciprocal_rank_fusion(vector_results, bm25_results)
for chunk in fused[:TOP_K]:
    print(f"  paper_id={chunk.paper_id} | section={chunk.section} | score={chunk.relevance_score:.4f}")
    print(f"  {chunk.text[:200]!r}")
    print()

print("=== Reranked (top 5) ===")
reranked = rerank(QUERY, fused, top_k=TOP_K)
for chunk in reranked[:TOP_K]:
    print(f"  paper_id={chunk.paper_id} | section={chunk.section} | score={chunk.relevance_score:.4f}")
    print(f"  {chunk.text[:200]!r}")
    print()
