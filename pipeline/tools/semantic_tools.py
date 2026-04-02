"""
Semantic tools - embedding-based search over the graph DB.
"""
from typing import Any


def vector_search(vector_retriever, embed_model, query: str, node_type: str = "Document", top_k: int = 5) -> list[dict]:
    """
    Embedding search over the DB. Finds semantically similar Document chunks.

    args:
        query: natural language search query
        node_type: currently "Document" (other types not yet indexed)
        top_k: number of results to return

    returns:
        list of dicts with text, source, page_number, similarity_score
    """
    from retriever_utils import add_similarity_scores

    try:
        vector_retriever.search_kwargs = {"k": top_k}
        results = vector_retriever.invoke(query)
        results = add_similarity_scores(results, query, embed_model)

        return [
            {
                "text": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page_number": doc.metadata.get("page_number"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "text_preview": doc.metadata.get("text_preview", ""),
                "similarity_score": round(float(doc.metadata.get("score", 0)), 4),
            }
            for doc in results
        ]
    except Exception as e:
        return [{"error": str(e)}]
