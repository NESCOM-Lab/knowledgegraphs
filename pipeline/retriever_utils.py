import numpy as np

# add similarity scores to docs 
# https://python.langchain.com/docs/how_to/add_scores_retriever/


def add_similarity_scores(docs, query, embed_model):
    query_embedding = embed_model.embed_query(query)
    query_norm = np.linalg.norm(query_embedding)


    for doc in docs:
        doc_embedding = embed_model.embed_query(doc.page_content)
        if doc_embedding:
            doc.metadata["score"] = np.dot(query_embedding, doc_embedding) / (query_norm * np.linalg.norm(doc_embedding))
        else:
            doc.metadata["score"] = None

    return docs
