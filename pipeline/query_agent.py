from retriever_utils import add_similarity_scores
from langchain_community.embeddings import OllamaEmbeddings
import os

class QueryAgent():
    def __init__(self, vect_retriever, embed_model):
        self.vector_retriever = vect_retriever
        self.embed_model = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"))


    def run(self, query, k_value) -> list:
        """
        Runs the query agent with the provided query on the vector retriever from neo4j DB
        """
        self.vector_retriever.search_kwargs = {"k": k_value} # change k value later (depends on length of context)
        results = self.vector_retriever.invoke(query)

        # add cosine similarity to results
        results = add_similarity_scores(results, query, self.embed_model)

        # print(len(results))
        # print("Retrieved chunk: ")
        # for doc in results:
        #     # print(doc.id)
        #     print("\tsource: " + doc.metadata['source'])
        #     print("\tpage number: " + str(doc.metadata['page_number']))
        #     print("\tpreview: " + doc.metadata['text_preview'])
        return results
    
    def run_unique(self, query, k_value) -> list:
        """
        Runs the query agent with query grouping by unique sources    
        (Takes longer since it gets more nodes)
        """

        self.vector_retriever.search_kwargs = {"k": k_value}
        try:
            results = self.vector_retriever.invoke(query)
        except Exception as e:
            print("Error running unique retriever: " + str(e))
        
         # add cosine similarity to results
        results = add_similarity_scores(results, query, self.embed_model)

        
        
        sources = {} 
        # store chunks by their source
        # for now only add 1 chunk per source
        for doc in results:
            curr_source = doc.metadata['source']
            if sources.get(curr_source, -1) == -1:
                sources[curr_source] = []
                sources[curr_source].append(doc)
        
        if len(sources) == 0:
            # only 1 source dominating
            print("Only 1 relevant source. Can't compare")
        else:
            print("Multiple sources")

        return results, sources



