from retriever_utils import add_similarity_scores

class QueryAgent():
    def __init__(self, vect_retriever, embed_model):
        self.vector_retriever = vect_retriever
        self.embed_model = embed_model


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

        
        
        
        for doc in results:
            print(doc.metadata['source'])




