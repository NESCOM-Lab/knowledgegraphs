

class QueryAgent():
    def __init__(self, vect_retriever):
        self.vector_retriever = vect_retriever


    def run(self, query):
        """
        Runs the query agent with the provided query on the vector retriever from neo4j DB
        """
        self.vector_retriever.search_kwargs = {"k": 1} # change k value later (depends on length of context)
        results = self.vector_retriever.invoke(query)

        # print(len(results))
        print("Retrieved chunk: ")
        for doc in results:
            # print(doc.id)
            print("\tsource: " + doc.metadata['source'])
            print("\tpage number: " + str(doc.metadata['page_number']))
            print("\tpreview: " + doc.metadata['text_preview'])
        return results


