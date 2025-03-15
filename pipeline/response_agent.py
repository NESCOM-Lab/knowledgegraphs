"""
Takes in context from retrieved chunk & retrieved graph information and outputs
response
"""

class LLMAgent():
    def __init__(self, llm):
        


    def run(self, query):
        """
        Answer the question based on the context below. Keep the answer short and concise. Respond "Unsure about answer" if not sure about the answer.
        Context: Teplizumab traces its roots to a New Jersey drug company called Ortho Pharmaceutical. There, scientists generated an early version of the antibody, dubbed OKT3. Originally sourced from mice, the molecule was able to bind to the surface of T cells and limit their cell-killing potential. In 1986, it was approved to help prevent organ rejection after kidney transplants, making it the first therapeutic antibody allowed for human use.
        Question: What was OKT3 originally sourced from?
        Answer:
        """
        
        # self.vector_retriever.search_kwargs = {"k": 1} # change k value later (depends on length of context)
        # results = self.vector_retriever.invoke(query)

        # # print(len(results))
        # print("Retrieved chunk: ")
        # for doc in results:
        #     # print(doc.id)
        #     print("\tsource: " + doc.metadata['source'])
        #     print("\tpage number: " + str(doc.metadata['page_number']))
        #     print("\tpreview: " + doc.metadata['text_preview'])
        # return results


