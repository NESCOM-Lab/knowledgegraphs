"""
Takes in context from retrieved chunk & retrieved graph information and outputs
response
"""

class LLMAgent():
    def __init__(self, llm):
        pass
        


    def run(self, query):
        prompt = f"""
        Answer the question based on the context below. Keep the answer short and concise. 
        Respond "Unsure about answer" if not sure about the answer.

        Context: 
        Question: {query}
        Answer:
        """
        