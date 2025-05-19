from typing import Dict, Any

class ComparisonAgent():
    def __init__(self, streamlit_obj):
        self.st = streamlit_obj
        self.prompt = """
        Answer the question based on the given source and its context. Do not hallucinate or make up information.
        If there is not enough information to answer the query based on the context, say so and explain what information you are missing.s[j]
        """
        pass
        

    def run(self, sources: Dict[str, Any], query) -> None:
        """
        Provides a response to each given source and makes comparisons
        """ 
        for source, chunk in sources.items():
            self.st.write("Responding for " + source)
            full_query = f"""
            Detailed Information: {chunk[0].page_content}
            Question: {query}
            """
            self.st.write(full_query)
        return  
        


