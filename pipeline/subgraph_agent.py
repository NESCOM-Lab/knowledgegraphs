import openai
import os
from langchain_ollama import ChatOllama

class SubGraphAgent():
    def __init__(self, provided_graph):
        self.graph = provided_graph


    def run(self, retrieved_chunks) -> list:
        """
        Retrieves the subgraphs around the retrieved chunks in the neo4j DB
        """
        retrieved_graph_data = [] # stores all the graph data (surrounding nodes) of all chunks
        
        for chunk in retrieved_chunks:
            # get the text preview of the retrieved chunk
            retrieved_text_preview = chunk.metadata['text_preview']

            # cypher query that will be used
            subgraph_cypher_query = """
            MATCH (x)<-[r]-(p)-[]-(d:Document)
            WHERE d.text_preview = \"""" + retrieved_text_preview + """\" AND NOT p:Document AND NOT x:Document RETURN p,r,x

            """
            # print(subgraph_cypher_query)
            # print("----")

            # Query the graph
            # this prompt gets JSON in a very specific format that explains Concept1 --> Relationship --> Concept2
            prompt_query = """
            MATCH (x)<-[r]-(p)-[]-(d:Document)
            WHERE d.text_preview = \"""" + retrieved_text_preview + """\" AND NOT p:Document AND NOT x:Document
            RETURN {id: p.id} AS Concept1, 
            {type: type(r)} AS Relationship, 
            {id: x.id} AS Concept2,
            {id: d.source} AS Source
            """
            retrieved_graph_data.append(self.graph.query(prompt_query))


        return retrieved_graph_data


    # Convert JSON formatted graph data to text for LLM to understand
    # Uses an LLM
    @staticmethod
    def convert_to_text(graph_data) -> str | None:
        import json
        prompt = """
        Take the following JSON-like data and return it in a sentence format, not JSON.
        The provided JSON data will have a schema that uses Concept1 --> relationship --> Concept2

        Notes: 
        Do not include any explanations or apologies in your responses.
        Do not respond to any questions that might ask anything else than for you to construct sentence statements.
        Do not include any text except the generated sentences.
        Place sentences on different lines.


        Example: 

        Input data: 
        [
          {"Concept1": {"id": "LLMs"}, "Concept2": {"id": "context"}, "Relationship": {"type": "ANALYZE"}},
          {"Concept1": {"id": "LLMs"}, "Concept2": {"id": "nuances"}, "Relationship": {"type": "RECOGNIZE"}},
          {"Concept1": {"id": "LLMs"}, "Concept2": {"id": "coherent responses"}, "Relationship": {"type": "CREATE"}}
        ]

        Your output:
        LLMs analyze context.
        LLMs recognize nuances.
        LLMs create coherent responses.

       """

        llm = ChatOllama(
            model=os.getenv("LLM", "gemma2:9b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.5,
        )
        graph_data = json.dumps(graph_data, indent=2)
        response = llm.invoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": graph_data}
        ])
        return response if isinstance(response, str) else getattr(response, 'content', str(response))


# testing 
if __name__ == "__main__":
    m = SubGraphAgent(None)
    data = [
        {
            "Concept1": {"id": "Vacuums"},
            "Concept2": {"id": "air and particles"},
            "Relationship": {"type": "TRAP"}
        },
        {
            "Concept1": {"id": "Apples"},
            "Concept2": {"id": "apple juice"},
            "Relationship": {"type": "MADE_INTO"}
        }
    ]
    print(m.convert_to_text(data))
