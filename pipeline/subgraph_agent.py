import openai
import os
from langchain_ollama import ChatOllama

class SubGraphAgent():
    def __init__(self, provided_graph):
        self.graph = provided_graph


    def run(self, retrieved_chunks) -> list:
        """
        Retrieves the subgraphs around the retrieved chunks in the neo4j DB
        Returns a list of graph data (one per chunk) for backward compatibility
        """
        retrieved_graph_data = [] # stores all the graph data (surrounding nodes) of all chunks
        
        for chunk in retrieved_chunks:
            # get the text preview of the retrieved chunk
            retrieved_text_preview = chunk.metadata['text_preview']

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

    def run_aggregated(self, retrieved_chunks, max_depth=2) -> dict:
        """
        Response that aggregates graph connections across all chunks
        and explores deeper relationships for better context.

        args:
            retrieved_chunks: List of document chunks from vector search
            max_depth: How many hops to explore (default is 2)

        output:
            dict with:
                - 'relationships': list of unique relationships
                - 'chunks': list of chunk texts
                - 'sources': set of source documents
        """
        all_relationships = []
        seen_relationships = set()  # remove duplicates
        chunk_texts = []
        sources = set()

        for chunk in retrieved_chunks:
            retrieved_text_preview = chunk.metadata['text_preview']
            chunk_texts.append(chunk.page_content)
            sources.add(chunk.metadata.get('source', 'unknown'))

            # Explore up to max_depth hops
            if max_depth == 1:
                cypher_query = """
                MATCH (x)<-[r]-(p)-[]-(d:Document)
                WHERE d.text_preview = $text_preview
                AND NOT p:Document AND NOT x:Document
                RETURN DISTINCT
                    p.id AS Concept1,
                    type(r) AS Relationship,
                    x.id AS Concept2,
                    d.source AS Source
                """
            else:
                # 2-hop query - explore deeper connections
                # Document -> Entity1 -> Entity2 -> Entity3
                cypher_query = """
                MATCH path = (d:Document)-[]-(e1)-[r1]-(e2)
                WHERE d.text_preview = $text_preview
                AND NOT e1:Document AND NOT e2:Document
                WITH DISTINCT e1, r1, e2, d.source AS source

                OPTIONAL MATCH (e2)-[r2]-(e3)
                WHERE NOT e3:Document AND e3.id <> e1.id

                WITH e1, r1, e2, r2, e3, source
                RETURN DISTINCT
                    e1.id AS Concept1,
                    type(r1) AS Relationship,
                    e2.id AS Concept2,
                    source AS Source

                UNION

                MATCH path = (d:Document)-[]-(e1)-[r1]-(e2)-[r2]-(e3)
                WHERE d.text_preview = $text_preview
                AND NOT e1:Document AND NOT e2:Document AND NOT e3:Document
                WITH DISTINCT e2, r2, e3, d.source AS source
                RETURN DISTINCT
                    e2.id AS Concept1,
                    type(r2) AS Relationship,
                    e3.id AS Concept2,
                    source AS Source
                LIMIT 100
                """

            # Execute query
            try:
                results = self.graph.query(cypher_query, params={"text_preview": retrieved_text_preview})

                # remove duplicate relationships
                for rel in results:
                    # Create unique key for this relationship
                    rel_key = (rel['Concept1'], rel['Relationship'], rel['Concept2'])

                    if rel_key not in seen_relationships:
                        seen_relationships.add(rel_key)
                        all_relationships.append({
                            'Concept1': {'id': rel['Concept1']},
                            'Relationship': {'type': rel['Relationship']},
                            'Concept2': {'id': rel['Concept2']},
                            'Source': {'id': rel['Source']}
                        })
            except Exception as e:
                print(f"Error querying subgraph: {e}")
                continue

        return {
            'relationships': all_relationships,
            'chunks': chunk_texts,
            'sources': sources,
            'num_relationships': len(all_relationships)
        }


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
