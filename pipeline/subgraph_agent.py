import openai

class SubGraphAgent():
    def __init__(self, provided_graph):
        self.graph = provided_graph


    def run(self, retrieved_chunks):
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
    def convert_to_text(graph_data):
        graph_data = [ (x['Concept1']['id'], x['Concept2']['id'], {'relationship': x['Relationship']['type']}) for x in graph_data ]
        graph_data = "".join(str(x) for x in graph_data) # convert to string

        prompt = """
        Take the following JSON-like data and convert it to a sentence format.
        The provided JSON data will have a schema that uses Concept1 --> relationship --> Concept2

        Example: 
        Input data: 
        1:[
            0:"LLMs"
            1:"context"
            2:"{
            "relationship":"ANALYZE"
            }"
        ]
        2:[
            0:"LLMs"
            1:"nuances"
            2:"{
            "relationship":"RECOGNIZE"
            }"
        ]
        3:[
            0:"LLMs"
            1:"coherent responses"
            2:"{
            "relationship":"CREATE"
            }"
        ]
        Output:
        LLMs analyze context.
        LLMs recognize nuances.
        LLMs create coherent responses.


        Note: Do not include any explanations or apologies in your responses.
        Do not respond to any questions that might ask anything else than for you to construct sentence statements.
        Do not include any text except the generated sentence statements.
        """

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": graph_data}
            ]
        )
        return response.choices[0].message.content


# testing 
if __name__ == "__main__":
    m = SubGraphAgent(None)
    data = [("Vacuums", "air and particles",{"relationship":"TRAP"}),
    ("Apples", "apple juice",{"relationship":"MADE_INTO"})]
    print(m.convert_to_text(data))
    print('welcome')

