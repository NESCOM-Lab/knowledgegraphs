

class SubGraphAgent():
    def __init__(self, provided_graph):
        self.graph = provided_graph


    def run(self, retrieved_chunks):
        """
        Retrieves the subgraph around the retrieved chunk in the neo4j DB
        """
        # get the text preview of the retrieved chunk
        retrieved_text_preview = retrieved_chunks[0].metadata['text_preview']

        # cypher query that will be used
        subgraph_cypher_query = """
        MATCH (x)<-[r]-(p)-[]-(d:Document)
        WHERE d.text_preview = \"""" + retrieved_text_preview + """\" AND NOT p:Document AND NOT x:Document RETURN p,r,x

        """
        print(subgraph_cypher_query)
        print("----")

        # Query the graph
        # this prompt gets JSON in a very specific format that explains Concept1 --> Relationship --> Concept2
        prompt_query = """
        MATCH (x)<-[r]-(p)-[]-(d:Document)
        WHERE d.text_preview = \"""" + retrieved_text_preview + """\" AND NOT p:Document AND NOT x:Document
        RETURN {id: p.id} AS Concept1, 
        {type: type(r)} AS Relationship, 
        {id: x.id} AS Concept2
        """
        print(self.graph.query(prompt_query))


