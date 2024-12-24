Right now graphrag2.ipynb queries for nodes using a prompt generation template
and the graph schema 

Future alternative: Use similarity search with user query on all nodes in the graph, find the most relevant 
top 2 nodes by their embeddings, and get all the connections to these nodes using a hard-coded
    cypher_query = f"MATCH (n {{id: '{node}'}}) RETURN n"