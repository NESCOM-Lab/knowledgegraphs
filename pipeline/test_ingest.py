import asyncio
import os
from langchain_openai import ChatOpenAI
from pipeline import *
from pipeline import neo4j_setup
from pipeline import load_llm_transformer
from pipeline import chunk_document
from pipeline import ingest_document
from pipeline import clean_graph

async def main():
    # Initialize neo4j
    print("Initializing neo4j")
    graph = neo4j_setup()


    # Initialize openai
    print("Initializing Gemini")
    api_key = os.getenv("OPENAI_API_KEY")
    # llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    llm_transformer, embed, vector_retriever = load_llm_transformer()
    
    print("Would you like to clean db?")
    while True:
        ans = int(input())
        if (ans == 1 or ans == 0):
            break
    if (ans == 1):
        clean_graph(graph)
        
        

    # Chunk the document
    print("Would you like to add a document? Enter 1 or 0")
    while True:
        ans = int(input())
        if (ans == 1 or ans == 0):
            break
    if (ans == 1):
        print("Chunking document")
        try:
            pdf_path = "paper.pdf"
            processed_chunks = chunk_document(pdf_path) 
        except:
            print("Invalid pdf path")


        # Ingest document
        print("Ingesting document")
        try:
            await ingest_document(processed_chunks, embed, llm_transformer, graph)
        except Exception as e:
            print(f"Error occured while ingesting document: {e}")


    # # Querying the system
    # query_agent = QueryAgent(vector_retriever)
    # subgraph_agent = SubGraphAgent(graph)

    # while True:
    #     print("Enter a query: ")
    #     user_query = input()
    #     print("Running query agent")
    #     retrieved_chunks = query_agent.run(user_query)
    #     print("----")
    #     print("Running subgraph agent")
    #     subgraph_agent.run(retrieved_chunks)
    #     print("----")









if __name__ == "__main__":
    asyncio.run(main())
    print("finished program")