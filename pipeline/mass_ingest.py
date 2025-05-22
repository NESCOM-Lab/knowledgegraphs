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
        
    
    pdf_dir = "PapersNeuro2" ############# Change to directory with files
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in PapersNeuro/")
        exit()

    # ask user if they want to ingest
    while True:
        try:
            ans = int(input("Would you like to ingest all documents in PapersNeuro/? Enter 1 for yes, 0 for no: "))
            if ans in [0, 1]:
                break
        except ValueError:
            continue

    if ans == 1:
        sources_file = "all_sources.txt"

        for file_name in pdf_files:
            pdf_path = os.path.join(pdf_dir, file_name)

            # Write source name to sources file
            with open(sources_file, "a") as file:
                file.write(file_name + "\n")

            print(f"Chunking document: {file_name}")
            try:
                processed_chunks = chunk_document(pdf_path)
            except Exception as e:
                print(f"Error during chunking {file_name}: {e}")
                continue

            print(f"Ingesting document: {file_name}")
            try:
                await ingest_document(processed_chunks, embed, llm_transformer, graph)
            except Exception as e:
                print(f"Error occurred while ingesting document {pdf_path}: {e}")


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