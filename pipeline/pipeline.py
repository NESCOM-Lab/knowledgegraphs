

# Neo4j setup
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv
import os
from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
import getpass
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio
from langchain_core.documents import Document

from query_agent import QueryAgent
from subgraph_agent import SubGraphAgent

def clean_graph(graph):
    query = """
    MATCH (n)
    DETACH DELETE n
    """
    graph.query(query)


def neo4j_setup():
    load_dotenv()
    neo_pass = os.getenv("NEO4J_PASS")
    neo_db_id = os.getenv("DB_ID")

    graph = Neo4jGraph(
        url=f"neo4j+s://{neo_db_id}.databases.neo4j.io",
        username="neo4j",
        password=neo_pass,
        enhanced_schema=True
        # refresh_schema=Fa lse
    )
    return graph


def chunk_document(pdf_path):
    # pdf_path = "resume2.pdf"
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    pages = loader.load() # load pages

    # chunk overlap is the shared context window between chunks--allows context to be maintained across chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)

    splits = text_splitter.split_documents(pages) # split the pages using LangChain's text_splitter

    processed_chunks = []

    for i, chunk in enumerate(splits):
        # Process the chunk
        metadata = {
            "chunk_id": i,
            "source": pdf_path,
            "page_number": chunk.metadata.get("page", None),
            "total_length": len(chunk.page_content),
            "text_preview": (
                chunk.page_content[:100] + "..."
                if len(chunk.page_content) > 100
                    else chunk.page_content
            ),


        }
        # Store the metadata for each chunk after processing
        processed_chunks.append({"text": chunk.page_content, "metadata": metadata})

    print(str(len(processed_chunks)) + " chunks processed")
    return processed_chunks

async def ingest_document(processed_chunks, embed, llm_transformer, graph):
    # Convert processed chunks to Langchain Document for Neo4j db
    docs = [
        Document(
            page_content=chunk['text'],
            metadata=chunk['metadata']
        )
        for chunk in processed_chunks
    ]

    # Function to process documents in batches
    async def process_batches(docs, batch_size=2, retry_delay=20, max_retries=5):
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size] # Use subset of documents
            retries = 0
            
            while retries < max_retries:
                try:
                    # generate embeddings for each doc
                    for doc in batch:
                        embedding = embed.embed_query(doc.page_content)
                        doc.metadata["embedding"] = embedding


                    graph_docs = await llm_transformer.aconvert_to_graph_documents(batch)
                    print(f"Processed batch {i // batch_size + 1}:")
                    
                    # Add to Neo4j
                    graph.add_graph_documents(graph_docs, include_source=True, baseEntityLabel=True)
                    print(f"Successfully added batch {i // batch_size + 1} to Neo4j")
                    break  # exit if retry works
                
                except Exception as e:
                    if "rate_limit_exceeded" in str(e):
                        retries += 1
                        print(f"Rate limit hit. Retrying batch {i // batch_size + 1} in {retry_delay} seconds... (Attempt {retries}/{max_retries})")
                        await asyncio.sleep(retry_delay)
                    else:
                        raise  # other error

    # Run batching process
    await process_batches(docs, batch_size=2)

# returns llm_tranformer, embedding model, and vector_retriever
def load_llm_transformer():
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo") # ChatOpenAI version
    additional_instructions = """
    When creating entities, add a "document_id" property to each node and set it to the document's unique ID.
    For example, if the document ID is "doc123", each created node should include document_id: "doc123".
    Query example: 
    CREATE (n:NodeLabel) 
    SET n.document_id = "doc123" 
    RETURN n
    """
    llm_transformer = LLMGraphTransformer(llm=llm, additional_instructions=additional_instructions, ignore_tool_usage=True)

    # Embeddings for later search queries
    embed = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_index = Neo4jVector.from_existing_graph(
        embedding=embed,
        search_type="vector",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )
    vector_retriever = vector_index.as_retriever()

    return llm_transformer, embed, vector_retriever
    

# Query neo4j with agents
def query_neo4j(user_prompt, query_agent, subgraph_agent):
    # Querying the system
    print("Enter a query: ")
    user_query = user_prompt
    print("Running query agent")
    retrieved_chunks = query_agent.run(user_query)
    print("----")
    print("Running subgraph agent")
    retrieved_graph_data = subgraph_agent.run(retrieved_chunks)
    print("----")
    return retrieved_chunks, retrieved_graph_data


async def main():
    # Initialize neo4j
    print("Initializing neo4j")
    graph = neo4j_setup()


    # Initialize openai
    print("Initializing OpenAI")
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    llm_transformer, embed, vector_retriever = load_llm_transformer()
    
    

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