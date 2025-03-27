import time
from pipeline import chunk_document, ingest_document
import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from pyvis.network import Network
import networkx as nx
import tempfile

from pipeline import *
from response_agent import *


# Load llm
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = OpenAI(api_key=OPENAI_API_KEY)


# Streamlit UI

st.set_page_config(page_title="Knowledge Graphs", layout='wide')
st.title("Knowledge Graphs")

# Session state variables

if "loaded_neo4j" not in st.session_state:
    st.session_state.loaded_neo4j = False

if "loaded_agents" not in st.session_state:
    st.session_state.loaded_neo4j = False

if "graph" not in st.session_state:
    st.session_state.graph = []

if "embed" not in st.session_state:
    st.session_state.embed = []

if "llm_transformer" not in st.session_state:
    st.session_state.llm_transformer = []

if "query_agent" not in st.session_state:
    st.session_state.query_agent = None

if "subgraph_agent" not in st.session_state:
    st.session_state.subgraph_agent = None

if "response_agent" not in st.session_state:
    st.session_state.response_agent = None


# chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# when app reruns display chat messages
def display_all_messages():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])



# Load neo4j
def load_neo4j():
    graph = neo4j_setup()
    # graph = []
    st.session_state.loaded_neo4j = True
    return graph


# Load agents
def load_agents(graph, vector_retriever):
    st.session_state.query_agent = QueryAgent(vector_retriever)
    st.session_state.subgraph_agent = SubGraphAgent(graph)
    st.session_state.response_agent = ResponseAgent(st)


# NetworkX graph for graph DB visualization
# https://python-textbook.pythonhumanities.com/06_sna/06_01_05_networkx_pyvis.html
def create_graph(edges):
    # process to format networkx uses
    edges = [ (x['Concept1']['id'], x['Concept2']['id'], {'relationship': x['Relationship']['type']}) for x in edges ]
    G = nx.Graph()
    # edges2 = [
    #     ('A', 'B', {'relationship': 'Edge1'}),
    #     ('B', 'C', {'relationship': 'Edge2'}),
    #     ('B', 'D', {'relationship': 'Edge3'})
    # ]
    G.add_edges_from(edges)
    # G.add_edge("a", "b")
    # G.add_edge("b", "c")
    # G.add_edge("c", "a")
    

    return G

def visualize_graph(G):
    net = Network(height="500px", width="100%", notebook=False)
    net.from_nx(G) # pass in graph

    # add relationship names to edges
    for edge in net.edges:
        u, v = edge["from"], edge["to"]
        edge_label = G[u][v].get("relationship", "") # get rship name
        edge["title"] = edge_label # hover text
        edge["label"] = edge_label # display on edge
        edge["color"] = "black"  # edge color
        # edge["font"] = {"size": 14, "color": "white", "face": "Arial"}  #  styling for edge labels
    
    # save graph as temp html file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        return tmp_file.name

# Main loop
col1, col2 = st.columns(spec=2, vertical_alignment="bottom") # aligns 2 columns together

# Streamlit PDF uploader
with col1:
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    st.write(uploaded_file.name if uploaded_file else "No file uploaded")
    if st.button("Ingest document"):
        if uploaded_file is not None:
            # save the uploaded file temporarily
            temp_pdf_path = uploaded_file.name
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.read())

            # ingest
            async def ingest_and_process():
                start_time=  time.time()
                with st.spinner(text="Ingesting document"):
                    # Chunk document
                    processed_chunks = chunk_document(temp_pdf_path)

                    # Ingest document
                    try:
                        await ingest_document(processed_chunks, 
                                              st.session_state.embed, 
                                              st.session_state.llm_transformer, 
                                              st.session_state.graph)
                    except Exception as e:
                        print(f"Error occurred while ingesting document: {e}")
                    finally:
                        if os.path.exists(temp_pdf_path):
                            os.remove(temp_pdf_path)
                end_time = time.time()
                st.write(f"**Ingested document in {end_time - start_time:.2f} seconds.**")

            # Run the async function
            import asyncio
            asyncio.run(ingest_and_process())

if st.session_state.loaded_neo4j is False:  
    with st.spinner(text="Running load_neo4j()"):
        st.session_state.graph = load_neo4j()
    with st.spinner(text="Loading pipeline"):
        # load the GraphRAG pipeline
        llm_transformer, embed, vector_retriever = load_llm_transformer() # uses a different version (ChatOpenAI)
        st.session_state.llm_transformer = llm_transformer
        st.session_state.embed = embed

        # loads agents into st.query_agent & st.subgraph_agent
        load_agents(st.session_state.graph, vector_retriever) # prob need to switch to session state
        st.session_state.loaded_agents = True
    

if st.session_state.loaded_neo4j and st.session_state.loaded_agents is True:
    if user_prompt := st.chat_input("Query your documents here"):
        
        with col1:
            display_all_messages()
            # display user's message in UI
            with st.chat_message("user"):
                st.markdown(user_prompt)
            
            # add user's message to chat history
            st.session_state.messages.append({"role": "user", "content": user_prompt})

            # display LLM's response
            # with st.chat_message("llm"):
            #     stream = llm.chat.completions.create(
            #         model="gpt-3.5-turbo",
            #         messages=[
            #             {"role": msg["role"], "content": msg["content"]}
            #             for msg in st.session_state.messages
            #         ],
            #         stream=True
            #     )
            #     response = st.write_stream(stream)
            
            # # add llm's response to chat history
            # st.session_state.messages.append({"role": "assistant", "content": response})

        
            with st.spinner(text="Searching"):
                results, retrieved_graph_data = query_neo4j(user_prompt, st.session_state.query_agent, st.session_state.subgraph_agent)
                # st.write(results)

            

        # Create retrieved graph
        with col2:
            graph_html = None
            with st.spinner(text="Generating graph"):
                G = create_graph(retrieved_graph_data)
                graph_html = visualize_graph(G)
                # unmounts (deleted) later?
            st.write(f"**Here is the graph I retrieved.**")
            
            # show graph in streamlit
            with open(graph_html, "r", encoding="utf-8") as f:
                graph_html_content = f.read()

            # style the pyvis html
            # graph_html_content = graph_html_content.replace(
            #     "<body>", 
            #     "<body style='background-color: black; color: white; border: none;'>"
            # )
            # custom_style = """
            # <style>
            #     #mynetwork {
            #         border: none !important;
            #     }
            # </style>
            # """
            # graph_html_content = graph_html_content.replace("</head>", custom_style + "</head>")
            # display the styled HTML in Streamlit
            st.components.v1.html(graph_html_content, height=550)

            os.remove(graph_html)

        with col1:
            # show concept reasoning from llm
            concept_text = ""
            with st.spinner(text="Reasoning"):
                concept_text = SubGraphAgent.convert_to_text(retrieved_graph_data)
                with st.expander("See context"):
                    st.write(concept_text)
            st.write(f"**This is the context I retrieved.**")


            st.write(f"**Finished reasoning.**")

            with st.spinner(text="Responding"):
                # st.session_state.response_agent.run(results[0].page_content, concept_text, user_prompt)
                final_answer = st.session_state.response_agent.run(results[0].page_content, concept_text, user_prompt)
                st.write(f"**{final_answer}**")

            # Cite the data
            for doc in results:
                source = f"Source: {doc.metadata['source']}" 
                page_n = f"Page number: {doc.metadata['page_number']}" 
                st.write(source)
                st.write(page_n)
            st.write(f"**Finished generation.**")
