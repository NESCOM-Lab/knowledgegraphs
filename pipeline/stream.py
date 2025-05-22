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
from comparison_agent import *


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

if "comparison_agent" not in st.session_state:
    st.session_state.comparison_agent = None

if "k_value" not in st.session_state:
    st.session_state.k_value = 1

if "compare_mode" not in st.session_state:
    st.session_state.compare_mode = False

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
    st.session_state.query_agent = QueryAgent(vector_retriever, st.session_state.embed)
    st.session_state.subgraph_agent = SubGraphAgent(graph)
    st.session_state.response_agent = ResponseAgent(st)
    st.session_state.comparison_agent = ComparisonAgent(st)


# NetworkX graph for graph DB visualization
# https://python-textbook.pythonhumanities.com/06_sna/06_01_05_networkx_pyvis.html
def create_graph(edges):
    # process to format networkx uses
    G = nx.Graph()

    for x in edges:
        c1 = x['Concept1']['id']
        c2 = x['Concept2']['id']
        relationship = x['Relationship']['type']
        source = x['Source']['id']  # e.g., 'paper1.pdf'

        # add the edge with its relationship
        G.add_edge(c1, c2, relationship=relationship)

        # add source to each concept node 
        G.nodes[c1]['source'] = source
        G.nodes[c2]['source'] = source

    return G

def visualize_graph(G):
    net = Network(height="500px", width="100%", notebook=False)
    net.from_nx(G, default_node_size=10) # pass in graph

    # add relationship names to edges
    for edge in net.edges:
        u, v = edge["from"], edge["to"]
        edge_label = G[u][v].get("relationship", "") # get rship name
        edge["title"] = edge_label # hover text
        edge["label"] = edge_label # display on edge
        edge["color"] = "black"  # edge color
        # edge["font"] = {"size": 14, "color": "white", "face": "Arial"}  #  styling for edge labels
    
    # add source names to nodes
    for node in net.nodes:
        node_id = node["id"]
        source = G.nodes[node_id].get("source", "unknown") # get source name
        node["title"] = source # hover text
    
    # save graph as temp html file
    from pathlib import Path
    tmp_dir = Path(tempfile.gettempdir())
    filename = "temp_graph.html"
    tmp_path = tmp_dir.joinpath(filename)

    # WORKAROUND: change into the temp dir
    old_cwd = os.getcwd()
    os.chdir(tmp_dir)

    try:
        net.write_html(filename)  # just give the name, not full path
    finally:
        os.chdir(old_cwd)

    return str(tmp_path)

# Main loop
col1, col2 = st.columns(spec=2, vertical_alignment="bottom") # aligns 2 columns together

# k-value for retriever agent
with col1:
    k_value = st.number_input("Number of chunks retrieved", min_value=1, max_value=10, value=1, step=1)
    st.session_state.k_value = k_value

# Streamlit PDF uploader
with col1:
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], accept_multiple_files=False)
    st.write(uploaded_file.name if uploaded_file else "No file uploaded")

    compare_toggle = st.toggle("Comparison mode")

    # toggle comparison mode
    if compare_toggle:
        st.write("Comparison mode on")
        st.session_state.compare_mode = True
    if not compare_toggle:
        st.session_state.compare_mode = False

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

# add zoom limits to the HTML networkx graph 
def add_zoom_limits(html_content):
    injection_js = """
    <script type="text/javascript">
    network.on("zoom", function (params) {
      const MIN_SCALE = 0.2;
      const MAX_SCALE = 2;
      const currentScale = network.getScale();
      if (currentScale < MIN_SCALE) {
        network.moveTo({ scale: MIN_SCALE });
      } else if (currentScale > MAX_SCALE) {
        network.moveTo({ scale: MAX_SCALE });
      }
    });
    </script>
    """
    return html_content.replace("</body>", injection_js + "\n</body>")

if st.session_state.loaded_neo4j and st.session_state.loaded_agents is True:
    if user_prompt := st.chat_input("Query your documents here"):
        if st.session_state.compare_mode:
            # Comparison Mode
            with col1:
                display_all_messages()
                # display user's message in UI
                with st.chat_message("user"):
                    st.markdown(user_prompt)
                
                with st.spinner(text="Searching"):
                    results, sources = query_neo4j(user_prompt, 10, 
                                                                st.session_state.query_agent, 
                                                                st.session_state.subgraph_agent, True)
                    # st.write(sources)

                # Display results
                with st.expander("See retrieved chunks"):
                    for doc in results:
                        st.write("Source: " + doc.metadata['source'])
                        st.write("Page #: " + str(doc.metadata['page_number']))
                        st.write("Text Preview: " + doc.metadata['text_preview'])
                        st.write("Similarity score: "  + str(doc.metadata['score']))
                
                # Compare results
                if len(sources) == 1:
                    # only 1 source dominating
                    st.write("Only 1 relevant source. Can't compare")
                else:
                    print("Multiple sources")
                    st.session_state.comparison_agent.run(sources, user_prompt)
                    


                    # with st.spinner(text="Reasoning"):
                    #     concept_tex       t = SubGraphAgent.convert_to_text(retrieved_graph_data[0]) # for now only use first chunk for context
                    #     with st.expander("See context"):
                    #         st.write(concept_text)
                    # st.write(f"**This is the context I retrieved.**")


                    st.write(f"**Finished reasoning.**")


                

                    
                


        if st.session_state.compare_mode == False:
            # Regular GraphRAG

            with col1:
                display_all_messages()
                # display user's message in UI
                with st.chat_message("user"):
                    st.markdown(user_prompt)
                
                
        
            
                with st.spinner(text="Searching"):
                    results, retrieved_graph_data = query_neo4j(user_prompt, st.session_state.k_value, 
                                                                st.session_state.query_agent, 
                                                                st.session_state.subgraph_agent)
                    # st.write(results)

                # Display results
                with st.expander("See retrieved chunks"):
                    for doc in results:
                        st.write("Source: " + doc.metadata['source'])
                        st.write("Page #: " + str(doc.metadata['page_number']))
                        st.write("Text Preview: " + doc.metadata['text_preview'])
                        st.write("Similarity score: "  + str(doc.metadata['score']))
                

            # Create retrieved graph
            with col2:
                graph_html = None
                with st.spinner(text="Generating graph"):
                    # cheese = retrieved_graph_data[1][0]
                    # cheese['Source']['id'] = "abi"
                    # st.write(cheese) # for debugging -- checks if different source can be shown in graph 
                    G = create_graph([item for sublist in retrieved_graph_data for item in sublist]) # use all chunks in the graph
                    graph_html = visualize_graph(G)
                    # unmounts (deleted) later?
                st.write(f"**Here is the graph I retrieved.**")
                
                # show graph in streamlit
                with open(graph_html, "r", encoding="utf-8") as f:
                    graph_html_content = f.read()
                
                graph_html_content = add_zoom_limits(graph_html_content)


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
                    concept_text = SubGraphAgent.convert_to_text(retrieved_graph_data[0]) # for now only use first chunk for context
                    with st.expander("See context"):
                        st.write(concept_text)
                st.write(f"**This is the context I retrieved.**")


                st.write(f"**Finished reasoning.**")


                # Respond and Cite the data
                for doc in results:
                    with st.spinner(text="Responding"):
                        # st.session_state.response_agent.run(results[0].page_content, concept_text, user_prompt)
                        final_answer = st.session_state.response_agent.run(doc.page_content, 
                                                                        concept_text, user_prompt)
                        # st.write(f"**{final_answer}**")
                    source = f"Source: {doc.metadata['source']}" 
                    page_n = f"Page number: {doc.metadata['page_number']}" 
                    st.write(source + ", " + page_n)

                st.write(f"**Finished response.**")
