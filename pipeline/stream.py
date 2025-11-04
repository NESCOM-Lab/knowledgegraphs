import time
from pipeline import chunk_document, ingest_document
import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
import networkx as nx
import plotly.graph_objects as go

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
    """Creates Plotly graph visualization with improved styling"""

    # Use spring layout for better node positioning
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # Create edge traces
    edge_traces = []
    edge_label_traces = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        # Get relationship type for this edge
        relationship = G[edge[0]][edge[1]].get('relationship', '')

        # Edge line
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(edge_trace)

        # Edge label (relationship) at midpoint
        edge_label = go.Scatter(
            x=[(x0 + x1) / 2],
            y=[(y0 + y1) / 2],
            mode='text',
            text=[relationship],
            textposition='middle center',
            textfont=dict(size=10, color='#666'),
            hoverinfo='text',
            hovertext=f"Relationship: {relationship}",
            showlegend=False
        )
        edge_label_traces.append(edge_label)

    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_hover = []
    node_colors = []

    # Group nodes by source for color coding
    sources = set(G.nodes[node].get('source', 'unknown') for node in G.nodes())
    source_colors = {}
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']
    for idx, source in enumerate(sources):
        source_colors[source] = colors[idx % len(colors)]

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        source = G.nodes[node].get('source', 'unknown')
        node_text.append(str(node))
        node_hover.append(f"<b>{node}</b><br>Source: {source}")
        node_colors.append(source_colors[source])

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        textfont=dict(size=10, color='#000'),
        hovertext=node_hover,
        hoverinfo='text',
        marker=dict(
            size=20,
            color=node_colors,
            line=dict(width=2, color='#fff')
        ),
        showlegend=False
    )

    # Create figure
    fig = go.Figure(data=edge_traces + edge_label_traces + [node_trace])

    # Update layout for better appearance
    fig.update_layout(
        title=dict(
            text='Knowledge Graph Visualization',
            font=dict(size=20, color='#333')
        ),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='#f8f9fa',
        height=600
    )

    return fig

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
    
    # toggle comparison mode
    compare_toggle = st.toggle("Comparison mode")
    if compare_toggle:
        st.session_state.compare_mode = True
    if not compare_toggle:
        st.session_state.compare_mode = False

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
                        # st.write("Text Preview: " + doc.metadata['text'])
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
                        # st.write("Text Preview: " + doc.metadata['text'])
                        st.write("Similarity score: "  + str(doc.metadata['score']))
                

            # Create retrieved graph
            with col2:
                with st.spinner(text="Generating graph"):
                    # limit # of edges seen in UI
                    all_edges = [item for sublist in retrieved_graph_data for item in sublist]

                    MAX_NODES = 40
                    # Collect edges that only involve up to MAX_NODES unique nodes
                    node_ids = set()
                    limited_edges = []
                    for edge in all_edges:
                        c1 = edge['Concept1']['id']
                        c2 = edge['Concept2']['id']
                        # Only add edge if we haven't exceeded MAX_NODES unique nodes
                        if len(node_ids) < MAX_NODES or (c1 in node_ids and c2 in node_ids):
                            limited_edges.append(edge)
                            node_ids.add(c1)
                            node_ids.add(c2)
                        if len(node_ids) >= MAX_NODES:
                            # Stop adding new nodes, but allow edges between already-included nodes
                            continue

                    G = create_graph(limited_edges)
                    fig = visualize_graph(G)

                st.write(f"**Here is the graph I retrieved.**")

                # Display the Plotly graph
                st.plotly_chart(fig, use_container_width=True)

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
                        print("*********8")
                        print(doc.page_content)
                        print("*********8")
                        print(concept_text)
                        print("*********8")
                        print(user_prompt)
                        print("***************")
                        final_answer = st.session_state.response_agent.run(doc.page_content, 
                                                                        concept_text, user_prompt)
                        # st.write(f"**{final_answer}**")
                    source = f"Source: {doc.metadata['source']}" 
                    page_n = f"Page number: {doc.metadata['page_number']}" 
                    st.write(source + ", " + page_n)

                st.write(f"**Finished response.**")
