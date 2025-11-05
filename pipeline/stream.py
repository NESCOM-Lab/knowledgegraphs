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

if "current_graph_fig" not in st.session_state:
    st.session_state.current_graph_fig = None

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
    k_value = st.number_input("Number of chunks retrieved", min_value=1, max_value=10, value=5, step=1)
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

                # Get chunks
                with st.spinner(text="Searching knowledge base..."):
                    results, _ = query_neo4j(user_prompt, st.session_state.k_value,
                                            st.session_state.query_agent,
                                            st.session_state.subgraph_agent)

                # Display retrieved chunks
                with st.expander(f"See {len(results)} retrieved chunks"):
                    for idx, doc in enumerate(results, 1):
                        st.write(f"**Chunk {idx}:**")
                        st.write(f"- Source: {doc.metadata['source']}")
                        st.write(f"- Page: {doc.metadata['page_number']}")
                        st.write(f"- Similarity: {doc.metadata['score']:.3f}")
                        st.write("---")

            # aggregate graph data from all chunks
            with col1:
                with st.spinner(text="Exploring knowledge graph..."):

                    aggregated_data = st.session_state.subgraph_agent.run_aggregated(
                        results,
                        max_depth=5 
                    )

                    st.write(f"**Found {aggregated_data['num_relationships']} unique relationships "
                            f"across {len(aggregated_data['sources'])} sources**")

            # visualize aggregated graph
            with col2:
                if aggregated_data['relationships']:
                    with st.spinner(text="Generating graph..."):
                        # Use aggregated relationships
                        all_edges = aggregated_data['relationships']

                        MAX_NODES = 40

                        # get nodes
                        node_ids = set()
                        limited_edges = []
                        for edge in all_edges:
                            c1 = edge['Concept1']['id']
                            c2 = edge['Concept2']['id']

                            # only add edge if we haven't exceeded MAX_NODES unique nodes
                            if len(node_ids) < MAX_NODES or (c1 in node_ids and c2 in node_ids):
                                limited_edges.append(edge)
                                node_ids.add(c1)
                                node_ids.add(c2)
                            if len(node_ids) >= MAX_NODES:
                                # stop adding new nodes but keep adding edges
                                continue

                        G = create_graph(limited_edges)
                        fig = visualize_graph(G)

                        # store figure in session state for download after agent response
                        st.session_state.current_graph_fig = fig

                    st.write(f"**Knowledge Graph ({len(limited_edges)} relationships)**")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No graph relationships found.")
                    st.session_state.current_graph_fig = None

            # Convert graph to text and generate aggregated response
            with col1:
                if aggregated_data['relationships']:
                    with st.spinner(text="Synthesizing knowledge graph context..."):
                        # convert to text
                        concept_text = SubGraphAgent.convert_to_text(aggregated_data['relationships'])

                        with st.expander("See relationship context"):
                            st.write(concept_text)

                    # aggregate all chunk texts
                    aggregated_chunk_text = "\n\n---\n\n".join([
                        f"[Chunk {i+1}]:\n{chunk}"
                        for i, chunk in enumerate(aggregated_data['chunks'])
                    ])

                    # citation list
                    sources_list = [
                        {'source': doc.metadata['source'], 'page': doc.metadata['page_number']}
                        for doc in results
                    ]

                    # reason with all context
                    with st.spinner(text="Reasoning..."):
                        final_answer = st.session_state.response_agent.run_aggregated(
                            aggregated_chunk_text,
                            concept_text,
                            user_prompt,
                            sources_list
                        )

                    st.write(f"**Response generated from {len(results)} chunks.**")
                else:
                    st.write("No graph relationships found.")
                    aggregated_chunk_text = "\n\n".join(aggregated_data['chunks'])
                    sources_list = [
                        {'source': doc.metadata['source'], 'page': doc.metadata['page_number']}
                        for doc in results
                    ]
                    final_answer = st.session_state.response_agent.run_aggregated(
                        aggregated_chunk_text,
                        "",
                        user_prompt,
                        sources_list
                    )

            # Download SVG button after response
            with col2:
                if st.session_state.current_graph_fig is not None:
                    try:
                        svg_bytes = st.session_state.current_graph_fig.to_image(format="svg")
                        st.download_button(
                            label="Download graph as SVG",
                            data=svg_bytes,
                            file_name="knowledge_graph.svg",
                            mime="image/svg+xml",
                            key=f"download_svg_{len(st.session_state.messages)}"  # use a unique key for each query
                        )
                    except Exception as e:
                        st.warning(f"SVG export requires kaleido: pip install kaleido")
