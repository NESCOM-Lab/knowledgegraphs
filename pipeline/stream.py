import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI

from pipeline import *



# Load llm
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = OpenAI(api_key=OPENAI_API_KEY)


# Streamlit UI

st.set_page_config(page_title="Knowledge Graphs")
st.title("Knowledge graphs")

# Session state variables

if "loaded_neo4j" not in st.session_state:
    st.session_state.loaded_neo4j = False

if "graph" not in st.session_state:
    st.session_state.graph = []

if "query_agent" not in st.session_state:
    st.session_state.query_agent = None

if "subgraph_agent" not in st.session_state:
        st.session_state.subgraph_agent = None


# chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# when app reruns display chat messages
def display_all_messages():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
display_all_messages()



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


# Main loop
if st.session_state.loaded_neo4j is False:  
    with st.spinner(text="Running load_neo4j()"):
        st.session_state.graph = load_neo4j()
    

if st.session_state.loaded_neo4j is True:
    if user_prompt := st.chat_input("Query your documents here"):

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

        # run the GraphRAG pipeline
        llm_transformer, embed, vector_retriever = load_llm_transformer() # uses a different version (ChatOpenAI)

        # loads agents into st.query_agent & st.subgraph_agent
        load_agents(st.session_state.graph, vector_retriever) # prob need to switch to session state

        res = query_neo4j(user_prompt, st.session_state.query_agent, st.session_state.subgraph_agent)
        st.write(res)



        print("finished")
