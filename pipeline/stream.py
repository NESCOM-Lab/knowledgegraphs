import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI



# Load llm
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = OpenAI(api_key=OPENAI_API_KEY)


# Streamlit UI

st.set_page_config(page_title="Knowledge Graphs")
st.title("Knowledge graphs")



# chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# when app reruns display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])



if user_prompt := st.chat_input("Query your documents here"):

    # display user's message in UI
    with st.chat_message("user"):
        st.markdown(user_prompt)
    
    # add user's message to chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # display LLM's response
    with st.chat_message("llm"):
        stream = llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages
            ],
            stream=True
        )
        response = st.write_stream(stream)
    
    # add llm's response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
