import streamlit as st



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
    response = f"wsg "
    with st.chat_message("llm"):
        st.markdown(response)
    
    st.session_state.messages.append({"role": "llm", "content": response})
