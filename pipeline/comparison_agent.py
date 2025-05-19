from typing import Dict, Any
import openai

class ComparisonAgent():
    def __init__(self, streamlit_obj):
        self.st = streamlit_obj
        self.prompt = """
        Answer the question using only the information from the provided source.
        Do not add or infer beyond what is explicitly stated.
        If the source lacks enough information to answer, say so and specify what’s missing.
        Begin your response with: 'Based on the paper "Paper name" by "Author",...
        """
        pass
        

    def run(self, sources: Dict[str, Any], query) -> None:
        """
        Provides a response to each given source and makes comparisons
        """ 
        for source, chunk in sources.items():
            self.st.write("Responding for " + source)
            full_query = f"""
            Source: {source} 
            Source's Information: {chunk[0].page_content}
            Question: {query}
            """
            
            # original chunk information 
            with self.st.expander("See original information"):
                self.st.write(chunk[0].page_content)



            # add user's question
            self.st.session_state.messages.append({"role": "user", "content": full_query})

            # display LLM's response with streaming
            response = ""
            with self.st.chat_message("assistant"):
                stream = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": self.prompt}] + [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in self.st.session_state.messages
                    ],
                    stream=True
                )
                response = self.st.write_stream(stream)

            # remove user's full query (w/ detailed info) and add just the question
            self.st.session_state.messages.pop()
            self.st.session_state.messages.append({"role": "user", "content": query})

            # add llm's response to chat history
            self.st.session_state.messages.append({"role": "assistant", "content": response})
        return  
        

