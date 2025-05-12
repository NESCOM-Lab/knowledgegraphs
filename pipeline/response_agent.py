"""
Takes in context from retrieved chunk & retrieved graph information and outputs
a response
"""

import openai
class ResponseAgent():
    def __init__(self, streamlit_obj):
        self.st = streamlit_obj
        # System prompt
        self.prompt = f"""
        Answer the question based on the contexts given and be detailed.
        There will be two contexts provided; the first will be detailed information, and the second will be
        concise contextual information that has been processed to describe the relationships between concepts.
        Respond "Unsure about answer" if not sure about the answer, or if the contextual information
        is not relevant to the query. Do not make up new information not provided in the context.



        Example of input:
        1. Detailed Information: 
        Apples are nutritious, round fruits that grow on apple trees (Malus domestica). They come in various colors, such as
        red, green, and yellow, and are rich in vitamins, fiber, and antioxidants. Apples are commonly eaten raw, 
        used in cooking, and made into products like apple juice, cider, and applesauce. They have a sweet to tart
        taste, depending on the variety, and are known for their health benefits, including aiding digestion and 
        supporting heart health. Vacuums are devices used to clean by suctioning dirt, dust, and debris from floors
        and other surfaces. They work by creating a partial vacuum that pulls in air and particles, which are then
        trapped in a bag or container for disposal. There are different types of vacuums, including upright,
        canister, handheld, and


        2. Relationship Context: 
        Vacuums are used to clean other surfaces. Vacuums are used to clean floors. Vacuums are used to clean surfaces.
        Vacuums work by creating a partial vacuum. Vacuums trap air and particles. Vacuums trap dirt, dust, and debris.
        Vacuums pull in particles. Vacuums pull in air. Vacuums trap in the bag. Vacuums trap in the container. 
        Vacuums work by creating a partial vacuum. Vacuums trap in the bag. Vacuums trap in the container. Apples contain vitamins. 
        Apples contain fiber. Apples contain antioxidants. Apples are used in apple juice. Apples are used in cooking. 
        Apples are used in cider. Apples are used in applesauce. Apples grow on apple trees. 
        Apples are made into apple juice. Apples are made into cider. Apples are made into applesauce. 
        Apples grow on apple trees. Apples contain vitamins. Apples contain fiber. Apples contain antioxidants. 
        Particles are trapped in the bag.

        Question: How do vacuums use air movement to trap and contain particles, and what are the different ways they store collected debris?
        
        Example of Output:
        Vacuums work by creating a partial vacuum, which allows them to pull in air and particles from floors and other surfaces. As air is drawn into the vacuum, it carries dirt, dust, and debris, which are then trapped inside the vacuum.

        Vacuums store collected debris in two main ways:

        Bag Storage -- The particles are trapped in the bag, which can be removed and disposed of when full.
        Container Storage -- The particles are trapped in the container, which can be emptied and reused.
        By using air movement to pull in and trap particles, vacuums effectively clean surfaces and floors.
        """
        


    # Combines contextual information (sentences) from graph with exact sentences from retrieved chunk 
    # and produces a final response
    def run(self, chunk_information, graph_context_information, query) -> str | None:

        # add all context together to query llm
        full_query = f"""
        Detailed Information: {chunk_information}
        Relationship Context: {graph_context_information}
        Question: {query}
        """

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": full_query}
            ]
        ) 
        
        # display LLM's response
        # with self.st.chat_message("assistant"):
        #     stream = openai.chat.completions.create(
        #         model="gpt-3.5-turbo",
        #         messages=[{"role": "system", "content": self.prompt}] + [
        #             {"role": msg["role"], "content": msg["content"]}
        #             for msg in self.st.session_state.messages
        #         ],
        #         stream=True
        #     )
        #     response = self.st.write_stream(stream)
        
        # add llm's response to chat history
        # self.st.session_state.messages.append({"role": "assistant", "content": response})

        return response.choices[0].message.content


        