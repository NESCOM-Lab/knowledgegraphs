# Knowledge Graphs
Extract knowledge graphs from papers and improve LLM reasoning with contextual understanding and a chain-of-thought system through HybridRAG.

## Architecture
![Architecture](https://github.com/user-attachments/assets/6d6c1b53-704f-4579-9bda-f5dc5328ee64)

## Models
PDF Ingestion (LLMGraphTransformer): Gemini-2.0-flash-lite <br/>
LLM streaming agents: GPT-4.1

## Usage
Setup API Keys in .env_example and rename it to .env <br>
Install requirements in virtual env with `pip install -r requirements.txt` <br/>
Cd into `/pipeline` and run `streamlit run stream.py`

## References
https://arxiv.org/abs/2501.00309v2
