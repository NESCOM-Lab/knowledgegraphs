"""
Helper tools - utilities for compressing graph data into LLM-readable summaries.
"""
import os


def summarize_subgraph(node_ids: list[str], graph, ollama_llm=None) -> str:
    """
    Compress a retrieved subgraph (by node ids) into a text summary.
    Used to shrink large graph context before reasoning.

    args:
        node_ids: list of entity node ids to include
        graph: Neo4j graph connection
        ollama_llm: optional ChatOllama instance for LLM summarization
                    (if None, returns structured text without LLM)

    returns:
        natural language summary of the subgraph
    """
    if not node_ids:
        return "No nodes provided to summarize."

    # Fetch relationships among these nodes
    cypher = """
    MATCH (a)-[r]-(b)
    WHERE a.id IN $node_ids AND b.id IN $node_ids
    RETURN DISTINCT a.id AS from_node, type(r) AS rel_type, b.id AS to_node
    LIMIT 100
    """
    try:
        rels = graph.query(cypher, params={"node_ids": node_ids})
    except Exception as e:
        return f"Error fetching subgraph: {e}"

    if not rels:
        # Fallback: just list the nodes
        return f"Nodes in subgraph: {', '.join(node_ids[:20])}"

    # Format as structured text first
    lines = [f"{r['from_node']} --[{r['rel_type']}]--> {r['to_node']}" for r in rels]
    structured = "\n".join(lines)

    if ollama_llm is None:
        # Return structured text without LLM
        return f"Subgraph relationships ({len(rels)} edges):\n{structured}"

    # Use Ollama LLM to compress into natural language
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        response = ollama_llm.invoke([
            SystemMessage(content=(
                "You are a neuroscience knowledge graph summarizer. "
                "Given a list of entity relationships from a knowledge graph, "
                "write a concise 3-5 sentence paragraph summarizing the key concepts "
                "and how they are connected. Focus on biological meaning."
            )),
            HumanMessage(content=f"Summarize these graph relationships:\n\n{structured}"),
        ])
        return response.content if isinstance(response.content, str) else str(response.content)
    except Exception as e:
        return f"Subgraph relationships:\n{structured}\n(LLM summarization failed: {e})"
