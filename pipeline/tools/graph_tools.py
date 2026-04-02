"""
Graph tools - direct Neo4j interaction.
Used when the agent knows exactly what it wants to query.
"""
from typing import Any


def query_graph(graph, cypher: str) -> list[dict]:
    """Execute arbitrary Cypher query. Returns list of result records."""
    try:
        return graph.query(cypher)
    except Exception as e:
        return [{"error": str(e)}]


def get_node(graph, node_id: str) -> dict:
    """
    Fetch a specific node by its id property with all its properties.
    Works for Document, Entity, or any node with an id field.
    """
    cypher = """
    MATCH (n)
    WHERE n.id = $node_id OR n.source = $node_id
    RETURN labels(n) AS labels, properties(n) AS props
    LIMIT 1
    """
    try:
        results = graph.query(cypher, params={"node_id": node_id})
        if results:
            r = results[0]
            props = r.get("props", {})
            props.pop("embedding", None)  # strip embedding vector - too large for context
            return {"labels": r.get("labels", []), "properties": props}
        return {"error": f"Node '{node_id}' not found"}
    except Exception as e:
        return {"error": str(e)}


def get_neighborhood(graph, node_id: str, depth: int = 2) -> dict:
    """
    Expand N hops out from a node. Returns a subgraph of nodes and relationships.
    depth: 1-3 recommended (deeper = more expensive)
    """
    depth = min(depth, 3)  # cap at 3 hops
    cypher = f"""
    MATCH path = (start)-[*1..{depth}]-(neighbor)
    WHERE start.id = $node_id OR start.source = $node_id
    AND NOT neighbor:Document
    WITH DISTINCT neighbor, relationships(path) AS rels
    RETURN
        neighbor.id AS node_id,
        labels(neighbor) AS node_labels,
        [r IN rels | {{type: type(r), start: startNode(r).id, end: endNode(r).id}}] AS relationships
    LIMIT 50
    """
    try:
        results = graph.query(cypher, params={"node_id": node_id})
        return {
            "center": node_id,
            "depth": depth,
            "nodes": [
                {
                    "id": r.get("node_id"),
                    "labels": r.get("node_labels"),
                    "relationships": r.get("relationships"),
                }
                for r in results
            ],
        }
    except Exception as e:
        return {"error": str(e)}


def get_path(graph, id_a: str, id_b: str) -> list[dict]:
    """
    Find shortest path(s) between two nodes. Critical for 'how are these connected?'
    Returns list of paths with intermediate nodes and relationships.
    """
    cypher = """
    MATCH path = shortestPath((a)-[*..6]-(b))
    WHERE (a.id = $id_a OR a.source = $id_a)
      AND (b.id = $id_b OR b.source = $id_b)
    RETURN
        [node IN nodes(path) | coalesce(node.id, node.source, 'unknown')] AS node_ids,
        [rel IN relationships(path) | type(rel)] AS rel_types,
        length(path) AS path_length
    LIMIT 5
    """
    try:
        results = graph.query(cypher, params={"id_a": id_a, "id_b": id_b})
        if not results:
            return [{"message": f"No path found between '{id_a}' and '{id_b}'"}]
        return [
            {
                "nodes": r.get("node_ids", []),
                "relationships": r.get("rel_types", []),
                "length": r.get("path_length"),
            }
            for r in results
        ]
    except Exception as e:
        return [{"error": str(e)}]


def get_schema(graph) -> dict:
    """
    Returns the current graph schema: node labels, relationship types, and property keys.
    Use this first to understand what's queryable.
    """
    try:
        # Node labels
        label_results = graph.query("CALL db.labels() YIELD label RETURN collect(label) AS labels")
        labels = label_results[0]["labels"] if label_results else []

        # Relationship types
        rel_results = graph.query(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) AS rel_types"
        )
        rel_types = rel_results[0]["rel_types"] if rel_results else []

        # Property keys
        prop_results = graph.query(
            "CALL db.propertyKeys() YIELD propertyKey RETURN collect(propertyKey) AS props"
        )
        props = prop_results[0]["props"] if prop_results else []
        # filter out embedding-related properties
        props = [p for p in props if "embedding" not in p.lower()]

        # Sample document sources (papers ingested)
        source_results = graph.query(
            "MATCH (d:Document) RETURN DISTINCT d.source AS source ORDER BY source LIMIT 20"
        )
        sources = [r["source"] for r in source_results if r.get("source")]

        return {
            "node_labels": labels,
            "relationship_types": rel_types,
            "property_keys": props,
            "ingested_papers": sources,
            "total_papers": len(sources),
        }
    except Exception as e:
        return {"error": str(e)}
