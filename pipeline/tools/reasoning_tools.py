"""
Reasoning tools - higher-level analysis built on graph + semantic search.
Used during VERIFY and EXPLORE phases.
"""
from typing import Any


def find_contradictions(graph, embed_model, vector_retriever, topic_or_claim: str, top_k: int = 8) -> list[dict]:
    """
    Find pairs of document chunks / entity relationships that potentially contradict each other
    on the same topic. Returns ranked list of contradiction candidates.

    Strategy:
    1. Vector search for chunks discussing the topic
    2. For each pair from different papers, find entities they share
    3. Flag entity pairs where the same entity has structurally different relationships
       (e.g., "INHIBITS" vs "ACTIVATES" for the same target)
    """
    from retriever_utils import add_similarity_scores

    try:
        vector_retriever.search_kwargs = {"k": top_k}
        results = vector_retriever.invoke(topic_or_claim)
        results = add_similarity_scores(results, topic_or_claim, embed_model)
    except Exception as e:
        return [{"error": f"Vector search failed: {e}"}]

    # Group by source paper
    by_source: dict[str, list] = {}
    for doc in results:
        src = doc.metadata.get("source", "unknown")
        by_source.setdefault(src, []).append(doc)

    if len(by_source) < 2:
        return [{"message": "Only one source found - need at least 2 papers to find contradictions"}]

    # For each chunk get its entity relationships from the graph
    def get_entity_rels(text_preview: str) -> list[dict]:
        cypher = """
        MATCH (d:Document)-[]-(e1)-[r]-(e2)
        WHERE d.text_preview = $preview
          AND NOT e1:Document AND NOT e2:Document
        RETURN DISTINCT e1.id AS entity, type(r) AS rel_type, e2.id AS target
        LIMIT 30
        """
        try:
            return graph.query(cypher, params={"preview": text_preview})
        except Exception:
            return []

    # Build entity→relationship map per paper
    paper_entity_rels: dict[str, dict[str, list[str]]] = {}
    for src, docs in by_source.items():
        paper_entity_rels[src] = {}
        for doc in docs:
            preview = doc.metadata.get("text_preview", "")
            if not preview:
                continue
            rels = get_entity_rels(preview)
            for r in rels:
                entity = r.get("entity", "")
                rel_type = r.get("rel_type", "")
                target = r.get("target", "")
                key = f"{entity}→{target}"
                paper_entity_rels[src].setdefault(key, []).append(rel_type)

    # Find contradictions: same entity pair, different relationship types across papers
    contradictions = []
    papers = list(paper_entity_rels.keys())
    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            pa, pb = papers[i], papers[j]
            shared_pairs = set(paper_entity_rels[pa].keys()) & set(paper_entity_rels[pb].keys())
            for pair in shared_pairs:
                rels_a = set(paper_entity_rels[pa][pair])
                rels_b = set(paper_entity_rels[pb][pair])
                if rels_a != rels_b:  # different relationship types = potential contradiction
                    contradictions.append({
                        "entity_pair": pair,
                        "paper_a": pa,
                        "paper_a_claims": list(rels_a),
                        "paper_b": pb,
                        "paper_b_claims": list(rels_b),
                        "conflict_type": "relationship_type_mismatch",
                    })

    # Also surface the raw text snippets as potential contradictions
    text_candidates = []
    for src, docs in by_source.items():
        for doc in docs[:2]:  # top 2 per source
            text_candidates.append({
                "source": src,
                "text_preview": doc.metadata.get("text_preview", ""),
                "similarity": round(float(doc.metadata.get("score", 0)), 4),
            })

    return {
        "topic": topic_or_claim,
        "structural_contradictions": contradictions[:10],
        "text_candidates_by_source": text_candidates,
        "papers_searched": list(by_source.keys()),
    }


def find_consensus(graph, embed_model, vector_retriever, topic: str, top_k: int = 8) -> dict:
    """
    Find claims/chunks that agree on a topic across multiple papers.
    Returns clusters of agreeing evidence.
    """
    from retriever_utils import add_similarity_scores

    try:
        vector_retriever.search_kwargs = {"k": top_k}
        results = vector_retriever.invoke(topic)
        results = add_similarity_scores(results, topic, embed_model)
    except Exception as e:
        return {"error": f"Vector search failed: {e}"}

    by_source: dict[str, list] = {}
    for doc in results:
        src = doc.metadata.get("source", "unknown")
        by_source.setdefault(src, []).append(doc)

    # Find shared entities across papers (consensus = same entity with same relationship type)
    def get_entity_rels(text_preview: str) -> list[dict]:
        cypher = """
        MATCH (d:Document)-[]-(e1)-[r]-(e2)
        WHERE d.text_preview = $preview
          AND NOT e1:Document AND NOT e2:Document
        RETURN DISTINCT e1.id AS entity, type(r) AS rel_type, e2.id AS target
        LIMIT 30
        """
        try:
            return graph.query(cypher, params={"preview": text_preview})
        except Exception:
            return []

    paper_claims: dict[str, set] = {}
    for src, docs in by_source.items():
        paper_claims[src] = set()
        for doc in docs[:2]:
            preview = doc.metadata.get("text_preview", "")
            rels = get_entity_rels(preview)
            for r in rels:
                claim = f"{r.get('entity')} {r.get('rel_type')} {r.get('target')}"
                paper_claims[src].add(claim)

    # Find claims shared across 2+ papers
    all_claims: dict[str, list[str]] = {}
    for src, claims in paper_claims.items():
        for claim in claims:
            all_claims.setdefault(claim, []).append(src)

    consensus_claims = [
        {"claim": claim, "supported_by": papers, "support_count": len(papers)}
        for claim, papers in all_claims.items()
        if len(papers) >= 2
    ]
    consensus_claims.sort(key=lambda x: x["support_count"], reverse=True)

    return {
        "topic": topic,
        "consensus_claims": consensus_claims[:20],
        "papers_searched": list(by_source.keys()),
        "total_consensus_claims": len(consensus_claims),
    }


def compare_papers(graph, paper_a: str, paper_b: str) -> dict:
    """
    Structured diff between two papers: shared entities, unique entities,
    opposing relationship types, and different methodologies mentioned.
    """
    # Shared entities
    shared_cypher = """
    MATCH (d1:Document {source: $pa})-[]-(e)-[]-(d2:Document {source: $pb})
    WHERE NOT e:Document
    RETURN DISTINCT e.id AS shared_entity, labels(e) AS entity_labels
    LIMIT 30
    """

    # Unique to paper_a
    unique_a_cypher = """
    MATCH (d1:Document {source: $pa})-[]-(e)
    WHERE NOT e:Document
      AND NOT EXISTS {
        MATCH (d2:Document {source: $pb})-[]-(e)
      }
    RETURN DISTINCT e.id AS entity, labels(e) AS entity_labels
    LIMIT 20
    """

    # Unique to paper_b
    unique_b_cypher = """
    MATCH (d2:Document {source: $pb})-[]-(e)
    WHERE NOT e:Document
      AND NOT EXISTS {
        MATCH (d1:Document {source: $pa})-[]-(e)
      }
    RETURN DISTINCT e.id AS entity, labels(e) AS entity_labels
    LIMIT 20
    """

    # Relationships in paper_a
    rels_a_cypher = """
    MATCH (d:Document {source: $pa})-[]-(e1)-[r]-(e2)
    WHERE NOT e1:Document AND NOT e2:Document
    RETURN DISTINCT e1.id AS from, type(r) AS rel, e2.id AS to
    LIMIT 30
    """

    # Relationships in paper_b
    rels_b_cypher = """
    MATCH (d:Document {source: $pb})-[]-(e1)-[r]-(e2)
    WHERE NOT e1:Document AND NOT e2:Document
    RETURN DISTINCT e1.id AS from, type(r) AS rel, e2.id AS to
    LIMIT 30
    """

    try:
        shared = graph.query(shared_cypher, params={"pa": paper_a, "pb": paper_b})
        unique_a = graph.query(unique_a_cypher, params={"pa": paper_a, "pb": paper_b})
        unique_b = graph.query(unique_b_cypher, params={"pa": paper_a, "pb": paper_b})
        rels_a = graph.query(rels_a_cypher, params={"pa": paper_a})
        rels_b = graph.query(rels_b_cypher, params={"pb": paper_b})

        # Find opposing relationships on shared entity pairs
        rels_a_set = {(r["from"], r["to"]): r["rel"] for r in rels_a}
        rels_b_set = {(r["from"], r["to"]): r["rel"] for r in rels_b}
        opposing = []
        for pair, rel_a in rels_a_set.items():
            rel_b = rels_b_set.get(pair)
            if rel_b and rel_b != rel_a:
                opposing.append({
                    "entity_pair": f"{pair[0]} → {pair[1]}",
                    "paper_a_relationship": rel_a,
                    "paper_b_relationship": rel_b,
                })

        return {
            "paper_a": paper_a,
            "paper_b": paper_b,
            "shared_entities": [r["shared_entity"] for r in shared],
            "unique_to_paper_a": [r["entity"] for r in unique_a],
            "unique_to_paper_b": [r["entity"] for r in unique_b],
            "paper_a_relationships": [{"from": r["from"], "rel": r["rel"], "to": r["to"]} for r in rels_a],
            "paper_b_relationships": [{"from": r["from"], "rel": r["rel"], "to": r["to"]} for r in rels_b],
            "opposing_claims": opposing,
        }
    except Exception as e:
        return {"error": str(e)}


def trace_evidence_chain(graph, text_preview: str, depth: int = 3) -> dict:
    """
    Walk backwards from a chunk/claim to find what it ultimately rests on.
    Traces: Document → entities → neighboring documents → their entities.
    Useful for assessing how well-supported a claim is.
    """
    depth = min(depth, 4)

    # Find the source document and trace outward
    cypher = f"""
    MATCH (d:Document)
    WHERE d.text_preview = $preview
    WITH d
    MATCH path = (d)-[*1..{depth}]-(neighbor)
    WHERE NOT ALL(n IN nodes(path) WHERE n:Document)
    WITH DISTINCT neighbor, d,
         [n IN nodes(path) | coalesce(n.id, n.source, 'unknown')] AS path_nodes,
         [r IN relationships(path) | type(r)] AS path_rels
    WHERE NOT neighbor:Document
    RETURN
        d.source AS source_paper,
        neighbor.id AS evidence_node,
        labels(neighbor) AS node_type,
        path_nodes AS chain,
        path_rels AS chain_relationships
    LIMIT 40
    """
    try:
        results = graph.query(cypher, params={"preview": text_preview})

        # Also find which other documents reference the same entities
        entities = list({r["evidence_node"] for r in results if r.get("evidence_node")})[:5]
        supporting_docs = []
        if entities:
            support_cypher = """
            MATCH (e)-[]-(d:Document)
            WHERE e.id IN $entities
            RETURN DISTINCT d.source AS paper, e.id AS entity
            LIMIT 20
            """
            support_results = graph.query(support_cypher, params={"entities": entities})
            supporting_docs = [
                {"paper": r["paper"], "shared_entity": r["entity"]}
                for r in support_results
            ]

        return {
            "source_chunk_preview": text_preview,
            "evidence_chain": [
                {
                    "node": r["evidence_node"],
                    "type": r.get("node_type"),
                    "chain": r.get("chain"),
                }
                for r in results[:20]
            ],
            "supporting_papers": supporting_docs,
            "chain_depth": depth,
        }
    except Exception as e:
        return {"error": str(e)}
