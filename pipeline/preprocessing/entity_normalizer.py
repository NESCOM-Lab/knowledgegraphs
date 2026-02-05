"""
normalize graph documents and map aliases before adding to DB
"""

import json
import unicodedata
from pathlib import Path


def normalize_string(text):
    """
    Deterministic string normalization:
    - Unicode normalization (Ca²⁺ → ca2+)
    - Casefold (aggressive lowercase for unicode)
    - Strip whitespace
    - Strip trailing punctuation/hyphens
    """
    if not text:
        return ""

    # Unicode normalization (NFKD = decompose combined chars)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))

    # Casefold (more aggressive than lower, handles unicode better)
    text = text.casefold()

    # Strip whitespace
    text = text.strip()

    # Strip trailing punctuation and hyphens
    text = text.rstrip('.,;:!?-_')

    return text


def load_aliases(alias_path=None):
    """Load alias dictionary from JSON file."""
    if alias_path is None:
        repo_root = Path(__file__).parent.parent.parent
        alias_path = repo_root / "ontology" / "aliases.json"

    if not Path(alias_path).exists():
        print(f"Warning: No aliases file at {alias_path}, using empty dict")
        return {}

    with open(alias_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_graph_documents(graph_docs, alias_map=None):
    """
    Normalize entities in graph documents before adding to Neo4j.

    Args:
        graph_docs: List of GraphDocument objects from LLM transformer
        alias_map: Dict mapping normalized_alias -> canonical_name (optional)

    Returns:
        List of GraphDocument objects with normalized entity names
    """
    # Load aliases if not provided
    if alias_map is None:
        alias_map = load_aliases()

    # Process each graph document
    normalized_docs = []

    for doc in graph_docs:

        normalized_nodes = []
        if doc.nodes:
            for node in doc.nodes:
                normalized_name = _normalize_entity(node.id, alias_map)
                node.id = normalized_name
                normalized_nodes.append(node)

        # Normalize relationships (source and target nodes)
        normalized_rels = []
        if doc.relationships:
            for rel in doc.relationships:
                rel.source.id = _normalize_entity(rel.source.id, alias_map)
                rel.target.id = _normalize_entity(rel.target.id, alias_map)
                normalized_rels.append(rel)

        # Update document
        doc.nodes = normalized_nodes
        doc.relationships = normalized_rels
        normalized_docs.append(doc)

    return normalized_docs


def _normalize_entity(entity_name, alias_map):
    """
    Internal helper: normalize and lookup canonical form.

    Args:
        entity_name: Original entity name
        alias_map: Dict of aliases

    Returns:
        Canonical name if found, otherwise original name
    """
    normalized = normalize_string(entity_name)

    # Lookup in alias map
    if normalized in alias_map:
        return alias_map[normalized]

    # No match, return original
    return entity_name
