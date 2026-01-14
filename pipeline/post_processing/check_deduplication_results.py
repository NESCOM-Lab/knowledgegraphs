"""
Check Deduplication Results
===========================

Helper script to analyze the state of the Neo4j graph before/after deduplication.

Usage:
    python check_deduplication_results.py
"""

import os
from collections import defaultdict
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()


def get_neo4j_connection():
    """Establish Neo4j connection."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    if not password:
        raise ValueError("NEO4J_PASSWORD not set in environment")

    return GraphDatabase.driver(uri, auth=(username, password)), database


def count_nodes_by_label(driver, database):
    """Count nodes grouped by label."""
    query = """
    MATCH (n)
    WHERE NOT n:Document
    RETURN labels(n)[0] AS label, count(n) AS count
    ORDER BY count DESC
    """

    with driver.session(database=database) as session:
        result = session.run(query)
        return [(record["label"], record["count"]) for record in result]


def find_potential_duplicates(driver, database, limit=20):
    """Find nodes with very similar names (simple string matching)."""
    query = """
    MATCH (n)
    WHERE NOT n:Document AND n.id IS NOT NULL
    WITH n.id AS name, collect(n) AS nodes, count(n) AS node_count
    WHERE node_count > 1
    RETURN name, node_count, [node in nodes | {id: elementId(node), label: labels(node)[0]}] AS nodes
    ORDER BY node_count DESC
    LIMIT $limit
    """

    with driver.session(database=database) as session:
        result = session.run(query, limit=limit)
        return list(result)


def get_graph_statistics(driver, database):
    """Get overall graph statistics."""
    stats_query = """
    MATCH (n) WHERE NOT n:Document
    WITH count(n) AS entity_count
    MATCH (d:Document)
    WITH entity_count, count(d) AS doc_count
    MATCH ()-[r]->()
    RETURN entity_count, doc_count, count(r) AS relationship_count
    """

    with driver.session(database=database) as session:
        result = session.run(stats_query)
        return result.single()


def find_nodes_by_name_pattern(driver, database, pattern, limit=50):
    """Find nodes matching a name pattern."""
    query = """
    MATCH (n)
    WHERE NOT n:Document
      AND n.id =~ $pattern
    RETURN n.id AS name, labels(n)[0] AS label, elementId(n) AS id
    ORDER BY name
    LIMIT $limit
    """

    with driver.session(database=database) as session:
        result = session.run(query, pattern=f"(?i).*{pattern}.*", limit=limit)
        return list(result)


def check_relationship_density(driver, database):
    """Check how connected nodes are."""
    query = """
    MATCH (n)
    WHERE NOT n:Document
    OPTIONAL MATCH (n)-[r]-()
    WITH n, count(r) AS rel_count
    RETURN
        count(n) AS total_nodes,
        avg(rel_count) AS avg_relationships,
        min(rel_count) AS min_relationships,
        max(rel_count) AS max_relationships
    """

    with driver.session(database=database) as session:
        result = session.run(query)
        return result.single()


def main():
    print("=" * 70)
    print("Neo4j Graph Deduplication Analysis")
    print("=" * 70)

    try:
        driver, database = get_neo4j_connection()
        print(f"✓ Connected to Neo4j database: {database}\n")

        # Overall statistics
        print("-" * 70)
        print("GRAPH STATISTICS")
        print("-" * 70)
        stats = get_graph_statistics(driver, database)
        print(f"Entity nodes: {stats['entity_count']:,}")
        print(f"Document nodes: {stats['doc_count']:,}")
        print(f"Relationships: {stats['relationship_count']:,}")

        # Relationship density
        print("\n" + "-" * 70)
        print("RELATIONSHIP DENSITY")
        print("-" * 70)
        density = check_relationship_density(driver, database)
        print(f"Average relationships per node: {density['avg_relationships']:.1f}")
        print(f"Min relationships: {density['min_relationships']}")
        print(f"Max relationships: {density['max_relationships']}")

        # Node count by label
        print("\n" + "-" * 70)
        print("NODES BY LABEL (Top 20)")
        print("-" * 70)
        labels = count_nodes_by_label(driver, database)
        for label, count in labels[:20]:
            print(f"  {label}: {count:,}")
        if len(labels) > 20:
            remaining = sum(count for _, count in labels[20:])
            print(f"  ... and {len(labels) - 20} more labels ({remaining:,} nodes)")

        # Exact duplicate names (should be 0 after deduplication)
        print("\n" + "-" * 70)
        print("EXACT DUPLICATE NAMES (Should be 0 after deduplication)")
        print("-" * 70)
        exact_dupes = find_potential_duplicates(driver, database, limit=10)
        if exact_dupes:
            print(f"⚠ Found {len(exact_dupes)} exact duplicate names:")
            for record in exact_dupes:
                name = record["name"]
                count = record["node_count"]
                nodes = record["nodes"]
                print(f"\n  '{name}' ({count} nodes):")
                for node in nodes:
                    print(f"    - {node['label']} [{node['id']}]")
        else:
            print("✓ No exact duplicate names found")

        # Interactive search
        print("\n" + "-" * 70)
        print("SEARCH FOR SPECIFIC ENTITIES")
        print("-" * 70)
        print("(Press Ctrl+C to exit)\n")

        while True:
            try:
                search_term = input("Enter search term (e.g., 'Parkinson', 'alpha'): ").strip()
                if not search_term:
                    continue

                results = find_nodes_by_name_pattern(driver, database, search_term, limit=50)

                if results:
                    print(f"\nFound {len(results)} matching nodes:")
                    for record in results:
                        print(f"  - {record['name']} [{record['label']}]")
                else:
                    print(f"No nodes found matching '{search_term}'")

                print()  # Empty line

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.close()


if __name__ == "__main__":
    main()
