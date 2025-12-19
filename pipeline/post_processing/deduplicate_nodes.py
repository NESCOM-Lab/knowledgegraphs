#!/usr/bin/env python3
"""
Neo4j Graph node deduplication 
Automatically detects and merges duplicate entity nodes in a Neo4j knowledge graph.

Process:
1. Extract all entity nodes (non-Document nodes) from Neo4j
2. Embed node names using Ollama embeddings
3. Cluster similar nodes using hierarchical clustering
4. Merge duplicates in Neo4j, preserving all relationships

Usage:
python deduplicate_nodes.py [--similarity-threshold 0.99] [--dry-run]
"""

import os
import sys
import argparse
import time
from typing import List, Dict
from collections import defaultdict

import numpy as np
from neo4j import GraphDatabase
from langchain_ollama import OllamaEmbeddings
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Neo4jDeduplicator:
    """Handles extraction, clustering, and merging of duplicate Neo4j nodes."""

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        embedding_model: str = "nomic-embed-text",
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database

        # Initialize Neo4j driver
        print(f"Connecting to Neo4j at {uri}...")
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

        # Initialize embedding model
        print(f"Initializing embedding model: {embedding_model}")
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=ollama_base_url,
        )

    def close(self):
        """Close Neo4j connection."""
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def extract_entity_nodes(self) -> List[Dict]:
        """
        Extract all entity nodes from Neo4j (excluding Document nodes).

        Returns:
            List of dicts with keys: element_id, label, name, properties
        """
        print("\n" + "="*70)
        print("STEP 1: Extracting entity nodes from Neo4j")
        print("="*70)

        query = """
        MATCH (n)
        WHERE NOT n:Document
        RETURN elementId(n) AS element_id,
               labels(n) AS labels,
               COALESCE(n.id, n.name, labels(n)[0]) AS name,
               properties(n) AS properties
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            nodes = []

            for record in result:
                element_id = record["element_id"]
                labels = record["labels"]
                name = record["name"]
                properties = record["properties"]

                # Skip nodes without any identifier
                if not name or not labels:
                    continue

                nodes.append({
                    "element_id": element_id,
                    "label": labels[0] if labels else "Entity",  # Primary label
                    "all_labels": labels,
                    "name": str(name),
                    "properties": properties,
                })

        print(f"**Extracted {len(nodes)} entity nodes")

        # Show label distribution
        label_counts = defaultdict(int)
        for node in nodes:
            label_counts[node["label"]] += 1

        print(f"\nLabel distribution:")
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1])[:20]:
            print(f"  {label}: {count}")
        if len(label_counts) > 20:
            print(f"  ... and {len(label_counts) - 20} more labels")

        return nodes

    def embed_node_names(self, nodes: List[Dict]) -> np.ndarray:
        """
        Embed all node names using Ollama embeddings.

        Args:
            nodes: List of node dictionaries

        Returns:
            numpy array of shape (n_nodes, embedding_dim)
        """
        print("\n" + "="*70)
        print("STEP 2: Embedding node names")
        print("="*70)

        node_names = [node["name"] for node in nodes]

        print(f"Embedding {len(node_names)} node names...")
        start_time = time.time()

        # Batch embedding for efficiency
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(node_names), batch_size):
            batch = node_names[i:i + batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)

            if (i + batch_size) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (i + batch_size) / elapsed
                print(f"  Progress: {i + batch_size}/{len(node_names)} ({rate:.1f} nodes/sec)")

        embeddings_array = np.array(all_embeddings)
        elapsed = time.time() - start_time

        print(f"Done embedding {len(node_names)} nodes in {elapsed:.2f}s")
        print(f"  Embedding shape: {embeddings_array.shape}")

        return embeddings_array

    def cluster_similar_nodes(
        self,
        nodes: List[Dict],
        embeddings: np.ndarray,
        similarity_threshold: float = 0.85,
    ) -> List[List[int]]:
        """
        Cluster nodes using hierarchical clustering based on cosine similarity.

        Args:
            nodes: List of node dictionaries
            embeddings: Node embeddings array
            similarity_threshold: Cosine similarity threshold (0-1)

        Returns:
            List of clusters, where each cluster is a list of node indices
        """
        print("\n" + "="*70)
        print("STEP 3: Clustering similar nodes")
        print("="*70)
        print(f"Similarity threshold: {similarity_threshold}")

        # Compute cosine similarity matrix
        print("Computing pairwise cosine similarities...")
        start_time = time.time()

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / (norms + 1e-10)

        # Cosine distance = 1 - cosine_similarity
        distances = pdist(normalized_embeddings, metric='cosine')

        elapsed = time.time() - start_time
        print(f"**Computed {len(distances)} pairwise distances in {elapsed:.2f}s")

        # Hierarchical clustering
        print("Performing hierarchical clustering...")
        linkage_matrix = linkage(distances, method='average')

        # Convert similarity threshold to distance threshold
        distance_threshold = 1 - similarity_threshold

        # Extract clusters
        cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')

        # Group nodes by cluster
        clusters_dict = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_labels):
            clusters_dict[cluster_id].append(idx)

        # Filter to only clusters with 2+ nodes (potential duplicates)
        duplicate_clusters = [
            cluster for cluster in clusters_dict.values()
            if len(cluster) >= 2
        ]

        print(f"***Found {len(duplicate_clusters)} clusters with duplicates")
        print(f"  Total duplicate nodes: {sum(len(c) for c in duplicate_clusters)}")

        # Show examples
        if duplicate_clusters:
            print("\nExample duplicate clusters:")
            for i, cluster_indices in enumerate(duplicate_clusters[:5]):
                cluster_nodes = [nodes[idx] for idx in cluster_indices]
                print(f"\n  Cluster {i+1} ({len(cluster_nodes)} nodes):")
                for node in cluster_nodes[:10]:  # Show max 10 per cluster
                    print(f"    - {node['name']} [{node['label']}]")
                if len(cluster_nodes) > 10:
                    print(f"    ... and {len(cluster_nodes) - 10} more")

        return duplicate_clusters

    def compute_cluster_statistics(
        self,
        clusters: List[List[int]],
        nodes: List[Dict],
        embeddings: np.ndarray,
    ) -> None:
        """Print detailed statistics about clusters."""
        print("\n" + "="*70)
        print("CLUSTER STATISTICS")
        print("="*70)

        for i, cluster_indices in enumerate(clusters[:10]):  # Show top 10
            cluster_nodes = [nodes[idx] for idx in cluster_indices]
            cluster_embeddings = embeddings[cluster_indices]

            # Compute pairwise similarities within cluster
            norms = np.linalg.norm(cluster_embeddings, axis=1, keepdims=True)
            normalized = cluster_embeddings / (norms + 1e-10)
            similarity_matrix = normalized @ normalized.T

            # Get mean similarity (excluding diagonal)
            mask = ~np.eye(len(cluster_indices), dtype=bool)
            mean_similarity = similarity_matrix[mask].mean()
            min_similarity = similarity_matrix[mask].min()

            print(f"\nCluster {i+1}:")
            print(f"  Size: {len(cluster_nodes)}")
            print(f"  Mean similarity: {mean_similarity:.3f}")
            print(f"  Min similarity: {min_similarity:.3f}")
            print(f"  Nodes:")
            for node in cluster_nodes:
                print(f"    - {node['name']} [{node['label']}]")

    def merge_duplicate_nodes(
        self,
        clusters: List[List[int]],
        nodes: List[Dict],
        dry_run: bool = False,
    ) -> int:
        """
        Merge duplicate nodes in Neo4j.

        For each cluster:
        1. Select the canonical node (most relationships or longest name)
        2. Merge all other nodes into the canonical node
        3. Transfer all relationships
        4. Delete duplicate nodes

        Args:
            clusters: List of clusters (each is list of node indices)
            nodes: List of all nodes
            dry_run: If True, only print what would be done

        Returns:
            Number of nodes merged
        """
        print("\n" + "="*70)
        if dry_run:
            print("STEP 4: Dry run - showing what would be merged")
        else:
            print("STEP 4: Merging duplicate nodes")
        print("="*70)

        total_merged = 0

        with self.driver.session(database=self.database) as session:
            for i, cluster_indices in enumerate(clusters):
                cluster_nodes = [nodes[idx] for idx in cluster_indices]

                # Select canonical node (prefer longer, more descriptive names)
                canonical = max(cluster_nodes, key=lambda n: len(n["name"]))
                duplicates = [n for n in cluster_nodes if n != canonical]

                if dry_run:
                    print(f"\nCluster {i+1}: Would merge {len(duplicates)} nodes into:")
                    print(f"  Canonical: {canonical['name']} [{canonical['label']}]")
                    print(f"  Duplicates:")
                    for dup in duplicates:
                        print(f"    - {dup['name']}")
                else:
                    print(f"\nCluster {i+1}: Merging {len(duplicates)} nodes into:")
                    print(f"  Canonical: {canonical['name']} [{canonical['label']}]")

                    for dup in duplicates:
                        try:
                            # Count relationships before merge (for reporting)
                            count_query = """
                            MATCH (duplicate)
                            WHERE elementId(duplicate) = $duplicate_id
                            OPTIONAL MATCH (duplicate)-[r1]->()
                            OPTIONAL MATCH ()-[r2]->(duplicate)
                            RETURN count(DISTINCT r1) AS outgoing, count(DISTINCT r2) AS incoming
                            """
                            count_result = session.run(count_query, duplicate_id=dup["element_id"])
                            counts = count_result.single()
                            out_count = counts["outgoing"] if counts else 0
                            in_count = counts["incoming"] if counts else 0

                            # Step 1: Transfer outgoing relationships
                            if out_count > 0:
                                transfer_out_query = """
                                MATCH (canonical)
                                WHERE elementId(canonical) = $canonical_id
                                MATCH (duplicate)
                                WHERE elementId(duplicate) = $duplicate_id
                                MATCH (duplicate)-[r]->(target)
                                WITH canonical, duplicate, r, target,
                                     type(r) AS rel_type, properties(r) AS rel_props
                                WHERE NOT EXISTS((canonical)-[]->(target))
                                CALL apoc.create.relationship(
                                    canonical, rel_type, rel_props, target
                                ) YIELD rel
                                RETURN count(rel) AS transferred
                                """

                                # Fallback without APOC - using dynamic relationship types is tricky
                                # We'll collect all relationship types first
                                get_rel_types_query = """
                                MATCH (duplicate)
                                WHERE elementId(duplicate) = $duplicate_id
                                MATCH (duplicate)-[r]->(target)
                                RETURN DISTINCT type(r) AS rel_type,
                                       elementId(target) AS target_id,
                                       properties(r) AS props
                                """
                                rel_types_result = session.run(
                                    get_rel_types_query,
                                    duplicate_id=dup["element_id"]
                                )

                                # Create each relationship individually
                                for rel_record in rel_types_result:
                                    rel_type = rel_record["rel_type"]
                                    target_id = rel_record["target_id"]
                                    props = rel_record["props"]

                                    # Use MERGE to avoid duplicates, dynamic relationship type requires APOC
                                    # So we'll use a generic approach
                                    create_rel_query = f"""
                                    MATCH (canonical)
                                    WHERE elementId(canonical) = $canonical_id
                                    MATCH (target)
                                    WHERE elementId(target) = $target_id
                                    MERGE (canonical)-[r:`{rel_type}`]->(target)
                                    SET r = $props
                                    """
                                    session.run(
                                        create_rel_query,
                                        canonical_id=canonical["element_id"],
                                        target_id=target_id,
                                        props=props
                                    )

                            # Step 2: Transfer incoming relationships
                            if in_count > 0:
                                get_in_rel_types_query = """
                                MATCH (duplicate)
                                WHERE elementId(duplicate) = $duplicate_id
                                MATCH (source)-[r]->(duplicate)
                                RETURN DISTINCT type(r) AS rel_type,
                                       elementId(source) AS source_id,
                                       properties(r) AS props
                                """
                                in_rel_types_result = session.run(
                                    get_in_rel_types_query,
                                    duplicate_id=dup["element_id"]
                                )

                                for rel_record in in_rel_types_result:
                                    rel_type = rel_record["rel_type"]
                                    source_id = rel_record["source_id"]
                                    props = rel_record["props"]

                                    create_in_rel_query = f"""
                                    MATCH (canonical)
                                    WHERE elementId(canonical) = $canonical_id
                                    MATCH (source)
                                    WHERE elementId(source) = $source_id
                                    MERGE (source)-[r:`{rel_type}`]->(canonical)
                                    SET r = $props
                                    """
                                    session.run(
                                        create_in_rel_query,
                                        canonical_id=canonical["element_id"],
                                        source_id=source_id,
                                        props=props
                                    )

                            # Step 3: Copy properties and delete duplicate
                            merge_props_query = """
                            MATCH (canonical)
                            WHERE elementId(canonical) = $canonical_id
                            MATCH (duplicate)
                            WHERE elementId(duplicate) = $duplicate_id
                            SET canonical += properties(duplicate)
                            WITH duplicate
                            DETACH DELETE duplicate
                            """
                            session.run(
                                merge_props_query,
                                canonical_id=canonical["element_id"],
                                duplicate_id=dup["element_id"],
                            )

                            rel_info = f" ({out_count} out, {in_count} in)" if (out_count + in_count) > 0 else ""
                            print(f"    * Merged: {dup['name']}{rel_info}")
                            total_merged += 1

                        except Exception as e:
                            print(f"    ✗ Error merging {dup['name']}: {e}")
                            continue

        if dry_run:
            print(f"\n Would merge {total_merged} duplicate nodes")
        else:
            print(f"\n Successfully merged {total_merged} duplicate nodes")

        return total_merged


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate entity nodes in Neo4j knowledge graph"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.90,
        help="Cosine similarity threshold for clustering (0-1, default: 0.90)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be merged without actually merging",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Only extract and cluster, don't merge (useful for analysis)",
    )

    args = parser.parse_args()

    # load config from environment
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")

    if not neo4j_password:
        print("ERROR: NEO4J_PASSWORD not set in environment")
        sys.exit(1)

    print("\n" + "="*70)
    print("Neo4j Node Deduplication Pipeline")
    print("="*70)
    print(f"Neo4j URI: {neo4j_uri}")
    print(f"Database: {neo4j_database}")
    print(f"Similarity threshold: {args.similarity_threshold}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE MERGE'}")

    try:
        with Neo4jDeduplicator(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            database=neo4j_database,
        ) as deduplicator:

            # Step 1: Extract nodes
            nodes = deduplicator.extract_entity_nodes()

            if not nodes:
                print("\nNo entity nodes found to deduplicate")
                return

            # Step 2: Embed node names
            embeddings = deduplicator.embed_node_names(nodes)

            # Step 3: Cluster similar nodes
            clusters = deduplicator.cluster_similar_nodes(
                nodes, embeddings, args.similarity_threshold
            )

            if not clusters:
                print("\nNo duplicate clusters found")
                return

            # Show detailed statistics
            deduplicator.compute_cluster_statistics(clusters, nodes, embeddings)

            # Step 4: Merge duplicates (if not in analysis-only mode)
            if not args.no_merge:
                deduplicator.merge_duplicate_nodes(
                    clusters, nodes, dry_run=args.dry_run
                )

            print("\n" + "="*70)
            print("Deduplication complete!")
            print("="*70)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
