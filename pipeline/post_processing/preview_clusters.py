#!/usr/bin/env python3
"""
Preview Duplicate Clusters
===========================

Interactive tool to preview and filter duplicate clusters before merging.
Helps find the optimal similarity threshold.

Usage:
    # Preview with default threshold (0.90)
    python preview_clusters.py

    # Try different thresholds
    python preview_clusters.py --similarity-threshold 0.80

    # Filter by label
    python preview_clusters.py --label Disease

    # Show only large clusters
    python preview_clusters.py --min-cluster-size 5
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

load_dotenv()


def extract_and_cluster(
    uri: str,
    username: str,
    password: str,
    database: str,
    similarity_threshold: float,
    label_filter: str = None,
):
    """Extract nodes, embed, and cluster them."""

    # Connect to Neo4j
    driver = GraphDatabase.driver(uri, auth=(username, password))
    embeddings_model = OllamaEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    # Extract nodes
    print("Extracting nodes from Neo4j...")
    query = """
    MATCH (n)
    WHERE NOT n:Document
    RETURN elementId(n) AS element_id,
           labels(n) AS labels,
           n.id AS name,
           properties(n) AS properties
    """

    with driver.session(database=database) as session:
        result = session.run(query)
        nodes = []

        for record in result:
            element_id = record["element_id"]
            labels = record["labels"]
            name = record["name"]
            properties = record["properties"]

            if not name:
                continue

            # Apply label filter if specified
            primary_label = labels[0] if labels else "Entity"
            if label_filter and primary_label != label_filter:
                continue

            nodes.append({
                "element_id": element_id,
                "label": primary_label,
                "all_labels": labels,
                "name": str(name),
                "properties": properties,
            })

    driver.close()

    if not nodes:
        print(f"No nodes found{f' with label {label_filter}' if label_filter else ''}")
        return [], []

    print(f"Extracted {len(nodes)} nodes")

    # Embed
    print("Embedding node names...")
    node_names = [node["name"] for node in nodes]
    batch_size = 100
    all_embeddings = []

    for i in range(0, len(node_names), batch_size):
        batch = node_names[i:i + batch_size]
        batch_embeddings = embeddings_model.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)

    embeddings_array = np.array(all_embeddings)
    print(f"Embedded {len(node_names)} nodes")

    # Cluster
    print("Clustering...")
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    normalized_embeddings = embeddings_array / (norms + 1e-10)
    distances = pdist(normalized_embeddings, metric='cosine')
    linkage_matrix = linkage(distances, method='average')
    distance_threshold = 1 - similarity_threshold
    cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')

    # Group nodes by cluster
    clusters_dict = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_labels):
        clusters_dict[cluster_id].append(idx)

    # Filter to clusters with 2+ nodes
    duplicate_clusters = [
        cluster for cluster in clusters_dict.values()
        if len(cluster) >= 2
    ]

    # Compute cluster statistics
    cluster_info = []
    for cluster_indices in duplicate_clusters:
        cluster_nodes = [nodes[idx] for idx in cluster_indices]
        cluster_embeddings = embeddings_array[cluster_indices]

        # Compute similarities
        norms = np.linalg.norm(cluster_embeddings, axis=1, keepdims=True)
        normalized = cluster_embeddings / (norms + 1e-10)
        similarity_matrix = normalized @ normalized.T
        mask = ~np.eye(len(cluster_indices), dtype=bool)

        mean_sim = similarity_matrix[mask].mean()
        min_sim = similarity_matrix[mask].min()

        cluster_info.append({
            "nodes": cluster_nodes,
            "mean_similarity": mean_sim,
            "min_similarity": min_sim,
            "size": len(cluster_nodes),
        })

    # Sort by cluster size (largest first)
    cluster_info.sort(key=lambda x: x["size"], reverse=True)

    return cluster_info, nodes


def display_clusters(
    cluster_info: List[Dict],
    min_cluster_size: int = 2,
    show_all: bool = False,
):
    """Display cluster information in a readable format."""

    # Filter by size
    filtered_clusters = [
        c for c in cluster_info
        if c["size"] >= min_cluster_size
    ]

    if not filtered_clusters:
        print(f"\nNo clusters found with size >= {min_cluster_size}")
        return

    print("\n" + "=" * 70)
    print(f"DUPLICATE CLUSTERS (showing {len(filtered_clusters)} clusters)")
    print("=" * 70)

    # Summary statistics
    total_duplicates = sum(c["size"] - 1 for c in filtered_clusters)
    print(f"\nTotal duplicate nodes that would be merged: {total_duplicates}")
    print(f"Canonical nodes that would remain: {len(filtered_clusters)}")

    # Show clusters
    num_to_show = len(filtered_clusters) if show_all else min(20, len(filtered_clusters))

    for i, cluster in enumerate(filtered_clusters[:num_to_show]):
        print(f"\n{'─' * 70}")
        print(f"Cluster {i+1} ({cluster['size']} nodes)")
        print(f"  Mean similarity: {cluster['mean_similarity']:.3f}")
        print(f"  Min similarity:  {cluster['min_similarity']:.3f}")
        print(f"  Nodes:")

        # Sort nodes by name length (canonical will be longest)
        sorted_nodes = sorted(cluster["nodes"], key=lambda n: len(n["name"]), reverse=True)
        canonical = sorted_nodes[0]
        duplicates = sorted_nodes[1:]

        print(f"    → CANONICAL: {canonical['name']} [{canonical['label']}]")
        for dup in duplicates:
            print(f"      • {dup['name']} [{dup['label']}]")

    if not show_all and len(filtered_clusters) > num_to_show:
        remaining = len(filtered_clusters) - num_to_show
        print(f"\n... and {remaining} more clusters")
        print(f"\nUse --show-all to see all clusters")


def main():
    parser = argparse.ArgumentParser(
        description="Preview duplicate clusters before merging"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.90,
        help="Cosine similarity threshold (0-1, default: 0.90)",
    )
    parser.add_argument(
        "--label",
        type=str,
        help="Filter to specific node label (e.g., 'Disease', 'Protein')",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="Minimum cluster size to display (default: 2)",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all clusters (not just first 20)",
    )

    args = parser.parse_args()

    # Load config
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")

    if not neo4j_password:
        print("ERROR: NEO4J_PASSWORD not set in environment")
        sys.exit(1)

    print("=" * 70)
    print("Preview Duplicate Clusters")
    print("=" * 70)
    print(f"Similarity threshold: {args.similarity_threshold}")
    if args.label:
        print(f"Label filter: {args.label}")
    print(f"Min cluster size: {args.min_cluster_size}")

    try:
        cluster_info, nodes = extract_and_cluster(
            neo4j_uri,
            neo4j_username,
            neo4j_password,
            neo4j_database,
            args.similarity_threshold,
            args.label,
        )

        display_clusters(
            cluster_info,
            args.min_cluster_size,
            args.show_all,
        )

        # Show label distribution
        if cluster_info:
            print("\n" + "=" * 70)
            print("LABEL DISTRIBUTION IN CLUSTERS")
            print("=" * 70)

            label_counts = defaultdict(int)
            for cluster in cluster_info:
                for node in cluster["nodes"]:
                    label_counts[node["label"]] += 1

            for label, count in sorted(label_counts.items(), key=lambda x: -x[1])[:10]:
                print(f"  {label}: {count}")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
